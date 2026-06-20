import torch
import numpy as np
import random
from contextlib import nullcontext
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
)
from peft import LoraConfig, TaskType, get_peft_model
from mgrpo.config import (
    GEN_MODEL, HF_TOKEN, USE_LORA, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    LR, WEIGHT_DECAY, MAX_PROMPT_TOKENS, SYSTEM_PROMPT
)

def module_device(module):
    try:
        return module.device
    except Exception:
        return next(module.parameters()).device

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def is_cuda_oom(exc):
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

def set_generation_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model_prompt(user_prompt, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

class GeneratorModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, token=HF_TOKEN)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL,
            torch_dtype=self.dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            token=HF_TOKEN,
        )
        self.model.config.use_cache = True

        if USE_LORA:
            lora_cfg = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()
        else:
            for p in self.model.parameters():
                p.requires_grad_(True)

        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

        base_config = self.model.get_base_model().config if hasattr(self.model, "get_base_model") else self.model.config
        self.GEN_HIDDEN = int(base_config.hidden_size)

    def adapter_disabled(self):
        if USE_LORA and hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        return nullcontext()

    @torch.inference_mode()
    def generate_responses_batch(self, user_prompt, profile, count=1, seed=None, batch_size=None):
        count = int(count)
        if count <= 0:
            return []
        if seed is not None:
            set_generation_seed(seed)

        batch_size = count if batch_size is None else max(1, min(int(batch_size), count))
        was_training = self.model.training
        self.model.eval()
        prompt_text = build_model_prompt(user_prompt, self.tokenizer)
        gen_device = module_device(self.model)
        responses = []

        try:
            start = 0
            while start < count:
                current_batch = min(batch_size, count - start)
                try:
                    inputs = self.tokenizer(
                        [prompt_text] * current_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=MAX_PROMPT_TOKENS,
                    )
                    inputs = to_device(inputs, gen_device)

                    out = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        do_sample=True,
                        temperature=float(profile.get("temperature", 0.65)),
                        top_p=float(profile.get("top_p", 0.90)),
                        repetition_penalty=float(profile.get("repetition_penalty", 1.05)),
                        max_new_tokens=int(profile.get("max_new_tokens", 160)),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        logits_processor=LogitsProcessorList([InfNanRemoveLogitsProcessor()]),
                    )
                except RuntimeError as exc:
                    if torch.cuda.is_available() and is_cuda_oom(exc) and batch_size > 1:
                        torch.cuda.empty_cache()
                        batch_size = max(1, batch_size // 2)
                        print(f"CUDA OOM during generation; retrying with generation batch={batch_size}")
                        continue
                    raise

                prompt_len = inputs["input_ids"].shape[1]
                for row in range(out.shape[0]):
                    new_ids = out[row, prompt_len:]
                    responses.append(self.tokenizer.decode(new_ids, skip_special_tokens=True).strip())
                start += current_batch
        finally:
            if was_training:
                self.model.train()

        return responses
