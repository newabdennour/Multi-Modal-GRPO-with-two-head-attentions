import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from mgrpo.config import PREF_MODEL, REWARD_BATCH_SIZE, PREF_MAX_TOKENS, MAX_PARALLEL_WORKERS, MIN_RESPONSE_TOKENS
from mgrpo.models.generator import module_device, to_device, is_cuda_oom

class RewardModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(PREF_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(PREF_MODEL).to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.enabled = True
        except Exception as e:
            print(f"Failed to load reward model: {e}")
            self.enabled = False

    def preference_rewards(self, user_prompt, responses):
        if not responses or not self.enabled:
            return [0.0] * len(responses)

        texts = [user_prompt.strip() + "\n\n" + (r or "").strip() for r in responses]
        scores = []
        pref_device = module_device(self.model)

        batch_size = max(1, min(REWARD_BATCH_SIZE, len(texts)))
        start = 0
        while start < len(texts):
            batch_texts = texts[start : start + batch_size]
            try:
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=PREF_MAX_TOKENS,
                )
                inputs = to_device(inputs, pref_device)
                with torch.inference_mode():
                    logits = self.model(**inputs).logits.float()
                    if logits.ndim == 2 and logits.shape[-1] > 1:
                        vals = logits[:, -1]
                    else:
                        vals = logits.view(-1)
            except RuntimeError as exc:
                if torch.cuda.is_available() and is_cuda_oom(exc) and batch_size > 1:
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    print(f"CUDA OOM during reward scoring; retrying with reward batch={batch_size}")
                    continue
                raise

            scores.extend(vals.detach().cpu().tolist())
            start += len(batch_texts)

        return [float(s) for s in scores]

def guard_penalty(user_prompt, response):
    response = (response or "").strip()
    words = response.split()
    penalty = 0.0

    if not response:
        penalty -= 4.0
    if len(words) < MIN_RESPONSE_TOKENS:
        penalty -= 2.0

    prompt_head = user_prompt.strip().lower()[:80]
    if prompt_head and prompt_head in response.lower():
        penalty -= 1.0

    lines = [line.strip().lower() for line in response.splitlines() if line.strip()]
    if len(lines) >= 4:
        repeated_lines = len(lines) - len(set(lines))
        penalty -= min(1.5, 0.25 * repeated_lines)

    if len(words) >= 30:
        distinct_ratio = len(set(w.lower() for w in words)) / max(1, len(words))
        if distinct_ratio < 0.35:
            penalty -= min(2.0, (0.35 - distinct_ratio) * 6.0)

    return float(penalty)

def score_responses(reward_model, user_prompt, responses):
    raw = reward_model.preference_rewards(user_prompt, responses)
    if len(responses) > 1 and MAX_PARALLEL_WORKERS > 1:
        workers = min(MAX_PARALLEL_WORKERS, len(responses))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            guards = list(pool.map(lambda r: guard_penalty(user_prompt, r), responses))
    else:
        guards = [guard_penalty(user_prompt, r) for r in responses]
    shaped = [r + g for r, g in zip(raw, guards)]
    return raw, shaped, guards
