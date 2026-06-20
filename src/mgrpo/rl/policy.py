import torch
import math
import numpy as np
from mgrpo.models.generator import module_device, build_model_prompt
from mgrpo.config import TRAIN_MAX_TOKENS, USE_LORA, EPS, KL_BETA, GRAD_CLIP, ATTENTION_TOPK_UPDATES

def trainable_l2_norm(params):
    total = 0.0
    with torch.no_grad():
        for p in params:
            total += float(p.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)

def build_train_tensors(user_prompt, response, gen_model):
    prompt_text = build_model_prompt(user_prompt, gen_model.tokenizer)
    prompt_ids = gen_model.tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_ids = gen_model.tokenizer((response or "").strip(), add_special_tokens=False).input_ids
    if gen_model.tokenizer.eos_token_id is not None:
        response_ids = response_ids + [gen_model.tokenizer.eos_token_id]
    if not response_ids:
        response_ids = [gen_model.tokenizer.eos_token_id or gen_model.tokenizer.pad_token_id]

    base_config = gen_model.model.get_base_model().config if hasattr(gen_model.model, "get_base_model") else gen_model.model.config
    max_total = min(TRAIN_MAX_TOKENS, getattr(base_config, "max_position_embeddings", TRAIN_MAX_TOKENS))
    
    if len(response_ids) >= max_total:
        response_ids = response_ids[: max_total - 1]
    keep_prompt = max(1, max_total - len(response_ids))
    if len(prompt_ids) > keep_prompt:
        prompt_ids = prompt_ids[-keep_prompt:]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)
    dev = module_device(gen_model.model)
    return (
        torch.tensor([input_ids], dtype=torch.long, device=dev),
        torch.tensor([attention_mask], dtype=torch.long, device=dev),
        torch.tensor([labels], dtype=torch.long, device=dev),
    )

def response_mean_logprob(user_prompt, response, gen_model, disable_adapter=False):
    input_ids, attention_mask, labels = build_train_tensors(user_prompt, response, gen_model)
    ctx = gen_model.adapter_disabled() if disable_adapter else gen_model.adapter_disabled()
    if not disable_adapter:
        ctx = torch.autograd.profiler.profile(enabled=False) # dummy ctx
        
    # Python 3.9+ nullcontext if we don't want to disable adapter
    from contextlib import nullcontext
    ctx = gen_model.adapter_disabled() if disable_adapter else nullcontext()
    
    with ctx:
        out = gen_model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    logits = out.logits[:, :-1, :].float()
    target_ids = input_ids[:, 1:]
    valid = labels[:, 1:] != -100
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    denom = valid.sum().clamp_min(1)
    return (token_logp * valid).sum() / denom

def normalize_advantages(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 1 or float(arr.std()) < 1e-6:
        out = np.zeros_like(arr)
        out[int(np.argmax(arr))] = 1.0
        return out
    lo, hi = np.percentile(arr, [5, 95])
    arr = np.clip(arr, lo, hi)
    return (arr - arr.mean()) / (arr.std() + 1e-6)

def select_policy_indices(shaped_rewards, preference_weights):
    rewards = np.asarray(shaped_rewards, dtype=np.float32)
    pref = np.asarray(preference_weights, dtype=np.float32)
    pref = pref / (pref.sum() + 1e-8)
    best_idx = int(np.argmax(rewards))
    positive_adv = np.maximum(normalize_advantages(rewards), 0.0)
    if float(positive_adv.max()) <= 1e-8:
        positive_adv[best_idx] = 1.0
    gains = positive_adv * np.maximum(0.10, pref / (pref.mean() + 1e-8))
    gains[best_idx] = max(gains[best_idx], 1.0)
    gains = gains / (gains.max() + 1e-8)
    ranked = list(np.argsort(gains)[::-1])
    train_indices = [int(i) for i in ranked if gains[i] > 1e-8][: max(1, ATTENTION_TOPK_UPDATES)]
    return train_indices, gains.astype(np.float32), best_idx

def update_policy(user_prompt, responses, shaped_rewards, preference_weights, gen_model):
    if not responses:
        return {"loss": 0.0, "trained": 0, "best_idx": None, "grad_norm": 0.0, "adapter_delta": 0.0}

    train_indices, advantages, best_idx = select_policy_indices(shaped_rewards, preference_weights)
    old_logps = {}
    base_logps = {}

    gen_model.model.eval()
    with torch.no_grad():
        for idx in train_indices:
            old_logps[idx] = response_mean_logprob(user_prompt, responses[idx], gen_model).detach()
            if USE_LORA:
                base_logps[idx] = response_mean_logprob(user_prompt, responses[idx], gen_model, disable_adapter=True).detach()
            else:
                base_logps[idx] = old_logps[idx]

    gen_model.model.train()
    gen_model.optimizer.zero_grad(set_to_none=True)
    losses = []
    for idx in train_indices:
        adv = float(advantages[idx])
        if abs(adv) < 1e-8:
            continue
        cur_logp = response_mean_logprob(user_prompt, responses[idx], gen_model)
        old_logp = old_logps[idx].to(cur_logp.device)
        base_logp = base_logps[idx].to(cur_logp.device)
        ratio = torch.exp(torch.clamp(cur_logp - old_logp, min=-5.0, max=5.0))
        adv_t = torch.tensor(adv, dtype=torch.float32, device=cur_logp.device)
        unclipped = ratio * adv_t
        clipped = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS) * adv_t
        policy_loss = -torch.minimum(unclipped, clipped)
        base_anchor = (cur_logp - base_logp).pow(2)
        losses.append(policy_loss + KL_BETA * base_anchor)

    if not losses:
        return {"loss": 0.0, "trained": 0, "best_idx": best_idx, "grad_norm": 0.0, "adapter_delta": 0.0}

    norm_before = trainable_l2_norm(gen_model.trainable_params)
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    grad_norm_t = torch.nn.utils.clip_grad_norm_(gen_model.trainable_params, GRAD_CLIP)
    gen_model.optimizer.step()
    norm_after = trainable_l2_norm(gen_model.trainable_params)
    return {
        "loss": float(total_loss.detach().cpu()),
        "trained": len(losses),
        "best_idx": best_idx,
        "grad_norm": float(grad_norm_t.detach().float().cpu()),
        "adapter_delta": norm_after - norm_before,
    }
