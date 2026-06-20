import torch
import numpy as np
from mgrpo.config import (
    USE_TWO_ATTENTION_HEADS, ATTN_LR, CANDIDATE_FEATURE_DIM, 
    N_MODALITIES, ATTN_REWARD_TEMP, ATTN_ENTROPY_LAMBDA, GRAD_CLIP,
    GENERATION_BATCH_SIZE, GROUP_SIZE, MODALITY_MAP, MODALITY_PROFILES
)
from mgrpo.models.generator import module_device

class AttentionTrainer:
    def __init__(self, generation_head, preference_head):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_head = generation_head
        self.preference_head = preference_head
        if USE_TWO_ATTENTION_HEADS:
            self.optimizer = torch.optim.AdamW(
                list(self.generation_head.parameters()) + list(self.preference_head.parameters()),
                lr=ATTN_LR,
            )
        else:
            self.optimizer = None

    def zscore_np(self, values):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0 or float(arr.std()) < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - arr.mean()) / (arr.std() + 1e-6)).astype(np.float32)

    def preference_target_distribution(self, shaped_rewards):
        rewards = torch.tensor(shaped_rewards, dtype=torch.float32, device=self.device)
        if rewards.numel() == 1:
            return torch.ones_like(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        return torch.softmax(rewards / ATTN_REWARD_TEMP, dim=0)

    def build_candidate_features(self, mod_ids, clf_probs, gen_scores, raw_rewards, shaped_rewards, guard_penalties, responses):
        raw_z = self.zscore_np(raw_rewards)
        shaped_z = self.zscore_np(shaped_rewards)
        guards = np.asarray(guard_penalties, dtype=np.float32)
        lengths = np.asarray([len((r or "").split()) for r in responses], dtype=np.float32)
        lengths = np.log1p(lengths) / np.log1p(max(float(lengths.max()), 1.0))

        rows = []
        for i, mid in enumerate(mod_ids):
            one_hot = np.zeros(N_MODALITIES, dtype=np.float32)
            one_hot[int(mid)] = 1.0
            extra = np.array([
                float(clf_probs[int(mid)]),
                float(gen_scores[int(mid)]),
                float(raw_z[i]),
                float(shaped_z[i]),
                float(guards[i]),
                float(lengths[i]),
            ], dtype=np.float32)
            rows.append(np.concatenate([one_hot, extra], axis=0))
        return torch.tensor(np.stack(rows), dtype=torch.float32, device=self.device)

    def preference_attention_distribution(self, candidate_features, shaped_rewards):
        target = self.preference_target_distribution(shaped_rewards)
        if not (USE_TWO_ATTENTION_HEADS and self.preference_head is not None):
            return target.detach().cpu().numpy().astype(np.float32), 0.0
        with torch.no_grad():
            weights, gate, _, _ = self.preference_head(candidate_features, target)
        return weights.detach().cpu().numpy().astype(np.float32), float(gate.detach().cpu())

    def train_attention_heads(self, mod_ids, clf_probs, gen_scores, raw_rewards, shaped_rewards, guard_penalties, responses):
        candidate_features = self.build_candidate_features(
            mod_ids, clf_probs, gen_scores, raw_rewards, shaped_rewards, guard_penalties, responses
        )
        target = self.preference_target_distribution(shaped_rewards).detach()
        info = {"gen_loss": 0.0, "pref_loss": 0.0, "gen_gate": 0.0, "pref_gate": 0.0}

        if not (USE_TWO_ATTENTION_HEADS and self.optimizer is not None):
            weights = target.detach().cpu().numpy().astype(np.float32)
            return candidate_features, weights, info

        self.optimizer.zero_grad(set_to_none=True)
        losses = []

        gen_dist, gen_gate, _, _ = self.generation_head(torch.tensor(clf_probs, dtype=torch.float32, device=self.device))
        idx_t = torch.tensor(mod_ids, dtype=torch.long, device=self.device)
        selected_gen = gen_dist[idx_t]
        selected_gen = selected_gen / selected_gen.sum().clamp_min(1e-8)
        gen_loss = -(target * torch.log(selected_gen + 1e-8)).sum()
        losses.append(gen_loss)
        info["gen_loss"] = float(gen_loss.detach().cpu())
        info["gen_gate"] = float(gen_gate.detach().cpu())

        pref_weights, pref_gate, _, _ = self.preference_head(candidate_features, target)
        pref_entropy = -(pref_weights * torch.log(pref_weights + 1e-8)).sum()
        pref_loss = -(target * torch.log(pref_weights + 1e-8)).sum() - ATTN_ENTROPY_LAMBDA * pref_entropy
        losses.append(pref_loss)
        info["pref_loss"] = float(pref_loss.detach().cpu())
        info["pref_gate"] = float(pref_gate.detach().cpu())

        total = torch.stack(losses).mean()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.generation_head.parameters()) + list(self.preference_head.parameters()),
            GRAD_CLIP,
        )
        self.optimizer.step()

        weights, pref_gate_after = self.preference_attention_distribution(candidate_features, shaped_rewards)
        info["pref_gate"] = pref_gate_after
        with torch.no_grad():
            _, gen_gate_after, _, _ = self.generation_head(torch.tensor(clf_probs, dtype=torch.float32, device=self.device))
        info["gen_gate"] = float(gen_gate_after.detach().cpu())
        return candidate_features, weights, info

