import torch
import torch.nn as nn
from mgrpo.config import N_MODALITIES, ATTN_HEAD_DIM, ATTN_NUM_HEADS

class ClassifierPromptGenerationAttention(nn.Module):
    """Attends over modality slots conditioned on classifier probabilities."""
    def __init__(self, n_modalities=N_MODALITIES, dim=ATTN_HEAD_DIM, heads=ATTN_NUM_HEADS):
        super().__init__()
        self.mod_emb = nn.Embedding(n_modalities, dim)
        self.prob_proj = nn.Linear(1, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.gate_logit = nn.Parameter(torch.tensor(-0.5))

    def forward(self, clf_probs):
        dev = self.mod_emb.weight.device
        probs = clf_probs.to(dev, dtype=torch.float32)
        probs = probs / probs.sum().clamp_min(1e-8)
        ids = torch.arange(probs.numel(), device=dev).unsqueeze(0)
        x = self.mod_emb(ids) + self.prob_proj(probs.view(1, -1, 1))
        y, attn_map = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        h = self.norm(x + y)
        learned = torch.softmax(self.score(h).squeeze(0).squeeze(-1), dim=-1)
        gate = torch.sigmoid(self.gate_logit)
        out = (1.0 - gate) * probs + gate * learned
        return out / out.sum().clamp_min(1e-8), gate, learned, attn_map


class UserPreferenceUpdateAttention(nn.Module):
    """Attends over candidate responses using preference/reward features."""
    def __init__(self, feature_dim, dim=ATTN_HEAD_DIM, heads=ATTN_NUM_HEADS):
        super().__init__()
        self.in_proj = nn.Linear(feature_dim, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.gate_logit = nn.Parameter(torch.tensor(-0.5))

    def forward(self, features, reward_prior):
        features = features.to(next(self.parameters()).device, dtype=torch.float32)
        reward_prior = reward_prior.to(features.device, dtype=torch.float32)
        reward_prior = reward_prior / reward_prior.sum().clamp_min(1e-8)
        x = self.in_proj(features).unsqueeze(0)
        y, attn_map = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        h = self.norm(x + y)
        learned = torch.softmax(self.score(h).squeeze(0).squeeze(-1), dim=-1)
        gate = torch.sigmoid(self.gate_logit)
        weights = (1.0 - gate) * reward_prior + gate * learned
        return weights / weights.sum().clamp_min(1e-8), gate, learned, attn_map
