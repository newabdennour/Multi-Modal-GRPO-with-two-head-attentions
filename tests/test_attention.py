import torch
import pytest
from mgrpo.models.attention import ClassifierPromptGenerationAttention, UserPreferenceUpdateAttention
from mgrpo.config import N_MODALITIES, ATTN_HEAD_DIM

def test_generation_attention():
    head = ClassifierPromptGenerationAttention(n_modalities=N_MODALITIES, dim=ATTN_HEAD_DIM)
    clf_probs = torch.rand(N_MODALITIES)
    clf_probs /= clf_probs.sum()
    
    dist, gate, learned, attn_map = head(clf_probs)
    
    assert dist.shape == (N_MODALITIES,)
    assert torch.isclose(dist.sum(), torch.tensor(1.0))
    assert 0.0 <= gate <= 1.0

def test_preference_attention():
    feature_dim = N_MODALITIES + 6
    head = UserPreferenceUpdateAttention(feature_dim=feature_dim, dim=ATTN_HEAD_DIM)
    
    # Batch of 5 candidates
    features = torch.rand(5, feature_dim)
    reward_prior = torch.rand(5)
    reward_prior /= reward_prior.sum()
    
    weights, gate, learned, attn_map = head(features, reward_prior)
    
    assert weights.shape == (5,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert 0.0 <= gate <= 1.0
