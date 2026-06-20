import os
from pathlib import Path

# Paths and models
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CLASSIFIER_PATH = "/mnt/main/prompt_classifier"
PREF_MODEL = "OpenAssistant/reward-model-deberta-v3-large-v2"
DATA_PATH = Path("prompts.jsonl")
OUTPUT_DIR = Path("two_head_attention_grpo_lora")

# Main controls
SEED = 42
DO_TRAIN = True
MAX_TRAIN_STEPS = None
DATASET_FRACTION = 0.10
TRAIN_FRACTION = 0.90
GROUP_SIZE = 5
GRPO_BASELINE_GROUP_SIZE = GROUP_SIZE
SAVE_EVERY = 100
PRINT_EVERY = 1
SUMMARY_EVERY = 10

# Generation and scoring
MAX_PROMPT_TOKENS = 1024
TRAIN_MAX_TOKENS = 1536
PREF_MAX_TOKENS = 512
REWARD_BATCH_SIZE = max(8, GROUP_SIZE + GRPO_BASELINE_GROUP_SIZE)
GENERATION_BATCH_SIZE = max(GROUP_SIZE, GRPO_BASELINE_GROUP_SIZE)
CPU_CORES = max(1, os.cpu_count() or 1)
MAX_PARALLEL_WORKERS = CPU_CORES
MIN_RESPONSE_TOKENS = 4

# LoRA policy update
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LR = 5e-5
WEIGHT_DECAY = 0.0
EPS = 0.20
KL_BETA = 0.03
GRAD_CLIP = 1.0
ATTENTION_TOPK_UPDATES = 2

# Attention heads
USE_TWO_ATTENTION_HEADS = True
ATTN_HEAD_DIM = 128
ATTN_NUM_HEADS = 4
ATTN_LR = 1e-4
ATTN_REWARD_TEMP = 0.75
ATTN_ENTROPY_LAMBDA = 0.005
UNIFORM_CANDIDATE_WEIGHT = 1.0 / GROUP_SIZE

# Modalities
NORMAL_MODALITY = 7
N_MODALITIES = 11

MODALITY_MAP = {
    0: "Concise",
    1: "Detailed",
    2: "Advanced",
    3: "Code_base",
    4: "Creative",
    5: "Formal",
    6: "Informal",
    7: "Normal",
    8: "Precise",
    9: "Reasoning",
    10: "Simple",
}

MODALITY_PROFILES = {
    0: dict(temperature=0.45, top_p=0.82, max_new_tokens=96, repetition_penalty=1.08),
    1: dict(temperature=0.70, top_p=0.90, max_new_tokens=192, repetition_penalty=1.05),
    2: dict(temperature=0.62, top_p=0.88, max_new_tokens=224, repetition_penalty=1.04),
    3: dict(temperature=0.35, top_p=0.82, max_new_tokens=256, repetition_penalty=1.03),
    4: dict(temperature=0.95, top_p=0.96, max_new_tokens=192, repetition_penalty=1.04),
    5: dict(temperature=0.50, top_p=0.86, max_new_tokens=160, repetition_penalty=1.06),
    6: dict(temperature=0.75, top_p=0.92, max_new_tokens=160, repetition_penalty=1.04),
    7: dict(temperature=0.65, top_p=0.90, max_new_tokens=160, repetition_penalty=1.05),
    8: dict(temperature=0.38, top_p=0.82, max_new_tokens=160, repetition_penalty=1.08),
    9: dict(temperature=0.55, top_p=0.88, max_new_tokens=256, repetition_penalty=1.04),
    10: dict(temperature=0.42, top_p=0.84, max_new_tokens=128, repetition_penalty=1.08),
}

SYSTEM_PROMPT = "You are a helpful assistant."
