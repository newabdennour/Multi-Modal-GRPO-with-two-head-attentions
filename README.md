# MGRPO — Two-Head Attention Prefix

GRPO training loop for causal language models where candidate selection and policy weighting are handled by two small learnable attention heads, trained online from the same RL batch. The user prompt is never rewritten or prefixed.

---

## What this is

Standard GRPO draws a group of completions under a fixed generation profile and steps the policy toward higher-reward responses. This notebook replaces that fixed profile with a learned routing mechanism: before sampling, a **generation attention head** picks which of 11 sampling profiles to try based on a prompt classifier's output distribution. Before the gradient step, a **preference update attention head** re-weights the candidates using reward features.

Both heads use a gated blend — at initialisation the gate is near zero so they stay close to their respective priors (classifier probabilities and reward rank). The gate grows as training progresses, transferring influence from the prior to the learned scores gradually.

The prompt text is never touched. No modality name, style instruction, or soft prefix is inserted. `build_model_prompt` receives only the raw user message and a fixed system prompt. Any reward gain over the GRPO baseline comes from sampling diversity and candidate selection, not prompt engineering.

---

## Architecture

```
user prompt
    │
    ├── prompt classifier ──► p(modality | prompt)
    │                                │
    │              ClassifierPromptGenerationAttention
    │              out = (1 - gate) · clf_probs + gate · learned_scores
    │                                │
    │              top-N profiles selected  (default N = 5)
    │
    ├── Qwen2.5-0.5B-Instruct + LoRA  ×  N profiles ──► N responses
    │
    ├── reward model (DeBERTa-v3) ──► r_i per response
    │
    │              UserPreferenceUpdateAttention
    │              out = (1 - gate) · softmax(r / τ) + gate · learned_weights
    │                                │
    │              per-candidate weights for the policy gradient
    │
    └── LoRA update — clipped PPO objective + KL anchor vs frozen base
```

The reference log-probability for the KL term is computed by calling `generator.disable_adapter()` on the live LoRA model. No second model copy is loaded.

---

## Modality profiles

Eleven sampling profiles are defined. They control decoding parameters only — nothing is injected into the prompt.

| ID | Name | Temperature | Top-p | Max new tokens | Rep. penalty |
|----|------|-------------|-------|----------------|--------------|
| 0  | Concise   | 0.45 | 0.82 | 96  | 1.08 |
| 1  | Detailed  | 0.70 | 0.90 | 192 | 1.05 |
| 2  | Advanced  | 0.62 | 0.88 | 224 | 1.04 |
| 3  | Code_base | 0.35 | 0.82 | 256 | 1.03 |
| 4  | Creative  | 0.95 | 0.96 | 192 | 1.04 |
| 5  | Formal    | 0.50 | 0.86 | 160 | 1.06 |
| 6  | Informal  | 0.75 | 0.92 | 160 | 1.04 |
| 7  | Normal    | 0.65 | 0.90 | 160 | 1.05 |
| 8  | Precise   | 0.38 | 0.82 | 160 | 1.08 |
| 9  | Reasoning | 0.55 | 0.88 | 256 | 1.04 |
| 10 | Simple    | 0.42 | 0.84 | 128 | 1.08 |

---

## Setup

```bash
pip install datasets transformers accelerate peft
```

Set your Hugging Face token before launching:

```bash
export HF_TOKEN=hf_...
```

Public models load without a token. The default generator (`Qwen/Qwen2.5-0.5B-Instruct`) is public. The reward model (`OpenAssistant/reward-model-deberta-v3-large-v2`) is also public.

Minimum ~8 GB VRAM is recommended. Generation and reward batch sizes halve automatically on CUDA OOM and retry without crashing.

---

## Data format

A JSONL file at `prompts.jsonl` with one prompt per line. The loader checks for `prompt`, `text`, `instruction`, and `user` keys in that order.

```jsonl
{"prompt": "Explain the tradeoffs between TCP and UDP."}
{"prompt": "Rewrite this function so it handles None inputs gracefully."}
```

---

## Configuration

Everything lives in the first cell of the notebook.

```python
GROUP_SIZE               = 5       # candidates per prompt on the NOVEL path
GRPO_BASELINE_GROUP_SIZE = 5       # candidates for the vanilla GRPO comparison
DATASET_FRACTION         = 0.10    # fraction of prompts.jsonl to sample
TRAIN_FRACTION           = 0.90    # train / val split

LORA_R                   = 16
LORA_ALPHA               = 32
LR                       = 5e-5
EPS                      = 0.20    # PPO clip ratio
KL_BETA                  = 0.03

ATTN_LR                  = 1e-4
ATTN_REWARD_TEMP         = 0.75    # softmax temperature over reward scores
ATTN_ENTROPY_LAMBDA      = 0.005   # entropy bonus on the preference head
ATTENTION_TOPK_UPDATES   = 2       # candidates that contribute to the gradient
```

Set `DO_TRAIN = False` to skip the training loop entirely and just load models.

---

## Checkpoints

Saved to `two_head_attention_grpo_lora/` in Hugging Face format. A `latest/` directory is always written at exit; intermediate checkpoints are saved every `SAVE_EVERY` steps (default 100).

```
two_head_attention_grpo_lora/
├── latest/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer.json  …
└── step-100/
    └── …
```

---

## Training log

Each step emits a single line comparing the GRPO baseline against the two-head NOVEL path:

```
  12/ 500 | GRPO R=-0.412 L=-0.031 | NOVEL R=-0.187 L=+0.014 mod=Reasoning   | dR=+0.225 avgdR=+0.183 WR= 75.0% last10=+0.201 | gen_gate=0.041 pref_w=0.237(+0.037) pref_gate=0.038 dW=+2.4e-04
```

- `dR` — reward delta (NOVEL − GRPO) for this step
- `avgdR` / `WR` — running average delta and win rate over all steps
- `gen_gate` / `pref_gate` — current gate values of each attention head (0 = prior-dominated, 1 = learned-dominated)
- `pref_w` — preference weight assigned to the best candidate; `(+x.xxx)` is the margin above uniform weight
- `dW` — L2 norm change in trainable LoRA parameters

A per-modality breakdown is printed at the end of training.

---

## Notes

The prompt classifier at `CLASSIFIER_PATH` is a local checkpoint that must be trained separately. Without it, routing falls back to uniform probabilities across modalities — the attention heads still train but start with a weaker prior signal.

Reward shaping adds small penalties for empty responses, prompt leakage, line-level repetition, and low token-type ratio. These are guards against obvious failures, not primary optimisation targets.

---

## References

**GRPO**
> Shao, Z., Wang, P., Zhu, Q., An, R., Song, R., Zhang, Y., ... & Guo, D. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.

**LoRA**
> Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. arXiv:2106.09685.

**Reward model**
> Köpf, A., Kilcher, Y., von Rütte, D., Anagnostidis, S., Tam, Z., Stevens, K., ... & Bossan, B. (2023). *OpenAssistant Conversations — Democratizing Large Language Model Alignment*. NeurIPS 2023. arXiv:2304.07327.

**Generator**
> Qwen Team. (2024). *Qwen2.5 Technical Report*. arXiv:2412.15115.
