import argparse
import sys
import numpy as np

from mgrpo.config import (
    DO_TRAIN, MAX_TRAIN_STEPS, SEED, NORMAL_MODALITY,
    MODALITY_PROFILES, MODALITY_MAP, GENERATION_BATCH_SIZE, GROUP_SIZE,
    GRPO_BASELINE_GROUP_SIZE, PRINT_EVERY, SUMMARY_EVERY, SAVE_EVERY
)
from mgrpo.data.dataset import load_data, get_user_prompt
from mgrpo.models.generator import GeneratorModel
from mgrpo.models.attention import ClassifierPromptGenerationAttention, UserPreferenceUpdateAttention
from mgrpo.rl.rewards import RewardModel, score_responses
from mgrpo.rl.policy import update_policy
from mgrpo.training.router import ModalityRouter
from mgrpo.training.trainer import AttentionTrainer
from mgrpo.utils.tracking import Tracker

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal GRPO Training")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()

    train_ds, val_ds = load_data(SEED)
    if args.smoke_test:
        train_ds = train_ds.select(range(min(5, len(train_ds))))
        print(f"Smoke test mode enabled. Running on {len(train_ds)} examples.")

    generator_model = GeneratorModel()
    reward_model = RewardModel()

    from mgrpo.config import CANDIDATE_FEATURE_DIM
    generation_head = ClassifierPromptGenerationAttention().to(generator_model.device)
    preference_head = UserPreferenceUpdateAttention(CANDIDATE_FEATURE_DIM).to(generator_model.device)

    router = ModalityRouter(generation_head)
    attention_trainer = AttentionTrainer(generation_head, preference_head)
    tracker = Tracker()

    if not DO_TRAIN:
        print("DO_TRAIN is False. Exiting.")
        return

    print("\n" + "=" * 88)
    print("Two-head attention GRPO training")
    print(f"comparison: GRPO(G={GRPO_BASELINE_GROUP_SIZE}) vs NOVEL(G={GROUP_SIZE}) | LoRA=True")
    print("NOVEL uses exactly two heads: generation attention + preference update attention")
    print("=" * 88)
    print("step | GRPO reward/loss | NOVEL reward/loss | reward delta/running win rate | attention state\n")

    steps_to_run = min(MAX_TRAIN_STEPS or len(train_ds), len(train_ds))
    
    for step, item in enumerate(train_ds, 1):
        if step > steps_to_run:
            break

        user_prompt = get_user_prompt(item)
        
        # 1. Baseline GRPO (Normal modality)
        grpo_prof = MODALITY_PROFILES[NORMAL_MODALITY]
        grpo_responses = generator_model.generate_responses_batch(
            user_prompt, grpo_prof, count=GRPO_BASELINE_GROUP_SIZE, batch_size=GENERATION_BATCH_SIZE
        )
        _, grpo_shaped, _ = score_responses(reward_model, user_prompt, grpo_responses)
        grpo_w = np.ones(len(grpo_shaped), dtype=np.float32) / max(1, len(grpo_shaped))
        grpo_info = update_policy(user_prompt, grpo_responses, grpo_shaped, grpo_w, generator_model)

        # 2. NOVEL Pipeline
        mod_ids, clf_probs, gen_scores, gen_gate_before = router.select_modalities(user_prompt, k=GROUP_SIZE)
        
        novel_responses = []
        for mid in mod_ids:
            prof = MODALITY_PROFILES[int(mid)]
            novel_responses.extend(generator_model.generate_responses_batch(
                user_prompt, prof, count=1, batch_size=1
            ))

        raw_r, shaped_r, guards = score_responses(reward_model, user_prompt, novel_responses)
        
        # Attention Training
        features, pref_w, attn_info = attention_trainer.train_attention_heads(
            mod_ids, clf_probs, gen_scores, raw_r, shaped_r, guards, novel_responses
        )

        # Policy Update
        novel_info = update_policy(user_prompt, novel_responses, shaped_r, pref_w, generator_model)

        # 3. Tracking
        best_grpo = float(np.max(grpo_shaped)) if grpo_shaped else 0.0
        best_novel = float(np.max(shaped_r)) if shaped_r else 0.0
        best_mid = int(mod_ids[np.argmax(shaped_r)]) if shaped_r else NORMAL_MODALITY
        best_weight = float(pref_w[np.argmax(shaped_r)]) if shaped_r else 0.0

        rec = {
            "grpo_reward": best_grpo,
            "grpo_loss": grpo_info["loss"],
            "novel_reward": best_novel,
            "novel_loss": novel_info["loss"],
            "best_modality": MODALITY_MAP[best_mid],
            "gen_gate": attn_info["gen_gate"],
            "pref_w": best_weight,
            "pref_gate": attn_info["pref_gate"],
            "adapter_delta": novel_info["adapter_delta"],
            "best_preference_weight": best_weight,
        }
        
        tracker.history.append(rec)
        tracker.per_mod[MODALITY_MAP[best_mid]]["novel"].append(best_novel)
        tracker.per_mod[MODALITY_MAP[best_mid]]["count"] += 1

        if step % PRINT_EVERY == 0:
            print(tracker.format_step_report(step, steps_to_run, rec))
            sys.stdout.flush()

        if step % SAVE_EVERY == 0 and not args.smoke_test:
            tracker.save_checkpoint(generator_model, tag=f"step_{step}")

    tracker.print_final_report()
    if not args.smoke_test:
        tracker.save_checkpoint(generator_model, tag="final")
        print("Training complete and final model saved.")

if __name__ == "__main__":
    main()
