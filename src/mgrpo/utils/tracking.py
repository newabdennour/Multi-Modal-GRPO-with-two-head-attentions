import numpy as np
from collections import defaultdict
from mgrpo.config import OUTPUT_DIR, MODALITY_MAP, UNIFORM_CANDIDATE_WEIGHT

class Tracker:
    def __init__(self):
        self.history = []
        self.per_mod = defaultdict(lambda: {"novel": [], "grpo": [], "count": 0})

    def avg(self, values):
        values = [v for v in values if v is not None and not np.isnan(v)]
        return float(np.mean(values)) if values else 0.0

    def paired_stats(self, window=None):
        rows = [h for h in self.history if not np.isnan(h["grpo_reward"])]
        if window is not None:
            rows = rows[-int(window):]
        if not rows:
            return {"n": 0, "avg_delta": 0.0, "win_rate": 0.0}
        deltas = [h["novel_reward"] - h["grpo_reward"] for h in rows]
        wins = [h["novel_reward"] > h["grpo_reward"] for h in rows]
        return {
            "n": len(rows),
            "avg_delta": float(np.mean(deltas)),
            "win_rate": 100.0 * float(np.mean(wins)),
        }

    def save_checkpoint(self, generator_model, tag="latest"):
        out_dir = OUTPUT_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        generator_model.model.save_pretrained(out_dir)
        generator_model.tokenizer.save_pretrained(out_dir)
        return out_dir

    def format_step_report(self, step, total, rec):
        delta = rec["novel_reward"] - rec["grpo_reward"]
        s = self.paired_stats()
        s10 = self.paired_stats(window=10)
        pref_signal = rec["best_preference_weight"] - UNIFORM_CANDIDATE_WEIGHT
        return (
            f"{step:>4}/{total} | "
            f"GRPO R={rec['grpo_reward']:+.3f} L={rec['grpo_loss']:+.3f} | "
            f"NOVEL R={rec['novel_reward']:+.3f} L={rec['novel_loss']:+.3f} mod={rec['best_modality']:<10} | "
            f"dR={delta:+.3f} avgdR={s['avg_delta']:+.3f} WR={s['win_rate']:>5.1f}% last10={s10['avg_delta']:+.3f} | "
            f"gen_gate={rec['gen_gate']:.3f} pref_w={rec['best_preference_weight']:.3f}({pref_signal:+.3f}) "
            f"pref_gate={rec['pref_gate']:.3f} dW={rec['adapter_delta']:+.1e}"
        )

    def print_final_report(self):
        if not self.history:
            print("No training history yet.")
            return
        grpo = [h["grpo_reward"] for h in self.history]
        novel = [h["novel_reward"] for h in self.history]
        s = self.paired_stats()
        print("\n" + "=" * 88)
        print(f"Two-head attention GRPO final report | steps={len(self.history)}")
        print("=" * 88)
        print(f"Mean GRPO reward       : {self.avg(grpo):+.4f}")
        print(f"Mean NOVEL reward      : {self.avg(novel):+.4f}")
        print(f"Mean delta NOVEL-GRPO  : {s['avg_delta']:+.4f}")
        print(f"NOVEL win rate vs GRPO : {s['win_rate']:.1f}%")
        print("-" * 88)
        for mid in sorted(MODALITY_MAP):
            name = MODALITY_MAP[mid]
            vals = self.per_mod[name]["novel"]
            if vals:
                print(f"  {name:<10} n={len(vals):>4} novel_avg={self.avg(vals):+.4f}")
        print("=" * 88 + "\n")
