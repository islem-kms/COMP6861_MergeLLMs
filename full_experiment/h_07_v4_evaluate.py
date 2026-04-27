"""
h_07_v4_evaluate.py — Evaluate v4 TIES 3-task sweep models

Evaluates all 16 TIES configs from the v4 sweep (λ × density grid)
on all 3 tasks: emotion accuracy, NLI accuracy, and ROUGE-L.

Output: ./results/metrics_v4_ties_sweep.csv

Run:
  python h_07_v4_evaluate.py 2>&1 | tee run_v4_eval.log

Estimated time: ~20-30 min per model × 16 models = ~5-8 hours total.
"""

import os, torch, pandas as pd
from h_07_evaluate import (
    load_from_sd,
    eval_emotion,
    eval_nli,
    eval_summarization,
    vtag,
)

# ── Sweep grid (must match g_06_v4_ties_sweep.py) ─

SCALING_FACTORS = [0.3, 0.5, 0.7, 1.0]
DENSITIES       = [0.2, 0.4, 0.6, 0.8]

V4_DIR   = "./state_dicts/v4"
CSV_PATH = "./results/metrics_v4_ties_sweep.csv"


def run_v4_eval():
    """Evaluate all v4 TIES sweep models on 3 tasks."""

    # Load any partial results so we can resume
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH, index_col=0)
        rows     = existing.to_dict(orient="index")
        print(f"Loaded {len(rows)} existing results from {CSV_PATH}")
    else:
        rows = {}

    total = len(SCALING_FACTORS) * len(DENSITIES)
    done  = 0

    for lam in SCALING_FACTORS:
        for d in DENSITIES:
            done += 1
            name  = f"ties_v4_l{vtag(lam)}_d{vtag(d)}"
            fname = f"sd_{name}.pt"
            p     = os.path.join(V4_DIR, fname)

            # Skip if already evaluated
            if name in rows:
                print(f"  [{done}/{total}] [skip] {name} already evaluated")
                continue

            if not os.path.exists(p):
                print(f"  [{done}/{total}] [skip] {p} not found")
                continue

            print(f"\n{'='*55}")
            print(f"[{done}/{total}] Evaluating: {name}  (λ={lam}, d={d})")
            print(f"{'='*55}")

            model   = load_from_sd(p)
            em_acc  = eval_emotion(model)
            nli_acc = eval_nli(model)
            rouge   = eval_summarization(model)

            rows[name] = {
                "scaling_factor":   lam,
                "density":          d,
                "emotion_accuracy": round(em_acc,          4),
                "nli_accuracy":     round(nli_acc,         4),
                "rouge1":           round(rouge["rouge1"], 4),
                "rouge2":           round(rouge["rouge2"], 4),
                "rougeL":           round(rouge["rougeL"], 4),
            }
            print(f"  Emotion Acc  : {em_acc:.4f}")
            print(f"  NLI Acc      : {nli_acc:.4f}")
            print(f"  ROUGE-1      : {rouge['rouge1']:.4f}")
            print(f"  ROUGE-L      : {rouge['rougeL']:.4f}")

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

            # Save after each model (crash-safe)
            df = pd.DataFrame(rows).T
            df.to_csv(CSV_PATH)
            print(f"  ✓ Partial save → {CSV_PATH}")

    # Final save
    df = pd.DataFrame(rows).T
    df.to_csv(CSV_PATH)
    print(f"\n\n{'='*55}")
    print(f"✓ v4 TIES sweep evaluation complete → {CSV_PATH}")
    print(f"{'='*55}")
    print(df.to_string())

    # Summary: find best config
    if len(df) > 0:
        # Compute a simple composite score (average of normalized metrics)
        print(f"\n── Best configs ──")
        best_emotion = df["emotion_accuracy"].idxmax()
        best_nli     = df["nli_accuracy"].idxmax()
        best_rouge   = df["rougeL"].idxmax()
        print(f"  Best emotion : {best_emotion}  "
              f"({df.loc[best_emotion, 'emotion_accuracy']:.4f})")
        print(f"  Best NLI     : {best_nli}  "
              f"({df.loc[best_nli, 'nli_accuracy']:.4f})")
        print(f"  Best ROUGE-L : {best_rouge}  "
              f"({df.loc[best_rouge, 'rougeL']:.4f})")

    return df


if __name__ == "__main__":
    run_v4_eval()
