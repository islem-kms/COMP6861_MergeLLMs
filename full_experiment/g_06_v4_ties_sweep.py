"""
g_06_v4_ties_sweep.py — v4: TIES 3-task parameter sweep

2D sweep over scaling_factor (λ) × density for 3-task TIES merging.
The v3 baseline (λ=0.5, d=0.4) was the worst 3-task merger.
This sweep explores whether a better hyperparameter config exists.

Grid:
  scaling_factor (λ) : 0.3, 0.5, 0.7, 1.0
  density            : 0.2, 0.4, 0.6, 0.8

  → 4 × 4 = 16 configurations
  → each state dict ~14 GB → ~224 GB total

Output: ./state_dicts/v4/sd_ties_v4_l{λ}_d{d}.pt

Run:
  python g_06_v4_ties_sweep.py 2>&1 | tee run_v4_merge.log
"""

import os, torch
from utils.model_utils import *

# Import the merge functions from the main merge script
from g_06_merge_methods import (
    load_all_state_dicts,
    ties_merge,
    skip_if_exists,
    vtag,
)


# ── Sweep grid ────────────────────────────────

SCALING_FACTORS = [0.3, 0.5, 0.7, 1.0]
DENSITIES       = [0.2, 0.4, 0.6, 0.8]

OUT_DIR = "./state_dicts/v4"
os.makedirs(OUT_DIR, exist_ok=True)


if __name__ == "__main__":

    # Load all state dicts into RAM (~48 GB)
    sds = load_all_state_dicts()

    if "nli" not in sds:
        raise RuntimeError(
            "NLI state dict not found — run c_03_finetune_nli.py and "
            "f_05_linearize_adapters.py first"
        )

    base_sd      = sds["base"]
    ft_sds_3task = [sds["emotion"], sds["summary"], sds["nli"]]

    total  = len(SCALING_FACTORS) * len(DENSITIES)
    done   = 0

    print(f"\n══ v4: TIES 3-task sweep ({total} configs) ══")
    print(f"  λ  ∈ {SCALING_FACTORS}")
    print(f"  d  ∈ {DENSITIES}")
    print(f"  Output: {OUT_DIR}/\n")

    for lam in SCALING_FACTORS:
        for d in DENSITIES:
            done += 1
            fname = f"sd_ties_v4_l{vtag(lam)}_d{vtag(d)}.pt"
            p     = os.path.join(OUT_DIR, fname)

            print(f"\n── [{done}/{total}] TIES λ={lam}, d={d} ──")

            if skip_if_exists(p):
                continue

            merged = ties_merge(base_sd, ft_sds_3task,
                                scaling_factor=lam, density=d)
            torch.save(merged, p)
            print(f"  ✓ saved → {p}")

            # Free the merged dict to reduce peak RAM
            del merged

    print(f"\n\n✓ All v4 TIES sweep merges saved to {OUT_DIR}/")
    print(f"  Next: python h_07_v4_evaluate.py")
