import torch, numpy as np, os
import matplotlib.pyplot as plt
import pandas as pd
from utils.model_utils import *
from g_06_merge_methods import (load_all_state_dicts, task_arithmetic,
                               breadcrumbs, sd_to_model, compute_task_vector)
from h_07_evaluate import eval_emotion, eval_summarization

os.makedirs("./results/plots", exist_ok=True)

# Load all state dicts once — keep in RAM for all analyses
# 36GB total — no problem with 125GB available
sds     = load_all_state_dicts()
base_sd = sds["base"]
ft_sds  = [sds["emotion"], sds["summary"]]


# ── 7.1  Lambda Sweep ─────────────────────────
# Most important experiment — put front-and-centre in your report

def lambda_sweep():
    lambdas        = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                      0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    emotion_scores = []
    rouge_scores   = []

    for lam in lambdas:
        print(f"  λ = {lam}")
        sd    = task_arithmetic(base_sd, ft_sds, scaling_factor=lam)
        model = sd_to_model(sd)
        emotion_scores.append(eval_emotion(model, n_samples=200))
        rouge_scores.append(
            eval_summarization(model, n_samples=100)["rougeL"]
        )
        del model;  torch.cuda.empty_cache()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lambdas, emotion_scores, "o-", color="steelblue",
            label="Emotion Accuracy",         linewidth=2, markersize=7)
    ax.plot(lambdas, rouge_scores,   "s-", color="darkorange",
            label="ROUGE-L (Summarization)",  linewidth=2, markersize=7)
    ax.set_xlabel("Scaling Factor λ", fontsize=12)
    ax.set_ylabel("Score",            fontsize=12)
    ax.set_title("Task Arithmetic: Performance vs. λ", fontsize=13)
    ax.legend(fontsize=11);  ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./results/plots/lambda_sweep.png", dpi=150)
    plt.show()
    print("✓ Saved lambda_sweep.png")


# ── 7.2  Task Vector Cosine Similarity ────────

def task_vector_similarity():
    """
    cos_sim ≈  0 → orthogonal tasks → merging works well
    cos_sim >  0 → aligned tasks    → may be redundant
    cos_sim <  0 → conflicting      → merging hurts both
    """
    tv_e = compute_task_vector(base_sd, sds["emotion"])
    tv_s = compute_task_vector(base_sd, sds["summary"])

    # Global similarity
    flat_e  = torch.cat([v.flatten() for v in tv_e.values()])
    flat_s  = torch.cat([v.flatten() for v in tv_s.values()])
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_e.unsqueeze(0), flat_s.unsqueeze(0)
    ).item()
    print(f"Global cosine similarity: {cos_sim:.6f}")

    # Layer-wise
    sims, mag_e, mag_s = [], [], []
    for key in base_sd:
        if "weight" not in key:
            continue
        e = tv_e[key].flatten().float()
        s = tv_s[key].flatten().float()
        if e.norm() < 1e-8 or s.norm() < 1e-8:
            continue
        sims.append(torch.nn.functional.cosine_similarity(
            e.unsqueeze(0), s.unsqueeze(0)).item())
        mag_e.append(e.norm().item())
        mag_s.append(s.norm().item())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    x = range(len(sims))

    axes[0].bar(x, sims,
                color=["green" if s > 0 else "red" for s in sims],
                alpha=0.7)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title(
        f"Layer-wise Task Vector Similarity  (global = {cos_sim:.4f})")
    axes[0].set_xticks([])

    axes[1].plot(x, mag_e, label="Emotion  ‖τ‖", alpha=0.8)
    axes[1].plot(x, mag_s, label="Summary  ‖τ‖", alpha=0.8)
    axes[1].set_xlabel("Layer (early → late)")
    axes[1].set_ylabel("L2 Norm")
    axes[1].set_title("Task Vector Magnitudes by Layer")
    axes[1].legend();  axes[1].set_xticks([])

    plt.tight_layout()
    plt.savefig("./results/plots/task_vector_similarity.png", dpi=150)
    plt.show()
    print("✓ Saved task_vector_similarity.png")
    return cos_sim


# ── 7.3  Sign Conflict Analysis ───────────────

def sign_conflict_analysis():
    """
    High conflict → tasks push same parameters in opposite directions.
    Explains why TIES outperforms Task Arithmetic when conflicts are high.
    """
    tv_e = compute_task_vector(base_sd, sds["emotion"])
    tv_s = compute_task_vector(base_sd, sds["summary"])

    layer_conflicts = {}
    for key in base_sd:
        if "weight" not in key:
            continue
        e, s = tv_e[key], tv_s[key]
        mask = (e.abs() > 1e-8) & (s.abs() > 1e-8)
        if mask.sum() == 0:
            continue
        rate = (e[mask].sign() != s[mask].sign()).float().mean().item()
        layer_conflicts[
            key.replace("model.", "").replace(".weight", "")
        ] = rate

    avg  = sum(layer_conflicts.values()) / len(layer_conflicts)
    vals = list(layer_conflicts.values())
    print(f"Average sign conflict rate: {avg:.4f}  ({avg*100:.1f}%)")
    print("  (50% = random | >50% = systematic opposition)")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(vals)), vals,
           color=["red" if v > 0.5 else "steelblue" for v in vals],
           alpha=0.75)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1,
               label="Random baseline (0.5)")
    ax.set_xlabel("Layer (early → late)")
    ax.set_ylabel("Sign Conflict Rate")
    ax.set_title("Parameter Sign Conflicts  (red = >50% conflict)")
    ax.legend();  ax.set_xticks([])
    plt.tight_layout()
    plt.savefig("./results/plots/sign_conflicts.png", dpi=150)
    plt.show()
    print("✓ Saved sign_conflicts.png")
    return avg


# ── 7.4  Density Sweep (Breadcrumbs) ──────────

def density_sweep():
    densities      = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    emotion_scores = []
    rouge_scores   = []

    for d in densities:
        print(f"  density = {d}")
        sd    = breadcrumbs(base_sd, ft_sds, scaling_factor=0.5, density=d)
        model = sd_to_model(sd)
        emotion_scores.append(eval_emotion(model, n_samples=200))
        rouge_scores.append(
            eval_summarization(model, n_samples=100)["rougeL"]
        )
        del model;  torch.cuda.empty_cache()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(densities, emotion_scores, "o-",
            label="Emotion Accuracy", linewidth=2)
    ax.plot(densities, rouge_scores,   "s-",
            label="ROUGE-L",          linewidth=2)
    ax.set_xlabel("Task Vector Density (fraction of weights kept)")
    ax.set_ylabel("Score")
    ax.set_title("Breadcrumbs: Performance vs. Sparsity")
    ax.legend();  ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./results/plots/density_sweep.png", dpi=150)
    plt.show()
    print("✓ Saved density_sweep.png")


# ── 7.5  Final Comparison Bar Chart ───────────

def plot_comparison():
    df     = pd.read_csv("./results/metrics.csv", index_col=0)
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].barh(df.index, df["emotion_accuracy"], color=colors)
    axes[0].set_xlabel("Accuracy");  axes[0].set_xlim(0, 1)
    axes[0].set_title("Emotion Classification Accuracy")
    for i, v in enumerate(df["emotion_accuracy"]):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    axes[1].barh(df.index, df["rougeL"], color=colors)
    axes[1].set_xlabel("ROUGE-L")
    axes[1].set_xlim(0, max(df["rougeL"]) * 1.2)
    axes[1].set_title("Summarization ROUGE-L")
    for i, v in enumerate(df["rougeL"]):
        axes[1].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

    plt.suptitle("Model Merging: All Methods Compared",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("./results/plots/final_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Saved final_comparison.png")


# ── Run all ───────────────────────────────────

if __name__ == "__main__":
    print("1. Task Vector Similarity\n");   task_vector_similarity()
    print("\n2. Sign Conflict Analysis\n");  sign_conflict_analysis()
    print("\n3. Lambda Sweep\n");            lambda_sweep()
    print("\n4. Density Sweep\n");           density_sweep()
    print("\n5. Final Comparison\n");        plot_comparison()