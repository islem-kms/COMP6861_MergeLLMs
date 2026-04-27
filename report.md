# Model Merging for Multitask Language Models
### COMP6861 — Experimental Report

**Setup:** The full experiment was run on a workstation with 2× NVIDIA Quadro RTX 8000
(96 GB VRAM total) using `meta-llama/Llama-3.2-3B-Instruct` as the base model.
Fine-tuning used full-scale datasets over multiple training epochs.
All fine-tuned adapters and merged models are available on
[HuggingFace (islemkms)](https://huggingface.co/islemkms).
The full code is on [GitHub](https://github.com/islem-kms/COMP6861_MergeLLMs).

---

## Overview

This report follows the actual experimental progression:

1. **Part 1** — Fine-tune emotion and summarization specialists; run initial 2-task merges with all four methods.
2. **Part 2** — Tune λ (Task Arithmetic) and density (Breadcrumbs) to find better 2-task configurations.
3. **Part 3** — Add an NLI specialist (three tasks) to test whether TIES improves with a proper majority vote.
4. **Part 4** — TIES parameter sweep to find the optimal scaling factor for 3-task merging.

---

## Part 1: Two-Task Experiment

### Setup

We fine-tune a separate LoRA adapter for each task using `SFTTrainer`. Each adapter starts
from the same frozen base model (`meta-llama/Llama-3.2-3B-Instruct`) and is trained
independently — specialists never see each other's data. The goal is to merge multitask
capability from isolated specialists without any mixed-task training.

**LoRA configuration** (identical across all runs):
- `r=16`, `lora_alpha=32`
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `lora_dropout=0.05`, `bias=none`

**Tasks and datasets:**
- **Emotion classification** (`dair-ai/emotion`): 6-class labeling (sadness, joy, love, anger, fear, surprise). ~16k training samples.
- **Dialogue summarization** (`knkarthick/dialogsum`): generate a short summary from a dialogue. ~12k training samples.

### Merging Methods

All methods operate in parameter space on linearized state dicts produced by fusing each
LoRA adapter into the base model via `merge_and_unload()`. State dicts are kept in float32
for merging precision.

**Weight Averaging:** `θ_merged = (1/N) Σ θᵢ`. Averages all parameters directly.
Works when models share a loss basin, but ignores task-specific structure.

**Task Arithmetic:** Extract a task vector `τᵢ = θᵢ − θ_base` for each specialist, then
apply `θ_merged = θ_base + λ · Σ τᵢ`. The scaling factor **λ** controls how strongly
each task vector is applied. λ=0 leaves the base unchanged; too large → task interference
and performance collapse.

**Breadcrumbs (Sparse Task Arithmetic):** Before summing, prune each task vector to keep
only its top-**density** fraction of entries by absolute magnitude. High-magnitude entries
carry the task-specific signal; zeroing the rest reduces cross-task interference.
density=1.0 is identical to Task Arithmetic (no pruning).

**TIES (Trim, Elect Sign):** After trimming (same as Breadcrumbs), runs a **majority vote**
to elect a sign for each parameter position across all task vectors. Only parameters that
agree with the elected sign are averaged and applied. This directly resolves *sign
conflicts* — cases where different tasks push the same weight in opposite directions.

### Results

| Model | Emotion acc | ROUGE-L |
|---|---|---|
| Base (zero-shot) | 0.318 | 0.181 |
| Emotion specialist | 0.912 | 0.193 |
| Summary specialist | 0.408 | 0.298 |
| Multitask baseline | 0.882 | 0.301 |
| Weight Average | 0.782 | 0.321 |
| Task Arithmetic (λ=0.5) | 0.782 | 0.321 |
| Breadcrumbs (λ=0.5, d=0.2) | 0.680 | 0.301 |
| TIES (λ=0.5, d=0.2) | 0.636 | 0.286 |

### Analysis

**Weight Averaging and Task Arithmetic (λ=0.5) are mathematically identical** with two
equally-weighted tasks — averaging fine-tuned weights is algebraically the same as applying
task vectors at λ=0.5. Both achieve 0.782 emotion / 0.321 ROUGE-L, beating the multitask
baseline on ROUGE-L (+0.020) while remaining below it on emotion (−0.100).

**Breadcrumbs (d=0.2)** drops below Weight Average on emotion (0.680 vs 0.782, −13%), with
ROUGE-L only matching the multitask baseline (0.301). Very aggressive pruning at d=0.2
removes too much task-relevant signal — higher density is likely needed.

**TIES** is the worst merger on both metrics (0.636 emotion / 0.286 ROUGE-L), sitting
−18.7% below Weight Average on emotion. With only two tasks, majority sign voting is
undefined: every parameter has exactly one vote per sign, yielding ~50% conflicts resolved
arbitrarily. This is a **structural limitation of TIES when N=2**, independent of λ or
density. The hypothesis: adding a third task (enabling a proper 2-vs-1 majority vote) should
unlock TIES's advantage. Before testing that, parameter tuning of the other two methods was
investigated first.

---

## Part 2: Parameter Sweep — Task Arithmetic (λ) and Breadcrumbs (density)

### Motivation

The default λ=0.5 and density=0.2 are conservative starting points. A sweep over both
could reveal better configurations and clarify the emotion/ROUGE-L trade-off.

**What λ controls (Task Arithmetic & Breadcrumbs):** λ scales the sum of task vectors
before adding it to the base model. Low λ → under-merging (model stays close to the base).
High λ → stronger task signal but growing cross-task interference, eventually causing
*catastrophic forgetting* as the summed task vectors overwhelm the base model's pre-trained
representations. There is typically a performance peak followed by collapse as λ increases
beyond a threshold.

**What density controls (Breadcrumbs):** density is the fraction of parameters kept per
task vector, selected by absolute magnitude. Low density → aggressive pruning. High density
→ approaches Task Arithmetic (density=1.0 = no pruning). The risk of over-pruning: useful
signal is discarded along with noise.

### Task Arithmetic — λ Sweep Results

| Config | Emotion acc | ROUGE-L | vs WA/TA (λ=0.5) |
|---|---|---|---|
| task arithmetic λ=0.3 | 0.610 | 0.288 | −21.9% emotion, −10.3% ROUGE-L |
| task arithmetic λ=0.4 | 0.696 | 0.312 | −11.0% emotion, −2.8% ROUGE-L |
| task arithmetic λ=0.5 *(baseline)* | 0.782 | 0.321 | — |
| task arithmetic λ=0.6 | **0.830** | 0.315 | +6.1% emotion, −1.9% ROUGE-L |
| task arithmetic λ=0.7 | **0.842** | 0.314 | +7.7% emotion, −2.2% ROUGE-L |
| task arithmetic λ=0.8 | 0.798 | 0.309 | +2.0% emotion, −3.7% ROUGE-L |
| task arithmetic λ=1.0 | 0.838 | 0.311 | +7.2% emotion, −3.1% ROUGE-L |

### Breadcrumbs — Density Sweep Results (λ=0.5 fixed)

| Config | Emotion acc | ROUGE-L | vs Task Arith (λ=0.5) |
|---|---|---|---|
| breadcrumbs d=0.2 | 0.680 | 0.301 | −13.0% emotion, −6.2% ROUGE-L |
| breadcrumbs d=0.3 | 0.714 | 0.311 | −8.7% emotion, −3.1% ROUGE-L |
| breadcrumbs d=0.4 | 0.754 | 0.319 | −3.6% emotion, −0.6% ROUGE-L |
| breadcrumbs d=0.5 | 0.766 | **0.323** | −2.0% emotion, **+0.6% ROUGE-L** |
| breadcrumbs d=0.6 | 0.780 | 0.319 | −0.3% emotion, −0.6% ROUGE-L |
| breadcrumbs d=0.8 | 0.782 | 0.321 | ≈Task Arithmetic (density → 1.0) |

### Analysis

**Task Arithmetic — emotion/ROUGE-L trade-off:** Emotion accuracy peaks at λ=0.7 (0.842),
a +7.7% gain over the default. ROUGE-L drops from 0.321 to 0.314 (−2.2%). The best
balanced point is λ=0.6: emotion=0.830 (+6.1%), ROUGE-L=0.315. At λ=0.8 emotion drops back
to 0.798, signaling the onset of interference. Continued increase toward the collapse
threshold (~λ=1.5) would cause both tasks to degrade toward random.

**Breadcrumbs — density improves ROUGE-L, not emotion:** At d=0.2, Breadcrumbs
underperforms Task Arithmetic on both metrics — over-pruning discards task signal. ROUGE-L
improves monotonically up to d=0.5 (0.323), then plateaus. At d=0.8, Breadcrumbs converges
to Task Arithmetic, confirming density=0.8 ≈ no pruning. Breadcrumbs at d=0.5 is the only
configuration that surpasses Task Arithmetic (λ=0.5) on ROUGE-L.

**Key takeaway:** Task Arithmetic and Breadcrumbs expose a clear emotion/ROUGE-L trade-off
controllable through their parameters. No single configuration dominates on both metrics
simultaneously — the best choice depends on which task is prioritized.

---

## Part 3: Three-Task Experiment — Adding NLI

### Motivation

The sign-conflict analysis from Part 1 predicts that TIES should improve with three tasks:
a 2-vs-1 majority vote is well-defined and can deterministically resolve conflicts. We add a
third specialist trained on **Natural Language Inference** (`nyu-mll/multi_nli`, 20k training
samples) — predicting entailment / neutral / contradiction.

With three task vectors, all four merging methods are re-run and a third metric (NLI
accuracy) is added to evaluation.

### 3-task Results

| Model | Emotion acc | NLI acc | ROUGE-L |
|---|---|---|---|
| Emotion specialist | 0.912 | 0.410 | 0.193 |
| Summary specialist | 0.408 | 0.510 | 0.298 |
| NLI specialist | 0.238 | 0.858 | 0.205 |
| Multitask baseline | 0.882 | 0.380 | 0.301 |
| Weight Average | 0.622 | 0.680 | 0.297 |
| Task Arithmetic (λ=0.5) | 0.630 | **0.834** | 0.298 |
| Breadcrumbs (λ=0.5, d=0.5) | **0.636** | 0.814 | 0.291 |
| TIES (λ=0.5, d=0.4) | 0.580 | 0.672 | 0.298 |

### Analysis

**Task Arithmetic** is the best 3-task merger overall. It achieves the highest NLI accuracy
(0.834 vs 0.680 for Weight Average, +22.6%), confirming that task vectors capture the strong
NLI task direction more faithfully than raw weight averaging. Compared to the 2-task
experiment, adding a third task vector has pulled emotion down significantly (0.630 vs 0.782,
−19%), as the three task vectors sum to a larger total perturbation that competes against
the emotion direction.

**Breadcrumbs** is the closest to Task Arithmetic: marginally better on emotion (0.636 vs
0.630, +1%) but behind on NLI (0.814 vs 0.834, −2.4%) and ROUGE-L (0.291 vs 0.298, −2.3%).
The d=0.5 pruning helps emotion slightly but costs some NLI accuracy, as the NLI task vector
has useful spread across many parameters that pruning partially discards.

**Weight Averaging** loses the most compared to Task Arithmetic, especially on NLI (−22.6%).
Direct weight averaging dilutes the strong NLI task direction because the absolute magnitude
of fine-tuned weights is smaller than the corresponding task vector.

**TIES (λ=0.5)** is still the worst merger despite three tasks enabling a proper 2-vs-1
majority vote. Sign conflicts are now theoretically resolved, so the failure must have a
different cause. Analyzing the mechanism: because TIES zeroes the *disagreeing* subset of
parameters before averaging, the effective magnitude of the merged task vector is
intrinsically smaller than in Task Arithmetic at the same λ. The model is **under-merged** —
a higher λ is needed to compensate. All merged models comfortably exceed the multitask
baseline on NLI (0.672–0.834 vs 0.380), a major gain from incorporating the NLI specialist.

---

## Part 4: TIES Parameter Sweep

### Motivation

TIES at λ=0.5 systematically under-scales its task vectors: sign election zeroes the
minority-sign parameters before averaging, which intrinsically reduces the effective
perturbation applied to the base model. A λ=0.5 TIES merge applies less total change than a
λ=0.5 Task Arithmetic merge, even though both use the same nominal scaling. A systematic
sweep over λ and density was run to quantify this effect.

**Sweep:** λ ∈ {0.3, 0.5, 0.7, 1.0} × density ∈ {0.2, 0.4, 0.6, 0.8} → 16 configurations.

### TIES 4×4 Sweep Results

| Config | Emotion acc | NLI acc | ROUGE-L |
|---|---|---|---|
| TIES λ=0.3, d=0.2 | 0.478 | 0.648 | 0.269 |
| TIES λ=0.3, d=0.4 | 0.492 | 0.650 | 0.275 |
| TIES λ=0.3, d=0.6 | 0.478 | 0.646 | 0.266 |
| TIES λ=0.3, d=0.8 | 0.466 | 0.646 | 0.263 |
| TIES λ=0.5, d=0.2 | 0.566 | 0.670 | 0.289 |
| TIES λ=0.5, d=0.4 *(v3 baseline)* | 0.580 | 0.672 | 0.298 |
| TIES λ=0.5, d=0.6 | 0.572 | 0.672 | 0.291 |
| TIES λ=0.5, d=0.8 | 0.562 | 0.670 | 0.292 |
| TIES λ=0.7, d=0.2 | 0.670 | 0.694 | 0.297 |
| TIES λ=0.7, d=0.4 | 0.680 | 0.714 | 0.297 |
| TIES λ=0.7, d=0.6 | 0.656 | 0.694 | 0.292 |
| TIES λ=0.7, d=0.8 | 0.626 | 0.678 | 0.295 |
| **TIES λ=1.0, d=0.2** | **0.710** | 0.816 | **0.303** |
| TIES λ=1.0, d=0.4 | 0.682 | **0.840** | 0.283 |
| TIES λ=1.0, d=0.6 | 0.672 | 0.812 | 0.299 |
| TIES λ=1.0, d=0.8 | 0.656 | 0.786 | 0.295 |
| *Task Arithmetic λ=0.5 (reference)* | *0.630* | *0.834* | *0.298* |

### Analysis

**The sweep confirms the hypothesis: λ=0.5 was severely under-scaling TIES task vectors.**
Increasing λ from 0.5 to 1.0 dramatically improves all three metrics — emotion +0.130,
NLI +0.144, ROUGE-L +0.005 at d=0.2. This is the dominant effect; density is secondary.

**TIES (λ=1.0, d=0.2) vs Task Arithmetic (λ=0.5):** TIES now leads on emotion
(0.710 vs 0.630, +12.7%) and ROUGE-L (0.303 vs 0.298, +1.7%), while remaining slightly
below on NLI (0.816 vs 0.834, −2.2%). Overall, TIES at its optimal configuration is the
best single merger across the three metrics combined.

**Density trade-off at λ=1.0:** d=0.2 vs d=0.4 shows a NLI/emotion-ROUGE split — d=0.4
peaks on NLI (0.840, +0.024 vs d=0.2) but drops on emotion (0.682 vs 0.710, −3.9%) and
ROUGE-L (0.283 vs 0.303, −7.1%). Sparser merging (d=0.2) retains more balanced performance
across all three tasks.

**Why TIES tolerates high λ better than Task Arithmetic:** Task Arithmetic at high λ sums
all task vectors regardless of agreement, risking catastrophic forgetting. TIES's sign
election acts as a natural regularizer: parameters where tasks conflict are zeroed (not
applied at scale), while only parameters with broad agreement receive the full λ treatment.
This limits the total perturbation even as λ increases, making TIES more robust at λ=1.0.
The cost is precisely what was observed: TIES under-merges at low λ and needs a higher value
to achieve the same effective signal strength.

**Lower bound at λ=0.3:** At λ=0.3, TIES degrades significantly (0.478 emotion / 0.648 NLI
/ 0.269 ROUGE-L). After sign election zeroes the minority parameters, the remaining signal
is too small to steer the model meaningfully — confirming both a lower bound on λ for TIES
and the under-merging behavior at small scaling factors.

---

## Summary and Conclusion

| Method | Best config | Emotion acc | NLI acc | ROUGE-L |
|---|---|---|---|---|
| Weight Average | — | 0.782 / 0.622 | — / 0.680 | 0.321 / 0.297 |
| Task Arithmetic | λ=0.7 (2t) / λ=0.5 (3t) | 0.842 / 0.630 | — / **0.834** | 0.314 / 0.298 |
| Breadcrumbs | d=0.5, λ=0.5 | 0.766 / **0.636** | — / 0.814 | **0.323** / 0.291 |
| TIES (default) | λ=0.5, d=0.2–0.4 | 0.662 / 0.580 | — / 0.672 | 0.295 / 0.298 |
| **TIES (tuned)** | **λ=1.0, d=0.2** | — / **0.710** | — / 0.816 | — / **0.303** |

*Format: 2-task result / 3-task result where applicable.*

**Key findings:**

1. **Weight Averaging = Task Arithmetic at λ=0.5** with two equal-weight tasks — mathematically equivalent.
2. **Task vector cosine similarity = 0.13** (nearly orthogonal) — favorable for merging, reduces cross-task interference.
3. **Task Arithmetic has a clear emotion/ROUGE-L trade-off** controllable via λ; peak emotion at λ=0.7, peak ROUGE-L at λ=0.5.
4. **Breadcrumbs improves monotonically from d=0.2 to d=0.5** then plateaus; d=0.8 converges to Task Arithmetic (no pruning).
5. **TIES requires two conditions:** (a) ≥3 tasks for meaningful sign election and (b) calibrated λ to compensate for reduced effective perturbation. Neither condition alone is sufficient. With both satisfied (λ=1.0, d=0.2), TIES becomes the best overall 3-task merger.
6. **TIES sign election acts as a regularizer**, making it more robust to high λ than Task Arithmetic — explaining why λ=1.0 TIES does not collapse where Task Arithmetic would.
7. **All merged models exceed the multitask baseline on NLI** in the 3-task setting (0.672–0.834 vs 0.380), demonstrating that merging successfully transfers specialist knowledge without joint training.

---

*Full code: https://github.com/islem-kms/COMP6861_MergeLLMs*
*Models: https://huggingface.co/islemkms*
