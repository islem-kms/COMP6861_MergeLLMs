import os, torch, copy
from transformers import AutoModelForCausalLM
from utils.model_utils import *


# ── Helpers ───────────────────────────────────

def load_all_state_dicts():
    """
    Load all state dicts simultaneously into RAM.
    4 × ~12GB = ~48GB — well within 125GB available.
    Eliminates the repeated load/del cycle from memory-constrained setups.
    NLI state dict loaded only if it exists (produced by f_05_linearize_adapters.py).
    """
    print("Loading all state dicts into RAM ...")
    sds = {
        "base":    torch.load(SD_BASE,     map_location="cpu"),
        "emotion": torch.load(SD_EMOTION,  map_location="cpu"),
        "summary": torch.load(SD_SUMMARY,  map_location="cpu"),
    }
    if os.path.exists(SD_NLI):
        sds["nli"] = torch.load(SD_NLI, map_location="cpu")
        print("  ✓ All loaded (including NLI)")
    else:
        print("  ✓ Loaded (NLI state dict not found — 3-task merges will be skipped)")
    return sds


def sd_to_model(state_dict):
    """Load a state dict into a model shell for inference."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="auto"
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_task_vector(base_sd, ft_sd):
    """τ = θ_finetuned − θ_base"""
    return {k: ft_sd[k].float() - base_sd[k].float() for k in base_sd}


# ── Method 1: Weight Averaging ────────────────

def weight_average(sds, weights=None):
    """
    θ_merged = Σ w_i * θ_i

    Simplest merge. Works when fine-tuned models remain
    in the same loss basin as the base model.
    Equal weights by default.
    """
    if weights is None:
        weights = [1.0 / len(sds)] * len(sds)

    merged = {}
    for key in sds[0]:
        merged[key] = sum(w * sd[key].float()
                          for w, sd in zip(weights, sds))

    print("✓ Weight Averaging complete")
    return merged


# ── Method 2: Task Arithmetic ─────────────────

def task_arithmetic(base_sd, ft_sds, scaling_factor=0.5):
    """
    θ_merged = θ_base + λ * Σ τ_i

    Task vectors (τ) encode what each fine-tuning run added.
    Summing and scaling them combines task capabilities.

    λ too small → behaves like base (under-merged)
    λ too large → tasks interfere, both degrade (over-merged)
    Optimal λ found via lambda sweep in Phase 6.
    """
    task_vectors = [compute_task_vector(base_sd, ft) for ft in ft_sds]

    merged = {}
    for key in base_sd:
        summed_tv   = sum(tv[key] for tv in task_vectors)
        merged[key] = base_sd[key].float() + scaling_factor * summed_tv

    print(f"✓ Task Arithmetic complete  (λ={scaling_factor})")
    return merged


# ── Method 3: Breadcrumbs ─────────────────────

def sparsify_task_vector(tv, density=0.2, protect_top=0.01):
    """
    Keep only the top `density` fraction by magnitude.
    Top `protect_top` fraction is always kept regardless.

    Most task vector signal lives in a small number of
    high-magnitude parameters — zeroing the rest reduces
    cross-task interference during summation.
    """
    sparse_tv = {}
    for key in tv:
        param = tv[key]
        flat  = param.abs().flatten()
        n     = flat.numel()

        keep_thresh    = flat.topk(max(1, int(n * density))).values.min()
        protect_thresh = flat.topk(max(1, int(n * protect_top))).values.min()

        mask          = ((param.abs() >= keep_thresh) |
                         (param.abs() >= protect_thresh))
        sparse_tv[key] = param * mask.float()

    total   = sum(v.numel() for v in tv.values())
    nonzero = sum((v != 0).sum().item() for v in sparse_tv.values())
    print(f"    Density: {nonzero/total:.3f}  "
          f"(nonzero={nonzero:,} / {total:,})")
    return sparse_tv


def breadcrumbs(base_sd, ft_sds, scaling_factor=0.5, density=0.2):
    """
    Sparse Task Arithmetic.
    Prune low-magnitude task vector entries before merging.
    Typically outperforms vanilla Task Arithmetic by reducing interference.
    """
    task_vectors = []
    for i, ft in enumerate(ft_sds):
        tv = compute_task_vector(base_sd, ft)
        print(f"  Sparsifying task vector {i+1}/{len(ft_sds)}:")
        task_vectors.append(sparsify_task_vector(tv, density=density))

    merged = {}
    for key in base_sd:
        summed      = sum(tv[key] for tv in task_vectors)
        merged[key] = base_sd[key].float() + scaling_factor * summed

    print(f"✓ Breadcrumbs complete  (λ={scaling_factor}, density={density})")
    return merged


# ── Method 4: TIES Merging ────────────────────

def ties_merge(base_sd, ft_sds, scaling_factor=0.5, density=0.2):
    """
    TIES: Trim → Elect sign → Merge

    Fixes the core failure mode of Task Arithmetic:
    when tasks push the same parameter in opposite directions
    they cancel out. TIES resolves this via majority sign voting.

    Step 1 TRIM:  Keep top-density% of each task vector
    Step 2 ELECT: Majority vote determines correct sign per parameter
    Step 3 MERGE: Average only values agreeing with elected sign
    """
    # Step 1: Trim
    trimmed = []
    for ft in ft_sds:
        tv = compute_task_vector(base_sd, ft)
        trimmed.append(sparsify_task_vector(tv, density=density))

    merged         = {}
    conflict_rates = []

    for key in base_sd:
        stacked   = torch.stack([tv[key] for tv in trimmed])   # [n, ...]

        # Step 2: Elect sign via majority vote
        sign_vote           = stacked.sign().sum(dim=0).sign()
        sign_vote[sign_vote == 0] = 1   # break ties positive

        # Track conflict rate for analysis output
        if stacked.shape[0] == 2:
            both_nz = (stacked[0] != 0) & (stacked[1] != 0)
            if both_nz.any():
                conflict = (stacked[0][both_nz].sign() !=
                            stacked[1][both_nz].sign()).float().mean().item()
                conflict_rates.append(conflict)

        # Step 3: Keep agreeing values, then mean
        agreed    = stacked * (stacked.sign() == sign_vote.unsqueeze(0)).float()
        count     = (agreed != 0).float().sum(dim=0).clamp(min=1)
        merged_tv = agreed.sum(dim=0) / count

        merged[key] = base_sd[key].float() + scaling_factor * merged_tv

    if conflict_rates:
        avg = sum(conflict_rates) / len(conflict_rates)
        print(f"  Avg sign conflict rate: {avg:.4f}  ({avg*100:.1f}%)")

    print(f"✓ TIES Merging complete  (λ={scaling_factor}, density={density})")
    return merged


# ── Method 5: Direct Adapter Merge (Strategy B) ──

def merge_adapters_direct(adapter_paths):
    """
    Merge LoRA adapters WITHOUT fusing into base weights first.
    Averages the low-rank A and B matrices directly.

    Comparison point for your report:
    does merging in adapter-space vs. full weight-space
    produce meaningfully different results?
    This is an open research question.
    """
    from peft import PeftModel

    print("\nDirect Adapter Merging (Strategy B)")
    base           = load_base_model(dtype=torch.float32, device="cpu")
    adapter_models = [PeftModel.from_pretrained(copy.deepcopy(base), p)
                      for p in adapter_paths]
    merged_model   = copy.deepcopy(adapter_models[0])

    for name, module in merged_model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        peers = [dict(am.named_modules()).get(name) for am in adapter_models[1:]]
        all_A = ([module.lora_A["default"].weight.data] +
                 [m.lora_A["default"].weight.data for m in peers if m])
        all_B = ([module.lora_B["default"].weight.data] +
                 [m.lora_B["default"].weight.data for m in peers if m])
        module.lora_A["default"].weight.data = torch.stack(all_A).mean(dim=0)
        module.lora_B["default"].weight.data = torch.stack(all_B).mean(dim=0)

    print("✓ Direct adapter merge complete")
    return merged_model


# ── Helpers ───────────────────────────────────

def skip_if_exists(path):
    if os.path.exists(path):
        print(f"  [skip] {path}")
        return True
    return False


def vtag(val):
    """Float → 2-digit string: 0.6 → '06', 1.0 → '10'"""
    return f"{int(round(val * 10)):02d}"


# ── Run all merges ────────────────────────────

if __name__ == "__main__":

    # Load everything into RAM at once (36GB — fine with 125GB available)
    sds     = load_all_state_dicts()
    base_sd = sds["base"]
    ft_sds  = [sds["emotion"], sds["summary"]]

    # ── v1: baseline merges (skip if already done) ──
    print("\n══ v1 merges ══")

    p = "./state_dicts/v1/sd_weight_avg.pt"
    if not skip_if_exists(p):
        print("\n── Method 1: Weight Averaging ──")
        torch.save(weight_average(ft_sds), p)

    p = "./state_dicts/v1/sd_task_arith.pt"
    if not skip_if_exists(p):
        print("\n── Method 2: Task Arithmetic (λ=0.5) ──")
        torch.save(task_arithmetic(base_sd, ft_sds, scaling_factor=0.5), p)

    p = "./state_dicts/v1/sd_breadcrumbs.pt"
    if not skip_if_exists(p):
        print("\n── Method 3: Breadcrumbs (λ=0.5, d=0.2) ──")
        torch.save(breadcrumbs(base_sd, ft_sds, scaling_factor=0.5, density=0.2), p)

    p = "./state_dicts/v1/sd_ties.pt"
    if not skip_if_exists(p):
        print("\n── Method 4: TIES (λ=0.5, d=0.2) ──")
        torch.save(ties_merge(base_sd, ft_sds, scaling_factor=0.5, density=0.2), p)

    print("\n✓ v1 complete")

    # ── v2: parameter sweeps ─────────────────────
    #
    # Task Arithmetic — vary λ, keep density fixed (no pruning)
    #   λ ∈ [0.3, 0.4, 0.6, 0.7, 0.8, 1.0]
    #   v1 did λ=0.5; stop at 1.0 (collapse observed at λ=1.5)
    #
    # Breadcrumbs — vary density, keep λ=0.5 fixed
    #   density ∈ [0.3, 0.4, 0.5, 0.6, 0.8]
    #   v1 d=0.2 was too aggressive; d≥0.5 expected to match Task Arithmetic
    #
    # TIES — same density sweep as Breadcrumbs, λ=0.5 fixed
    #   majority vote needs ≥3 tasks to shine, but higher density may help with 2

    TA_LAMBDAS   = [0.3, 0.4, 0.6, 0.7, 0.8, 1.0]
    BC_DENSITIES = [0.3, 0.4, 0.5, 0.6, 0.8]

    print("\n\n══ v2: Task Arithmetic λ sweep ══")
    for lam in TA_LAMBDAS:
        p = f"./state_dicts/v2/sd_task_arith_v2_{vtag(lam)}.pt"
        if not skip_if_exists(p):
            print(f"\n  λ = {lam}")
            torch.save(task_arithmetic(base_sd, ft_sds, scaling_factor=lam), p)
            print(f"  ✓ saved → {p}")

    print("\n\n══ v2: Breadcrumbs density sweep (λ=0.5) ══")
    for d in BC_DENSITIES:
        p = f"./state_dicts/v2/sd_breadcrumbs_v2_d{vtag(d)}.pt"
        if not skip_if_exists(p):
            print(f"\n  density = {d}")
            torch.save(breadcrumbs(base_sd, ft_sds, scaling_factor=0.5, density=d), p)
            print(f"  ✓ saved → {p}")

    print("\n\n══ v2: TIES density sweep (λ=0.5) ══")
    for d in BC_DENSITIES:
        p = f"./state_dicts/v2/sd_ties_v2_d{vtag(d)}.pt"
        if not skip_if_exists(p):
            print(f"\n  density = {d}")
            torch.save(ties_merge(base_sd, ft_sds, scaling_factor=0.5, density=d), p)
            print(f"  ✓ saved → {p}")

    print("\n✓ All v2 merges saved to ./state_dicts/v2/")

    # ── 3-task merges (requires NLI state dict) ───
    if "nli" not in sds:
        print("\n[skip] 3-task merges — run c_03_finetune_nli.py and "
              "f_05_linearize_adapters.py first")
    else:
        ft_sds_3task = [sds["emotion"], sds["summary"], sds["nli"]]

        print("\n\n══ 3-task merges (emotion + summary + NLI) ══")

        p = "./state_dicts/v3/sd_weight_avg_3task.pt"
        if not skip_if_exists(p):
            print("\n── Method 1: Weight Averaging (3-task) ──")
            torch.save(weight_average(ft_sds_3task), p)
            print(f"  ✓ saved → {p}")

        p = "./state_dicts/v3/sd_task_arith_3task.pt"
        if not skip_if_exists(p):
            print("\n── Method 2: Task Arithmetic λ=0.5 (3-task) ──")
            torch.save(task_arithmetic(base_sd, ft_sds_3task, scaling_factor=0.5), p)
            print(f"  ✓ saved → {p}")

        p = "./state_dicts/v3/sd_breadcrumbs_3task.pt"
        if not skip_if_exists(p):
            print("\n── Method 3: Breadcrumbs λ=0.5, d=0.5 (3-task) ──")
            torch.save(breadcrumbs(base_sd, ft_sds_3task, scaling_factor=0.5, density=0.5), p)
            print(f"  ✓ saved → {p}")

        p = "./state_dicts/v3/sd_ties_3task.pt"
        if not skip_if_exists(p):
            print("\n── Method 4: TIES λ=0.5, d=0.4 (3-task) ──")
            torch.save(ties_merge(base_sd, ft_sds_3task, scaling_factor=0.5, density=0.4), p)
            print(f"  ✓ saved → {p}")

        print("\n✓ All 3-task merges saved to ./state_dicts/v3/")