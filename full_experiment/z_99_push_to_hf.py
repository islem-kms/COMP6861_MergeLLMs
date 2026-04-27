"""
push_to_hf.py — Upload all v1 and v2 models to HuggingFace Hub

Repos created:
  islemkms/llama-3.2-3b-emotion-lora    (~36 MB)
  islemkms/llama-3.2-3b-summary-lora    (~36 MB)
  islemkms/llama-3.2-3b-multitask-lora  (~36 MB)
  islemkms/llama-3.2-3b-merged-v1       (~56 GB  — 4 merged state dicts)
  islemkms/llama-3.2-3b-merged-v2       (~224 GB — 16 merged state dicts)

Skipped:
  sd_base.pt        — redundant (Meta's original weights are on HF)
  sd_emotion.pt     — redundant (regenerable from adapter via f_05_linearize_adapters.py)
  sd_summary.pt     — redundant (same)
  sd_multitask.pt   — redundant (same)

Run with: python z_99_push_to_hf.py
Requires: huggingface-cli login  (or HF_TOKEN env var)
"""

import os
from huggingface_hub import HfApi

HF_USER = "islemkms"
api     = HfApi()


def already_on_hf(repo_id, filename):
    """Return True if filename exists in the repo (skip re-upload)."""
    try:
        existing = {f.rfilename for f in api.list_repo_files(repo_id, repo_type="model")}
        return filename in existing
    except Exception:
        return False


# ── 1. Fine-tuned LoRA adapters ───────────────
# Already in standard PEFT format — upload the whole folder.
# Training checkpoints (checkpoint-*) are skipped to save space.

ADAPTERS = [
    (f"{HF_USER}/llama-3.2-3b-emotion-lora",   "./adapters/emotion"),
    (f"{HF_USER}/llama-3.2-3b-summary-lora",   "./adapters/summary"),
    (f"{HF_USER}/llama-3.2-3b-multitask-lora", "./adapters/multitask"),
]

for repo_id, folder in ADAPTERS:
    print(f"\nPushing adapter → {repo_id}")
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    if already_on_hf(repo_id, "adapter_config.json"):
        print(f"  [skip] already on HF")
        continue
    api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*"],
        commit_message="v1: initial upload",
    )
    print(f"  ✓ {repo_id}")


# ── 2. Merged state dicts ─────────────────────
# Four merging strategies, all 2-task (emotion + summary).
# Each file is ~14 GB; total ~56 GB — HF handles large files natively.

MERGED_REPO = f"{HF_USER}/llama-3.2-3b-merged-v1"
api.create_repo(MERGED_REPO, repo_type="model", exist_ok=True, private=False)

MERGED_FILES = [
    ("sd_weight_avg.pt",  "./state_dicts/v1/sd_weight_avg.pt"),
    ("sd_task_arith.pt",  "./state_dicts/v1/sd_task_arith.pt"),
    ("sd_breadcrumbs.pt", "./state_dicts/v1/sd_breadcrumbs.pt"),
    ("sd_ties.pt",        "./state_dicts/v1/sd_ties.pt"),
]

for path_in_repo, local_path in MERGED_FILES:
    if already_on_hf(MERGED_REPO, path_in_repo):
        print(f"  [skip] {path_in_repo} already on HF")
        continue
    size_gb = os.path.getsize(local_path) / 1e9
    print(f"\nUploading {path_in_repo}  ({size_gb:.1f} GB) ...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=MERGED_REPO,
        repo_type="model",
        commit_message=f"v1: {path_in_repo}",
    )
    print(f"  ✓ {path_in_repo}")


print(f"\n\n✓ All v1 models pushed.")


# ── 3. v2 merged state dicts ──────────────────
# Parameter sweep: 6 Task Arithmetic (λ) + 5 Breadcrumbs (density) + 5 TIES (density)
# Each file ~14 GB; 16 files total ~224 GB.

MERGED_REPO_V2 = f"{HF_USER}/llama-3.2-3b-merged-v2"
api.create_repo(MERGED_REPO_V2, repo_type="model", exist_ok=True, private=False)

V2_DIR = "./state_dicts/v2"
v2_files = sorted(f for f in os.listdir(V2_DIR) if f.endswith(".pt"))

for filename in v2_files:
    if already_on_hf(MERGED_REPO_V2, filename):
        print(f"  [skip] {filename} already on HF")
        continue
    local_path = os.path.join(V2_DIR, filename)
    size_gb    = os.path.getsize(local_path) / 1e9
    print(f"\nUploading {filename}  ({size_gb:.1f} GB) ...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=MERGED_REPO_V2,
        repo_type="model",
        commit_message=f"v2: {filename}",
    )
    print(f"  ✓ {filename}")

print(f"\n\n✓ All v2 models pushed.")
print(f"  https://huggingface.co/{HF_USER}")


# ── 4. v4 TIES sweep state dicts ──────────────
# 2D sweep over scaling_factor × density for 3-task TIES.
# 16 configs, each ~14 GB.

V4_DIR = "./state_dicts/v4"
if os.path.isdir(V4_DIR):
    MERGED_REPO_V4 = f"{HF_USER}/llama-3.2-3b-merged-v4"
    api.create_repo(MERGED_REPO_V4, repo_type="model", exist_ok=True, private=False)

    v4_files = sorted(f for f in os.listdir(V4_DIR) if f.endswith(".pt"))

    if v4_files:
        print(f"\n\n══ Pushing v4 TIES sweep ({len(v4_files)} files) ══")
        for filename in v4_files:
            if already_on_hf(MERGED_REPO_V4, filename):
                print(f"  [skip] {filename} already on HF")
                continue
            local_path = os.path.join(V4_DIR, filename)
            size_gb    = os.path.getsize(local_path) / 1e9
            print(f"\nUploading {filename}  ({size_gb:.1f} GB) ...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=MERGED_REPO_V4,
                repo_type="model",
                commit_message=f"v4: {filename}",
            )
            print(f"  ✓ {filename}")

        print(f"\n\n✓ All v4 models pushed.")
    else:
        print(f"\n[skip] {V4_DIR} is empty — run g_06_v4_ties_sweep.py first")
else:
    print(f"\n[skip] {V4_DIR} not found — run g_06_v4_ties_sweep.py first")

print(f"\n  https://huggingface.co/{HF_USER}")

