"""
Download pre-trained adapters and merged state dicts from HuggingFace.

Usage:
    python download_models.py                        # download everything
    python download_models.py --adapters-only        # LoRA adapters only
    python download_models.py --repo <repo_id>       # one specific repo
    python download_models.py --out ./my_models      # custom output directory

Available repos:
    Adapters:
        islemkms/llama-3.2-3b-emotion-lora
        islemkms/llama-3.2-3b-summary-lora
        islemkms/llama-3.2-3b-nli-lora
        islemkms/llama-3.2-3b-multitask-lora

    Merged state dicts:
        islemkms/llama-3.2-3b-merged-v1   (2-task, default params)
        islemkms/llama-3.2-3b-merged-v2   (2-task parameter sweep, 16 configs)
        islemkms/llama-3.2-3b-merged-v3   (3-task, default params)
        islemkms/llama-3.2-3b-merged-v4   (TIES 4x4 sweep, best: l10_d02)
"""

import argparse
import os
from pathlib import Path

ADAPTER_REPOS = [
    "islemkms/llama-3.2-3b-emotion-lora",
    "islemkms/llama-3.2-3b-summary-lora",
    "islemkms/llama-3.2-3b-nli-lora",
    "islemkms/llama-3.2-3b-multitask-lora",
]

MERGED_REPOS = [
    "islemkms/llama-3.2-3b-merged-v1",
    "islemkms/llama-3.2-3b-merged-v2",
    "islemkms/llama-3.2-3b-merged-v3",
    "islemkms/llama-3.2-3b-merged-v4",
]


def download_repo(repo_id: str, out_dir: Path) -> None:
    from huggingface_hub import snapshot_download, list_repo_files

    repo_name = repo_id.split("/")[-1]
    local_dir = out_dir / repo_name

    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"  [skip] {repo_id} — already at {local_dir}")
        return

    print(f"  Downloading {repo_id} → {local_dir} ...")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    files = list(local_dir.rglob("*"))
    print(f"  Done. {len([f for f in files if f.is_file()])} files saved.")


def main():
    parser = argparse.ArgumentParser(description="Download models from HuggingFace")
    parser.add_argument("--repo", default=None,
                        help="Download a single repo by ID (e.g. islemkms/llama-3.2-3b-merged-v4)")
    parser.add_argument("--adapters-only", action="store_true",
                        help="Download LoRA adapters only (skip merged state dicts)")
    parser.add_argument("--merged-only", action="store_true",
                        help="Download merged state dicts only (skip adapters)")
    parser.add_argument("--out", default="./downloaded_models",
                        help="Output directory (default: ./downloaded_models)")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install huggingface_hub")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.repo:
        repos = [args.repo]
    elif args.adapters_only:
        repos = ADAPTER_REPOS
    elif args.merged_only:
        repos = MERGED_REPOS
    else:
        repos = ADAPTER_REPOS + MERGED_REPOS

    print(f"\nDownloading {len(repos)} repo(s) to {out_dir.resolve()}\n")
    for repo in repos:
        download_repo(repo, out_dir)

    print("\nAll downloads complete.")
    print("\nTo load a merged state dict:")
    print("  import torch")
    print("  from transformers import AutoModelForCausalLM")
    print('  model = AutoModelForCausalLM.from_pretrained(')
    print('      "meta-llama/Llama-3.2-3B-Instruct",')
    print('      torch_dtype=torch.bfloat16, device_map="auto")')
    print('  sd = torch.load(')
    print(f'      "{out_dir}/llama-3.2-3b-merged-v4/sd_ties_v4_l10_d02.pt",')
    print('      map_location="cpu", weights_only=True)')
    print('  model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()})')
    print()
    print("To load a LoRA adapter:")
    print("  from transformers import AutoModelForCausalLM")
    print("  from peft import PeftModel")
    print('  base = AutoModelForCausalLM.from_pretrained(')
    print('      "meta-llama/Llama-3.2-3B-Instruct",')
    print('      torch_dtype=torch.bfloat16, device_map="auto")')
    print(f'  model = PeftModel.from_pretrained(base, "{out_dir}/llama-3.2-3b-emotion-lora")')


if __name__ == "__main__":
    main()
