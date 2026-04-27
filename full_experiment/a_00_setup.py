# 00_setup.py

import os
import torch
import psutil
from huggingface_hub import login

# ── Step 1: HuggingFace Login ─────────────────
# Run this once. After the first login your token is cached
# at ~/.cache/huggingface/token and you won't need to run it again.
login(token="TOKEN_GOES_HERE")


# ── Step 2: Verify GPUs ───────────────────────
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"GPU count       : {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}  VRAM={props.total_memory/1e9:.1f}GB")


# ── Step 3: Verify RAM & Disk ─────────────────
ram = psutil.virtual_memory()
print(f"\nTotal RAM : {ram.total/1e9:.1f}GB")
print(f"Free  RAM : {ram.available/1e9:.1f}GB")

disk = psutil.disk_usage(os.path.expanduser("~"))
print(f"Free Disk : {disk.free/1e9:.1f}GB")


# ── Step 4: Enable both GPUs ──────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print("\n✓ Both GPUs enabled")


# ── Step 5: Create project directories ───────
dirs = [
    "./adapters/emotion",
    "./adapters/summary",
    "./adapters/multitask",
    "./state_dicts",
    "./results/plots",
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"  ✓ {d}")

print("\n✓ Setup complete. Run 01_prepare_datasets.py next.")