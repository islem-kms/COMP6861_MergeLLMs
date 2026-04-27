import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from utils.model_utils import *


def linearize(adapter_path, save_path):
    """
    Fuses LoRA into base weights:
        W_merged = W_base + (alpha / r) * B @ A

    Loaded on GPU (cuda:0) — 12GB float32 fits easily in 48GB VRAM.
    Previously this had to be done on CPU; not needed here.
    """
    print(f"\nLinearizing: {adapter_path}")
    base  = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cuda:0"         # GPU is fine — 48GB available
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()    # fuses adapter, returns plain nn.Module

    torch.save(model.state_dict(), save_path)
    size = os.path.getsize(save_path) / 1e9
    print(f"  ✓ Saved → {save_path}  ({size:.1f} GB)")

    del model, base
    torch.cuda.empty_cache()


# ── Save base model weights (no adapter) ──────
print("Saving base weights ...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, device_map="cuda:0"
)
torch.save(base.state_dict(), SD_BASE)
del base
torch.cuda.empty_cache()
print(f"  ✓ Base saved  ({os.path.getsize(SD_BASE)/1e9:.1f} GB)")

# ── Linearize each adapter ────────────────────
linearize(ADAPTER_EMOTION,   SD_EMOTION)
linearize(ADAPTER_SUMMARY,   SD_SUMMARY)
linearize(ADAPTER_MULTITASK, SD_MULTITASK)

if os.path.exists(ADAPTER_NLI + "/adapter_config.json"):
    linearize(ADAPTER_NLI, SD_NLI)
else:
    print(f"\n[skip] NLI adapter not found — run c_03_finetune_nli.py first")

# Each .pt file ≈ 12GB — total disk usage ≈ 60GB with NLI (well within 976GB free)
print("\n✓ All state dicts ready.")