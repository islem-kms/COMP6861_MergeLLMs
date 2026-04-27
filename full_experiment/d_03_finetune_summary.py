from utils.model_utils import *
from b_01_prepare_datasets import summary_ds, summary_raw

model_summary = finetune(
    train_dataset=summary_ds["train"],
    eval_dataset=summary_ds["validation"],
    output_dir=ADAPTER_SUMMARY,
    num_epochs=3,
)

# ── Sanity check ──────────────────────────────
print("\n--- Sanity check: Summary specialist ---")
sample = summary_raw["test"][0]
prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "Write a concise one or two sentence summary.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    f"Dialogue:\n{sample['dialogue']}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)
pred = generate(model_summary, prompt, max_new_tokens=MAX_TARGET_LEN)
print(f"  Reference : {sample['summary']}")
print(f"  Predicted : {pred}")