from utils.model_utils import *
from b_01_prepare_datasets import nli_ds

model_nli = finetune(
    train_dataset=nli_ds["train"],
    eval_dataset=nli_ds["validation"],
    output_dir=ADAPTER_NLI,
    num_epochs=3,
)

# ── Sanity check ──────────────────────────────
print("\n--- Sanity check: NLI specialist ---")
test_cases = [
    ("A man is eating pizza.",          "The man is consuming food.",       "entailment"),
    ("A black race car starts up.",     "A man is driving down a road.",    "neutral"),
    ("A woman is walking her dog.",     "The woman has no pets.",           "contradiction"),
    ("Children are playing outside.",   "Kids are having fun in the park.", "entailment"),
]
for premise, hypothesis, expected in test_cases:
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "Determine the relationship between the premise and hypothesis. "
        "Reply with one word: entailment, neutral, or contradiction.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    pred   = generate(model_nli, prompt, max_new_tokens=5).lower()
    status = "✓" if expected in pred else "✗"
    print(f"  {status}  Expected={expected:15s}  Got={pred}")
