from utils.model_utils import *
from b_01_prepare_datasets import emotion_ds

model_emotion = finetune(
    train_dataset=emotion_ds["train"],
    eval_dataset=emotion_ds["validation"],
    output_dir=ADAPTER_EMOTION,
    num_epochs=3,
)

# ── Sanity check ──────────────────────────────
print("\n--- Sanity check: Emotion specialist ---")
test_cases = [
    ("I just got promoted, this is the best day ever!", "joy"),
    ("I miss my grandmother so much.",                  "sadness"),
    ("This is absolutely outrageous.",                  "anger"),
    ("I think there is someone in the house.",          "fear"),
]
for text, expected in test_cases:
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "Classify the emotion. Reply with one word: "
        "sadness, joy, love, anger, fear, or surprise.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Text: {text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    pred = generate(model_emotion, prompt, max_new_tokens=5).lower()
    status = "✓" if expected in pred else "✗"
    print(f"  {status}  Expected={expected:8s}  Got={pred}")