from datasets import load_dataset, concatenate_datasets, DatasetDict
from utils.model_utils import *


# ─────────────────────────────────────────────
# 2.1  Emotion Classification
# dair-ai/emotion — train=16000, val=2000, test=2000
# Labels: 0=sadness 1=joy 2=love 3=anger 4=fear 5=surprise
# ─────────────────────────────────────────────

emotion_raw = load_dataset("dair-ai/emotion", "split")

def format_emotion(example):
    """
    Formats as instruction-following. Assistant outputs exactly
    one label word — clean string-match evaluation.
    """
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a sentiment analysis assistant. "
        "Classify the emotion in the given text. "
        "Reply with exactly one word: sadness, joy, love, anger, fear, or surprise."
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Text: {example['text']}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{LABEL_MAP[example['label']]}"
        "<|eot_id|>"
    )
    return {"full_text": prompt, "label_str": LABEL_MAP[example["label"]]}

emotion_ds = emotion_raw.map(
    format_emotion,
    remove_columns=emotion_raw["train"].column_names,
    # num_proc=8      # use multiple CPU cores for fast preprocessing
)
print("Emotion sample:\n", emotion_ds["train"][0]["full_text"])


# ─────────────────────────────────────────────
# 2.2  Summarization
# knkarthick/dialogsum — train=12460, val=500, test=1500
# ─────────────────────────────────────────────

summary_raw = load_dataset("knkarthick/dialogsum")

def format_summary(example):
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a summarization assistant. "
        "Write a concise one or two sentence summary of the dialogue."
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Dialogue:\n{example['dialogue']}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{example['summary']}"
        "<|eot_id|>"
    )
    return {"full_text": prompt, "reference_summary": example["summary"]}

summary_ds = summary_raw.map(
    format_summary,
    remove_columns=summary_raw["train"].column_names,
    # num_proc=8
)
print("Summary sample:\n", summary_ds["train"][0]["full_text"])


# ─────────────────────────────────────────────
# 2.3  Multi-task dataset (interleaved)
# ─────────────────────────────────────────────

# Subsample emotion to match summary size and balance training
emotion_sub     = emotion_ds["train"].shuffle(seed=42).select(range(12000))
multitask_train = concatenate_datasets(
    [emotion_sub, summary_ds["train"]]
).shuffle(seed=42)
multitask_val   = concatenate_datasets(
    [emotion_ds["validation"], summary_ds["validation"]]
).shuffle(seed=42)

print(f"\nDataset sizes:")
print(f"  Emotion   — train={len(emotion_ds['train'])}, "
      f"val={len(emotion_ds['validation'])}")
print(f"  Summary   — train={len(summary_ds['train'])}, "
      f"val={len(summary_ds['validation'])}")
print(f"  Multitask — train={len(multitask_train)}, "
      f"val={len(multitask_val)}")


# ─────────────────────────────────────────────
# 2.4  Natural Language Inference
# nyu-mll/multi_nli — train=392702, validation_matched=9815
# Labels: 0=entailment, 1=neutral, 2=contradiction
# Note: some examples carry label=-1 (annotation artifact) — filtered out
# ─────────────────────────────────────────────

nli_raw = load_dataset("nyu-mll/multi_nli")

def format_nli(example):
    """
    Formats as instruction-following. Assistant outputs exactly
    one label word — clean string-match evaluation.
    """
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a natural language inference assistant. "
        "Given a premise and a hypothesis, determine their relationship. "
        "Reply with exactly one word: entailment, neutral, or contradiction."
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Premise: {example['premise']}\n"
        f"Hypothesis: {example['hypothesis']}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{NLI_LABEL_MAP[example['label']]}"
        "<|eot_id|>"
    )
    return {"full_text": prompt, "label_str": NLI_LABEL_MAP[example["label"]]}

# Filter out unlabeled examples (label == -1)
nli_train_raw = nli_raw["train"].filter(
    lambda ex: ex["label"] != -1, num_proc=1
)
nli_val_raw = nli_raw["validation_matched"].filter(
    lambda ex: ex["label"] != -1, num_proc=1
)

# Subsample training to 20000 for speed; keep 1000 validation examples
nli_train_sub = nli_train_raw.shuffle(seed=42).select(range(20000))
nli_val_sub   = nli_val_raw.shuffle(seed=42).select(range(1000))

nli_ds = DatasetDict({
    "train":      nli_train_sub.map(
        format_nli,
        remove_columns=nli_train_sub.column_names,
        num_proc=1,
    ),
    "validation": nli_val_sub.map(
        format_nli,
        remove_columns=nli_val_sub.column_names,
        num_proc=1,
    ),
})
print("NLI sample:\n", nli_ds["train"][0]["full_text"])
print(f"  NLI       — train={len(nli_ds['train'])}, "
      f"val={len(nli_ds['validation'])}")