import torch, evaluate, pandas as pd, os
from transformers import AutoModelForCausalLM
from peft import PeftModel
from utils.model_utils import *
from b_01_prepare_datasets import emotion_raw, summary_raw, nli_raw

rouge_metric = evaluate.load("rouge")
os.makedirs("./results", exist_ok=True)


# ── Model loaders ─────────────────────────────

def load_from_sd(sd_path):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="auto"
    )
    model.load_state_dict(torch.load(sd_path, map_location="cpu"))
    model.eval()
    return model


def load_from_adapter(adapter_path):
    base  = load_base_model(dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


# ── Prompt builders ───────────────────────────

def build_emotion_prompt(text):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "Classify the emotion. Reply with one word: "
        "sadness, joy, love, anger, fear, or surprise.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Text: {text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def build_summary_prompt(dialogue):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "Write a concise one or two sentence summary of the dialogue.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Dialogue:\n{dialogue}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def build_nli_prompt(premise, hypothesis):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "Determine the relationship between the premise and hypothesis. "
        "Reply with one word: entailment, neutral, or contradiction.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


# ── Task evaluations ──────────────────────────

def eval_emotion(model, n_samples=500):
    dataset = emotion_raw["test"].select(
        range(min(n_samples, len(emotion_raw["test"])))
    )
    correct = 0
    for ex in dataset:
        pred    = generate(model, build_emotion_prompt(ex["text"]),
                           max_new_tokens=5).lower()
        matched = next((l for l in LABEL_MAP.values() if l in pred), None)
        correct += int(matched == LABEL_MAP[ex["label"]])
    return correct / len(dataset)


def eval_nli(model, n_samples=500):
    dataset = nli_raw["validation_matched"].filter(
        lambda ex: ex["label"] != -1, num_proc=1
    ).select(range(min(n_samples, 9815)))
    correct = 0
    for ex in dataset:
        pred    = generate(model,
                           build_nli_prompt(ex["premise"], ex["hypothesis"]),
                           max_new_tokens=5).lower()
        matched = next((l for l in NLI_LABEL_MAP.values() if l in pred), None)
        correct += int(matched == NLI_LABEL_MAP[ex["label"]])
    return correct / len(dataset)


def eval_summarization(model, n_samples=200):
    dataset = summary_raw["test"].select(
        range(min(n_samples, len(summary_raw["test"])))
    )
    preds, refs = [], []
    for ex in dataset:
        pred = generate(model, build_summary_prompt(ex["dialogue"]),
                        max_new_tokens=MAX_TARGET_LEN)
        preds.append(pred)
        refs.append(ex["summary"])
    return rouge_metric.compute(predictions=preds, references=refs)


# ── Full evaluation pipeline ──────────────────

def vtag(val):
    """Float → 2-digit string: 0.6 → '06', 1.0 → '10'"""
    return f"{int(round(val * 10)):02d}"


def run_eval(models_dict, csv_path):
    rows = {}
    for name, cfg in models_dict.items():
        if not os.path.exists(cfg["path"]):
            print(f"  [skip] {cfg['path']} not found")
            continue
        print(f"\n{'='*55}\nEvaluating: {name}\n{'='*55}")

        model  = (load_from_sd(cfg["path"]) if cfg["type"] == "sd"
                  else load_from_adapter(cfg["path"]))
        em_acc = eval_emotion(model)
        rouge  = eval_summarization(model)

        rows[name] = {
            "emotion_accuracy": round(em_acc,          4),
            "rouge1":           round(rouge["rouge1"], 4),
            "rouge2":           round(rouge["rouge2"], 4),
            "rougeL":           round(rouge["rougeL"], 4),
        }
        print(f"  Emotion Acc  : {em_acc:.4f}")
        print(f"  ROUGE-1      : {rouge['rouge1']:.4f}")
        print(f"  ROUGE-L      : {rouge['rougeL']:.4f}")

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows).T
    df.to_csv(csv_path)
    print(f"\nSaved → {csv_path}")
    print(df.to_string())
    return df


# ── v1 models ─────────────────────────────────

MODELS_V1 = {
    "base_zero_shot":       {"type": "sd",      "path": SD_BASE},
    "emotion_specialist":   {"type": "adapter", "path": ADAPTER_EMOTION},
    "summary_specialist":   {"type": "adapter", "path": ADAPTER_SUMMARY},
    "multitask_baseline":   {"type": "adapter", "path": ADAPTER_MULTITASK},
    "weight_average":       {"type": "sd",      "path": "./state_dicts/v1/sd_weight_avg.pt"},
    "task_arithmetic_0.5":  {"type": "sd",      "path": "./state_dicts/v1/sd_task_arith.pt"},
    "breadcrumbs_d02":      {"type": "sd",      "path": "./state_dicts/v1/sd_breadcrumbs.pt"},
    "ties_d02":             {"type": "sd",      "path": "./state_dicts/v1/sd_ties.pt"},
}

CSV_V1 = "./results/metrics.csv"
if not os.path.exists(CSV_V1):
    print("\n\n══ Evaluating v1 models ══")
    run_eval(MODELS_V1, CSV_V1)
else:
    print(f"[skip] {CSV_V1} already exists")


# ── v2 models: parameter sweep ────────────────
#
# Task Arithmetic  — λ ∈ {0.3, 0.4, 0.6, 0.7, 0.8, 1.0}  (v1 was 0.5)
# Breadcrumbs      — density ∈ {0.3, 0.4, 0.5, 0.6, 0.8}  (v1 was 0.2, λ=0.5 fixed)
# TIES             — density ∈ {0.3, 0.4, 0.5, 0.6, 0.8}  (v1 was 0.2, λ=0.5 fixed)

TA_LAMBDAS   = [0.3, 0.4, 0.6, 0.7, 0.8, 1.0]
BC_DENSITIES = [0.3, 0.4, 0.5, 0.6, 0.8]

MODELS_V2 = {}

for lam in TA_LAMBDAS:
    name = f"task_arith_v2_{vtag(lam)}"
    MODELS_V2[name] = {"type": "sd",
                        "path": f"./state_dicts/v2/sd_{name}.pt"}

for d in BC_DENSITIES:
    name = f"breadcrumbs_v2_d{vtag(d)}"
    MODELS_V2[name] = {"type": "sd",
                        "path": f"./state_dicts/v2/sd_{name}.pt"}

for d in BC_DENSITIES:
    name = f"ties_v2_d{vtag(d)}"
    MODELS_V2[name] = {"type": "sd",
                        "path": f"./state_dicts/v2/sd_{name}.pt"}

CSV_V2 = "./results/metrics_v2.csv"
if not os.path.exists(CSV_V2):
    print("\n\n══ Evaluating v2 models ══")
    run_eval(MODELS_V2, CSV_V2)
else:
    print(f"[skip] {CSV_V2} already exists")


# ── 3-task evaluation ─────────────────────────
#
# Adds nli_accuracy column alongside emotion_accuracy and rougeL.
# Only runs when metrics_3task.csv does not exist yet.

def run_eval_3task(models_dict, csv_path):
    rows = {}
    for name, cfg in models_dict.items():
        if not os.path.exists(cfg["path"]):
            print(f"  [skip] {cfg['path']} not found")
            continue
        print(f"\n{'='*55}\nEvaluating: {name}\n{'='*55}")

        model   = (load_from_sd(cfg["path"]) if cfg["type"] == "sd"
                   else load_from_adapter(cfg["path"]))
        em_acc  = eval_emotion(model)
        nli_acc = eval_nli(model)
        rouge   = eval_summarization(model)

        rows[name] = {
            "emotion_accuracy": round(em_acc,          4),
            "nli_accuracy":     round(nli_acc,         4),
            "rouge1":           round(rouge["rouge1"], 4),
            "rouge2":           round(rouge["rouge2"], 4),
            "rougeL":           round(rouge["rougeL"], 4),
        }
        print(f"  Emotion Acc  : {em_acc:.4f}")
        print(f"  NLI Acc      : {nli_acc:.4f}")
        print(f"  ROUGE-1      : {rouge['rouge1']:.4f}")
        print(f"  ROUGE-L      : {rouge['rougeL']:.4f}")

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows).T
    df.to_csv(csv_path)
    print(f"\nSaved → {csv_path}")
    print(df.to_string())
    return df


MODELS_3TASK = {
    "emotion_specialist":      {"type": "adapter", "path": ADAPTER_EMOTION},
    "summary_specialist":      {"type": "adapter", "path": ADAPTER_SUMMARY},
    "nli_specialist":          {"type": "adapter", "path": ADAPTER_NLI},
    "multitask_baseline":      {"type": "adapter", "path": ADAPTER_MULTITASK},
    "weight_avg_3task":        {"type": "sd",      "path": "./state_dicts/v3/sd_weight_avg_3task.pt"},
    "task_arith_3task":        {"type": "sd",      "path": "./state_dicts/v3/sd_task_arith_3task.pt"},
    "breadcrumbs_3task":       {"type": "sd",      "path": "./state_dicts/v3/sd_breadcrumbs_3task.pt"},
    "ties_3task":              {"type": "sd",      "path": "./state_dicts/v3/sd_ties_3task.pt"},
}

CSV_3TASK = "./results/metrics_3task.csv"
if not os.path.exists(CSV_3TASK):
    if os.path.exists(ADAPTER_NLI + "/adapter_config.json"):
        print("\n\n══ Evaluating 3-task models ══")
        run_eval_3task(MODELS_3TASK, CSV_3TASK)
    else:
        print(f"[skip] NLI adapter not found — run c_03_finetune_nli.py first")
else:
    print(f"[skip] {CSV_3TASK} already exists")