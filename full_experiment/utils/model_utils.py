# utils/model_utils.py

import os
import torch

os.makedirs("./adapters/emotion",   exist_ok=True)
os.makedirs("./adapters/summary",   exist_ok=True)
os.makedirs("./adapters/multitask", exist_ok=True)
os.makedirs("./adapters/nli",       exist_ok=True)
os.makedirs("./state_dicts/v1",     exist_ok=True)
os.makedirs("./state_dicts/v2",     exist_ok=True)
os.makedirs("./state_dicts/v3",     exist_ok=True)
os.makedirs("./state_dicts/v4",     exist_ok=True)
os.makedirs("./results/plots",      exist_ok=True)

# ── Paths ─────────────────────────────────────
MODEL_NAME        = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_EMOTION   = "./adapters/emotion"
ADAPTER_SUMMARY   = "./adapters/summary"
ADAPTER_MULTITASK = "./adapters/multitask"
ADAPTER_NLI       = "./adapters/nli"
SD_BASE           = "./state_dicts/v1/sd_base.pt"
SD_EMOTION        = "./state_dicts/v1/sd_emotion.pt"
SD_SUMMARY        = "./state_dicts/v1/sd_summary.pt"
SD_MULTITASK      = "./state_dicts/v1/sd_multitask.pt"
SD_NLI            = "./state_dicts/v1/sd_nli.pt"

# ── Task config ───────────────────────────────
MAX_INPUT_LEN  = 384
MAX_TARGET_LEN = 64
LABEL_MAP      = {0:"sadness", 1:"joy",  2:"love",
                  3:"anger",   4:"fear", 5:"surprise"}
LABEL_MAP_INV  = {v: k for k, v in LABEL_MAP.items()}

NLI_LABEL_MAP     = {0: "entailment", 1: "neutral", 2: "contradiction"}
NLI_LABEL_MAP_INV = {v: k for k, v in NLI_LABEL_MAP.items()}


# ── Tokenizer ─────────────────────────────────
# Loaded lazily so importing model_utils never triggers a model download
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _tokenizer.pad_token    = _tokenizer.eos_token
        _tokenizer.padding_side = "right"
    return _tokenizer


# ── LoRA config ───────────────────────────────
def get_lora_config():
    from peft import LoraConfig, TaskType
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )


# ── Model loader ──────────────────────────────
def load_base_model(dtype=None, device="auto"):
    from transformers import AutoModelForCausalLM
    if dtype is None:
        dtype = torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device,
    )


# ── Fine-tuning function ──────────────────────
def finetune(train_dataset, eval_dataset, output_dir,
             num_epochs=3, lr=2e-4):
    # Heavy imports only happen when finetune() is actually called
    from transformers import TrainingArguments
    from peft import get_peft_model
    from trl import SFTTrainer                 # ← moved here

    tokenizer = get_tokenizer()
    model     = load_base_model(dtype=torch.bfloat16)
    model     = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        # dataloader_num_workers=8,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        tokenizer=tokenizer,
        dataset_text_field="full_text",
        max_seq_length=MAX_INPUT_LEN,
        packing=False,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✓ Adapter saved → {output_dir}")
    return model


# ── Inference helper ──────────────────────────
def generate(model, prompt, max_new_tokens=20):
    tokenizer = get_tokenizer()
    inputs    = tokenizer(prompt, return_tensors="pt",
                          truncation=True,
                          max_length=MAX_INPUT_LEN).to(model.device)
    n_input   = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    return get_tokenizer().decode(
        out[0][n_input:], skip_special_tokens=True
    ).strip()