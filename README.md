# COMP6861 — Model Merging for Multitask Language Models
#### Islem Khemissi [LinkedIn](https://ca.linkedin.com/in/islem-khemissi) 

Replication package for the model merging project.
Base model: `meta-llama/Llama-3.2-3B-Instruct` (full experiment) /
`Qwen/Qwen2.5-0.5B-Instruct` (Colab demo notebook).

---

## Repository Layout

```
COMP6861_MergeLLMs/
├── README.md                          ← this file
├── report.md                          ← full experimental narrative
├── download_models.py                 ← download pre-trained adapters & merged models from HuggingFace
├── notebook/
│   ├── model_merging_demo.ipynb           ← clean demo notebook (run on Google Colab)
│   └── model_merging_demo_executed.ipynb  ← executed version with all cell outputs
└── full_experiment/
    ├── requirements.txt
    ├── utils/
    │   └── model_utils.py             ← shared constants, loaders, finetune(), generate()
    ├── a_00_setup.py                  ← environment verification
    ├── b_01_prepare_datasets.py       ← dataset loading and formatting
    ├── c_02_finetune_emotion.py       ← emotion specialist fine-tuning
    ├── c_03_finetune_nli.py           ← NLI specialist fine-tuning
    ├── d_03_finetune_summary.py       ← summarization specialist fine-tuning
    ├── e_04_finetune_multitask.py     ← multitask baseline fine-tuning
    ├── f_05_linearize_adapters.py     ← fuse LoRA adapters → full-rank state dicts
    ├── g_06_merge_methods.py          ← all merging methods (v1 + v2 sweep + 3-task)
    ├── g_06_v4_ties_sweep.py          ← TIES 4×4 hyperparameter sweep (v4)
    ├── h_07_evaluate.py               ← evaluation: emotion accuracy, ROUGE-L, NLI accuracy
    ├── h_07_v4_evaluate.py            ← evaluation for the v4 TIES sweep
    ├── i_08_analysis.py               ← plots and analysis
    ├── z_99_push_to_hf.py             ← push adapters and merged models to HuggingFace
    └── results/
        ├── metrics.csv                ← v1 evaluation (2-task, default params)
        ├── metrics_v2.csv             ← v2 parameter sweep (2-task)
        ├── metrics_3task.csv          ← 3-task evaluation (v3)
        └── metrics_v4_ties_sweep.csv  ← TIES 4×4 sweep results (v4)
```

---

## Server Specifications (Full Experiment)

| Component | Spec |
|---|---|
| GPU | 2× NVIDIA Quadro RTX 8000 (48 GB VRAM each, 96 GB total) |
| CPU | AMD EPYC 7401P (48 threads) |
| RAM | 125 GB |
| Disk | 976 GB free |
| OS | Linux |
| Python | 3.12.8 |
| CUDA | 12.0 |
| PyTorch | 2.10.0 |

All state dicts were loaded simultaneously during merging (125 GB RAM made this feasible without streaming).
No quantization was used — bfloat16 for training/inference, float32 for merging.

---

## Option A — Run the Demo Notebook on Google Colab (recommended)

The notebook (`notebook/model_merging_demo.ipynb`) is self-contained and runs on a free T4 GPU.
It uses `Qwen/Qwen2.5-0.5B-Instruct` (no HuggingFace token required) and completes in ~45–60 minutes.

1. Upload `notebook/model_merging_demo.ipynb` to Google Drive.
2. Open with Google Colab → **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**.
4. When done: **File → Download → Download .ipynb** (saves all cell outputs).

The executed version with full output traces is provided at
`notebook/model_merging_demo_executed.ipynb`.

---

## Option B — Reproduce the Full Experiment

### Hardware requirements
- GPU with ≥ 24 GB VRAM for fine-tuning (tested on 2× NVIDIA Quadro RTX 8000, 48 GB each)
- ≥ 32 GB RAM for merging (state dicts loaded in float32)
- ~200 GB free disk space

### Setup

```bash
git clone https://github.com/islem-kms/COMP6861_MergeLLMs
cd ConvAI/v2

python -m venv .env
source .env/bin/activate
pip install -r full_experiment/requirements.txt

cd full_experiment
python a_00_setup.py      # verify environment and log in to HuggingFace
```

> **Note:** `meta-llama/Llama-3.2-3B-Instruct` is gated. You need a HuggingFace account
> with access approved at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct.

### Step-by-step execution

Run each script in order. Use `screen` to survive SSH disconnects:

```bash
screen -S convai
# then run commands below; Ctrl+A D to detach, screen -r convai to re-attach
```

**1 — Prepare datasets**
```bash
python b_01_prepare_datasets.py
```

**2 — Fine-tune specialists** (run sequentially; each uses ~24 GB VRAM)
```bash
python c_02_finetune_emotion.py
python d_03_finetune_summary.py
python c_03_finetune_nli.py
python e_04_finetune_multitask.py   # optional multitask baseline
```

**3 — Linearize adapters** (CPU, ~20 min)
```bash
python f_05_linearize_adapters.py
```

**4 — Merge (2-task v1 + v2 sweep + 3-task v3)**
```bash
python g_06_merge_methods.py
```

**5 — Evaluate (v1 + v2 + 3-task)**
```bash
python h_07_evaluate.py
```

**6 — TIES parameter sweep (v4)**
```bash
python g_06_v4_ties_sweep.py && python h_07_v4_evaluate.py
```

**7 — Generate plots**
```bash
python i_08_analysis.py
```

All scripts use a **skip-if-exists** pattern: re-running after an interruption safely
skips completed steps without recomputing.

---

## Pre-trained Models on HuggingFace

All fine-tuned adapters and merged state dicts are publicly available.
Use `download_models.py` to fetch them locally (see below).

### Fine-tuned LoRA adapters (Llama-3.2-3B-Instruct)

| Model | HuggingFace URL |
|---|---|
| Emotion specialist | https://huggingface.co/islemkms/llama-3.2-3b-emotion-lora |
| Summarization specialist | https://huggingface.co/islemkms/llama-3.2-3b-summary-lora |
| NLI specialist | https://huggingface.co/islemkms/llama-3.2-3b-nli-lora |
| Multitask baseline | https://huggingface.co/islemkms/llama-3.2-3b-multitask-lora |

### Merged full-weight state dicts

| Repo | Contents | HuggingFace URL |
|---|---|---|
| merged-v1 | 2-task merges, default params | https://huggingface.co/islemkms/llama-3.2-3b-merged-v1 |
| merged-v2 | 2-task parameter sweep (16 configs) | https://huggingface.co/islemkms/llama-3.2-3b-merged-v2 |
| merged-v3 | 3-task merges, default params | https://huggingface.co/islemkms/llama-3.2-3b-merged-v3 |
| merged-v4 | TIES 4×4 sweep (λ × density) | https://huggingface.co/islemkms/llama-3.2-3b-merged-v4 |

### Downloading and using a merged model

```bash
python download_models.py --repo islemkms/llama-3.2-3b-merged-v4 --out ./downloaded_models
```

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
sd = torch.load("downloaded_models/sd_ties_v4_l10_d02.pt", map_location="cpu", weights_only=True)
model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()})
model.eval()
```

---

## Key Results

### 2-task experiment (emotion + summarization) — Llama-3.2-3B-Instruct

| Model | Emotion acc | ROUGE-L |
|---|---|---|
| Base (zero-shot) | 0.318 | 0.181 |
| Emotion specialist | 0.912 | 0.193 |
| Summary specialist | 0.408 | 0.298 |
| Multitask baseline | 0.882 | 0.301 |
| Weight Average | 0.782 | 0.321 |
| Task Arithmetic λ=0.7 | **0.842** | 0.314 |
| Breadcrumbs d=0.5 | 0.766 | **0.323** |
| TIES d=0.4 | 0.662 | 0.295 |

### 3-task experiment (emotion + summarization + NLI) — Llama-3.2-3B-Instruct

| Model | Emotion acc | NLI acc | ROUGE-L |
|---|---|---|---|
| Task Arithmetic λ=0.5 | 0.630 | **0.834** | 0.298 |
| Breadcrumbs λ=0.5, d=0.5 | **0.636** | 0.814 | 0.291 |
| TIES λ=1.0, d=0.2 *(v4 best)* | 0.710 | 0.816 | **0.303** |
| TIES λ=0.5, d=0.4 *(v3 default)* | 0.580 | 0.672 | 0.298 |

---

## Citation / Attribution

Experiment conducted on a workstation with 2× NVIDIA Quadro RTX 8000 (96 GB VRAM).
Full codebase: https://github.com/islem-kms/COMP6861_MergeLLMs
Models: https://huggingface.co/islemkms
