# TISER — Temporal Information Semantic Extraction and Reasoning

Pipeline for temporal reasoning over the TISER dataset: preprocessing, fine-tuning with LoRA, single-prompt inference (ablation), and multi-stage Actor–Critic inference. Supports English and Italian, multiple prompt variants, and CSV logging with per-dataset metrics (EM, F1, SM).

---

## Features

- **Dataset preprocessing**: Hierarchical stratified sampling to create train/test subsets while preserving per-dataset and per-context distribution.
- **Fine-tuning**: LoRA-based fine-tuning on TISER with completion-only training; automatic device selection (CUDA, MPS, CPU).
- **Single-prompt inference**: Ablation over prompt variants (standard, only_reasoning, only_timeline, no_reflection, no_timeline, no_reasoning, all_stages) with incremental saving, `--resume`, and OOM recovery.
- **Actor–Critic inference**: Multi-stage pipeline (Actor → Critic → Solver) in 3-stage or 2-stage mode, with optional few-shot prompting and CSV outputs.
- **Metrics**: Exact Match (EM), token F1, and Soft Match (SM) with per-dataset and overall aggregation.

---

## Project structure

```
DNLP-TISER/
├── README.md
├── requirements.txt
├── config (paths and models via src/config.py)
├── data/
│   ├── raw/                    # Raw JSONL (e.g. TISER_train_demo.json, TISER_test_demo.json)
│   └── processed/              # Sampled subsets (e.g. TISER_*_demo_10pct.json)
├── experiments/
│   ├── results/                # CSV results from inference and actor-critic
│   └── logs/
├── checkpoints/                # LoRA / model checkpoints (config references)
├── finetuned_models/           # Fine-tuned LoRA adapters (output of run_finetuning.py)
├── scripts/
│   ├── run_preprocessing_dataset.py   # Dataset subset creation
│   ├── run_finetuning.py              # LoRA fine-tuning
│   ├── run_inference.py               # Prompt ablation (single model)
│   └── run_actor_critic.py            # Actor–Critic multi-stage inference
├── src/
│   ├── config.py               # Paths, model names, generation defaults
│   ├── data/
│   │   ├── preprocessing.py    # Hierarchical sampling, TISERPreprocessor
│   │   └── tiser_dataset.py    # TiserExample, load_tiser_file, TiserDataset
│   ├── models/
│   │   └── base_model.py       # LLMWrapper (HF + LoRA, chat template)
│   └── utils/
│       ├── metrics.py          # EM, F1, SM, compute_metrics
│       ├── parsing.py          # extract_answer, extract_section
│       └── prompts.py         # TISER and ablation prompt templates
└── tools/                      # Optional (e.g. translation chunker)
```

---

## Requirements and installation

- Python 3.10+
- PyTorch 2.4+, transformers, datasets, accelerate  
- peft, trl (fine-tuning)  
- pandas, tqdm (data and logging)

Install from the project root:

```bash
cd DNLP-TISER
pip install -r requirements.txt
```

For GPU training/inference, install a CUDA-enabled PyTorch build. For Apple Silicon, MPS is used when available.

---

## Data format

TISER data are JSONL files: one JSON object per line (or a single JSON array). Each record should include:

| Field          | Description |
|----------------|-------------|
| `dataset_name` | Source dataset (e.g. `tgqa_split_train`, `timeqa_easy_test`) |
| `question_id`  | Unique question id (e.g. `story42_Q1_0`) |
| `question`     | Natural language question |
| `context`      | Temporal context (or derived from `prompt` if missing) |
| `answer`       | Ground-truth answer |
| `prompt`       | Full prompt shown to the model (instructions + question + context) |
| `output`       | (Training only) Model target output (reasoning + answer in tags) |

Place raw files under `data/raw/` (e.g. `TISER_train_demo.json`, `TISER_test_demo.json`). Processed subsets will go under `data/processed/` (see Configuration).

---

## Configuration

Paths and models are centralized in `src/config.py`:

- **Paths**: `PROJECT_ROOT`, `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `RESULTS_DIR`, `CHECKPOINTS_DIR`, etc.
- **Models**: `DEV_MODEL_NAME` (e.g. Qwen 1.5B for local dev), `TRAIN_MODEL_NAME_EN` / `TRAIN_MODEL_NAME_IT` (e.g. Qwen 7B), `CRITIC_MODEL_NAME_EN`.
- **Generation**: `GEN_MAX_NEW_TOKENS`, `GEN_TEMPERATURE`, `GEN_TOP_P`.

`get_model_name(mode="dev"|"train", lang="en"|"it", role="actor"|"critic")` returns the HuggingFace model name used by the scripts.

---

## Usage

All commands are intended to be run from the **project root** (`DNLP-TISER/`).

### 1. Preprocessing (create subsets)

Create stratified subsets of the TISER dataset (single file, directory, or multiple ratios):

```bash
# Single file, 10% retention
python scripts/run_preprocessing_dataset.py \
  --input data/raw/TISER_train_demo.json \
  --output data/processed/TISER_train_demo_10pct.json \
  --ratio 0.1

# All files in data/raw → data/processed with one ratio
python scripts/run_preprocessing_dataset.py \
  --input-dir data/raw \
  --output-dir data/processed \
  --ratio 0.1 \
  --seed 42

# Multiple ratios for one input file
python scripts/run_preprocessing_dataset.py \
  --input data/raw/TISER_train_demo.json \
  --output-prefix data/processed/TISER_train_demo \
  --ratios 0.05 0.1 0.25
```

Defaults: `--input-dir` = `RAW_DIR`, `--output-dir` = `PROCESSED_DIR` when using directory mode.

### 2. Fine-tuning (LoRA)

Fine-tune a causal LM on TISER with LoRA. Default data path: `data/processed/TISER_train_demo_10pct.json`. Save fine-tuned models under `finetuned_models/` to keep them separate from other checkpoints.

```bash
# Save to finetuned_models/
python scripts/run_finetuning.py --output finetuned_models/qwen_finetuned

# Custom data and epochs
python scripts/run_finetuning.py \
  --data data/processed/TISER_train_demo_10pct.json \
  --output finetuned_models/qwen_finetuned \
  --epochs 5

# LoRA and sequence length
python scripts/run_finetuning.py \
  --output finetuned_models/qwen_lora \
  --lora-r 32 \
  --lora-alpha 64 \
  --max-seq-length 4096
```

Checkpoints and tokenizer are saved under the path passed to `--output`. Use `finetuned_models/<run_name>` for inference with `--lora` or `--lora-path`.

### 3. Single-prompt inference (ablation)

Run multiple prompt variants on the same test set. Default test file: `data/processed/TISER_test_demo_10pct.json`. Results are written under `experiments/results/`.

```bash
# Default test file and tag
python scripts/run_inference.py --tag ablation

# Custom test file and resume
python scripts/run_inference.py \
  --test-file data/processed/TISER_test_demo_10pct.json \
  --tag ablation \
  --resume \
  --save-every 50

# Evaluate on raw output (fallback extractor) and use a fine-tuned LoRA adapter
python scripts/run_inference.py \
  --eval-on-raw \
  --lora-path finetuned_models/qwen_finetuned
```

Outputs: per-variant CSVs (e.g. `ablation_ablation_standard.csv`) and a single summary CSV (e.g. `ablation_summary_ablation.csv`) with columns `tag`, `variant`, `dataset_name`, `n`, `em`, `f1`.

### 4. Actor–Critic inference

Multi-stage pipeline: Actor (draft) → Critic (reflection) → Solver (final answer). Supports 3-stage (Actor, Critic, Solver) or 2-stage (Actor, Critic+Solver), optional few-shot prompting, and timeline format (list or table). Default test file: `data/processed/TISER_test_demo_10pct.json`.

```bash
# Default test file, 3-stage, tag
python scripts/run_actor_critic.py --tag base_run

# 2-stage (faster)
python scripts/run_actor_critic.py --pipeline-mode 2-stage --tag base_2stage

# Few-shot for Critic
python scripts/run_actor_critic.py --use-few-shot --tag base_fewshot

# With LoRA adapter (e.g. from finetuned_models/)
python scripts/run_actor_critic.py \
  --lora finetuned_models/qwen_finetuned \
  --tag ft_run
```

Results: detailed CSV (e.g. `actor_critic_results_<tag>.csv`) and summary CSV (e.g. `actor_critic_summary_<tag>.csv`) with per-dataset and overall EM, F1, SM.

---

## Outputs and metrics

- **Preprocessing**: Writes JSONL under the path given with `--output` or under `--output-dir` / `--output-prefix`.
- **Fine-tuning**: Model and tokenizer under `--output` (e.g. `finetuned_models/qwen_finetuned`).
- **Inference (ablation)**:  
  - `experiments/results/ablation_<tag>_<variant>.csv` (per-variant details).  
  - `experiments/results/ablation_summary_<tag>.csv` (vertical summary: tag, variant, dataset_name, n, em, f1).
- **Actor–Critic**:  
  - `experiments/results/actor_critic_results_<tag>.csv` (per-example details).  
  - `experiments/results/actor_critic_summary_<tag>.csv` (per-dataset and overall EM, F1, SM).

Metrics:

- **EM**: Exact Match after basic normalization.
- **F1**: Token-level F1 (precision/recall over tokens).
- **SM**: Soft Match (entity/number identity + token inclusion); see `src/utils/metrics.py`.

---

## Default data files

Scripts that need a test or train file use, by default, paths under `data/` as defined in `src/config.py`:

- **Test (inference / actor-critic)**: `data/processed/TISER_test_demo_10pct.json`
- **Train (fine-tuning)**: `data/processed/TISER_train_demo_10pct.json`

Ensure these files exist (e.g. by running preprocessing) or override with `--test-file` / `--data`.

---

## License and references

If you use the TISER dataset or this codebase, please cite the original TISER dataset and model sources (e.g. Hugging Face model cards, dataset papers) as appropriate. This README does not imply a specific license for the project; see repository metadata or source files for license information.
