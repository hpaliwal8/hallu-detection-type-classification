# HalluDetectionTypeClassification

Comparative hallucination analysis across multiple models using HotpotQA (primary) and TruthfulQA (stress test).
See `PROJECT_PLAN.md`, `WEEKLY_PLAN.md`, and `TODO.md` for the research design.

> **Project rename note:** This repository was previously named `hdc-rag` / `HDC-RAG`. The GitHub URL was renamed in May 2026; old clone URLs continue to redirect. If you have a local clone at `~/source/hdc-rag/`, see the "Renaming the local clone" section below.

## Requirements

- **Python 3.12** (the pinned `spacy` and `thinc` versions only have prebuilt wheels for 3.10–3.14). Confirm with `python3.12 --version` before creating the venv.
- macOS, Linux, or WSL.
- For inference: NVIDIA T4 (16 GB, with 4-bit quantization) or A100 (recommended).

## Quick Start

1. Create a virtualenv **explicitly with Python 3.12** and install deps:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# Sanity check — must print 3.12.x
python --version

# Use the venv's own pip to avoid system-pip resolving deps for an older Python
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **Why explicit `python3.12`?** Running `python3 -m venv .venv` may pick up the system Python (often 3.9 on macOS), which causes `spacy`/`thinc` install to fail with `No matching distribution found for thinc>=8.3.12` because those versions require Python 3.10+. Always create the venv with `python3.12`.

### Renaming the local clone

If you previously cloned the repo at `~/source/hdc-rag/`, rename and recreate the venv:

```bash
cd ~/source
mv hdc-rag HalluDetectionTypeClassification
cd HalluDetectionTypeClassification

# Old .venv has hardcoded /hdc-rag paths — must recreate
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Update git remote
git remote set-url origin https://github.com/hpaliwal8/hallu-detection-type-classification.git
```

2. Download HotpotQA (primary) and create a dataset file:

```bash
python scripts/prepare_hotpotqa.py --config config/default.yaml
```

3. Download TruthfulQA (stress-test set) and create a dataset file:

```bash
python scripts/prepare_truthfulqa.py --config config/default.yaml
```

Optional flags include `--subset_size`, `--dataset_name`, `--split`, and `--no_stratify`.

## Local LLM Health Check

Verify Ollama is running and the configured model is available:

```bash
python scripts/ollama_check.py --config config/default.yaml
```

## Project Layout

- `src/data/`: dataset loaders
- `src/baseline/`: baseline answer generator
- `src/utils/`: LLM utilities, IO helpers
- `scripts/`: CLI entrypoints
- `config/`: YAML config

## Notes

- Baseline generation is wired for Ollama (`llm.provider: ollama`).
- LLM model selection is centralized in `config/default.yaml` under `llm`.
- To switch to the dev model without editing config, set `HDC_RAG_USE_DEV_MODEL=1`.
- For local inference, this project supports Ollama via `llm.provider: ollama`.

## Week 3 — Labeling + Analysis

### 1. Clean raw inference outputs

Strips the prompt prefix from model answers (artifact of full-sequence decoding):

```bash
python scripts/clean_outputs.py \
  --input data/outputs/hotpotqa_phi4_results.jsonl \
          data/outputs/hotpotqa_mistral7b.jsonl \
          data/outputs/hotpotqa_qwen25.jsonl \
          data/outputs/hotpotqa_llama31.jsonl \
  --output_dir data/outputs/cleaned/
```

### 2. Label outputs with NLI + heuristics

Joins experiment outputs with HotpotQA dataset, runs DeBERTa-MNLI, and assigns hallucination types:

```bash
python scripts/label_outputs.py \
  --input data/outputs/cleaned/hotpotqa_phi4_results.jsonl \
  --hotpotqa data/processed/hotpotqa.jsonl \
  --output data/outputs/labeled/hotpotqa_phi4_labeled.jsonl

python scripts/label_outputs.py \
  --input data/outputs/cleaned/hotpotqa_mistral7b.jsonl \
  --hotpotqa data/processed/hotpotqa.jsonl \
  --output data/outputs/labeled/hotpotqa_mistral7b_labeled.jsonl

python scripts/label_outputs.py \
  --input data/outputs/cleaned/hotpotqa_qwen25.jsonl \
  --hotpotqa data/processed/hotpotqa.jsonl \
  --output data/outputs/labeled/hotpotqa_qwen25_labeled.jsonl

python scripts/label_outputs.py \
  --input data/outputs/cleaned/hotpotqa_llama31.jsonl \
  --hotpotqa data/processed/hotpotqa.jsonl \
  --output data/outputs/labeled/hotpotqa_llama31_labeled.jsonl
```

### 3. Compute metrics

Outputs EM, F1, hallucination rate, abstention rate, and type breakdown per model × prompt. Also flags overconfident abstention failures:

```bash
python scripts/compute_metrics.py \
  --input data/outputs/labeled/hotpotqa_phi4_labeled.jsonl \
  --output data/outputs/metrics/hotpotqa_phi4_metrics.json
```

## Next Steps

- Add NLI-based labeling and analysis scripts.
 
## Experiments

Run all models × prompt variants:

```bash
python scripts/run_experiments.py --config config/default.yaml --limit 50
```

To run only one dataset:

```bash
python scripts/run_experiments.py --config config/default.yaml --dataset hotpotqa --limit 50
```

### HF/Colab Support

You can run Hugging Face models in Colab by setting `provider: hf` in `config/default.yaml` and using HF model IDs.
For example:

```yaml
experiments:
  models:
    - id: qwen2.5-7b-instruct
      provider: hf
      model: Qwen/Qwen2.5-7B-Instruct
```

In Colab, install deps:

```bash
pip install transformers accelerate torch
```

Optional: enable 4‑bit/8‑bit quantization in `experiments.hf` (requires `bitsandbytes`).

Legacy RAG code is archived under `legacy_rag/`.
