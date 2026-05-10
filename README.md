# HalluDetectionTypeClassification

Comparative hallucination analysis across multiple models using HotpotQA (primary) and TruthfulQA (stress test).
See `PROJECT_PLAN.md`, `WEEKLY_PLAN.md`, and `TODO.md` for the research design.

## Quick Start

1. Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
