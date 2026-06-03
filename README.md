# HalluDetectionTypeClassification

Comparative hallucination detection and type classification across four LLMs on HotpotQA (multi-hop) and TruthfulQA (factual stress test), under three prompting strategies.

> **Project rename note:** This repository was previously named `hdc-rag` / `HDC-RAG`. The GitHub URL was renamed in May 2026; old clone URLs continue to redirect. If you have a local clone at `~/source/hdc-rag/`, see [Renaming the local clone](#renaming-the-local-clone) below.

---

## Overview

This project investigates how hallucination rates and hallucination types shift across:
- **4 models**: Mistral-7B, LLaMA-3.1-8B, Qwen-2.5-7B, Phi-4-mini
- **3 prompt variants**: Plain (zero-shot), Reasoning (chain-of-thought), Abstain (model may refuse)
- **2 datasets**: HotpotQA (113K multi-hop questions) and TruthfulQA (~800 factual questions)

Answers are labeled using a multi-phase NLI + heuristic pipeline and validated through manual review.

---

## Hallucination Taxonomy

Each model answer is classified into one of:

| Label | Meaning |
|---|---|
| `supported` | Answer is supported by the evidence |
| `abstained` | Model correctly refused to answer |
| `contradiction_to_evidence` | Answer directly contradicts the evidence |
| `entity_error` | Answer references a wrong named entity |
| `attribute_error` | Entity is correct but an associated attribute (e.g. date, number) is wrong |
| `multi_hop_reasoning_error` | Answer fails to correctly chain reasoning across evidence hops |
| `unsupported_inference` | Answer makes claims not grounded in any evidence |

---

## Requirements

- **Python 3.12** (spaCy/thinc wheels require 3.10–3.14). Confirm with `python3.12 --version`.
- macOS, Linux, or WSL
- For inference: NVIDIA T4 (16 GB, 4-bit quantization) or A100 (recommended)

---

## Quick Start

### 1. Create virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# Must print 3.12.x
python --version

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **Why `python3.12` explicitly?** Running `python3 -m venv` may pick up system Python (often 3.9 on macOS), which causes spaCy/thinc to fail with `No matching distribution found`. Always create the venv with `python3.12`.

### 2. Prepare datasets

```bash
python scripts/prepare_hotpotqa.py --config config/default.yaml
python scripts/prepare_truthfulqa.py --config config/default.yaml
```

Optional flags: `--subset_size`, `--dataset_name`, `--split`, `--no_stratify`

### 3. Run experiments

Runs all 4 models × 3 prompts on both datasets:

```bash
python scripts/run_experiments.py --config config/default.yaml
```

To limit records during development:

```bash
python scripts/run_experiments.py --config config/default.yaml --dataset hotpotqa --limit 50
```

---

## Labeling Pipeline

The labeling pipeline runs DeBERTa-MNLI for NLI classification and applies six phases of heuristic corrections to reduce false positives.

### Phase overview

| Phase | Fix |
|---|---|
| 1 — Signal Override | Multi-signal support check (token recall + cosine + bidirectional NLI) to stop mislabeling verbose correct answers |
| 2 — Score Argmax | Score-based type classification via argmax instead of sequential if/elif |
| 3 — spaCy NER | Named entity recognition via spaCy instead of capitalized-word heuristics |
| 4 — Attribute Gate | Attribute error only fires when the entity itself is confirmed correct |
| 5 — Overlap Gate | Multi-hop error only fires when the answer engages with the evidence |
| 6 — Manual Validation | Stratified 100-record human review with FP rate computation |

### Step 1: Clean raw inference outputs

Strips prompt prefixes from model answers (artifact of full-sequence decoding):

```bash
python scripts/clean_outputs.py \
  --input data/outputs/hotpotqa_phi4_results.jsonl \
          data/outputs/hotpotqa_mistral7b.jsonl \
          data/outputs/hotpotqa_qwen25.jsonl \
          data/outputs/hotpotqa_llama31.jsonl \
  --output_dir data/outputs/cleaned/
```

### Step 2: Label all models (recommended)

Re-run the full labeling pipeline for all four models at once:

```bash
python scripts/relabel_all.py              # writes to *_labeled_v2.jsonl
python scripts/relabel_all.py --suffix v3  # custom suffix
```

Or label a single model manually:

```bash
python scripts/label_outputs.py \
  --input data/outputs/cleaned/hotpotqa_mistral7b.jsonl \
  --hotpotqa data/processed/hotpotqa.jsonl \
  --output data/outputs/labeled/hotpotqa_mistral7b_labeled_v2.jsonl
```

Labeled files are written to `data/outputs/labeled/`.

### Step 3: Compute metrics

```bash
python scripts/compute_metrics.py \
  --input data/outputs/labeled/hotpotqa_phi4_labeled_v2.jsonl \
  --output data/outputs/metrics/hotpotqa_phi4_metrics.json
```

### Step 4: Compute type breakdown

```bash
python scripts/type_breakdown.py
```

### Step 5: Plot results

```bash
python scripts/plot_results.py       # general plots
python scripts/plot_report_figures.py  # report-quality figures → data/outputs/plots/
```

---

## Manual Validation (Phase 6)

Generate a stratified 100-record sample for human review:

```bash
python scripts/sample_for_review.py --output data/review_sample.csv
```

Fill the `your_label` column in the CSV with `correct` or `wrong`, then compute false positive rates:

```bash
python scripts/sample_for_review.py --compute-fp data/review_sample.csv
```

---

## Verify Poster Stats

Compute every number shown on the poster and compare against actual data:

```bash
python scripts/compute_poster_stats.py
python scripts/compute_poster_stats.py --fp-csv data/review_sample_edited.csv
```

Outputs:
- Hallucination rate per model × prompt (Panel 2)
- Abstention failure counts (Panel 6)
- Type distribution by prompt for Mistral-7B (Panel 7 top)
- FP rates from manual review (Panel 7 bottom)

---

## Key Results (HotpotQA, v2 labels)

### Hallucination rate by model × prompt

| Model | Plain | Reasoning | Abstain |
|---|---|---|---|
| Mistral-7B | 65.8% | 48.8% | 9.6% |
| LLaMA-3.1 | 31.2% | 38.6% | 4.4% |
| Qwen-2.5 | 67.4% | 48.2% | 2.2% |
| Phi-4 | 74.6% | 65.6% | 2.6% |

### Abstention failure (plain hallucinated, abstain correctly refused)

| Model | Failure count |
|---|---|
| Phi-4 | 349 |
| Qwen-2.5 | 319 |
| Mistral-7B | 277 |
| LLaMA-3.1 | 137 |

### Phase 6 manual validation FP rates (100 samples)

| Type | FP Rate |
|---|---|
| entity_error | 37.0% |
| contradiction_to_evidence | 51.6% |
| unsupported_inference | 55.6% |
| attribute_error | 65.6% |
| multi_hop_reasoning_error | 100.0% |
| **OVERALL** | **53.0%** |

---

## Local LLM Health Check

Verify Ollama is running and the configured model is available:

```bash
python scripts/ollama_check.py --config config/default.yaml
```

---

## HF / Colab Support

Set `provider: hf` in `config/default.yaml` and use HF model IDs:

```yaml
experiments:
  models:
    - id: qwen2.5-7b-instruct
      provider: hf
      model: Qwen/Qwen2.5-7B-Instruct
```

In Colab:

```bash
pip install transformers accelerate torch
```

Optional: enable 4-bit/8-bit quantization in `experiments.hf` (requires `bitsandbytes`).

---

## Project Layout

```
scripts/
  prepare_hotpotqa.py       # Download and preprocess HotpotQA
  prepare_truthfulqa.py     # Download and preprocess TruthfulQA
  run_experiments.py        # Run all models × prompts
  clean_outputs.py          # Strip prompt prefixes from raw outputs
  label_outputs.py          # Label a single model's outputs
  relabel_all.py            # Re-label all 4 models in one command
  compute_metrics.py        # EM, F1, hallucination/abstention rates
  type_breakdown.py         # Hallucination type counts per model
  sample_for_review.py      # Phase 6 — generate/evaluate manual review CSV
  compute_poster_stats.py   # Verify all poster numbers against real data
  plot_results.py           # Exploratory plots
  plot_report_figures.py    # Report-quality figures
  extract_examples.py       # Pull representative examples from labeled files
  preview_outputs.py        # Quick terminal preview of output files
  ollama_check.py           # Verify Ollama connectivity

src/
  data/                     # Dataset loaders (HotpotQA, TruthfulQA)
  baseline/                 # Baseline answer generator
  labeling/
    nli.py                  # DeBERTa-MNLI wrapper
    heuristics.py           # Phase 1–5 correction logic + classify_hallucination_type
    embeddings.py           # Sentence-transformer cosine similarity (all-MiniLM-L6-v2)
    entities.py             # spaCy NER wrapper
    evidence.py             # Evidence utilities
  metrics/                  # EM, F1, hallucination rate computation
  config/                   # YAML config loader
  utils/                    # LLM client, IO helpers, logging

config/
  default.yaml              # Models, prompts, paths, HF settings

data/
  processed/                # hotpotqa.jsonl, truthfulqa.jsonl
  outputs/
    cleaned/                # Prompt-stripped model outputs
    labeled/                # *_labeled_v2.jsonl (current), *_labeled.jsonl (v1)
    metrics/                # Per-model JSON metrics
    plots/                  # PNG figures

legacy_rag/                 # Archived RAG prototype (not used in current work)
```

---

## Renaming the local clone

If you previously cloned the repo at `~/source/hdc-rag/`:

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
