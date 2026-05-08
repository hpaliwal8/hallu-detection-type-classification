# Comparative Hallucination Analysis Study

## Objective

Design and execute a comparative study of hallucinations across four LLMs on two QA datasets (HotpotQA primary, TruthfulQA secondary). The study will classify hallucination types, identify prompt‑sensitivity patterns, and report cross‑model comparisons.

## Research Questions

1. What types of hallucinations does each model produce?
2. Under what conditions do hallucinations increase or decrease?
3. Do specific prompt cues reduce hallucination?
4. Which model gives the best tradeoff between accuracy, abstention, and hallucination rate?

## Datasets

- **Primary:** HotpotQA (dev split, 400–600 questions)
- **Secondary:** TruthfulQA (100 questions, stress test)

## Models (4)

- Phi-4-mini-instruct (3.8B)
- Mistral-7B-Instruct-v0.3 (7B)
- Qwen2.5-7B-Instruct (7B)
- Llama-3.1-8B-Instruct (8B)

## Hallucination Taxonomy

### `supported`
The model's answer is entailed by the evidence. Not a hallucination.

### `abstained`
The model correctly declined to answer (e.g. "I don't know", "I'm not sure", "insufficient evidence"). Not a hallucination — this is the desired behavior under uncertainty.

### `contradiction_to_evidence`
The model's answer directly conflicts with the supporting facts. Detected by NLI (contradiction label). Example: evidence says "founded in 1755", model says "founded in 1800".

### `entity_error`
The model names a person, place, or thing that does not appear in the evidence at all. Detected by checking if named entities in the answer are present in the evidence string.

### `attribute_error`
The correct entity is mentioned but a property is wrong — typically a date, number, or location. Detected by finding entities that appear in both answer and evidence, but with mismatched numeric/date values.

### `multi_hop_reasoning_error`
The model gets individual facts right but chains them incorrectly across hops. Specific to HotpotQA `bridge` and `comparison` questions where multiple evidence passages must be combined. Detected when question type is multi-hop and evidence is partially matched but NLI is neutral.

### `unsupported_inference`
Catch-all for neutral NLI cases where no specific heuristic fires. The answer goes beyond the evidence without directly contradicting it. If this label dominates (>40% of hallucinations), heuristics need improvement via manual spot-checking.

### `overconfident_abstention_failure`
Cross-cutting flag: the model hallucinated an answer when it should have abstained. Cannot be determined per-record — requires comparing the `plain` and `abstain` prompt results for the same (model, question) pair. Implemented as a second pass in `compute_metrics.py`:
- Plain prompt → hallucinated
- Abstain prompt → abstained
- If both true → flag the plain prompt record as `overconfident_abstention_failure`

## Labeling Methodology (NLI + Heuristics)

Use a **DeBERTa‑MNLI** cross‑encoder as an NLI judge.

- **Premise:** HotpotQA supporting facts (concatenated evidence).
- **Hypothesis:** model answer.

**Decision flow:**
- Entailment → supported (not hallucinated).
- Contradiction → hallucinated (type = contradiction).
- Neutral → hallucinated (type assigned by heuristics below).

**Heuristic typing (for neutral/unsupported):**
- Abstention check: if answer says “I don’t know” / “insufficient evidence” → abstained.
- If answer entity not in evidence → entity error / unsupported inference.
- If entity present but attribute mismatch (date/number/location) → attribute error.
- If HotpotQA question is bridge/comparison and evidence has parts but answer neutral → multi‑hop reasoning error.
- Otherwise → unsupported inference.
- `overconfident_abstention_failure` is computed in `compute_metrics.py` as a cross-record second pass, not during per-record labeling.

## Prompt Variants (3)

1. Plain QA prompt  
2. Abstention prompt: “If unsure, say ‘I don’t know’.”  
3. Reasoning + abstention: “Use step-by-step reasoning; if evidence is insufficient, say so.”  

## Evaluation Metrics

- Exact Match / token-level F1 (HotpotQA)
- Hallucination rate
- Abstention rate
- Hallucination type distribution
- Prompt sensitivity (hallucination shift by prompt variant)
- Per-model confusion table (supported / unsupported / contradictory)

## Environment

- **Inference:** Google Colab Pro (T4 preferred, A100 optional)
- **Local:** VS Code for preprocessing, analysis, and plotting

## Outputs

- Per-model result tables
- Hallucination type distribution charts
- Prompt‑sensitivity comparisons
- Error analysis examples


Per model - per prompting mode hallucination type classification:

For HotpotQA:
Model                     Prompt      Total   Hall% | Type breakdown (% of hallucinations)
--------------------------------------------------------------------------------------------------------------
llama-3.1-8b-instruct     abstain       500    6.0% | multi_hop_reasoning_error=36.7%, contradiction_to_evidence=33.3%, entity_error=16.7%, attribute_error=13.3%
llama-3.1-8b-instruct     plain         500   40.2% | contradiction_to_evidence=60.7%, multi_hop_reasoning_error=21.4%, attribute_error=10.4%, entity_error=7.5%
llama-3.1-8b-instruct     reasoning     500   58.0% | attribute_error=40.7%, contradiction_to_evidence=29.0%, multi_hop_reasoning_error=24.1%, entity_error=6.2%
mistral-7b-instruct       abstain       500   15.8% | contradiction_to_evidence=38.0%, multi_hop_reasoning_error=32.9%, attribute_error=17.7%, entity_error=11.4%
mistral-7b-instruct       plain         500   92.4% | contradiction_to_evidence=48.3%, multi_hop_reasoning_error=23.2%, attribute_error=18.6%, entity_error=10.0%
mistral-7b-instruct       reasoning     500   75.4% | attribute_error=57.0%, multi_hop_reasoning_error=26.5%, contradiction_to_evidence=11.4%, entity_error=5.0%
phi-4-mini-instruct       abstain       500    3.0% | contradiction_to_evidence=73.3%, entity_error=13.3%, multi_hop_reasoning_error=6.7%, attribute_error=6.7%
phi-4-mini-instruct       plain         500   87.6% | contradiction_to_evidence=68.0%, multi_hop_reasoning_error=14.6%, attribute_error=11.4%, entity_error=5.9%
phi-4-mini-instruct       reasoning     500   81.8% | contradiction_to_evidence=44.0%, attribute_error=29.3%, multi_hop_reasoning_error=19.6%, entity_error=7.1%
qwen2.5-7b-instruct       abstain       500    4.0% | contradiction_to_evidence=45.0%, multi_hop_reasoning_error=40.0%, attribute_error=10.0%, entity_error=5.0%
qwen2.5-7b-instruct       plain         500   91.0% | contradiction_to_evidence=48.1%, attribute_error=28.1%, multi_hop_reasoning_error=15.6%, entity_error=8.1%
qwen2.5-7b-instruct       reasoning     500   74.4% | attribute_error=91.7%, contradiction_to_evidence=8.1%, multi_hop_reasoning_error=0.3%

TruthfulQA

Model                     Prompt      Total   Hall% | Type breakdown (% of hallucinations)
--------------------------------------------------------------------------------------------------------------
llama-3.1-8b-instruct     abstain       120   11.7% | unsupported_inference=78.6%, contradiction_to_evidence=21.4%
llama-3.1-8b-instruct     plain         120   91.7% | unsupported_inference=70.9%, contradiction_to_evidence=29.1%
llama-3.1-8b-instruct     reasoning     120   80.0% | unsupported_inference=86.5%, contradiction_to_evidence=13.5%
mistral-7b-instruct       abstain       120   21.7% | unsupported_inference=84.6%, contradiction_to_evidence=15.4%
mistral-7b-instruct       plain         120   95.8% | unsupported_inference=89.6%, contradiction_to_evidence=10.4%
mistral-7b-instruct       reasoning     120   75.8% | unsupported_inference=96.7%, contradiction_to_evidence=3.3%
phi-4-mini-instruct       abstain       120    4.2% | unsupported_inference=60.0%, contradiction_to_evidence=40.0%
phi-4-mini-instruct       plain         120   93.3% | unsupported_inference=73.2%, contradiction_to_evidence=26.8%
phi-4-mini-instruct       reasoning     120   78.3% | unsupported_inference=85.1%, contradiction_to_evidence=14.9%
qwen2.5-7b-instruct       abstain       120   28.3% | unsupported_inference=82.4%, contradiction_to_evidence=17.6%
qwen2.5-7b-instruct       plain         120   87.5% | unsupported_inference=88.6%, contradiction_to_evidence=11.4%
qwen2.5-7b-instruct       reasoning     120   75.0% | unsupported_inference=96.7%, contradiction_to_evidence=3.3%