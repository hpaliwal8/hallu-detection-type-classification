# Labeling Pipeline Corrections

This document tracks identified weaknesses in the current labeling pipeline and the planned fixes to reduce false positives in both supported/hallucinated classification and hallucination type assignment.

## Identified Issues

### Issue 1 — Verbose correct answers labeled as hallucinated
DeBERTa-MNLI labels correct-but-verbose answers as `neutral`, which the heuristics treat as hallucinated.

**Examples observed:**
- Mistral correctly answers "peach" → labeled `entity_error`
- Mistral correctly answers "Camel Up is a board game" → labeled `entity_error`
- Qwen correctly answers "Naomi Campbell" → labeled `entity_error`
- Mistral correctly answers "yes, both novelists" → labeled `multi_hop_reasoning_error`
- Mistral correctly identifies "Romantic era" → labeled `multi_hop_reasoning_error`

### Issue 2 — Type heuristics fire on first match, not strongest signal
The current `classify_hallucination_type` runs sequential `if/elif/else` checks. A weak signal that fires first wins over a stronger signal that would fire later.

### Issue 3 — Naive entity detection
`entity_error` uses capitalized-word matching as a proxy for named entities. Catches sentence starters and common nouns as false positives.

### Issue 4 — Naive attribute detection
`attribute_error` fires on any number-string mismatch, even when the entity itself is wrong (which would be `entity_error`).

### Issue 5 — Multi-hop label fires on inventions
`multi_hop_reasoning_error` fires whenever a question is bridge/comparison. Doesn't require the model to have engaged with the evidence at all.

### Issue 6 — No way to measure actual FP rate
Without manual validation, the FP rate is unknown. Cannot make defensible claims in the write-up.

---

## Planned Fixes

### Phase 1 — Reduce supported/hallucinated false positives

**Goal:** Stop labeling correct verbose answers as hallucinated.

**Changes to `src/labeling/heuristics.py`:**
Add `is_supported_by_signals` check that runs before type classification:
- Recall between answer and reference > 0.50 → supported
- Embedding cosine similarity > 0.60 → supported
- Bidirectional NLI both entailment → supported

If any two of these three agree on supported, override NLI's neutral/contradiction label.

**New dependency:** `sentence-transformers`

Summary of Phase 1 changes: 
sentence-transformers>=3.0 added to requirements.txt
src/labeling/embeddings.py — new module wrapping all-MiniLM-L6-v2 for cosine similarity
Key architectural change vs. the original plan
The spec assumed signals would compare answer vs. evidence. The actual implementation compares answer vs. gold reference answer (reference_answer field from the HotpotQA processed data). This was necessary because the evidence is a multi-sentence passage — all three signals degrade badly against long text. The gold reference is always a short phrase, which is what the spec implicitly assumed.

What changed in the data (Mistral, HotpotQA)
supported: 119 → 416 (+297)
attribute_error: 315 → 167 (−148)
multi_hop_reasoning_error: 233 → 119 (−114)
entity_error: 74 → 53 (−21)
contradiction_to_evidence: 296 → 282 (−14)

---


Completion comments:
Implemented signal-based override; in practice the input cleaning fix accounted for most observed gains.


### Phase 2 — Score-based type classification

**Goal:** Pick the most confident type, not the first one that fires.

Rewrite `classify_hallucination_type` to compute confidence scores for each candidate:

```
scores = {
    "contradiction_to_evidence":   nli_result["prob_contradiction"],
    "attribute_error":             attribute_score(answer, evidence),
    "entity_error":                entity_score(answer, evidence),
    "multi_hop_reasoning_error":   multihop_score(answer, evidence, q_type),
}
```

Pick `argmax(scores)`. If `max(scores) < 0.5`, label `unsupported_inference` (uncertain).

---

### Phase 3 — Better entity detection

**Goal:** Reduce `entity_error` false positives.

Use spaCy NER (`en_core_web_sm`):
- Extract named entities from the answer
- Check if each appears in evidence
- Score = fraction of answer entities not in evidence

**New dependency:** `spacy` + `en_core_web_sm`

---

### Phase 4 — Smarter attribute detection

**Goal:** `attribute_error` only fires when entity is correct but a property is wrong.

Detection logic:
- Find entities in answer that are also in evidence (correct entity)
- For those entities, find numbers/dates nearby in answer vs evidence
- Mismatch counts only if entity is present in both texts

This eliminates the current bug where any number mismatch fires.

---

### Phase 5 — Multi-hop signal requires partial correctness

**Goal:** Don't label invented answers as multi-hop errors.

Detection logic:
- Compute token overlap between answer and each evidence sentence
- If max overlap > 0.4 AND question is bridge/comparison → multi-hop error
- If max overlap < 0.2 → not engaging with evidence; falls through to `unsupported_inference` or `entity_error`

---

### Phase 6 — Manual validation sample

**Goal:** Measure actual FP rate so the write-up can be honest.

New script `scripts/sample_for_review.py`:
- Sample 100 records stratified by hallucination type
- Output CSV with columns: `id`, `type`, `answer`, `reference`, `your_label`, `notes`
- Manually fill `your_label` (1–2 hours)
- Compute per-type FP rate

Enables a claim in the write-up like:
> Manual validation on 100 samples showed FP rates of: contradiction 3%, entity_error 12%, attribute_error 8%, multi-hop 22%.

---

### Phase 7 — Optional GPT-4 tiebreaker

**Goal:** Resolve ambiguous cases that improved heuristics can't handle.

For records with low max score from Phase 2, send to GPT-4:

```
Given this question, evidence, and model answer, classify the error type:
- contradiction_to_evidence
- entity_error
- attribute_error
- multi_hop_reasoning_error
- unsupported_inference (catch-all)
```

~500 records × $0.005 ≈ $2.50 total.

**New dependency:** `openai`

---

## File Changes Summary

| File | Change |
|---|---|
| `src/labeling/heuristics.py` | Rewrite with score-based logic |
| `src/labeling/embeddings.py` | New — sentence-transformer wrapper |
| `src/labeling/entities.py` | New — spaCy NER wrapper |
| `src/labeling/nli.py` | Add bidirectional NLI option |
| `scripts/sample_for_review.py` | New — manual validation sampler |
| `scripts/gpt4_tiebreaker.py` | New — Phase 7, optional |
| `requirements.txt` | Add `sentence-transformers`, `spacy`, `openai` |

---

## Time Estimate

| Phase | Code time | Relabeling time |
|---|---|---|
| 1 + 2 | ~2 hours | ~3 hours |
| 3 + 4 + 5 | ~3 hours | ~3 hours |
| 6 | ~1 hour | ~2 hours (manual) |
| 7 | ~30 min | ~10 min (API) |

**Total: ~12–15 hours**

---

## Implementation Order

1. Phase 1 (signals override) + Phase 2 (score-based types) as a single rewrite
2. Re-run labeling, re-extract examples, evaluate
3. If still many FPs in entity/attribute/multi-hop: Phase 3 + 4 + 5
4. Phase 6 (manual validation) regardless — needed for the write-up
5. Phase 7 only if budget remains and FP rate still unsatisfactory
