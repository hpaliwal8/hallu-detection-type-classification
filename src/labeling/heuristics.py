import re
from typing import Dict, Any

from src.labeling import nli as _nli
from src.labeling import embeddings as _emb

ABSTAIN_PHRASES = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i'm not aware",
    "i am not aware",
    "i don't have information",
    "i do not have information",
    "i don't have any information",
    "i don't have enough information",
    "i'm unable to verify",
    "i am unable to verify",
    "unable to verify",
    "cannot verify",
    "cannot definitively",
    "insufficient evidence",
    "insufficient information",
    "cannot determine",
    "cannot be determined",
    "not enough information",
    "no information available",
    "could you provide more context",
    "could you provide more details",
    "unclear",
]

NUMBER_PATTERN = re.compile(r"\b\d{1,4}\b")

_EMBED_SIM_THRESHOLD = 0.60
_REFERENCE_RECALL_THRESHOLD = 0.50
_STRIP_CHARS = ".,;:?!\"'()-"


def _is_abstention(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in ABSTAIN_PHRASES)


def _reference_recall(answer: str, reference: str) -> float:
    """Fraction of reference tokens that appear in the answer.
    Works correctly when the answer is verbose and the reference is short."""
    r_tokens = {w.strip(_STRIP_CHARS) for w in reference.lower().split()}
    a_tokens = {w.strip(_STRIP_CHARS) for w in answer.lower().split()}
    r_tokens.discard("")
    if not r_tokens:
        return 0.0
    return len(r_tokens & a_tokens) / len(r_tokens)


def is_supported_by_signals(answer: str, reference: str) -> bool:
    """Return True if at least 2 of 3 independent signals agree the answer is supported.
    Compares the model answer against the short gold reference answer, not the full
    evidence passage (which would make F1 and cosine artificially low)."""
    signals = 0

    if _reference_recall(answer, reference) > _REFERENCE_RECALL_THRESHOLD:
        signals += 1

    if _emb.cosine_similarity(answer, reference) > _EMBED_SIM_THRESHOLD:
        signals += 1

    if signals < 2 and _nli.is_bidirectional_entailment(answer, reference):
        signals += 1

    return signals >= 2


def _entity_score(answer: str, evidence: str) -> float:
    """Fraction of capitalized words in the answer that are absent from the evidence."""
    evidence_lower = evidence.lower()
    words = answer.split()
    capitalized = [w.strip(_STRIP_CHARS) for w in words if w and w[0].isupper()]
    if not capitalized:
        return 0.0
    missing = [w for w in capitalized if w.lower() not in evidence_lower]
    return len(missing) / len(capitalized)


def _attribute_score(answer: str, evidence: str) -> float:
    """Fraction of numbers in the answer that do not appear in the evidence."""
    answer_numbers = set(NUMBER_PATTERN.findall(answer))
    if not answer_numbers:
        return 0.0
    evidence_numbers = set(NUMBER_PATTERN.findall(evidence))
    missing = answer_numbers - evidence_numbers
    return len(missing) / len(answer_numbers)


def _multihop_score(question_type: str) -> float:
    """Fixed prior for multi-hop questions. Phase 5 will replace this with overlap signal."""
    return 0.5 if question_type in ("bridge", "comparison") else 0.0


def classify_hallucination_type(
    nli_result: Dict[str, Any],
    answer: str,
    evidence: str,
    question_type: str = "",
    reference: str = "",
) -> str:
    if nli_result["nli_label"] == "entailment":
        return "supported"

    if _is_abstention(answer):
        return "abstained"

    # For neutral: a high reference recall alone is sufficient — if the model's answer
    # contains the gold answer tokens, DeBERTa's neutral is almost certainly a false
    # positive caused by verbosity. Single-word gold answers have low cosine against
    # verbose answers, so requiring 2-of-3 would never fire.
    if nli_result["nli_label"] == "neutral" and reference:
        if _reference_recall(answer, reference) > _REFERENCE_RECALL_THRESHOLD:
            return "supported"

    # For contradiction (or neutral without a reference): require 2-of-3 signals so we
    # don't accidentally suppress a real error.
    if nli_result["nli_label"] in ("neutral", "contradiction"):
        signal_ref = reference if reference else evidence
        if is_supported_by_signals(answer, signal_ref):
            return "supported"

    scores = {
        "contradiction_to_evidence": nli_result["prob_contradiction"],
        "attribute_error":           _attribute_score(answer, evidence),
        "entity_error":              _entity_score(answer, evidence),
        "multi_hop_reasoning_error": _multihop_score(question_type),
    }

    best_type = max(scores, key=scores.get)
    if scores[best_type] < 0.5:
        return "unsupported_inference"
    return best_type
