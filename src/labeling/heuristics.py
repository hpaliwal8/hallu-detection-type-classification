import re
from typing import Dict, Any

from src.labeling import nli as _nli
from src.labeling import embeddings as _emb
from src.labeling import entities as _ent

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
    return _ent.entity_score(answer, evidence)


def _sentences(text: str):
    return re.split(r'(?<=[.!?])\s+', text)


def _attribute_score(answer: str, evidence: str) -> float:
    """Fraction of numbers — in answer sentences containing a correct entity — that are
    absent from the evidence. Returns 0.0 when the entity itself is wrong so that
    entity_error wins instead of attribute_error."""
    # Only consider numbers near entities that are confirmed present in evidence.
    answer_entities = _ent.extract_named_entities(answer)
    correct_entities = [e for e in answer_entities if e.lower() in evidence.lower()]
    if not correct_entities:
        return 0.0

    # Collect numbers from answer sentences that contain at least one correct entity.
    relevant_numbers = set()
    for sent in _sentences(answer):
        if any(e.lower() in sent.lower() for e in correct_entities):
            relevant_numbers |= set(NUMBER_PATTERN.findall(sent))

    if not relevant_numbers:
        return 0.0

    evidence_numbers = set(NUMBER_PATTERN.findall(evidence))
    missing = relevant_numbers - evidence_numbers
    return len(missing) / len(relevant_numbers)


def _tokenize(text: str) -> set:
    return {w.strip(_STRIP_CHARS) for w in text.lower().split()} - {""}


def _max_sentence_overlap(answer: str, evidence: str) -> float:
    """Max Jaccard token overlap between the answer and any single evidence sentence."""
    a_tokens = _tokenize(answer)
    if not a_tokens:
        return 0.0
    best = 0.0
    for sent in _sentences(evidence):
        e_tokens = _tokenize(sent)
        if not e_tokens:
            continue
        overlap = len(a_tokens & e_tokens) / len(a_tokens | e_tokens)
        if overlap > best:
            best = overlap
    return best


def _multihop_score(answer: str, evidence: str, question_type: str) -> float:
    """Score is gated on evidence engagement: the model must overlap with at least one
    evidence sentence to be labeled multi_hop rather than unsupported_inference."""
    if question_type not in ("bridge", "comparison"):
        return 0.0
    overlap = _max_sentence_overlap(answer, evidence)
    if overlap < 0.2:
        return 0.0
    return min(overlap / 0.4, 1.0) * 0.5


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
        "multi_hop_reasoning_error": _multihop_score(answer, evidence, question_type),
    }

    best_type = max(scores, key=scores.get)
    if scores[best_type] < 0.5:
        return "unsupported_inference"
    return best_type
