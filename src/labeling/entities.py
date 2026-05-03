from typing import List

_nlp = None


def _load_model() -> None:
    global _nlp
    if _nlp is not None:
        return
    import spacy
    _nlp = spacy.load("en_core_web_sm")


def extract_named_entities(text: str) -> List[str]:
    """Return a list of named entity strings found in text using spaCy NER."""
    _load_model()
    doc = _nlp(text)
    return [ent.text for ent in doc.ents]


def entity_score(answer: str, evidence: str) -> float:
    """Fraction of named entities in the answer that are absent from the evidence.
    Uses spaCy NER instead of capitalized-word matching to avoid false positives
    from sentence starters and common nouns."""
    entities = extract_named_entities(answer)
    if not entities:
        return 0.0
    evidence_lower = evidence.lower()
    missing = [e for e in entities if e.lower() not in evidence_lower]
    return len(missing) / len(entities)
