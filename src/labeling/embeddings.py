from typing import Optional

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _load_model() -> None:
    global _model
    if _model is not None:
        return
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(_MODEL_NAME)


def cosine_similarity(text_a: str, text_b: str) -> float:
    _load_model()
    embeddings = _model.encode([text_a, text_b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))
