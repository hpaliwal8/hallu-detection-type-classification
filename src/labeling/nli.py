from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MODEL_NAME = "microsoft/deberta-large-mnli"
_tokenizer = None
_model = None


def _load_model() -> None:
    global _tokenizer, _model
    if _model is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.to("cuda")


def _classify_one(premise: str, hypothesis: str) -> Dict[str, float]:
    _load_model()

    inputs = _tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze()
    id2label = _model.config.id2label
    label_to_idx = {v.lower(): k for k, v in id2label.items()}

    entailment_idx = label_to_idx.get("entailment", 0)
    neutral_idx = label_to_idx.get("neutral", 1)
    contradiction_idx = label_to_idx.get("contradiction", 2)

    predicted_idx = probs.argmax().item()

    return {
        "nli_label": id2label[predicted_idx].lower(),
        "confidence": probs[predicted_idx].item(),
        "prob_entailment": probs[entailment_idx].item(),
        "prob_neutral": probs[neutral_idx].item(),
        "prob_contradiction": probs[contradiction_idx].item(),
    }


def is_bidirectional_entailment(text_a: str, text_b: str) -> bool:
    forward = _classify_one(text_a, text_b)
    backward = _classify_one(text_b, text_a)
    return forward["nli_label"] == "entailment" and backward["nli_label"] == "entailment"


def classify(premise: str, hypothesis: str) -> Dict[str, float]:
    return _classify_one(premise, hypothesis)
