#!/usr/bin/env python
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.labeling.evidence import extract_evidence
from src.labeling.nli import classify
from src.labeling.heuristics import classify_hallucination_type, _is_abstention
from src.utils.io import read_jsonl, ensure_dir
from src.utils.logging import setup_logging


def load_index(path: str) -> dict:
    index = {}
    for row in read_jsonl(path):
        index[row["id"]] = row
    return index


def label_hotpotqa(record: dict, hotpotqa_row: dict) -> dict:
    answer = record.get("answer", "")
    supporting_facts = hotpotqa_row.get("supporting_facts", [])
    context = hotpotqa_row.get("context", {})
    question_type = hotpotqa_row.get("type", "")
    reference_answer = hotpotqa_row.get("reference_answer", "")

    evidence = extract_evidence(supporting_facts, context)
    nli_result = classify(evidence, answer)
    hallucination_type = classify_hallucination_type(
        nli_result=nli_result,
        answer=answer,
        evidence=evidence,
        question_type=question_type,
        reference=reference_answer,
    )
    is_hallucinated = hallucination_type not in ("supported", "abstained")

    return {
        **record,
        "evidence": evidence,
        "nli_label": nli_result["nli_label"],
        "confidence": nli_result["confidence"],
        "prob_entailment": nli_result["prob_entailment"],
        "prob_neutral": nli_result["prob_neutral"],
        "prob_contradiction": nli_result["prob_contradiction"],
        "hallucination_type": hallucination_type,
        "is_hallucinated": is_hallucinated,
    }


def label_truthfulqa(record: dict, truthfulqa_row: dict) -> dict:
    answer = record.get("answer", "")
    reference = truthfulqa_row.get("reference_answer", "")
    category = truthfulqa_row.get("category", "")

    if not reference:
        return {**record, "hallucination_type": "unknown", "is_hallucinated": False}

    nli_result = classify(reference, answer)

    if _is_abstention(answer):
        hallucination_type = "abstained"
    elif nli_result["nli_label"] == "entailment":
        hallucination_type = "supported"
    elif nli_result["nli_label"] == "contradiction":
        hallucination_type = "contradiction_to_evidence"
    else:
        hallucination_type = "unsupported_inference"

    is_hallucinated = hallucination_type not in ("supported", "abstained")

    return {
        **record,
        "category": category,
        "nli_label": nli_result["nli_label"],
        "confidence": nli_result["confidence"],
        "prob_entailment": nli_result["prob_entailment"],
        "prob_neutral": nli_result["prob_neutral"],
        "prob_contradiction": nli_result["prob_contradiction"],
        "hallucination_type": hallucination_type,
        "is_hallucinated": is_hallucinated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Experiment output JSONL file.")
    parser.add_argument("--hotpotqa", default=None, help="Processed HotpotQA JSONL file.")
    parser.add_argument("--truthfulqa", default=None, help="Processed TruthfulQA JSONL file.")
    parser.add_argument("--output", required=True, help="Labeled output JSONL file.")
    args = parser.parse_args()

    setup_logging()

    hotpotqa_index = load_index(args.hotpotqa) if args.hotpotqa else {}
    truthfulqa_index = load_index(args.truthfulqa) if args.truthfulqa else {}
    ensure_dir(os.path.dirname(args.output))

    with open(args.output, "w", encoding="utf-8") as out_f:
        for record in read_jsonl(args.input):
            row_id = record.get("id")
            dataset = record.get("dataset", "")

            if dataset == "hotpotqa" and row_id in hotpotqa_index:
                labeled = label_hotpotqa(record, hotpotqa_index[row_id])
            elif dataset == "truthfulqa" and row_id in truthfulqa_index:
                labeled = label_truthfulqa(record, truthfulqa_index[row_id])
            else:
                labeled = record

            out_f.write(json.dumps(labeled) + "\n")
            out_f.flush()

    print(f"Labeled outputs written to {args.output}")


if __name__ == "__main__":
    main()
