#!/usr/bin/env python
"""Phase 6 — Manual validation sample.

Usage:
    # Generate the sample CSV:
    python scripts/sample_for_review.py --output data/review_sample.csv

    # After manually filling your_label, compute FP rates:
    python scripts/sample_for_review.py --compute-fp data/review_sample.csv
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

LABELED_FILES = {
    "mistral7b": "data/outputs/labeled/hotpotqa_mistral7b_labeled_v2.jsonl",
    "llama31":   "data/outputs/labeled/hotpotqa_llama31_labeled_v2.jsonl",
    "qwen25":    "data/outputs/labeled/hotpotqa_qwen25_labeled_v2.jsonl",
    "phi4":      "data/outputs/labeled/hotpotqa_phi4_labeled_v2.jsonl",
}

SAMPLE_TYPES = [
    "contradiction_to_evidence",
    "attribute_error",
    "entity_error",
    "multi_hop_reasoning_error",
    "unsupported_inference",
]

TOTAL_SAMPLES = 100
CSV_FIELDS = [
    "id", "model", "question_type", "hallucination_type",
    "answer", "evidence", "reference",
    "your_label", "notes",
]


def load_all_records():
    records = []
    for model, path in LABELED_FILES.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping.")
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                row["_model"] = model
                records.append(row)
    return records


def stratified_sample(records, n=TOTAL_SAMPLES, seed=42):
    random.seed(seed)
    by_type = defaultdict(list)
    for r in records:
        t = r.get("hallucination_type", "")
        if t in SAMPLE_TYPES:
            by_type[t].append(r)

    # Count total eligible records to compute proportions
    total = sum(len(v) for v in by_type.values())
    if total == 0:
        return []

    sampled = []
    for t in SAMPLE_TYPES:
        pool = by_type[t]
        if not pool:
            continue
        quota = max(1, round(n * len(pool) / total))
        sampled.extend(random.sample(pool, min(quota, len(pool))))

    # Trim or top-up to exactly n if rounding drifted
    random.shuffle(sampled)
    return sampled[:n]


def generate(output_path):
    records = load_all_records()
    sample = stratified_sample(records)

    counts = Counter(r.get("hallucination_type") for r in sample)
    print(f"Sampled {len(sample)} records:")
    for t, c in sorted(counts.items()):
        print(f"  {t:<35} {c}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in sample:
            writer.writerow({
                "id":               r.get("id", ""),
                "model":            r.get("_model", ""),
                "question_type":    r.get("question_type", r.get("type", "")),
                "hallucination_type": r.get("hallucination_type", ""),
                "answer":           r.get("answer", ""),
                "evidence":         r.get("evidence", ""),
                "reference":        r.get("reference_answer", ""),
                "your_label":       "",
                "notes":            "",
            })

    print(f"\nCSV written to {output_path}")
    print("Fill the 'your_label' column with: correct / wrong")
    print("Then run:  python scripts/sample_for_review.py --compute-fp", output_path)


def compute_fp(csv_path):
    by_type = defaultdict(lambda: {"total": 0, "fp": 0})

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("your_label", "").strip().lower()
            if label not in ("correct", "wrong"):
                continue
            t = row.get("hallucination_type", "unknown")
            by_type[t]["total"] += 1
            if label == "wrong":
                by_type[t]["fp"] += 1

    reviewed = sum(v["total"] for v in by_type.values())
    if reviewed == 0:
        print("No filled rows found. Make sure your_label is 'correct' or 'wrong'.")
        return

    print(f"\nFP rates from {reviewed} reviewed records:")
    print("%-35s %6s %6s %8s" % ("type", "total", "FP", "FP rate"))
    print("-" * 60)
    for t in SAMPLE_TYPES:
        if t not in by_type:
            continue
        d = by_type[t]
        rate = d["fp"] / d["total"] if d["total"] else 0.0
        print("%-35s %6d %6d %7.1f%%" % (t, d["total"], d["fp"], rate * 100))

    total_fp = sum(v["fp"] for v in by_type.values())
    print("-" * 60)
    print("%-35s %6d %6d %7.1f%%" % ("OVERALL", reviewed, total_fp, total_fp / reviewed * 100))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", help="Path to write the sample CSV")
    group.add_argument("--compute-fp", metavar="CSV", help="Path to filled CSV to compute FP rates")
    args = parser.parse_args()

    if args.output:
        generate(args.output)
    else:
        compute_fp(args.compute_fp)


if __name__ == "__main__":
    main()
