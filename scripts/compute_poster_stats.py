#!/usr/bin/env python
"""Compute every stat shown on the poster and print a verification report.

Usage:
    python scripts/compute_poster_stats.py
    python scripts/compute_poster_stats.py --fp-csv data/review_sample.csv
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

LABELED_FILES = {
    "Mistral-7B": "data/outputs/labeled/hotpotqa_mistral7b_labeled_v2.jsonl",
    "LLaMA-3.1":  "data/outputs/labeled/hotpotqa_llama31_labeled_v2.jsonl",
    "Qwen-2.5":   "data/outputs/labeled/hotpotqa_qwen25_labeled_v2.jsonl",
    "Phi-4":      "data/outputs/labeled/hotpotqa_phi4_labeled_v2.jsonl",
}

PROMPT_ORDER  = ["plain", "reasoning", "abstain"]
HALLUC_TYPES  = [
    "contradiction_to_evidence",
    "attribute_error",
    "entity_error",
    "multi_hop_reasoning_error",
    "unsupported_inference",
]
NON_HALLUC    = {"supported", "abstained"}


def load_all():
    records = []
    for model_label, path in LABELED_FILES.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                r["_model_label"] = model_label
                records.append(r)
    return records


def halluc_rate(recs):
    if not recs:
        return 0.0
    halluc = sum(1 for r in recs if r["hallucination_type"] not in NON_HALLUC)
    return halluc / len(recs) * 100


def sep(char="=", n=70):
    print(char * n)


# ── Panel 2: Hallucination rate by model × prompt ────────────────────────────
def panel2(records):
    sep()
    print("PANEL 2 — Hallucination Rate by Model and Prompt")
    sep()
    by_model_prompt = defaultdict(list)
    for r in records:
        by_model_prompt[(r["_model_label"], r.get("prompt_id", "unknown"))].append(r)

    header = f"{'Model':<14}" + "".join(f"{p:>12}" for p in PROMPT_ORDER)
    print(header)
    print("-" * (14 + 12 * len(PROMPT_ORDER)))
    for model in LABELED_FILES:
        row = f"{model:<14}"
        for p in PROMPT_ORDER:
            rate = halluc_rate(by_model_prompt[(model, p)])
            row += f"{rate:>11.1f}%"
        print(row)
    print()


# ── Panel 6: Abstention failure counts ───────────────────────────────────────
def panel6(records):
    sep()
    print("PANEL 6 — Abstention Failure: Plain Hallucinated & Abstain Correctly Refused")
    sep()
    # Group by model → question id → prompt_id
    by_model = defaultdict(lambda: defaultdict(dict))
    for r in records:
        by_model[r["_model_label"]][r["id"]][r.get("prompt_id", "unknown")] = r

    print(f"{'Model':<14}  {'Failure Count':>15}")
    print("-" * 35)
    for model in LABELED_FILES:
        count = 0
        for qid, prompts in by_model[model].items():
            plain_r   = prompts.get("plain")
            abstain_r = prompts.get("abstain")
            if plain_r is None or abstain_r is None:
                continue
            plain_halluc   = plain_r["hallucination_type"] not in NON_HALLUC
            abstain_ok     = abstain_r["hallucination_type"] == "abstained"
            if plain_halluc and abstain_ok:
                count += 1
        print(f"{model:<14}  {count:>15}")
    print()


# ── Panel 7 top: Type distribution by prompt (Mistral-7B) ────────────────────
def panel7_top(records):
    sep()
    print("PANEL 7 TOP — Type Distribution by Prompt (Mistral-7B only)")
    sep()
    mistral = [r for r in records if r["_model_label"] == "Mistral-7B"]
    by_prompt = defaultdict(list)
    for r in mistral:
        if r["hallucination_type"] not in NON_HALLUC:
            by_prompt[r.get("prompt_id", "unknown")].append(r["hallucination_type"])

    all_types = HALLUC_TYPES
    header = f"{'Type':<35}" + "".join(f"{p:>10}" for p in PROMPT_ORDER)
    print(header)
    print("-" * (35 + 10 * len(PROMPT_ORDER)))
    for t in all_types:
        row = f"{t:<35}"
        for p in PROMPT_ORDER:
            pool = by_prompt[p]
            pct = pool.count(t) / len(pool) * 100 if pool else 0.0
            row += f"{pct:>9.1f}%"
        print(row)
    print()
    print("(Totals per prompt — hallucinated records only):")
    for p in PROMPT_ORDER:
        print(f"  {p}: {len(by_prompt[p])} hallucinated records")
    print()


# ── Panel 7 bottom: FP rates from manual review CSV ──────────────────────────
def panel7_bottom(csv_path):
    sep()
    print(f"PANEL 7 BOTTOM — Phase 6 FP Rates  (from {csv_path})")
    sep()
    by_type = defaultdict(lambda: {"total": 0, "fp": 0})
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("your_label", "").strip().lower()
            if label not in ("correct", "wrong"):
                continue
            t = row.get("hallucination_type", "unknown")
            by_type[t]["total"] += 1
            if label == "wrong":
                by_type[t]["fp"] += 1

    reviewed = sum(v["total"] for v in by_type.values())
    if reviewed == 0:
        print("No filled rows found.")
        return

    print(f"{'Type':<35} {'Total':>6} {'FP':>6} {'FP rate':>9}")
    print("-" * 60)
    for t in HALLUC_TYPES:
        if t not in by_type:
            continue
        d = by_type[t]
        rate = d["fp"] / d["total"] * 100 if d["total"] else 0.0
        print(f"{t:<35} {d['total']:>6} {d['fp']:>6} {rate:>8.1f}%")
    total_fp = sum(v["fp"] for v in by_type.values())
    print("-" * 60)
    print(f"{'OVERALL':<35} {reviewed:>6} {total_fp:>6} {total_fp/reviewed*100:>8.1f}%")
    print()


# ── Overall hallucination counts (quick sanity check) ────────────────────────
def summary(records):
    sep()
    print("SUMMARY — Record counts and type breakdown per model")
    sep()
    by_model = defaultdict(list)
    for r in records:
        by_model[r["_model_label"]].append(r)

    for model, recs in by_model.items():
        counts = Counter(r["hallucination_type"] for r in recs)
        total  = len(recs)
        print(f"\n{model}  (n={total})")
        for t, c in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {t:<35} {c:>5}  ({c/total*100:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-csv", default="data/review_sample.csv",
                        help="Path to filled review CSV for FP rate computation")
    args = parser.parse_args()

    print("\nLoading labeled files...")
    records = load_all()
    print(f"Loaded {len(records)} total records.\n")

    summary(records)
    panel2(records)
    panel6(records)
    panel7_top(records)

    if os.path.exists(args.fp_csv):
        panel7_bottom(args.fp_csv)
    else:
        print(f"FP CSV not found at {args.fp_csv} — skipping Panel 7 bottom.")
        print("Run: python scripts/sample_for_review.py --compute-fp <your_csv>")


if __name__ == "__main__":
    main()
