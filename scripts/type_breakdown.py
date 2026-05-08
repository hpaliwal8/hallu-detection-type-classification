#!/usr/bin/env python
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io import read_jsonl, ensure_dir
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Labeled JSONL files.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    setup_logging()

    groups = defaultdict(list)
    for path in args.inputs:
        for r in read_jsonl(path):
            groups[(r.get("model_id", ""), r.get("prompt_id", ""))].append(r)

    rows = []
    print(f"{'Model':<25} {'Prompt':<10} {'Total':>6} {'Hall%':>7} | Type breakdown (% of hallucinations)")
    print("-" * 110)

    for (model, prompt), records in sorted(groups.items()):
        halls = [r for r in records if r.get("is_hallucinated")]
        total = len(records)
        hall_rate = len(halls) / total if total else 0.0

        type_counts = Counter(r.get("hallucination_type", "unknown") for r in halls)
        breakdown_pct = {
            t: round(c / len(halls) * 100, 1) for t, c in type_counts.most_common()
        } if halls else {}

        breakdown_str = ", ".join(f"{t}={p}%" for t, p in breakdown_pct.items())
        print(f"{model:<25} {prompt:<10} {total:>6} {hall_rate*100:>6.1f}% | {breakdown_str}")

        rows.append({
            "model_id": model,
            "prompt_id": prompt,
            "total": total,
            "hallucinated": len(halls),
            "hallucination_rate": round(hall_rate, 4),
            "type_breakdown_pct": breakdown_pct,
            "type_counts": dict(type_counts),
        })

    if args.output:
        ensure_dir(os.path.dirname(args.output))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
