#!/usr/bin/env python
"""Re-run label_outputs.py for all four HotpotQA models.

Usage:
    python scripts/relabel_all.py              # writes to *_labeled_v2.jsonl
    python scripts/relabel_all.py --suffix v3  # writes to *_labeled_v3.jsonl
"""
import argparse
import os
import subprocess
import sys

MODELS = [
    ("hotpotqa_mistral7b",    "data/outputs/cleaned/hotpotqa_mistral7b.jsonl"),
    ("hotpotqa_llama31",      "data/outputs/cleaned/hotpotqa_llama31.jsonl"),
    ("hotpotqa_qwen25",       "data/outputs/cleaned/hotpotqa_qwen25.jsonl"),
    ("hotpotqa_phi4",         "data/outputs/cleaned/hotpotqa_phi4_results.jsonl"),
]

HOTPOTQA = "data/processed/hotpotqa.jsonl"
OUTPUT_DIR = "data/outputs/labeled"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="v2", help="Output file suffix (default: v2)")
    args = parser.parse_args()

    for name, input_path in MODELS:
        output_path = os.path.join(OUTPUT_DIR, f"{name}_labeled_{args.suffix}.jsonl")
        print(f"\n{'='*60}")
        print(f"Labeling: {name}  →  {output_path}")
        print(f"{'='*60}")

        result = subprocess.run(
            [sys.executable, "scripts/label_outputs.py",
             "--input", input_path,
             "--hotpotqa", HOTPOTQA,
             "--output", output_path],
            check=False,
        )

        if result.returncode != 0:
            print(f"ERROR: labeling failed for {name} (exit code {result.returncode})")
            sys.exit(result.returncode)

    print("\nAll models labeled successfully.")


if __name__ == "__main__":
    main()
