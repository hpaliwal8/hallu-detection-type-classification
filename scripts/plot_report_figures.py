#!/usr/bin/env python
"""Publication-quality plots for the project report. Produces three figures
with infographic-style design (bold colors, value labels, clean layout)."""

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import read_jsonl, ensure_dir
from src.utils.logging import setup_logging


PROMPT_ORDER = ["plain", "abstain", "reasoning"]
PROMPT_LABELS = {"plain": "Plain (Zero-shot)", "abstain": "Abstain (Option)", "reasoning": "Reasoning (CoT)"}
MODEL_ORDER = [
    "phi-4-mini-instruct",
    "mistral-7b-instruct",
    "qwen2.5-7b-instruct",
    "llama-3.1-8b-instruct",
]
MODEL_LABELS = {
    "phi-4-mini-instruct": "Phi-4-mini",
    "mistral-7b-instruct": "Mistral-7B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B",
    "llama-3.1-8b-instruct": "Llama-3.1-8B",
}

# Bold, infographic-style palette
COLOR_PLAIN = "#D7263D"      # red
COLOR_ABSTAIN = "#F5B700"    # gold/yellow
COLOR_REASONING = "#1B3B6F"  # deep navy

TYPE_COLORS = {
    "contradiction_to_evidence": "#D7263D",   # red
    "attribute_error": "#F5B700",             # gold
    "entity_error": "#1B3B6F",                # navy
    "multi_hop_reasoning_error": "#7A8B99",   # muted slate
    "unsupported_inference": "#B0B7BF",       # light grey
}
TYPE_ORDER = [
    "contradiction_to_evidence",
    "attribute_error",
    "entity_error",
    "multi_hop_reasoning_error",
    "unsupported_inference",
]


def load_records(paths):
    rows = []
    for p in paths:
        for r in read_jsonl(p):
            rows.append(r)
    return pd.DataFrame(rows)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888")
    ax.spines["bottom"].set_color("#888")
    ax.grid(axis="y", color="#EEE", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_hallucination_rate_grouped(df, out_path):
    """Motivation panel: grouped bars of Hall% by model x prompt."""
    rate = (
        df.groupby(["model_id", "prompt_id"])["is_hallucinated"]
        .mean()
        .unstack("prompt_id")
        .reindex(MODEL_ORDER)
    )

    x = np.arange(len(MODEL_ORDER))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars_plain = ax.bar(x - width, rate["plain"] * 100, width,
                        label="Plain (Zero-shot)", color=COLOR_PLAIN, zorder=2)
    bars_reason = ax.bar(x, rate["reasoning"] * 100, width,
                         label="Reasoning (CoT)", color=COLOR_REASONING, zorder=2)
    bars_abstain = ax.bar(x + width, rate["abstain"] * 100, width,
                          label="Abstain (Option)", color=COLOR_ABSTAIN, zorder=2)

    for bars in (bars_plain, bars_reason, bars_abstain):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11)
    ax.set_ylabel("Hallucination Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Hallucination Rate by Model and Prompt",
                 fontsize=14, fontweight="bold", color="#1B3B6F", pad=30)
    ax.text(0.5, 1.02, "(Higher is Worse)",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=10, color="#666", style="italic")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_type_shifts_stacked(df, model_id, out_path):
    """Analysis panel: stacked horizontal 100% bars of type distribution by prompt
    for one model."""
    sub = df[(df["model_id"] == model_id) & df["is_hallucinated"]].copy()

    rows = []
    for prompt in PROMPT_ORDER:
        prompt_records = sub[sub["prompt_id"] == prompt]
        total = len(prompt_records)
        if total == 0:
            rows.append({"prompt": prompt, **{t: 0 for t in TYPE_ORDER}})
            continue
        counts = Counter(prompt_records["hallucination_type"])
        rows.append({
            "prompt": prompt,
            **{t: counts.get(t, 0) / total * 100 for t in TYPE_ORDER},
        })
    pivot = pd.DataFrame(rows).set_index("prompt").reindex(PROMPT_ORDER)

    fig, ax = plt.subplots(figsize=(11, 4))
    left = np.zeros(len(PROMPT_ORDER))
    for t in TYPE_ORDER:
        vals = pivot[t].values
        bars = ax.barh(np.arange(len(PROMPT_ORDER)), vals, left=left,
                       color=TYPE_COLORS[t], edgecolor="white",
                       linewidth=1.5, label=t.replace("_", " "))
        for i, v in enumerate(vals):
            if v >= 4:
                ax.text(left[i] + v / 2, i, f"{v:.1f}%",
                        ha="center", va="center", fontsize=9.5,
                        fontweight="bold",
                        color="white" if t in ("contradiction_to_evidence", "entity_error") else "#222")
        left += vals

    ax.set_yticks(np.arange(len(PROMPT_ORDER)))
    ax.set_yticklabels([PROMPT_LABELS[p] for p in PROMPT_ORDER], fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of Hallucinations (%)", fontsize=11)
    ax.set_title(f"Type Distribution Shifts with Prompt ({MODEL_LABELS[model_id]})",
                 fontsize=14, fontweight="bold", color="#1B3B6F", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False, fontsize=9.5)
    ax.invert_yaxis()
    style_axes(ax)
    ax.grid(axis="x", color="#EEE", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_hallucination_rate_by_type(df, out_path):
    """Combined: grouped bars (model x prompt) where each bar is stacked by type.
    Bar height = hallucination rate; segments = type share of all records."""
    # For each (model, prompt), compute type_count / total_records (so segments sum to Hall%)
    rows = []
    for model in MODEL_ORDER:
        for prompt in PROMPT_ORDER:
            sub = df[(df["model_id"] == model) & (df["prompt_id"] == prompt)]
            total = len(sub)
            if total == 0:
                continue
            counts = Counter(sub[sub["is_hallucinated"]]["hallucination_type"])
            row = {"model": model, "prompt": prompt}
            for t in TYPE_ORDER:
                row[t] = counts.get(t, 0) / total * 100
            rows.append(row)
    pivot = pd.DataFrame(rows)

    x = np.arange(len(MODEL_ORDER))
    width = 0.27
    offsets = {"plain": -width, "reasoning": 0, "abstain": width}
    prompt_short = {"plain": "P", "reasoning": "R", "abstain": "A"}

    fig, ax = plt.subplots(figsize=(13, 6))

    light_text_types = {"contradiction_to_evidence", "entity_error", "multi_hop_reasoning_error"}
    MIN_SEGMENT_FOR_LABEL = 5.0  # % — below this we hide the label to avoid clutter

    for prompt in PROMPT_ORDER:
        bottom = np.zeros(len(MODEL_ORDER))
        for t in TYPE_ORDER:
            heights = [
                pivot[(pivot["model"] == m) & (pivot["prompt"] == prompt)][t].iloc[0]
                if len(pivot[(pivot["model"] == m) & (pivot["prompt"] == prompt)]) > 0 else 0
                for m in MODEL_ORDER
            ]
            label = t.replace("_", " ") if prompt == "plain" else None
            ax.bar(x + offsets[prompt], heights, width,
                   bottom=bottom, color=TYPE_COLORS[t],
                   edgecolor="white", linewidth=0.6,
                   label=label, zorder=2)

            for i, h in enumerate(heights):
                if h >= MIN_SEGMENT_FOR_LABEL:
                    text_color = "white" if t in light_text_types else "#222"
                    ax.text(
                        i + offsets[prompt],
                        bottom[i] + h / 2,
                        f"{h:.0f}%",
                        ha="center", va="center",
                        fontsize=7.5, fontweight="bold",
                        color=text_color,
                        zorder=3,
                    )

            bottom += np.array(heights)

        # total Hall% label on top of each stacked bar
        for i, m in enumerate(MODEL_ORDER):
            row = pivot[(pivot["model"] == m) & (pivot["prompt"] == prompt)]
            if row.empty:
                continue
            total = sum(row[t].iloc[0] for t in TYPE_ORDER)
            ax.text(i + offsets[prompt], total + 1.5, f"{total:.0f}%",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                    color="#1B3B6F")
            # P/R/A label sits just below the axis line; model names go
            # further down via tick pad below so the two rows don't overlap.
            ax.annotate(
                prompt_short[prompt],
                xy=(i + offsets[prompt], 0), xycoords=("data", "axes fraction"),
                xytext=(0, -8), textcoords="offset points",
                ha="center", va="top", fontsize=9, color="#666",
                annotation_clip=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11)
    ax.tick_params(axis="x", pad=18)  # push model names below the P/R/A row
    ax.set_ylabel("Hallucination Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Hallucination Type Composition by Model and Prompt",
                 fontsize=14, fontweight="bold", color="#1B3B6F", pad=30)
    ax.text(0.5, 1.02,
            "Each bar's height = hallucination rate; segments = type share of all records. "
            "P=Plain, R=Reasoning, A=Abstain.",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=9.5, color="#666", style="italic")
    ax.set_ylim(-5, 110)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=5, frameon=False, fontsize=9.5)
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_overconfident_abstention(df, out_path):
    """Key Results panel: horizontal bars of overconfident abstention failure counts."""
    groups = defaultdict(dict)
    for _, r in df.iterrows():
        groups[(r["model_id"], r["id"])][r["prompt_id"]] = r

    counts = {m: 0 for m in MODEL_ORDER}
    for (model, _qid), prompts in groups.items():
        plain = prompts.get("plain")
        abstain = prompts.get("abstain")
        if plain is None or abstain is None:
            continue
        if plain.get("is_hallucinated") and abstain.get("hallucination_type") == "abstained":
            counts[model] += 1

    sorted_models = sorted(MODEL_ORDER, key=lambda m: counts[m], reverse=True)
    values = [counts[m] for m in sorted_models]
    labels = [MODEL_LABELS[m] for m in sorted_models]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(np.arange(len(sorted_models)), values,
                   color=COLOR_REASONING, edgecolor="white", linewidth=1.2, zorder=2)

    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_width() + max(values) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{v}", va="center", ha="left",
                fontsize=12, fontweight="bold", color="#1B3B6F")

    ax.set_yticks(np.arange(len(sorted_models)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Failure Counts (Questions)", fontsize=11)
    ax.set_title("Overconfident Abstention Failure",
                 fontsize=14, fontweight="bold", color="#1B3B6F", pad=30)
    ax.text(0.5, 1.02, "Plain Hallucinated, Abstain Correctly Refused",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=10, color="#666", style="italic")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    style_axes(ax)
    ax.grid(axis="x", color="#EEE", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Labeled HotpotQA JSONL files (one per model).")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--type_shifts_model", default="mistral-7b-instruct",
                        help="Which model's type-shift plot to render.")
    args = parser.parse_args()

    setup_logging()
    ensure_dir(args.out_dir)

    df = load_records(args.inputs)

    plot_hallucination_rate_grouped(
        df, os.path.join(args.out_dir, "report_hallucination_rate.png"))
    plot_hallucination_rate_by_type(
        df, os.path.join(args.out_dir, "report_hallucination_rate_by_type.png"))
    plot_type_shifts_stacked(
        df, args.type_shifts_model,
        os.path.join(args.out_dir, "report_type_shifts.png"))
    plot_overconfident_abstention(
        df, os.path.join(args.out_dir, "report_overconfident_abstention.png"))

    print(f"Report figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
