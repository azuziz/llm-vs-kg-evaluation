#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# Paths
# -------------------------
ROOT = Path(__file__).resolve().parents[1]   # /gpt/7_gpt
OUT = ROOT / "out"
PLOTS = OUT / "plots"

METRICS_CSV = OUT / "step2_metrics_per_run.csv"

# -------------------------
# Load data
# -------------------------
def load_df() -> pd.DataFrame:
    df = pd.read_csv(METRICS_CSV)
    # ensure numeric
    for col in ["jaccard", "precision_like", "recall_like", "f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

# -------------------------
# Plot A: F1 boxplot per question
# -------------------------
def plot_f1_boxplot(df: pd.DataFrame) -> None:
    PLOTS.mkdir(exist_ok=True)

    grouped = [g["f1"].values for _, g in df.groupby("question_id")]
    labels = [qid for qid, _ in df.groupby("question_id")]

    plt.figure(figsize=(8, 4))
    plt.boxplot(grouped, labels=labels, showfliers=True)
    plt.ylabel("F1 score")
    plt.xlabel("Question")
    plt.title("GPT vs ROBOKOP — F1 distribution per question")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "f1_boxplot_per_question.png", dpi=200)
    plt.close()

# -------------------------
# Plot B: Precision vs Recall scatter
# -------------------------
def plot_precision_recall(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 6))

    for qid, g in df.groupby("question_id"):
        plt.scatter(
            g["precision_like"],
            g["recall_like"],
            label=qid,
            alpha=0.7,
        )

    plt.xlabel("Precision-like (|∩| / |GPT|)")
    plt.ylabel("Recall-like (|∩| / |ROBOKOP|)")
    plt.title("Precision vs Recall per run")
    plt.legend(title="Question")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "precision_vs_recall.png", dpi=200)
    plt.close()

# -------------------------
# Plot C: Jaccard boxplot per question
# -------------------------
def plot_jaccard_boxplot(df: pd.DataFrame) -> None:
    grouped = [g["jaccard"].values for _, g in df.groupby("question_id")]
    labels = [qid for qid, _ in df.groupby("question_id")]

    plt.figure(figsize=(8, 4))
    plt.boxplot(grouped, labels=labels, showfliers=True)
    plt.ylabel("Jaccard similarity")
    plt.xlabel("Question")
    plt.title("GPT vs ROBOKOP — Jaccard overlap per question")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "jaccard_boxplot_per_question.png", dpi=200)
    plt.close()

# -------------------------
# Main
# -------------------------
def main() -> None:
    df = load_df()
    plot_f1_boxplot(df)
    plot_precision_recall(df)
    plot_jaccard_boxplot(df)
    print(f"Plots written to {PLOTS}")

if __name__ == "__main__":
    main()
