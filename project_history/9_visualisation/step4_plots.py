#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_csv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Per-question F1 boxplot
    # -------------------------
    plt.figure(figsize=(14, 6))
    df.boxplot(column="f1", by="question_id", rot=90)
    plt.title("F1 distribution per question (100 GPT runs each)")
    plt.suptitle("")
    plt.ylabel("F1 score")
    plt.tight_layout()
    plt.savefig(args.outdir / "f1_boxplot_per_question.png", dpi=200)
    plt.close()

    # -------------------------
    # 2) Global F1 histogram
    # -------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(df["f1"], bins=30)
    plt.xlabel("F1 score")
    plt.ylabel("Number of runs")
    plt.title("Global F1 distribution (all questions, all runs)")
    plt.tight_layout()
    plt.savefig(args.outdir / "f1_histogram_global.png", dpi=200)
    plt.close()

    # -------------------------
    # 3) Global Jaccard histogram
    # -------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(df["jaccard"], bins=30)
    plt.xlabel("Jaccard similarity")
    plt.ylabel("Number of runs")
    plt.title("Global Jaccard distribution")
    plt.tight_layout()
    plt.savefig(args.outdir / "jaccard_histogram_global.png", dpi=200)
    plt.close()

    # -------------------------
    # 4) Per-question summary table
    # -------------------------
    summary = (
        df.groupby("question_id")
        .agg(
            mean_f1=("f1", "mean"),
            median_f1=("f1", "median"),
            mean_jaccard=("jaccard", "mean"),
            zero_f1_rate=("f1", lambda x: (x == 0).mean()),
        )
        .reset_index()
    )

    summary.to_csv(args.outdir / "per_question_summary.csv", index=False)

    # -------------------------
    # 5) Scatter: median F1 vs zero-F1-rate
    # -------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(summary["median_f1"], summary["zero_f1_rate"])
    plt.xlabel("Median F1 (per question)")
    plt.ylabel("Zero-F1 rate (fraction of runs with F1 = 0)")
    plt.title("Stability vs Performance\nMedian F1 vs Zero-F1 Rate")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.outdir / "scatter_medianF1_vs_zeroF1rate.png", dpi=200)
    plt.close()

    print("Plots and summary written to:", args.outdir)


if __name__ == "__main__":
    main()
