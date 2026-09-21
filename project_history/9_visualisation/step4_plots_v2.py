#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def qsort_key(q: str):
    q = str(q).strip()
    if q.startswith("Q") and q[1:].isdigit():
        return int(q[1:])
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--bottomk", type=int, default=20)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics_csv)

    # Ensure numeric
    df["question_id"] = df["question_id"].astype(str).str.strip()
    for c in ["jaccard", "precision_like", "recall_like", "f1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["question_id", "f1", "jaccard", "precision_like", "recall_like"])
    if df.empty:
        raise ValueError("metrics_csv loaded but has no usable rows after cleaning.")

    # ---- Per-question summary (this becomes your "big picture table") ----
    summary = (
        df.groupby("question_id", as_index=False)
          .agg(
              mean_f1=("f1", "mean"),
              median_f1=("f1", "median"),
              p25_f1=("f1", lambda x: x.quantile(0.25)),
              p75_f1=("f1", lambda x: x.quantile(0.75)),
              iqr_f1=("f1", lambda x: x.quantile(0.75) - x.quantile(0.25)),
              mean_jaccard=("jaccard", "mean"),
              median_jaccard=("jaccard", "median"),
              zero_f1_rate=("f1", lambda x: (x == 0).mean()),
              zero_jaccard_rate=("jaccard", lambda x: (x == 0).mean()),
              mean_precision=("precision_like", "mean"),
              mean_recall=("recall_like", "mean"),
          )
    )

    # Rank questions by median F1, then by mean F1
    summary = summary.sort_values(["median_f1", "mean_f1"], ascending=[False, False]).reset_index(drop=True)
    summary["rank_median_f1"] = summary.index + 1

    summary_out = args.outdir / "per_question_summary_ranked.csv"
    summary.to_csv(summary_out, index=False)

    # ---- Big picture plot: ranked median F1 for ALL questions ----
    plt.figure(figsize=(10, 4))
    plt.plot(summary["rank_median_f1"], summary["median_f1"])
    plt.xlabel("Question rank by median F1 (1=best)")
    plt.ylabel("Median F1 (over 100 runs)")
    plt.title("Per-question performance overview (median F1 ranked)")
    plt.tight_layout()
    plt.savefig(args.outdir / "ranked_median_f1_all_questions.png", dpi=200)
    plt.close()

    # Histogram of per-question medians (163 values)
    plt.figure(figsize=(6, 4))
    plt.hist(summary["median_f1"], bins=30)
    plt.xlabel("Per-question median F1")
    plt.ylabel("Number of questions")
    plt.title("Distribution of per-question median F1")
    plt.tight_layout()
    plt.savefig(args.outdir / "hist_median_f1_per_question.png", dpi=200)
    plt.close()

    # ---- Small scale plots: Top-K and Bottom-K boxplots ----
    def boxplot_subset(question_ids, title, filename):
        order = sorted(question_ids, key=qsort_key)
        data = [df.loc[df["question_id"] == q, "f1"].values for q in order]
        plt.figure(figsize=(max(10, len(order) * 0.45), 5))
        plt.boxplot(data, labels=order, vert=True)
        plt.xticks(rotation=90)
        plt.ylabel("F1")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(args.outdir / filename, dpi=200)
        plt.close()

    topk = summary.head(args.topk)["question_id"].tolist()
    bottomk = summary.tail(args.bottomk)["question_id"].tolist()

    boxplot_subset(topk, f"Top {args.topk} questions by median F1", f"f1_boxplot_top{args.topk}.png")
    boxplot_subset(bottomk, f"Bottom {args.bottomk} questions by median F1", f"f1_boxplot_bottom{args.bottomk}.png")

    # Optional: most variable (largest IQR)
    most_var = summary.sort_values("iqr_f1", ascending=False).head(20)["question_id"].tolist()
    boxplot_subset(most_var, "Most variable questions (top 20 by F1 IQR)", "f1_boxplot_most_variable20.png")

    # ---- Big picture table snippets (top/bottom) ----
    summary.head(30).to_csv(args.outdir / "table_top30_by_median_f1.csv", index=False)
    summary.tail(30).to_csv(args.outdir / "table_bottom30_by_median_f1.csv", index=False)

    print("Wrote readable plots + ranked tables to:", args.outdir)
    print("Key table:", summary_out)


if __name__ == "__main__":
    main()
