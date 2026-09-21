#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


F1_CANDIDATES = [
    "mean_f1",
    "mean_f1_match",
    "mean_f1match",
    "mean_f1_match_rate",
    "mean_f1match_rate",
    "f1",
    "f1_mean",
]


def find_f1_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)

    # Exact matches first
    for c in F1_CANDIDATES:
        if c in cols:
            return c

    # Heuristic: any column that contains 'f1' and also contains 'mean'
    heur = [c for c in cols if ("f1" in c.lower()) and ("mean" in c.lower())]
    if len(heur) == 1:
        return heur[0]

    # Fallback: any column that contains 'f1'
    heur2 = [c for c in cols if "f1" in c.lower()]
    if len(heur2) == 1:
        return heur2[0]

    return None


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def read_variant(path: str, variant: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    f1_col = find_f1_column(df)
    if f1_col is None:
        print(f"\nERROR: Could not find an F1 column in {path}")
        print("Columns found:")
        for c in df.columns:
            print(f"  - {c}")
        print("\nFix options:")
        print("  1) Rename your F1 column to 'mean_f1', OR")
        print("  2) Add your exact column name to F1_CANDIDATES in this script.")
        sys.exit(1)

    if "question_id" not in df.columns:
        print(f"\nERROR: 'question_id' column missing in {path}")
        sys.exit(1)

    out = df.copy()
    out["variant"] = variant
    out[f1_col] = pd.to_numeric(out[f1_col], errors="coerce")
    out = out.rename(columns={f1_col: "mean_f1"})
    out = out.dropna(subset=["mean_f1"])

    return out[["question_id", "variant", "mean_f1"]]


def plot_boxplot(df_all: pd.DataFrame, outpath: str, order: List[str]) -> None:
    data = [df_all.loc[df_all["variant"] == v, "mean_f1"].values for v in order]

    fig = plt.figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111)

    # Boxplot with sensible whiskers (prevents "flat" look)
    ax.boxplot(
        data,
        tick_labels=order,   # Matplotlib >=3.9
        showfliers=False,
        whis=(5, 95),        # robust whiskers; or delete for default 1.5*IQR
    )

    # Zoom y-axis to reveal structure (based on global percentile)
    all_vals = np.concatenate([x for x in data if len(x)])
    upper = float(np.nanpercentile(all_vals, 99))
    ax.set_ylim(0, max(0.05, upper * 1.10))

    # Overlay jittered points to show mass at 0 / near 0
    for i, v in enumerate(order, start=1):
        y = df_all.loc[df_all["variant"] == v, "mean_f1"].values
        x = np.random.normal(i, 0.06, size=len(y))
        ax.plot(x, y, marker=".", linestyle="None", markersize=2, alpha=0.35)

    ax.set_title("Per-question mean F1 by prompt variant")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Mean F1 (per question; averaged over runs)")

    # Annotate medians (more precision than 0.00)
    medians = [np.median(x) if len(x) else np.nan for x in data]
    for i, m in enumerate(medians, start=1):
        if np.isfinite(m):
            ax.text(i, m + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_zero_f1(df_all: pd.DataFrame, outpath: str, order: List[str]) -> None:
    rows = []
    for v in order:
        sub = df_all[df_all["variant"] == v]
        n = len(sub)
        k = int((sub["mean_f1"] == 0).sum())
        p = k / n if n else np.nan
        lo, hi = wilson_ci(k, n) if n else (np.nan, np.nan)
        rows.append((v, n, k, p, lo, hi))

    stats = pd.DataFrame(
        rows,
        columns=["variant", "n_questions", "n_zero_f1", "zero_f1_rate", "ci_lo", "ci_hi"],
    )

    fig = plt.figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111)

    x = np.arange(len(order))
    y = stats["zero_f1_rate"].values
    yerr = np.vstack([(y - stats["ci_lo"].values), (stats["ci_hi"].values - y)])

    ax.bar(x, y, yerr=yerr, capsize=5)
    ax.set_xticks(x, order)
    ax.set_ylim(0, 1)
    ax.set_title("Zero-F1 rate by prompt variant (per question)")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Zero-F1 rate (share of questions with mean F1 = 0)")

    for i, (p, n) in enumerate(zip(y, stats["n_questions"].values)):
        if np.isfinite(p):
            ax.text(i, p + 0.03, f"{p:.2f}\n(n={n})", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Plot key metrics (F1 distribution + zero-F1 rate) across v1–v4."
    )
    ap.add_argument("--indir", default=".", help="Directory containing v1..v4_per_question_metrics.tsv")
    ap.add_argument("--outdir", default="plots", help="Output directory for PNG figures")
    ap.add_argument("--prefix", default="key_metrics", help="Filename prefix for output figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = {
        "v1": os.path.join(args.indir, "v1_per_question_metrics.tsv"),
        "v2": os.path.join(args.indir, "v2_per_question_metrics.tsv"),
        "v3": os.path.join(args.indir, "v3_per_question_metrics.tsv"),
        "v4": os.path.join(args.indir, "v4_per_question_metrics.tsv"),
    }

    for v, p in files.items():
        if not os.path.exists(p):
            print(f"ERROR: missing file: {p}")
            sys.exit(1)

    dfs = [read_variant(files[v], v) for v in ["v1", "v2", "v3", "v4"]]
    df_all = pd.concat(dfs, ignore_index=True)

    order = ["v1", "v2", "v3", "v4"]

    out_box = os.path.join(args.outdir, f"{args.prefix}_mean_f1_boxplot.png")
    out_zero = os.path.join(args.outdir, f"{args.prefix}_zero_f1_rate.png")

    plot_boxplot(df_all, out_box, order)
    plot_zero_f1(df_all, out_zero, order)

    print(f"Wrote:\n  {out_box}\n  {out_zero}")


if __name__ == "__main__":
    main()