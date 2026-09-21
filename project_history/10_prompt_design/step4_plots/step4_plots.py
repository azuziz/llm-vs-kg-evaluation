#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_run", type=Path, default=ROOT / "out" / "step3" / "per_run_metrics.csv")
    ap.add_argument("--per_question", type=Path, default=ROOT / "out" / "step3" / "per_question_metrics.csv")
    ap.add_argument("--outdir", type=Path, default=ROOT / "out" / "plots")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    per_run = pd.read_csv(args.per_run)
    per_q = pd.read_csv(args.per_question)

    # Ensure ordering
    prompt_order = ["v1", "v2", "v3", "v4"]
    per_run["prompt_version"] = pd.Categorical(per_run["prompt_version"], categories=prompt_order, ordered=True)
    per_q["prompt_version"] = pd.Categorical(per_q["prompt_version"], categories=prompt_order, ordered=True)

    # -------------------------
    # 1) F1 boxplot by prompt
    # -------------------------
    plt.figure()
    per_run.boxplot(column="f1", by="prompt_version", grid=False)
    plt.title("F1 distribution by prompt version (per-run)")
    plt.suptitle("")
    plt.xlabel("prompt_version")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(args.outdir / "f1_boxplot_by_prompt.png", dpi=200)
    plt.close()

    # -------------------------
    # 2) Jaccard boxplot by prompt
    # -------------------------
    plt.figure()
    per_run.boxplot(column="jaccard", by="prompt_version", grid=False)
    plt.title("Jaccard distribution by prompt version (per-run)")
    plt.suptitle("")
    plt.xlabel("prompt_version")
    plt.ylabel("Jaccard")
    plt.tight_layout()
    plt.savefig(args.outdir / "jaccard_boxplot_by_prompt.png", dpi=200)
    plt.close()

    # ---------------------------------------------------
    # 3) Per-question mean F1: prompt vs baseline (v1)
    # ---------------------------------------------------
    # pivot to wide: rows=question_id, cols=prompt_version, values=mean_f1
    wide = per_q.pivot(index="question_id", columns="prompt_version", values="mean_f1")

    # Make scatter plots v2/v3/v4 vs v1
    for pv in ["v2", "v3", "v4"]:
        if "v1" not in wide.columns or pv not in wide.columns:
            continue
        x = wide["v1"]
        y = wide[pv]

        plt.figure()
        plt.scatter(x, y)
        plt.xlabel("mean F1 (v1 baseline)")
        plt.ylabel(f"mean F1 ({pv})")
        plt.title(f"Per-question mean F1: {pv} vs v1")
        # diagonal reference
        mn = float(min(x.min(), y.min()))
        mx = float(max(x.max(), y.max()))
        plt.plot([mn, mx], [mn, mx])
        plt.tight_layout()
        plt.savefig(args.outdir / f"mean_f1_scatter_{pv}_vs_v1.png", dpi=200)
        plt.close()

    # ---------------------------------------------------
    # 4) Delta F1 per question (prompt - v1)
    # ---------------------------------------------------
    # build long delta table
    deltas = []
    for pv in ["v2", "v3", "v4"]:
        if "v1" in wide.columns and pv in wide.columns:
            d = (wide[pv] - wide["v1"]).dropna()
            for qid, val in d.items():
                deltas.append({"prompt_version": pv, "question_id": qid, "delta_f1": val})
    ddf = pd.DataFrame(deltas)

    if not ddf.empty:
        ddf["prompt_version"] = pd.Categorical(ddf["prompt_version"], categories=["v2", "v3", "v4"], ordered=True)

        plt.figure()
        ddf.boxplot(column="delta_f1", by="prompt_version", grid=False)
        plt.title("ΔF1 per question (prompt - v1)")
        plt.suptitle("")
        plt.xlabel("prompt_version")
        plt.ylabel("ΔF1")
        plt.axhline(0)
        plt.tight_layout()
        plt.savefig(args.outdir / "delta_f1_boxplot_vs_v1.png", dpi=200)
        plt.close()

    # ---------------------------------------------------
    # 5) Summary table (printed)
    # ---------------------------------------------------
    # mean F1 per prompt (per-run)
    summary = (
        per_run.groupby("prompt_version", observed=True)["f1"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values("prompt_version")
    )
    summary_path = args.outdir / "summary_f1_by_prompt.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[OK] wrote plots to {args.outdir}")
    print(f"[OK] wrote summary table to {summary_path}")


if __name__ == "__main__":
    main()
