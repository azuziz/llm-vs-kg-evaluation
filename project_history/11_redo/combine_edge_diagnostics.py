#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robokop_csv", required=True, type=Path)
    ap.add_argument("--v1_csv", required=True, type=Path)
    ap.add_argument("--v2_csv", required=True, type=Path)
    ap.add_argument("--v3_csv", required=True, type=Path)
    ap.add_argument("--v4_csv", required=True, type=Path)
    ap.add_argument("--out_combined_csv", required=True, type=Path)
    ap.add_argument("--out_summary_md", required=True, type=Path)
    args = ap.parse_args()

    rob = pd.read_csv(args.robokop_csv)
    v1  = pd.read_csv(args.v1_csv)
    v2  = pd.read_csv(args.v2_csv)
    v3  = pd.read_csv(args.v3_csv)
    v4  = pd.read_csv(args.v4_csv)

    # v1 might not have prompt_version col; normalize
    if "prompt_version" not in v1.columns:
        v1["prompt_version"] = "v1"

    def pivot(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        keep = ["question_id", "mean_edges", "zero_edge_rate", "median_edges", "max_edges"]
        df2 = df[keep].copy()
        df2 = df2.rename(columns={
            "mean_edges": f"{tag}_mean_edges",
            "zero_edge_rate": f"{tag}_zero_edge_rate",
            "median_edges": f"{tag}_median_edges",
            "max_edges": f"{tag}_max_edges",
        })
        return df2

    out = rob.copy()
    out = out.merge(pivot(v1, "v1"), on="question_id", how="left")
    out = out.merge(pivot(v2, "v2"), on="question_id", how="left")
    out = out.merge(pivot(v3, "v3"), on="question_id", how="left")
    out = out.merge(pivot(v4, "v4"), on="question_id", how="left")

    args.out_combined_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_combined_csv, index=False)

    # Write a small markdown summary
    def summarize(df: pd.DataFrame, tag: str) -> str:
        return (
            f"- **{tag}**: mean(mean_edges)={df[f'{tag}_mean_edges'].mean():.3f}, "
            f"median(mean_edges)={df[f'{tag}_mean_edges'].median():.3f}, "
            f"mean(zero_edge_rate)={df[f'{tag}_zero_edge_rate'].mean():.3f}"
        )

    md = []
    md.append("# Edge output diagnostics\n")
    md.append("Per-question edge-count summaries (163 questions).\n")
    md.append("## GPT output volume\n")
    md.append(summarize(out, "v1"))
    md.append(summarize(out, "v2"))
    md.append(summarize(out, "v3"))
    md.append(summarize(out, "v4"))
    md.append("\n## ROBOKOP density\n")
    md.append(f"- **ROBOKOP**: mean edges per question = {out['robokop_n_edges'].mean():.3f}, "
              f"median = {out['robokop_n_edges'].median():.3f}, "
              f"min={out['robokop_n_edges'].min()}, max={out['robokop_n_edges'].max()}\n")

    args.out_summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary_md.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] wrote {args.out_combined_csv}")
    print(f"[OK] wrote {args.out_summary_md}")


if __name__ == "__main__":
    main()
