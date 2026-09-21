#!/usr/bin/env python3
"""
Compute predicate frequency counts + fractions from:
  A) ROBOKOP triples TSV (subject, predicate, object)  [deduplicated baseline]
  B) GPT JSONL runs with an "edges" list containing {"predicate": "..."} entries

Outputs:
  - predicate_counts_<prefix>_full.csv
  - top20_<prefix>_predicates.csv
Optionally:
  - top20_<prefix>_predicates.png
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd


ROBOKOP_PRED_COL_CANDIDATES = [
    "predicate", "pred", "relation", "edge_predicate", "biolink_predicate"
]


def detect_predicate_column(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in ROBOKOP_PRED_COL_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    # fallback: assume 2nd column = predicate in (subj, pred, obj)
    if df.shape[1] >= 2:
        return df.columns[1]
    raise ValueError("Could not detect predicate column (file has <2 columns).")


def compute_counts(pred_series: pd.Series) -> pd.DataFrame:
    pred_series = pred_series.dropna().astype(str)
    counts = pred_series.value_counts(dropna=False)
    total = int(counts.sum())
    out = (
        counts.rename("count")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "predicate"})
    )
    out["fraction"] = out["count"] / total if total > 0 else 0.0
    return out


def save_outputs(df_counts: pd.DataFrame, outdir: Path, prefix: str, make_plot: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    full_path = outdir / f"predicate_counts_{prefix}_full.csv"
    top20_path = outdir / f"top20_{prefix}_predicates.csv"

    df_counts.to_csv(full_path, index=False)

    df_top20 = df_counts.head(20).copy()
    df_top20.to_csv(top20_path, index=False)

    if make_plot:
        # matplotlib only if requested
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = plt.gca()

        # horizontal is usually more readable for long predicate strings
        ax.barh(df_top20["predicate"][::-1], df_top20["fraction"][::-1])
        ax.set_xlabel("fraction")
        ax.set_ylabel("predicate")
        ax.set_title(f"Top 20 predicates ({prefix})")
        plt.tight_layout()

        plot_path = outdir / f"top20_{prefix}_predicates.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)


def load_robokop_tsv(path: Path, sep: str, has_header: bool) -> pd.DataFrame:
    if has_header:
        df = pd.read_csv(path, sep=sep, dtype=str)
    else:
        df = pd.read_csv(path, sep=sep, header=None, dtype=str)
    return df


def load_gpt_jsonl_predicates(path: Path) -> List[str]:
    preds: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

            # expected: obj["edges"] is a list of dicts with key "predicate"
            edges = obj.get("edges", None)
            if edges is None:
                # some runs store under "output" or similar; try a couple common fallbacks
                edges = obj.get("output", {}).get("edges", None)

            if edges is None:
                continue

            if not isinstance(edges, list):
                continue

            for e in edges:
                if isinstance(e, dict):
                    p = e.get("predicate", None)
                    if p is not None:
                        preds.append(str(p))
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robokop-tsv", type=str, help="Deduplicated ROBOKOP triples TSV (subject, predicate, object).")
    ap.add_argument("--gpt-jsonl", type=str, help="GPT runs JSONL with edges[].predicate.")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory.")
    ap.add_argument("--robokop-prefix", type=str, default="robokop_dedup", help="Prefix label for ROBOKOP outputs.")
    ap.add_argument("--gpt-prefix", type=str, default="gpt", help="Prefix label for GPT outputs.")
    ap.add_argument("--sep", type=str, default="\t", help="TSV separator for ROBOKOP file (default: tab).")
    ap.add_argument("--no-header", action="store_true", help="Set if ROBOKOP TSV has no header row.")
    ap.add_argument("--plot", action="store_true", help="Also save top-20 bar plot PNG(s).")

    args = ap.parse_args()
    outdir = Path(args.outdir)

    if not args.robokop_tsv and not args.gpt_jsonl:
        raise SystemExit("Provide at least one of --robokop-tsv or --gpt-jsonl")

    if args.robokop_tsv:
        rob_path = Path(args.robokop_tsv)
        df = load_robokop_tsv(rob_path, sep=args.sep, has_header=(not args.no_header))
        pred_col = detect_predicate_column(df)
        df_counts = compute_counts(df[pred_col]).sort_values("count", ascending=False).reset_index(drop=True)
        save_outputs(df_counts, outdir, args.robokop_prefix, make_plot=args.plot)

        total = int(df_counts["count"].sum())
        n_pred = df_counts.shape[0]
        print(f"[ROBOKOP] total edges = {total:,} | unique predicates = {n_pred:,}")
        print(f"[ROBOKOP] wrote: {outdir / f'predicate_counts_{args.robokop_prefix}_full.csv'}")
        print(f"[ROBOKOP] wrote: {outdir / f'top20_{args.robokop_prefix}_predicates.csv'}")

    if args.gpt_jsonl:
        gpt_path = Path(args.gpt_jsonl)
        preds = load_gpt_jsonl_predicates(gpt_path)
        df_counts = compute_counts(pd.Series(preds, dtype="string")).sort_values("count", ascending=False).reset_index(drop=True)
        save_outputs(df_counts, outdir, args.gpt_prefix, make_plot=args.plot)

        total = int(df_counts["count"].sum())
        n_pred = df_counts.shape[0]
        print(f"[GPT] total edges = {total:,} | unique predicates = {n_pred:,}")
        print(f"[GPT] wrote: {outdir / f'predicate_counts_{args.gpt_prefix}_full.csv'}")
        print(f"[GPT] wrote: {outdir / f'top20_{args.gpt_prefix}_predicates.csv'}")


if __name__ == "__main__":
    main()
