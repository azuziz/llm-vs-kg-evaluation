#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gpt_runs_jsonl",
        required=True,
        type=Path,
        help="GPT runs JSONL (one row per call, with edge_keys).",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        type=Path,
        help="Output CSV with per-question edge statistics.",
    )
    args = ap.parse_args()

    # question_id -> list of n_edges per run
    edge_counts = defaultdict(list)

    with args.gpt_runs_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            qid = obj["question_id"]
            n_edges = obj.get("n_edges")

            # Fallback if n_edges missing
            if n_edges is None:
                ek = obj.get("edge_keys", [])
                n_edges = len(ek) if isinstance(ek, list) else 0

            edge_counts[qid].append(n_edges)

    rows = []
    for qid, counts in edge_counts.items():
        s = pd.Series(counts)
        rows.append(
            {
                "question_id": qid,
                "n_runs": len(s),
                "mean_edges": s.mean(),
                "median_edges": s.median(),
                "min_edges": s.min(),
                "max_edges": s.max(),
                "zero_edge_rate": (s == 0).mean(),
            }
        )

    df = pd.DataFrame(rows).sort_values("question_id")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"[OK] Wrote per-question edge stats to {args.out_csv}")
    print()
    print("Global summary:")
    print(df[["mean_edges", "zero_edge_rate"]].describe())


if __name__ == "__main__":
    main()
