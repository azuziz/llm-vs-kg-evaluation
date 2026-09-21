#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_csv", required=True, type=Path,
                    help="CSV with columns including question_id, edges_json, status")
    ap.add_argument("--prompt_version", required=True,
                    help="Label to write into output (v2/v3/v4)")
    ap.add_argument("--out_csv", required=True, type=Path)
    args = ap.parse_args()

    q_to_counts = {}

    with args.runs_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status") != "ok":
                continue
            qid = (row.get("question_id") or "").strip()
            if not qid:
                continue
            raw = (row.get("edges_json") or "").strip()
            if not raw:
                n_edges = 0
            else:
                try:
                    obj = json.loads(raw)
                    edges = obj.get("edges", [])
                    n_edges = len(edges) if isinstance(edges, list) else 0
                except json.JSONDecodeError:
                    n_edges = 0

            q_to_counts.setdefault(qid, []).append(n_edges)

    rows = []
    for qid, counts in q_to_counts.items():
        s = pd.Series(counts)
        rows.append({
            "prompt_version": args.prompt_version,
            "question_id": qid,
            "n_runs": int(s.shape[0]),
            "mean_edges": float(s.mean()),
            "median_edges": float(s.median()),
            "min_edges": int(s.min()),
            "max_edges": int(s.max()),
            "zero_edge_rate": float((s == 0).mean()),
        })

    df = pd.DataFrame(rows).sort_values("question_id")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv}")
    print(df[["mean_edges","zero_edge_rate"]].describe())


if __name__ == "__main__":
    main()
