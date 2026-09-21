#!/usr/bin/env python3
"""
Union-over-runs evaluation of GPT edge outputs vs ROBOKOP baseline.

Input 1 (JSONL): lines like
{"question_id":"Q1","run_id":3,"edge_keys":[["S","P","O"], ...], ...}

Input 2 (TSV/CSV): ROBOKOP edges with columns at least:
question_id, subject, predicate, object

Outputs:
- Per-question metrics for union(G) vs K
- Per-question mean-over-runs metrics
- Summary stats including zero-F1 rates for:
    * mean-over-runs (F1_mean == 0)
    * union-over-runs (F1_union == 0)
  plus "never-match rate" under union (intersection_size == 0)
- Plots (PNG) into output directory
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd
import matplotlib.pyplot as plt

Edge = Tuple[str, str, str]


def safe_precision(intersection_n: int, g_n: int) -> float:
    # define as 0 if |G|=0
    if g_n == 0:
        return 0.0
    return intersection_n / g_n


def safe_recall(intersection_n: int, k_n: int) -> float:
    # define as 0 if |K|=0 (your convention)
    if k_n == 0:
        return 0.0
    return intersection_n / k_n


def safe_f1(p: float, r: float) -> float:
    # harmonic mean; 0 if both 0
    if p == 0.0 and r == 0.0:
        return 0.0
    denom = p + r
    if denom == 0.0:
        return 0.0
    return 2.0 * p * r / denom


def edge_from_list(x) -> Edge | None:
    """
    edge_keys entries are typically ["SUBJ_CURIE","biolink:...","OBJ_CURIE"].
    Returns (s,p,o) as a tuple, or None if malformed.
    """
    if not isinstance(x, list) or len(x) != 3:
        return None
    s, p, o = x
    if not (isinstance(s, str) and isinstance(p, str) and isinstance(o, str)):
        return None
    return (s.strip(), p.strip(), o.strip())


def load_gpt_jsonl(jsonl_path: str) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error in {jsonl_path} line {line_no}: {e}") from e

            if "question_id" not in obj or "run_id" not in obj:
                raise RuntimeError(f"Missing question_id/run_id in {jsonl_path} line {line_no}")

            edge_keys = obj.get("edge_keys", [])
            edges: List[Edge] = []
            if isinstance(edge_keys, list):
                for ek in edge_keys:
                    e3 = edge_from_list(ek)
                    if e3 is not None:
                        edges.append(e3)

            rows.append(
                {
                    "question_id": str(obj["question_id"]),
                    "run_id": int(obj["run_id"]),
                    "edges": edges,
                    "n_edges": int(obj.get("n_edges", len(edges))),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No rows loaded from {jsonl_path}")
    return df


def load_robokop_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tsv", ".tab"]:
        df = pd.read_csv(path, sep="\t", dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)

    required = {"question_id", "subject", "predicate", "object"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"ROBOKOP file missing required columns: {sorted(missing)}")

    for c in ["question_id", "subject", "predicate", "object"]:
        df[c] = df[c].astype(str).str.strip()
    return df


def build_robokop_sets(robokop_df: pd.DataFrame) -> Dict[str, Set[Edge]]:
    K: Dict[str, Set[Edge]] = defaultdict(set)
    for row in robokop_df.itertuples(index=False):
        q = getattr(row, "question_id")
        s = getattr(row, "subject")
        p = getattr(row, "predicate")
        o = getattr(row, "object")
        K[q].add((s, p, o))
    return K


def build_gpt_union_sets(gpt_df: pd.DataFrame) -> Dict[str, Set[Edge]]:
    G_union: Dict[str, Set[Edge]] = defaultdict(set)
    for row in gpt_df.itertuples(index=False):
        q = row.question_id
        for e in row.edges:
            G_union[q].add(e)
    return G_union


def build_gpt_run_sets(gpt_df: pd.DataFrame) -> Dict[Tuple[str, int], Set[Edge]]:
    G_run: Dict[Tuple[str, int], Set[Edge]] = defaultdict(set)
    for row in gpt_df.itertuples(index=False):
        key = (row.question_id, int(row.run_id))
        for e in row.edges:
            G_run[key].add(e)
    return G_run


def per_question_union_metrics(
    G_union: Dict[str, Set[Edge]],
    K: Dict[str, Set[Edge]],
    questions: Iterable[str],
) -> pd.DataFrame:
    out = []
    for q in questions:
        g = G_union.get(q, set())
        k = K.get(q, set())
        inter = g.intersection(k)

        p = safe_precision(len(inter), len(g))
        r = safe_recall(len(inter), len(k))
        f1 = safe_f1(p, r)

        denom = len(g.union(k))
        j = (len(inter) / denom) if denom > 0 else 0.0

        out.append(
            {
                "question_id": q,
                "G_union_size": len(g),
                "K_size": len(k),
                "intersection_size": len(inter),
                "P_union": p,
                "R_union": r,
                "F1_union": f1,
                "J_union": j,
            }
        )
    return pd.DataFrame(out)


def per_question_mean_over_runs_metrics(
    G_run: Dict[Tuple[str, int], Set[Edge]],
    K: Dict[str, Set[Edge]],
    questions: Iterable[str],
    expected_runs: int | None = None,
) -> pd.DataFrame:
    """
    Compute P,R,F1 per run, then mean over runs per question.
    """
    acc = defaultdict(list)

    runs_by_q = defaultdict(set)
    for (q, r) in G_run.keys():
        runs_by_q[q].add(r)

    for q in questions:
        k = K.get(q, set())
        run_ids = sorted(runs_by_q.get(q, set()))
        # expected_runs is a sanity check; do not enforce here
        _ = expected_runs

        for r in run_ids:
            g = G_run.get((q, r), set())
            inter = g.intersection(k)

            p = safe_precision(len(inter), len(g))
            rc = safe_recall(len(inter), len(k))
            f1 = safe_f1(p, rc)

            acc[q].append((p, rc, f1, len(g), len(inter)))

        if len(acc[q]) == 0:
            # Should not happen; treat as zeros
            acc[q].append((0.0, 0.0, 0.0, 0, 0))

    out = []
    for q, vals in acc.items():
        ps = [v[0] for v in vals]
        rs = [v[1] for v in vals]
        f1s = [v[2] for v in vals]
        g_sizes = [v[3] for v in vals]
        inter_sizes = [v[4] for v in vals]

        out.append(
            {
                "question_id": q,
                "runs_observed": len(vals),
                "G_mean_size": sum(g_sizes) / len(g_sizes),
                "intersection_mean_size": sum(inter_sizes) / len(inter_sizes),
                "P_mean": sum(ps) / len(ps),
                "R_mean": sum(rs) / len(rs),
                "F1_mean": sum(f1s) / len(f1s),
            }
        )
    return pd.DataFrame(out)


def save_plots(df: pd.DataFrame, outdir: str, prefix: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Histogram: union F1
    plt.figure()
    plt.hist(df["F1_union"].astype(float).values, bins=30)
    plt.xlabel("F1 (union over runs)")
    plt.ylabel("Number of questions")
    plt.title("Distribution of union-over-runs F1")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_hist_f1_union.png"), dpi=200)
    plt.close()

    # Scatter: P_union vs R_union
    plt.figure()
    plt.scatter(df["P_union"].astype(float).values, df["R_union"].astype(float).values, s=10)
    plt.xlabel("Precision-like containment (union)")
    plt.ylabel("Recall-like containment (union)")
    plt.title("Union-over-runs P vs R")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_scatter_p_vs_r_union.png"), dpi=200)
    plt.close()

    # Histogram: union Jaccard
    plt.figure()
    plt.hist(df["J_union"].astype(float).values, bins=30)
    plt.xlabel("Jaccard(G_union, K)")
    plt.ylabel("Number of questions")
    plt.title("Distribution of union-over-runs Jaccard")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_hist_jaccard_union.png"), dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-jsonl", required=True, help="GPT outputs JSONL")
    ap.add_argument("--robokop", required=True, help="ROBOKOP edges table (TSV/CSV)")
    ap.add_argument("--outdir", default="union_eval_out", help="Output directory")
    ap.add_argument("--expected-runs", type=int, default=None, help="e.g., 100 (optional sanity check)")
    ap.add_argument("--prefix", default="v1", help="Prefix for output files")
    args = ap.parse_args()

    gpt_df = load_gpt_jsonl(args.gpt_jsonl)
    rob_df = load_robokop_table(args.robokop)

    K = build_robokop_sets(rob_df)
    G_union = build_gpt_union_sets(gpt_df)
    G_run = build_gpt_run_sets(gpt_df)

    # Question universe: anything in either source
    questions = sorted(set(gpt_df["question_id"].unique()).union(set(rob_df["question_id"].unique())))

    union_df = per_question_union_metrics(G_union, K, questions)
    mean_df = per_question_mean_over_runs_metrics(G_run, K, questions, expected_runs=args.expected_runs)

    merged = union_df.merge(mean_df, on="question_id", how="left")

    # zero-F1 rates:
    # mean-over-runs: fraction of questions with mean F1 over runs == 0
    # union-over-runs: fraction of questions with union F1 == 0 (i.e., no baseline triple ever matched)
    zero_f1_rate_mean = (merged["F1_mean"].fillna(0.0) == 0.0).mean()
    zero_f1_rate_union = (merged["F1_union"].fillna(0.0) == 0.0).mean()

    # "never matched any baseline triple" under union aggregation
    never_match_rate_union = (merged["intersection_size"].fillna(0).astype(int) == 0).mean()

    summary = {
        "n_questions": int(len(merged)),

        "zero_f1_rate_mean_over_runs": float(zero_f1_rate_mean),
        "zero_f1_rate_union_over_runs": float(zero_f1_rate_union),
        "never_match_rate_union_over_runs": float(never_match_rate_union),

        "P_union_mean": float(merged["P_union"].mean()),
        "R_union_mean": float(merged["R_union"].mean()),
        "F1_union_mean": float(merged["F1_union"].mean()),
        "J_union_mean": float(merged["J_union"].mean()),

        "P_mean_mean": float(merged["P_mean"].mean()),
        "R_mean_mean": float(merged["R_mean"].mean()),
        "F1_mean_mean": float(merged["F1_mean"].mean()),
    }

    os.makedirs(args.outdir, exist_ok=True)

    merged_path = os.path.join(args.outdir, f"{args.prefix}_per_question_metrics.tsv")
    merged.to_csv(merged_path, sep="\t", index=False)

    summary_path = os.path.join(args.outdir, f"{args.prefix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_plots(merged, args.outdir, args.prefix)

    print("Wrote:", merged_path)
    print("Wrote:", summary_path)
    print("Zero-F1 rate (mean-over-runs):", summary["zero_f1_rate_mean_over_runs"])
    print("Zero-F1 rate (union-over-runs):", summary["zero_f1_rate_union_over_runs"])
    print("Never-match rate (union-over-runs):", summary["never_match_rate_union_over_runs"])


if __name__ == "__main__":
    main()