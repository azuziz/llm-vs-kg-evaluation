#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


# ============================================================
# Core stability math
# ============================================================

def jaccard(a: set, b: set) -> float:
    # match your existing behavior: empty vs empty is identical (1.0)
    if not a and not b:
        return 1.0
    u = len(a | b)
    return (len(a & b) / u) if u else 1.0


def mean_pairwise_jaccard(sets: List[set]) -> float:
    n = len(sets)
    if n <= 1:
        return float("nan")
    total = 0.0
    m = 0
    for i, j in combinations(range(n), 2):
        total += jaccard(sets[i], sets[j])
        m += 1
    return total / m if m else float("nan")


def per_question_metrics(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    df_runs must have:
      question_id (str), edge_count (int), pred_set (set), triple_set (set)
    Each row = one run.
    """
    out_rows = []
    for qid, g in df_runs.groupby("question_id", dropna=False):
        edge_counts = g["edge_count"].tolist()
        pred_sets = g["pred_set"].tolist()
        triple_sets = g["triple_set"].tolist()

        n_runs = len(edge_counts)
        mean_cnt = sum(edge_counts) / n_runs if n_runs else float("nan")
        var_cnt = sum((c - mean_cnt) ** 2 for c in edge_counts) / n_runs if n_runs else float("nan")
        std_cnt = math.sqrt(var_cnt) if var_cnt == var_cnt else float("nan")
        abst = sum(1 for c in edge_counts if c == 0) / n_runs if n_runs else float("nan")

        out_rows.append({
            "question_id": str(qid) if qid is not None else None,
            "n_runs": n_runs,
            "edge_count_mean": mean_cnt,
            "edge_count_var": var_cnt,
            "edge_count_std": std_cnt,
            "abstention_rate": abst,
            "predicate_set_stability": mean_pairwise_jaccard(pred_sets),
            "triple_set_stability": mean_pairwise_jaccard(triple_sets),
        })

    cols = [
        "question_id", "n_runs",
        "edge_count_mean", "edge_count_var", "edge_count_std",
        "abstention_rate", "predicate_set_stability", "triple_set_stability"
    ]
    return pd.DataFrame(out_rows)[cols].sort_values("question_id").reset_index(drop=True)


def stability_summary(df_per_q: pd.DataFrame) -> pd.DataFrame:
    """
    Match your v2_stability_summary.tsv style: transpose of describe()
    with index = metric names and columns count/mean/std/min/25%/50%/75%/max
    """
    metrics = [
        "edge_count_mean",
        "edge_count_std",
        "abstention_rate",
        "predicate_set_stability",
        "triple_set_stability",
    ]
    desc = df_per_q[metrics].describe(percentiles=[0.25, 0.5, 0.75]).T
    return desc


# ============================================================
# Parsers
# ============================================================

def _triples_from_edge_keys(edge_keys: Any) -> List[Tuple[str, str, str]]:
    """
    edge_keys: list of [subj, pred, obj] triples (as in v1)
    """
    if edge_keys is None or not isinstance(edge_keys, list):
        return []
    triples: List[Tuple[str, str, str]] = []
    for t in edge_keys:
        if isinstance(t, list) and len(t) == 3:
            s, p, o = t
            triples.append((str(s), str(p), str(o)))
    return triples


def load_v1_ndjson(path: Path) -> pd.DataFrame:
    """
    v1 file is NDJSON (one JSON object per line), e.g.:
      {"question_id":"Q1", ..., "edge_keys":[["S","P","O"]], "n_edges":1, ...}
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: invalid JSON at line {ln}: {e}") from e

            if not isinstance(rec, dict):
                continue

            qid = rec.get("question_id")
            triples = _triples_from_edge_keys(rec.get("edge_keys", []))

            pred_set = {p for (_, p, _) in triples if p}
            triple_set = {f"{s}\t{p}\t{o}" for (s, p, o) in triples if s and p and o}

            # prefer explicit n_edges, else derive from edge_keys
            n_edges = rec.get("n_edges")
            edge_count = int(n_edges) if isinstance(n_edges, (int, float)) else len(triples)

            rows.append({
                "question_id": str(qid) if qid is not None else None,
                "edge_count": edge_count,
                "pred_set": pred_set,
                "triple_set": triple_set,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"{path}: parsed 0 records")
    if "question_id" not in df.columns:
        raise ValueError(f"{path}: missing question_id")
    return df


def _parse_edges_json_field(s: Any) -> List[Dict[str, Any]]:
    """
    v2-v4 edges_json is a JSON string like:
      {"edges":[{"predicate":"...","subject":"...","object":"..."}, ...]}
    """
    if s is None:
        return []
    if isinstance(s, float) and math.isnan(s):
        return []
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return []

    obj = json.loads(s)  # fail loudly if malformed
    edges = obj.get("edges", [])
    if not isinstance(edges, list):
        return []
    return [e for e in edges if isinstance(e, dict)]


def load_v234_csv(path: Path, require_ok: bool = True) -> pd.DataFrame:
    """
    v2-v4 CSV columns:
      prompt_version,question_id,run_id,subject_curie,object_curie,edges_json,status,error
    """
    df = pd.read_csv(path, dtype=str)
    if "question_id" not in df.columns:
        raise ValueError(f"{path}: missing question_id column")
    if "edges_json" not in df.columns:
        raise ValueError(f"{path}: missing edges_json column")

    if require_ok and "status" in df.columns:
        df = df[df["status"].fillna("") == "ok"].copy()

    rows = []
    for _, r in df.iterrows():
        qid = r["question_id"]
        edges = _parse_edges_json_field(r["edges_json"])

        triples: List[Tuple[str, str, str]] = []
        for e in edges:
            p = e.get("predicate")
            s = e.get("subject")
            o = e.get("object")
            if p and s and o:
                triples.append((str(s), str(p), str(o)))

        pred_set = {p for (_, p, _) in triples if p}
        triple_set = {f"{s}\t{p}\t{o}" for (s, p, o) in triples if s and p and o}

        rows.append({
            "question_id": str(qid) if qid is not None else None,
            "edge_count": len(triples),
            "pred_set": pred_set,
            "triple_set": triple_set,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"{path}: parsed 0 records (maybe filtered by status?)")
    return out


# ============================================================
# Agreement TSV merge
# ============================================================

def load_agreement_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "question_id" not in df.columns:
        raise ValueError(f"{path}: missing question_id column")
    for c in df.columns:
        if c == "question_id":
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="v1 NDJSON file or v2-v4 CSV file.")
    ap.add_argument("--format", required=True, choices=["v1_ndjson", "v234_csv"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--agreement-tsv", default=None)
    ap.add_argument("--include-non-ok", action="store_true",
                    help="For CSV: include non-ok rows (default filters status==ok).")
    args = ap.parse_args()

    runs_path = Path(args.runs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.format == "v1_ndjson":
        df_runs = load_v1_ndjson(runs_path)
    else:
        df_runs = load_v234_csv(runs_path, require_ok=(not args.include_non_ok))

    df_per_q = per_question_metrics(df_runs)
    df_sum = stability_summary(df_per_q)

    per_q_path = outdir / f"{args.prefix}_stability_per_question.tsv"
    sum_path = outdir / f"{args.prefix}_stability_summary.tsv"
    df_per_q.to_csv(per_q_path, sep="\t", index=False)
    df_sum.to_csv(sum_path, sep="\t")

    print(f"[{args.prefix}] wrote: {per_q_path}")
    print(f"[{args.prefix}] wrote: {sum_path}")

    if args.agreement_tsv:
        df_agr = load_agreement_tsv(Path(args.agreement_tsv))
        merged = df_per_q.merge(df_agr, on="question_id", how="left")
        merged_path = outdir / f"{args.prefix}_stability_with_agreement.tsv"
        merged.to_csv(merged_path, sep="\t", index=False)
        print(f"[{args.prefix}] wrote: {merged_path}")


if __name__ == "__main__":
    main()