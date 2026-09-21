#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


Edge = Tuple[str, str, str]  # (subject, predicate, object)


def load_inverse_map(path: Optional[Path]) -> Dict[str, str]:
    """
    Load a TSV with columns: predicate, inverse_predicate
    Returns a dict mapping predicate -> inverse_predicate, and also adds reverse entries.
    """
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"inverse map not found: {path}")

    df = pd.read_csv(path, sep="\t")
    required = {"predicate", "inverse_predicate"}
    if not required.issubset(df.columns):
        raise ValueError(f"inverse map must have columns {required}, got {set(df.columns)}")

    m: Dict[str, str] = {}
    for _, r in df.iterrows():
        p = str(r["predicate"]).strip()
        inv = str(r["inverse_predicate"]).strip()
        if p and inv:
            m[p] = inv
            m[inv] = p  # ensure symmetric lookup
    return m


def load_robokop_edges(path: Path) -> Dict[str, Set[Edge]]:
    """
    Load ROBOKOP triples grouped by question_id.
    Expected columns (flexible names):
      - question_id (or question)
      - subject_curie (or subject)
      - predicate
      - object_curie (or object)
    """
    df = pd.read_csv(path, sep="\t")
    # flexible column mapping
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"question_id", "question"}:
            colmap["question_id"] = c
        elif lc in {"subject_curie", "subject"}:
            colmap["subject"] = c
        elif lc == "predicate":
            colmap["predicate"] = c
        elif lc in {"object_curie", "object"}:
            colmap["object"] = c

    missing = {"question_id", "subject", "predicate", "object"} - set(colmap.keys())
    if missing:
        raise ValueError(
            f"ROBOKOP TSV missing columns {missing}. Found columns: {list(df.columns)}"
        )

    out: Dict[str, Set[Edge]] = {}
    for _, r in df.iterrows():
        q = str(r[colmap["question_id"]]).strip()
        s = str(r[colmap["subject"]]).strip()
        p = str(r[colmap["predicate"]]).strip()
        o = str(r[colmap["object"]]).strip()
        out.setdefault(q, set()).add((s, p, o))
    return out


def parse_edges_json(edges_json: str) -> List[Edge]:
    """
    Parse the edges_json column. Expected like {"edges":[{"predicate":..., "subject":..., "object":...}, ...]}
    """
    if not isinstance(edges_json, str) or not edges_json:
        return []
    try:
        obj = json.loads(edges_json)
    except json.JSONDecodeError:
        return []
    edges = obj.get("edges", [])
    out: List[Edge] = []
    if isinstance(edges, list):
        for e in edges:
            if not isinstance(e, dict):
                continue
            p = str(e.get("predicate", "")).strip()
            s = str(e.get("subject", "")).strip()
            o = str(e.get("object", "")).strip()
            if p and s and o:
                out.append((s, p, o))
    return out


def normalize_edges(
    edges: Iterable[Edge],
    inverse_map: Dict[str, str],
    allow_inverse: bool,
) -> Set[Edge]:
    """
    Normalize into a set. If allow_inverse is True, we will store both the edge and its inverse form
    (swapping subject/object and mapping predicate->inverse predicate when available).
    This makes intersection robust to direction differences.
    """
    out: Set[Edge] = set()
    for s, p, o in edges:
        out.add((s, p, o))
        if allow_inverse:
            invp = inverse_map.get(p)
            if invp:
                out.add((o, invp, s))
    return out


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    jaccard: float
    n_pred: int
    n_ref: int
    n_inter: int


def compute_metrics(pred: Set[Edge], ref: Set[Edge]) -> Metrics:
    inter = pred.intersection(ref)
    n_pred = len(pred)
    n_ref = len(ref)
    n_inter = len(inter)

    precision = (n_inter / n_pred) if n_pred > 0 else 0.0
    recall = (n_inter / n_ref) if n_ref > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    union = pred.union(ref)
    jaccard = (n_inter / len(union)) if len(union) > 0 else 0.0

    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        n_pred=n_pred,
        n_ref=n_ref,
        n_inter=n_inter,
    )


def summarize_per_question(run_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize metrics per question_id: mean/median, p25/p75/IQR, and zero-f1 rate.
    """
    def q25(x): return x.quantile(0.25)
    def q75(x): return x.quantile(0.75)

    grouped = run_df.groupby(["prompt_version", "question_id"], as_index=False)

    summary = grouped.agg(
        mean_f1=("f1", "mean"),
        median_f1=("f1", "median"),
        p25_f1=("f1", q25),
        p75_f1=("f1", q75),
        mean_jaccard=("jaccard", "mean"),
        median_jaccard=("jaccard", "median"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_n_pred=("n_pred", "mean"),
        mean_n_ref=("n_ref", "mean"),
    )
    summary["iqr_f1"] = summary["p75_f1"] - summary["p25_f1"]

    # zero-f1 rate (fraction of runs with f1 == 0)
    z = run_df.assign(zero_f1=(run_df["f1"] == 0.0).astype(int))
    zrate = z.groupby(["prompt_version", "question_id"], as_index=False).agg(
        zero_f1_rate=("zero_f1", "mean")
    )
    summary = summary.merge(zrate, on=["prompt_version", "question_id"], how="left")
    return summary


def summarize_per_prompt(run_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across questions/runs per prompt.
    """
    grouped = run_df.groupby("prompt_version", as_index=False)
    out = grouped.agg(
        mean_f1=("f1", "mean"),
        median_f1=("f1", "median"),
        mean_jaccard=("jaccard", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        zero_f1_rate=("f1", lambda x: float((x == 0.0).mean())),
        mean_n_pred=("n_pred", "mean"),
        mean_n_ref=("n_ref", "mean"),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="~/gpt/12_stats")
    ap.add_argument("--robokop", type=Path, required=True, help="robokop_triples.tsv")
    ap.add_argument("--prompts", nargs="+", required=True, help="e.g. v2 v3 v4 (and/or v1)")
    ap.add_argument("--inverse_map", type=Path, default=None, help="inverse_predicates.tsv (optional)")
    ap.add_argument(
        "--allow_inverse",
        action="store_true",
        help="If set, allow inverse predicate matching via inverse_map.",
    )
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    inv_map = load_inverse_map(args.inverse_map)
    ref_by_q = load_robokop_edges(args.robokop)

    rows = []

    for pv in args.prompts:
        runs_path = args.root / pv / "gpt_runs.csv"
        if not runs_path.exists():
            raise FileNotFoundError(f"missing: {runs_path}")

        df = pd.read_csv(runs_path)
        required = {"prompt_version", "question_id", "run_id", "edges_json"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{runs_path} missing columns {missing}. Found: {list(df.columns)}")

        # only successful
        if "status" in df.columns:
            df = df[df["status"] == "ok"].copy()

        for _, r in df.iterrows():
            qid = str(r["question_id"]).strip()
            run_id = int(r["run_id"])
            pv_row = str(r["prompt_version"]).strip()

            pred_edges = parse_edges_json(r["edges_json"])
            pred_set = normalize_edges(pred_edges, inv_map, allow_inverse=args.allow_inverse)

            ref_set = ref_by_q.get(qid, set())
            # also normalize reference if inverse matching is enabled
            ref_set_norm = normalize_edges(ref_set, inv_map, allow_inverse=args.allow_inverse)

            m = compute_metrics(pred_set, ref_set_norm)
            rows.append(
                {
                    "prompt_version": pv_row,
                    "question_id": qid,
                    "run_id": run_id,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "jaccard": m.jaccard,
                    "n_pred": m.n_pred,
                    "n_ref": m.n_ref,
                    "n_inter": m.n_inter,
                }
            )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    run_metrics = pd.DataFrame(rows).sort_values(["prompt_version", "question_id", "run_id"])
    run_metrics.to_csv(outdir / "run_metrics.csv", index=False)

    per_q = summarize_per_question(run_metrics)
    per_q.to_csv(outdir / "per_question_summary.csv", index=False)

    per_prompt = summarize_per_prompt(run_metrics)
    per_prompt.to_csv(outdir / "per_prompt_summary.csv", index=False)

    print(f"[OK] wrote {outdir/'run_metrics.csv'}")
    print(f"[OK] wrote {outdir/'per_question_summary.csv'}")
    print(f"[OK] wrote {outdir/'per_prompt_summary.csv'}")


if __name__ == "__main__":
    main()
