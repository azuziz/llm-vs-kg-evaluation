#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]

ROBOKOP = ROOT / "data" / "robokop_triples_clean.tsv"
GPT_EDGES = ROOT / "out" / "step2" / "gpt_edges_long.tsv"

OUT_PER_RUN = ROOT / "out" / "step3" / "per_run_metrics.csv"
OUT_PER_Q = ROOT / "out" / "step3" / "per_question_metrics.csv"


def load_robokop(path: Path) -> Dict[str, Set[str]]:
    """
    Loads robokop triples and returns by question_id a set of edge_keys:
      predicate|subject|object
    Expected columns in robokop_triples_clean.tsv include: question_id, predicate, subject, object
    """
    by_q: Dict[str, Set[str]] = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            qid = row["question_id"]
            pred = row["predicate"]
            subj = row["subject"]
            obj = row["object"]
            by_q[qid].add(f"{pred}|{subj}|{obj}")
    return by_q


def load_gpt_edges(path: Path) -> Dict[Tuple[str, str, str], Set[str]]:
    """
    Returns mapping (prompt_version, question_id, run_id) -> set(edge_key)
    """
    m: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            key = (row["prompt_version"], row["question_id"], row["run_id"])
            m[key].add(row["edge_key"])
    return m


def metrics(gpt: Set[str], rob: Set[str]) -> Tuple[int, int, int, float, float, float, float]:
    n_g = len(gpt)
    n_r = len(rob)
    inter = len(gpt & rob)
    union = len(gpt | rob)

    jacc = (inter / union) if union else 0.0
    prec = (inter / n_g) if n_g else 0.0
    rec = (inter / n_r) if n_r else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return n_g, n_r, inter, jacc, prec, rec, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robokop", type=Path, default=ROBOKOP)
    ap.add_argument("--gpt_edges", type=Path, default=GPT_EDGES)
    ap.add_argument("--out_per_run", type=Path, default=OUT_PER_RUN)
    ap.add_argument("--out_per_question", type=Path, default=OUT_PER_Q)
    args = ap.parse_args()

    args.out_per_run.parent.mkdir(parents=True, exist_ok=True)

    rob = load_robokop(args.robokop)
    gpt = load_gpt_edges(args.gpt_edges)

    # per-run
    per_run_rows: List[List] = []
    for (pv, qid, rid), gset in gpt.items():
        rset = rob.get(qid, set())
        n_g, n_r, inter, jacc, prec, rec, f1 = metrics(gset, rset)
        per_run_rows.append([pv, qid, rid, n_g, n_r, inter, jacc, prec, rec, f1])

    with open(args.out_per_run, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_version", "question_id", "run_id", "n_gpt", "n_robokop", "n_intersection",
                    "jaccard", "precision_like", "recall_like", "f1"])
        w.writerows(per_run_rows)

    # per-question aggregated (mean over runs)
    by_pq = defaultdict(list)
    for pv, qid, rid, n_g, n_r, inter, jacc, prec, rec, f1 in per_run_rows:
        by_pq[(pv, qid)].append((n_g, inter, jacc, prec, rec, f1, n_r))

    with open(args.out_per_question, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_version", "question_id",
                    "n_runs", "n_robokop",
                    "mean_n_gpt", "mean_intersection",
                    "mean_jaccard", "mean_precision_like", "mean_recall_like", "mean_f1"])
        for (pv, qid), vals in sorted(by_pq.items()):
            n_runs = len(vals)
            n_robokop = vals[0][6] if vals else 0
            mean_n_gpt = sum(v[0] for v in vals) / n_runs
            mean_inter = sum(v[1] for v in vals) / n_runs
            mean_j = sum(v[2] for v in vals) / n_runs
            mean_p = sum(v[3] for v in vals) / n_runs
            mean_r = sum(v[4] for v in vals) / n_runs
            mean_f1 = sum(v[5] for v in vals) / n_runs
            w.writerow([pv, qid, n_runs, n_robokop, mean_n_gpt, mean_inter, mean_j, mean_p, mean_r, mean_f1])

    print(f"[OK] wrote {args.out_per_run}")
    print(f"[OK] wrote {args.out_per_question}")


if __name__ == "__main__":
    main()
