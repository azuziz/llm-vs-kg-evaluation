#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

Triple = Tuple[str, str, str]


def canonical_edge(curie1: str, predicate: str, curie2: str) -> Triple:
    a, b = sorted([curie1, curie2])
    return (a, predicate, b)


def safe_div(n: int, d: int) -> float:
    return 0.0 if d == 0 else n / d


def f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0.0 else (2.0 * p * r) / (p + r)


def detect_columns(fieldnames: List[str]) -> Dict[str, str]:
    """
    Robustly map a TSV's columns to required logical fields.
    We need question_id, subject, predicate, object.
    """
    f = set(fieldnames)

    # question id possibilities
    q_candidates = ["question_id", "qid", "question", "pair_id"]
    s_candidates = ["subject", "subj", "subject_curie", "subject_id", "source", "source_id"]
    o_candidates = ["object", "obj", "object_curie", "object_id", "target", "target_id"]
    p_candidates = ["predicate", "pred", "relation", "edge_predicate"]

    def pick(cands: List[str]) -> str:
        for c in cands:
            if c in f:
                return c
        raise ValueError(f"Could not find any of {cands} in columns: {fieldnames}")

    return {
        "question_id": pick(q_candidates),
        "subject": pick(s_candidates),
        "predicate": pick(p_candidates),
        "object": pick(o_candidates),
    }


def load_robokop_by_question(path: Path) -> Dict[str, Set[Triple]]:
    """
    Loads ROBOKOP triples and canonicalizes them to (min_id, predicate, max_id).
    Returns: question_id -> set(triples)
    """
    by_q: Dict[str, Set[Triple]] = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise ValueError("ROBOKOP TSV has no header row.")
        col = detect_columns(r.fieldnames)

        for row in r:
            qid = row[col["question_id"]].strip()
            s = row[col["subject"]].strip()
            p = row[col["predicate"]].strip()
            o = row[col["object"]].strip()
            if not qid or not s or not p or not o:
                continue
            by_q[qid].add(canonical_edge(s, p, o))

    return by_q


def iter_gpt_runs(jsonl_path: Path) -> Iterable[dict]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt_jsonl", required=True, type=Path)
    ap.add_argument("--robokop_tsv", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    args = ap.parse_args()

    robokop = load_robokop_by_question(args.robokop_tsv)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id", "run_id",
            "n_gpt", "n_robokop", "n_intersection",
            "jaccard", "precision_like", "recall_like", "f1"
        ])

        nrows = 0
        for rec in iter_gpt_runs(args.gpt_jsonl):
            qid = str(rec["question_id"])
            run_id = int(rec["run_id"])

            gpt_set: Set[Triple] = set(tuple(x) for x in rec.get("edge_keys", []))
            rob_set: Set[Triple] = robokop.get(qid, set())

            inter = gpt_set.intersection(rob_set)
            union = gpt_set.union(rob_set)

            n_gpt = len(gpt_set)
            n_rob = len(rob_set)
            n_i = len(inter)

            j = safe_div(len(inter), len(union))
            p = safe_div(n_i, n_gpt)     # |∩| / |GPT|
            r = safe_div(n_i, n_rob)     # |∩| / |ROBOKOP|
            f1v = f1(p, r)

            w.writerow([qid, run_id, n_gpt, n_rob, n_i, f"{j:.6f}", f"{p:.6f}", f"{r:.6f}", f"{f1v:.6f}"])
            nrows += 1

        print(f"Wrote {nrows} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
