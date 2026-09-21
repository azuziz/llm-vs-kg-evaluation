#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_tsv", required=True, type=Path,
                    help="Pairs TSV with columns: question_id, subject_curie, object_curie")
    ap.add_argument("--robokop_triples_tsv", required=True, type=Path,
                    help="ROBOKOP triples TSV with columns for subject/object + predicate")
    ap.add_argument("--out_csv", required=True, type=Path)
    args = ap.parse_args()

    # Load pairs: question_id -> (a,b) canonical
    q_to_pair = {}
    with args.pairs_tsv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        need = {"question_id", "subject_curie", "object_curie"}
        if not need.issubset(set(r.fieldnames or [])):
            raise ValueError(f"pairs_tsv must have columns {need}, got {r.fieldnames}")
        for row in r:
            qid = row["question_id"].strip()
            a = row["subject_curie"].strip()
            b = row["object_curie"].strip()
            if not qid or not a or not b:
                continue
            pair = tuple(sorted((a, b)))
            q_to_pair[qid] = pair

    # Determine subject/object columns in triples TSV
    with args.robokop_triples_tsv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        cols = [c.lower() for c in (r.fieldnames or [])]

    def find_col(name_candidates):
        for cand in name_candidates:
            if cand in cols:
                return (r.fieldnames or [])[cols.index(cand)]
        return None

    subj_col = find_col(["subject", "subject_curie", "subject_id"])
    obj_col  = find_col(["object", "object_curie", "object_id"])
    pred_col = find_col(["predicate", "biolink_predicate", "predicate_id", "edge_predicate"])

    if not subj_col or not obj_col:
        raise ValueError(f"Could not detect subject/object columns in {args.robokop_triples_tsv}")

    # Count unique triples per canonical pair
    pair_to_triples = defaultdict(set)

    with args.robokop_triples_tsv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = (row.get(subj_col) or "").strip()
            o = (row.get(obj_col) or "").strip()
            if not s or not o:
                continue
            pair = tuple(sorted((s, o)))
            p = (row.get(pred_col) or "").strip() if pred_col else ""
            # store unique triple; if no predicate column detected, store just endpoints
            key = (s, p, o) if pred_col else (s, o)
            pair_to_triples[pair].add(key)

    rows = []
    for qid, pair in sorted(q_to_pair.items(), key=lambda x: x[0]):
        rows.append({
            "question_id": qid,
            "robokop_n_edges": len(pair_to_triples.get(pair, set()))
        })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_id", "robokop_n_edges"])
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
