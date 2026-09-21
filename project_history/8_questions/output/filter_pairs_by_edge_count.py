#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple


OUT_FIELDS = [
    "question_id",
    "query_edge_key",
    "edge_id",
    "subject",
    "predicate",
    "object",
    "subject_label",
    "object_label",
    "provenance",
]


def iter_input_files(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            files.append(Path(p))
    # de-duplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help='One or more input globs, e.g. "/8_questions/output/robokop_triples*.tsv"',
    )
    ap.add_argument(
        "--out_tsv",
        required=True,
        help='Output TSV, e.g. "/8_questions/output/robokop_triples_filtered.tsv"',
    )
    ap.add_argument("--min_edges", type=int, default=1, help="Minimum edges per question (inclusive). Default=1")
    ap.add_argument("--max_edges", type=int, default=10, help="Maximum edges per question (inclusive). Default=10")
    args = ap.parse_args()

    in_files = iter_input_files(args.inputs)
    if not in_files:
        raise SystemExit(f"No input files matched: {args.inputs}")

    # Preserve question appearance order across files
    # old_qid -> list of rows (each row is dict)
    rows_by_qid: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()

    for fp in in_files:
        with fp.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            if not r.fieldnames:
                continue

            missing = [c for c in OUT_FIELDS if c not in r.fieldnames]
            if missing:
                raise SystemExit(f"{fp}: missing required columns: {missing}")

            for row in r:
                old_qid = (row.get("question_id") or "").strip()
                if not old_qid:
                    continue
                if old_qid not in rows_by_qid:
                    rows_by_qid[old_qid] = []
                # Keep only the required columns, in case inputs have extras
                rows_by_qid[old_qid].append({k: row.get(k, "") for k in OUT_FIELDS})

    # Filter by edge count per question_id
    kept_old_qids: List[str] = []
    for old_qid, qrows in rows_by_qid.items():
        n_edges = len(qrows)
        if args.min_edges <= n_edges <= args.max_edges:
            kept_old_qids.append(old_qid)

    # Reassign new question IDs in order of appearance among kept questions
    old_to_new: Dict[str, str] = {}
    for i, old_qid in enumerate(kept_old_qids, start=1):
        old_to_new[old_qid] = f"Q{i}"

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        w = csv.DictWriter(out_f, fieldnames=OUT_FIELDS, delimiter="\t")
        w.writeheader()

        for old_qid in kept_old_qids:
            new_qid = old_to_new[old_qid]
            for row in rows_by_qid[old_qid]:
                row_out = dict(row)
                row_out["question_id"] = new_qid
                w.writerow(row_out)


if __name__ == "__main__":
    main()
