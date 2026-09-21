#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


PAIR_FIELDS = [
    "question_id",
    "subject_text",
    "subject_curie",
    "subject_label",
    "object_text",
    "object_curie",
    "object_label",
    "notes",
]


def qid_sort_key(qid: str) -> Tuple[int, str]:
    """
    Sort QIDs like Q1, Q2, ..., Q10 numerically when possible.
    Falls back to lexicographic.
    """
    m = re.match(r"^[Qq](\d+)$", (qid or "").strip())
    if m:
        return (int(m.group(1)), qid)
    return (10**9, qid)


def count_edges_by_question(triples_path: Path) -> Dict[str, int]:
    """
    Counts rows per question_id in a robokop_triples*.tsv file.
    Each row is treated as an edge instance (as written by your collector).
    """
    counts: Dict[str, int] = {}
    with triples_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames or "question_id" not in r.fieldnames:
            raise SystemExit(f"{triples_path}: missing header or question_id column")
        for row in r:
            qid = (row.get("question_id") or "").strip()
            if not qid:
                continue
            counts[qid] = counts.get(qid, 0) + 1
    return counts


def load_pairs_by_question(pairs_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Loads the original pairs file as a mapping: question_id -> row dict.
    """
    out: Dict[str, Dict[str, str]] = {}
    with pairs_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise SystemExit(f"{pairs_path}: missing header")
        missing = [c for c in PAIR_FIELDS if c not in r.fieldnames]
        if missing:
            raise SystemExit(f"{pairs_path}: missing required columns: {missing}")
        for row in r:
            qid = (row.get("question_id") or "").strip()
            if not qid:
                continue
            # keep only expected columns
            out[qid] = {k: row.get(k, "") for k in PAIR_FIELDS}
    return out


def parse_batch_args(batch_args: List[str]) -> List[Tuple[Path, Path]]:
    """
    Expects repeated pairs of arguments:
      --batch TRIPLES.tsv PAIRS.tsv
    """
    if len(batch_args) % 2 != 0:
        raise SystemExit("Each --batch must supply exactly two paths: TRIPLES.tsv PAIRS.tsv")
    batches: List[Tuple[Path, Path]] = []
    for i in range(0, len(batch_args), 2):
        batches.append((Path(batch_args[i]), Path(batch_args[i + 1])))
    return batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--batch",
        nargs=2,
        action="append",
        metavar=("TRIPLES_TSV", "PAIRS_TSV"),
        required=True,
        help="Provide a triples TSV and its corresponding original pairs TSV. Repeat for each batch.",
    )
    ap.add_argument("--out_pairs_tsv", required=True, help="Output filtered pairs TSV")
    ap.add_argument("--min_edges", type=int, default=1)
    ap.add_argument("--max_edges", type=int, default=10)
    args = ap.parse_args()

    # args.batch is a list of [ [triples, pairs], [triples, pairs], ... ]
    batches: List[Tuple[Path, Path]] = [(Path(t), Path(p)) for t, p in args.batch]

    kept_rows: List[Dict[str, str]] = []

    for triples_path, pairs_path in batches:
        if not triples_path.exists():
            raise SystemExit(f"Triples file not found: {triples_path}")
        if not pairs_path.exists():
            raise SystemExit(f"Pairs file not found: {pairs_path}")

        edge_counts = count_edges_by_question(triples_path)
        pairs_by_qid = load_pairs_by_question(pairs_path)

        # Filter qids by edge count
        kept_qids = [
            qid for qid, n in edge_counts.items()
            if args.min_edges <= n <= args.max_edges
        ]
        kept_qids.sort(key=qid_sort_key)

        # Pull those rows from the matching pairs file
        for qid in kept_qids:
            if qid not in pairs_by_qid:
                raise SystemExit(
                    f"Batch mismatch: {triples_path} contains {qid} "
                    f"but {pairs_path} does not."
                )
            kept_rows.append(dict(pairs_by_qid[qid]))

    # Renumber question_id sequentially in final output
    for i, row in enumerate(kept_rows, start=1):
        row["question_id"] = f"Q{i}"

    out_path = Path(args.out_pairs_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        w = csv.DictWriter(out_f, fieldnames=PAIR_FIELDS, delimiter="\t")
        w.writeheader()
        for row in kept_rows:
            w.writerow({k: row.get(k, "") for k in PAIR_FIELDS})


if __name__ == "__main__":
    main()
