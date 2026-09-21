#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Set, Tuple


PAIRS_FIELDS = [
    "question_id",
    "subject_text",
    "subject_curie",
    "subject_label",
    "object_text",
    "object_curie",
    "object_label",
    "notes",
]

TRIPLES_QID_FIELD = "question_id"


def read_filtered_question_ids(filtered_triples_path: Path) -> List[str]:
    """
    Return question_ids in the order they first appear in robokop_triples_filtered.tsv.
    """
    qids: List[str] = []
    seen: Set[str] = set()

    with filtered_triples_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames or TRIPLES_QID_FIELD not in r.fieldnames:
            raise SystemExit(f"{filtered_triples_path}: missing '{TRIPLES_QID_FIELD}' column")

        for row in r:
            qid = (row.get(TRIPLES_QID_FIELD) or "").strip()
            if not qid:
                continue
            if qid not in seen:
                qids.append(qid)
                seen.add(qid)

    return qids


def iter_pairs_files(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            files.append(Path(p))

    # de-duplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for fp in files:
        if fp not in seen:
            uniq.append(fp)
            seen.add(fp)
    return uniq


def build_pairs_index(pairs_files: List[Path]) -> Dict[str, Dict[str, str]]:
    """
    Map question_id -> full pairs row dict (only the PAIRS_FIELDS).
    Raises if a question_id appears more than once with conflicting content.
    """
    idx: Dict[str, Dict[str, str]] = {}

    for fp in pairs_files:
        with fp.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            if not r.fieldnames:
                continue
            missing = [c for c in PAIRS_FIELDS if c not in r.fieldnames]
            if missing:
                raise SystemExit(f"{fp}: missing required columns: {missing}")

            for row in r:
                qid = (row.get("question_id") or "").strip()
                if not qid:
                    continue
                normalized = {k: (row.get(k) or "") for k in PAIRS_FIELDS}

                if qid not in idx:
                    idx[qid] = normalized
                else:
                    # sanity: if duplicated qid exists, it must be identical
                    if idx[qid] != normalized:
                        raise SystemExit(
                            f"Conflicting duplicate question_id '{qid}' found.\n"
                            f"First:  {idx[qid]}\n"
                            f"Second: {normalized}\n"
                            f"File: {fp}"
                        )

    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--filtered_triples",
        required=True,
        help="Path to robokop_triples_filtered.tsv",
    )
    ap.add_argument(
        "--pairs_inputs",
        nargs="+",
        required=True,
        help='One or more globs for original node-pair TSVs, e.g. "robokop_100_pairs*.tsv"',
    )
    ap.add_argument(
        "--out_pairs",
        required=True,
        help="Output TSV with node pairs corresponding to filtered triples",
    )
    ap.add_argument(
        "--require_exact_count",
        type=int,
        default=0,
        help="If >0, enforce exactly this many unique question_ids (sanity check). Example: 115",
    )
    args = ap.parse_args()

    filtered_triples_path = Path(args.filtered_triples)
    out_pairs_path = Path(args.out_pairs)

    qids_in_filtered = read_filtered_question_ids(filtered_triples_path)
    if args.require_exact_count and len(qids_in_filtered) != args.require_exact_count:
        raise SystemExit(
            f"Filtered triples contain {len(qids_in_filtered)} unique question_ids, "
            f"expected {args.require_exact_count}."
        )

    pairs_files = iter_pairs_files(args.pairs_inputs)
    if not pairs_files:
        raise SystemExit(f"No pairs files matched: {args.pairs_inputs}")

    pairs_idx = build_pairs_index(pairs_files)

    missing = [qid for qid in qids_in_filtered if qid not in pairs_idx]
    if missing:
        # Fail loudly: mismatch would break your downstream alignment
        raise SystemExit(
            f"ERROR: {len(missing)} question_ids from filtered triples were not found in the pairs files.\n"
            f"Examples: {missing[:20]}"
        )

    # Write output pairs in the same question_id order as filtered triples.
    out_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with out_pairs_path.open("w", newline="", encoding="utf-8") as out_f:
        w = csv.DictWriter(out_f, fieldnames=PAIRS_FIELDS, delimiter="\t")
        w.writeheader()
        for qid in qids_in_filtered:
            w.writerow(pairs_idx[qid])

    # Final sanity check: number of output rows == number of unique filtered qids
    # (and therefore should be 115 in your case)
    print(f"Wrote {len(qids_in_filtered)} node pairs to: {out_pairs_path}")


if __name__ == "__main__":
    main()
