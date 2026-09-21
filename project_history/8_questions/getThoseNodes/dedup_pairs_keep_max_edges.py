#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List


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


RE_EDGES = re.compile(r"(?:^|[|;,\s])ROBOKOP_EDGES=(\d+)(?:$|[|;,\s])")


def norm(x: str) -> str:
    return (x or "").strip()


def undirected_key(subj_curie: str, obj_curie: str) -> Tuple[str, str]:
    a, b = sorted([norm(subj_curie), norm(obj_curie)])
    return (a, b)


def parse_edges(notes: str) -> Optional[int]:
    """
    Extract ROBOKOP_EDGES=N from notes, if present.
    Returns None if not found.
    """
    m = RE_EDGES.search(notes or "")
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pairs_tsv", required=True, help="Input pairs TSV (e.g., robokop_pairs_filtered.tsv)")
    ap.add_argument("--out_pairs_tsv", required=True, help="Output deduplicated pairs TSV")
    args = ap.parse_args()

    in_path = Path(args.in_pairs_tsv)
    out_path = Path(args.out_pairs_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # key -> (best_score, first_seen_index, row)
    best: Dict[Tuple[str, str], Tuple[int, int, Dict[str, str]]] = {}

    rows_in_order: List[Dict[str, str]] = []

    with in_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise SystemExit(f"{in_path}: missing header")

        missing = [c for c in PAIR_FIELDS if c not in r.fieldnames]
        if missing:
            raise SystemExit(f"{in_path}: missing required columns: {missing}")

        for i, row in enumerate(r):
            row2 = {k: row.get(k, "") for k in PAIR_FIELDS}
            rows_in_order.append(row2)

    for i, row in enumerate(rows_in_order):
        s = norm(row["subject_curie"])
        o = norm(row["object_curie"])
        if not s or not o:
            continue

        key = undirected_key(s, o)

        # score = ROBOKOP_EDGES, default to -1 if missing (so any present wins)
        n_edges = parse_edges(row.get("notes", ""))  # type: ignore[arg-type]
        score = n_edges if n_edges is not None else -1

        if key not in best:
            best[key] = (score, i, row)
        else:
            cur_score, cur_i, _ = best[key]
            # keep max score; tie-break: earliest occurrence
            if score > cur_score or (score == cur_score and i < cur_i):
                best[key] = (score, i, row)

    # Preserve a stable output ordering: by first occurrence of the kept representative
    kept = sorted(best.values(), key=lambda t: t[1])  # sort by first_seen_index

    # Renumber question_id
    out_rows: List[Dict[str, str]] = []
    for j, (_, _, row) in enumerate(kept, start=1):
        row_out = dict(row)
        row_out["question_id"] = f"Q{j}"
        out_rows.append(row_out)

    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        w = csv.DictWriter(out_f, fieldnames=PAIR_FIELDS, delimiter="\t")
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row.get(k, "") for k in PAIR_FIELDS})


if __name__ == "__main__":
    main()
