#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any


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


def norm(x: str) -> str:
    return (x or "").strip()


def iter_files(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            files.append(Path(p))
    seen = set()
    uniq: List[Path] = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def load_original_pairs(pairs_files: List[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for fp in pairs_files:
        with fp.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            if not r.fieldnames:
                continue
            missing = [c for c in PAIR_FIELDS if c not in r.fieldnames]
            if missing:
                raise SystemExit(f"{fp}: missing required columns: {missing}")
            for row in r:
                s = norm(row.get("subject_curie", ""))
                o = norm(row.get("object_curie", ""))
                if not s or not o:
                    continue
                rows.append({k: row.get(k, "") for k in PAIR_FIELDS})
    return rows


def read_triples_grouped(triples_path: Path) -> "OrderedDict[str, List[Tuple[str, str]]]":
    """
    Returns ordered mapping: qid -> list of (subject_curie, object_curie) edges from triples file.
    """
    by_q: "OrderedDict[str, List[Tuple[str, str]]]" = OrderedDict()

    with triples_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise SystemExit(f"{triples_path}: empty or missing header")
        for col in ["question_id", "subject", "object"]:
            if col not in r.fieldnames:
                raise SystemExit(f"{triples_path}: missing required column: {col}")

        for row in r:
            qid = norm(row.get("question_id", ""))
            s = norm(row.get("subject", ""))
            o = norm(row.get("object", ""))
            if not qid or not s or not o:
                continue
            if qid not in by_q:
                by_q[qid] = []
            by_q[qid].append((s, o))

    if not by_q:
        raise SystemExit(f"{triples_path}: no usable rows found")
    return by_q


def score_pair(freq: Counter, a: str, b: str) -> int:
    return int(freq.get(a, 0) + freq.get(b, 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples_filtered", required=True, help="robokop_triples_filtered.tsv")
    ap.add_argument("--pairs_inputs", nargs="+", required=True, help='e.g. "robokop_100_pairs*.tsv"')
    ap.add_argument("--out_pairs_tsv", required=True, help="output recovered pairs TSV")
    ap.add_argument(
        "--topk_debug",
        type=int,
        default=0,
        help="If >0, print the top-k most frequent CURIEs per question to stderr for debugging.",
    )
    args = ap.parse_args()

    triples_path = Path(args.triples_filtered)
    pairs_files = iter_files(args.pairs_inputs)
    if not pairs_files:
        raise SystemExit(f"No pairs files matched: {args.pairs_inputs}")

    original_pairs = load_original_pairs(pairs_files)
    triples_by_q = read_triples_grouped(triples_path)

    # Build a quick lookup from undirected key -> candidate original rows (usually 1)
    idx: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in original_pairs:
        s = norm(row["subject_curie"])
        o = norm(row["object_curie"])
        a, b = sorted([s, o])
        idx[(a, b)].append(row)

    recovered: List[Dict[str, str]] = []
    failures: List[str] = []

    for qid, edges in triples_by_q.items():
        # frequency of CURIEs in subjects/objects
        freq: Counter = Counter()
        node_set = set()
        for s, o in edges:
            freq[s] += 1
            freq[o] += 1
            node_set.add(s)
            node_set.add(o)

        if args.topk_debug and args.topk_debug > 0:
            topk = freq.most_common(args.topk_debug)
            print(f"[{qid}] top CURIEs: {topk}", file=sys.stderr)  # noqa: F821

        # Candidate original pairs are those whose two CURIEs both appear somewhere in this question's KG edges
        candidates: List[Tuple[int, Dict[str, str]]] = []
        for (a, b), rows in idx.items():
            if a in node_set and b in node_set:
                # choose the first row as representative if duplicates are identical
                rep = rows[0]
                # score by frequency
                s = score_pair(freq, a, b)
                candidates.append((s, rep))

        if not candidates:
            failures.append(f"{qid}: no original pair found whose CURIEs both appear in this question")
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score = candidates[0][0]
        best = [r for sc, r in candidates if sc == best_score]

        if len(best) != 1:
            # still ambiguous: try tie-break using exact orientation appearance (optional heuristic)
            # count how often (subject_curie, object_curie) appears as an oriented edge endpoints
            def oriented_bonus(row: Dict[str, str]) -> int:
                s0 = norm(row["subject_curie"])
                o0 = norm(row["object_curie"])
                bonus = 0
                for s, o in edges:
                    if s == s0 and o == o0:
                        bonus += 1
                return bonus

            best2 = sorted(best, key=oriented_bonus, reverse=True)
            if oriented_bonus(best2[0]) > oriented_bonus(best2[1]):
                best = [best2[0]]
            else:
                # cannot resolve uniquely
                ex = [(norm(r["subject_curie"]), norm(r["object_curie"])) for r in best[:10]]
                failures.append(
                    f"{qid}: ambiguous best match (score={best_score}), candidates (up to 10): {ex}"
                )
                continue

        row_out = dict(best[0])
        row_out["question_id"] = qid  # enforce same qid as filtered triples
        recovered.append(row_out)

    if failures:
        raise SystemExit("Failed to recover pairs for some questions:\n" + "\n".join("  " + x for x in failures))

    # Sanity check: one pair per question_id
    if len(recovered) != len(triples_by_q):
        raise SystemExit(
            f"Sanity check failed: recovered={len(recovered)} != questions_in_triples={len(triples_by_q)}"
        )

    out_path = Path(args.out_pairs_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        w = csv.DictWriter(out_f, fieldnames=PAIR_FIELDS, delimiter="\t")
        w.writeheader()
        for row in recovered:
            w.writerow({k: row.get(k, "") for k in PAIR_FIELDS})


if __name__ == "__main__":
    import sys
    main()
