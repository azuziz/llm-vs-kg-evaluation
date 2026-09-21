#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def norm(x: str) -> str:
    return (x or "").strip()


def load_pairs(pairs_tsv: Path) -> Dict[str, Tuple[str, str]]:
    q_to_pair: Dict[str, Tuple[str, str]] = {}
    with pairs_tsv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        required = {"question_id", "subject_curie", "object_curie"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise SystemExit(f"{pairs_tsv}: missing columns {sorted(missing)}")

        for row in r:
            q = norm(row.get("question_id", ""))
            s = norm(row.get("subject_curie", ""))
            o = norm(row.get("object_curie", ""))
            if q and s and o:
                q_to_pair[q] = (s, o)
    if not q_to_pair:
        raise SystemExit(f"{pairs_tsv}: no pairs loaded")
    return q_to_pair


def load_triples(triples_tsv: Path) -> Dict[str, List[Tuple[str, str]]]:
    by_q: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    with triples_tsv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        required = {"question_id", "subject", "object"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise SystemExit(f"{triples_tsv}: missing columns {sorted(missing)}")

        for row in r:
            q = norm(row.get("question_id", ""))
            s = norm(row.get("subject", ""))
            o = norm(row.get("object", ""))
            if q and s and o:
                by_q[q].append((s, o))
    if not by_q:
        raise SystemExit(f"{triples_tsv}: no triples loaded")
    return by_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples_filtered", required=True)
    ap.add_argument("--pairs_filtered", required=True)
    ap.add_argument("--out_report", required=True)
    ap.add_argument("--topk", type=int, default=5, help="Top-k node frequency to report")
    args = ap.parse_args()

    triples_by_q = load_triples(Path(args.triples_filtered))
    pairs_by_q = load_pairs(Path(args.pairs_filtered))

    out_fields = [
        "question_id",
        "pair_subject_curie",
        "pair_object_curie",
        "n_triples_rows",
        "n_unique_nodes_in_triples",
        "freq_subject",
        "freq_object",
        "direct_endpoint_edges",
        "subject_rank",
        "object_rank",
        "top_nodes",
        "flag_suspicious",
    ]

    out_path = Path(args.out_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as out_f:
        w = csv.DictWriter(out_f, fieldnames=out_fields, delimiter="\t")
        w.writeheader()

        for qid, edges in triples_by_q.items():
            if qid not in pairs_by_q:
                raise SystemExit(f"Pair file missing question_id {qid}")

            ps, po = pairs_by_q[qid]
            freq = Counter()
            nodes = set()

            direct = 0
            for s, o in edges:
                freq[s] += 1
                freq[o] += 1
                nodes.add(s)
                nodes.add(o)
                if (s == ps and o == po) or (s == po and o == ps):
                    direct += 1

            # ranking (1 = most frequent)
            ranked = [curie for curie, _ in freq.most_common()]
            rank_s = ranked.index(ps) + 1 if ps in freq else 10**9
            rank_o = ranked.index(po) + 1 if po in freq else 10**9

            top_nodes = ";".join([f"{c}:{n}" for c, n in freq.most_common(args.topk)])

            # Heuristic flags:
            # - direct endpoints never appear AND one or both nodes are not in top-10 by frequency
            suspicious = (direct == 0) and (rank_s > 10 or rank_o > 10)

            w.writerow(
                {
                    "question_id": qid,
                    "pair_subject_curie": ps,
                    "pair_object_curie": po,
                    "n_triples_rows": len(edges),
                    "n_unique_nodes_in_triples": len(nodes),
                    "freq_subject": freq.get(ps, 0),
                    "freq_object": freq.get(po, 0),
                    "direct_endpoint_edges": direct,
                    "subject_rank": rank_s if rank_s != 10**9 else "",
                    "object_rank": rank_o if rank_o != 10**9 else "",
                    "top_nodes": top_nodes,
                    "flag_suspicious": "YES" if suspicious else "",
                }
            )


if __name__ == "__main__":
    main()
