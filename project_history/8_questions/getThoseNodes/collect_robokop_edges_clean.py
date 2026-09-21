#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DEFAULT_ENDPOINT = "https://robokop-automat.apps.renci.org/robokopkg/query"

PAIR_REQUIRED = [
    "question_id",
    "subject_curie",
    "subject_label",
    "object_curie",
    "object_label",
]


TRIPLES_OUT_FIELDS = [
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


def build_one_hop_trapi_query(subject_curie: str, object_curie: str) -> Dict[str, Any]:
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": [subject_curie]},
                    "n1": {"ids": [object_curie]},
                },
                "edges": {
                    "e0": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": None,
                    }
                },
            }
        }
    }


def extract_provenance(edge: Dict[str, Any]) -> str:
    parts: List[str] = []
    sources = edge.get("sources")
    if isinstance(sources, list):
        for s in sources:
            rid = s.get("resource_id")
            role = s.get("resource_role")
            if rid and role:
                parts.append(f"{rid}({role})")
            elif rid:
                parts.append(str(rid))
        if parts:
            return ";".join(parts)

    attrs = edge.get("attributes")
    if isinstance(attrs, list):
        for a in attrs:
            atype = a.get("attribute_type_id")
            val = a.get("value")
            if atype and "knowledge_source" in str(atype):
                if isinstance(val, str):
                    parts.append(val)
                elif isinstance(val, list):
                    for v in val:
                        if isinstance(v, dict) and v.get("resource_id"):
                            rid = v.get("resource_id")
                            role = v.get("resource_role")
                            parts.append(f"{rid}({role})" if role else rid)

    return ";".join(parts)


def post_trapi(
    session: requests.Session,
    endpoint: str,
    payload: Dict[str, Any],
    timeout: int,
    retries: int,
    backoff_sec: float,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = session.post(endpoint, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff_sec * attempt)
            else:
                raise
    assert last_err is not None
    raise last_err


def read_pairs(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise SystemExit(f"{path}: missing header")
        missing = [c for c in PAIR_REQUIRED if c not in r.fieldnames]
        if missing:
            raise SystemExit(f"{path}: missing required columns: {missing}")
        for row in r:
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_tsv", required=True, help="Deduplicated pairs TSV (163 rows)")
    ap.add_argument("--out_triples_tsv", required=True, help="Output triples TSV")
    ap.add_argument("--out_counts_tsv", required=True, help="Output edge counts TSV")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--sleep", type=float, default=0.1, help="Sleep between calls (seconds)")
    ap.add_argument(
        "--enforce_range",
        action="store_true",
        help="If set, only write triples for questions whose current edge count is 1-10; others are skipped (still logged in counts).",
    )
    ap.add_argument("--min_edges", type=int, default=1)
    ap.add_argument("--max_edges", type=int, default=10)
    args = ap.parse_args()

    pairs_path = Path(args.pairs_tsv)
    pairs = read_pairs(pairs_path)

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    out_triples = Path(args.out_triples_tsv)
    out_counts = Path(args.out_counts_tsv)
    out_triples.parent.mkdir(parents=True, exist_ok=True)
    out_counts.parent.mkdir(parents=True, exist_ok=True)

    with out_triples.open("w", newline="", encoding="utf-8") as tf, out_counts.open("w", newline="", encoding="utf-8") as cf:
        tw = csv.DictWriter(tf, fieldnames=TRIPLES_OUT_FIELDS, delimiter="\t")
        tw.writeheader()

        cw_fields = ["question_id", "subject_curie", "object_curie", "n_edges", "status"]
        cw = csv.DictWriter(cf, fieldnames=cw_fields, delimiter="\t")
        cw.writeheader()

        for row in pairs:
            qid = (row.get("question_id") or "").strip()
            subj = (row.get("subject_curie") or "").strip()
            obj = (row.get("object_curie") or "").strip()
            subj_label_in = (row.get("subject_label") or "").strip()
            obj_label_in = (row.get("object_label") or "").strip()

            if not qid or not subj or not obj:
                cw.writerow({"question_id": qid, "subject_curie": subj, "object_curie": obj, "n_edges": 0, "status": "SKIP_BAD_ROW"})
                continue

            payload = build_one_hop_trapi_query(subj, obj)

            try:
                resp = post_trapi(
                    session=session,
                    endpoint=args.endpoint,
                    payload=payload,
                    timeout=args.timeout,
                    retries=args.retries,
                    backoff_sec=args.backoff,
                )
            except Exception as e:
                print(f"[{qid}] ERROR querying {subj} -> {obj}: {e}", file=sys.stderr)
                cw.writerow({"question_id": qid, "subject_curie": subj, "object_curie": obj, "n_edges": 0, "status": "ERROR"})
                if args.sleep:
                    time.sleep(args.sleep)
                continue

            message = (resp or {}).get("message", {})
            kg = message.get("knowledge_graph", {}) if isinstance(message, dict) else {}
            kg_nodes = kg.get("nodes", {}) if isinstance(kg, dict) else {}
            kg_edges = kg.get("edges", {}) if isinstance(kg, dict) else {}

            if not isinstance(kg_edges, dict):
                cw.writerow({"question_id": qid, "subject_curie": subj, "object_curie": obj, "n_edges": 0, "status": "NO_EDGES_DICT"})
                if args.sleep:
                    time.sleep(args.sleep)
                continue

            n_edges = len(kg_edges)
            in_range = args.min_edges <= n_edges <= args.max_edges

            status = "OK"
            if args.enforce_range and not in_range:
                status = "OUT_OF_RANGE_SKIPPED"

            cw.writerow({"question_id": qid, "subject_curie": subj, "object_curie": obj, "n_edges": n_edges, "status": status})

            if (not args.enforce_range) or in_range:
                for edge_id, e in kg_edges.items():
                    if not isinstance(e, dict):
                        continue
                    e_subj = e.get("subject", "")
                    e_obj = e.get("object", "")
                    pred = e.get("predicate", "")

                    subj_label = ""
                    obj_label = ""
                    if isinstance(kg_nodes, dict):
                        subj_label = (kg_nodes.get(e_subj, {}) or {}).get("name", "") if e_subj else ""
                        obj_label = (kg_nodes.get(e_obj, {}) or {}).get("name", "") if e_obj else ""

                    # fallback to input labels if KG lacks them
                    if not subj_label:
                        subj_label = subj_label_in
                    if not obj_label:
                        obj_label = obj_label_in

                    prov = extract_provenance(e)

                    tw.writerow(
                        {
                            "question_id": qid,
                            "query_edge_key": "e0",
                            "edge_id": edge_id,
                            "subject": e_subj,
                            "predicate": pred,
                            "object": e_obj,
                            "subject_label": subj_label,
                            "object_label": obj_label,
                            "provenance": prov,
                        }
                    )

            if args.sleep:
                time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
