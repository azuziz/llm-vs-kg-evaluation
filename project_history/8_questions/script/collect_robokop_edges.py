#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


DEFAULT_ENDPOINT = "https://robokop-automat.apps.renci.org/robokopkg/query"


def build_one_hop_trapi_query(subject_curie: str, object_curie: str) -> Dict[str, Any]:
    """
    TRAPI message for a single-edge (one-hop) query:
      n0 (subject ids) --e0--> n1 (object ids)
    Predicates are left null to retrieve all predicates between the pair.
    """
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
    """
    Prefer TRAPI 'sources' if present (TRAPI 1.4+ commonly uses this).
    Format: infores:xxx(role);infores:yyy(role)
    Falls back to scanning attributes for knowledge source fields if needed.
    """
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

    # Fallback: try to infer from attributes if sources absent
    attrs = edge.get("attributes")
    if isinstance(attrs, list):
        for a in attrs:
            atype = a.get("attribute_type_id")
            val = a.get("value")
            # Sometimes knowledge sources appear as strings or lists of dicts.
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
    timeout: int = 120,
    retries: int = 3,
    backoff_sec: float = 2.0,
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


def iter_pairs_tsv(path: str) -> Iterable[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        required = {
            "question_id",
            "subject_curie",
            "subject_label",
            "object_curie",
            "object_label",
        }
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in input TSV: {sorted(missing)}")
        for row in r:
            yield row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_tsv", required=True, help="Input TSV (robokop_100_pairs.tsv)")
    ap.add_argument("--out_tsv", required=True, help="Output TSV (robokop_triples1.tsv)")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"ROBOKOP TRAPI query endpoint (default: {DEFAULT_ENDPOINT})")
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep (seconds) between calls")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout per request (seconds)")
    args = ap.parse_args()

    out_fields = [
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

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    with open(args.out_tsv, "w", newline="", encoding="utf-8") as out_f:
        w = csv.DictWriter(out_f, fieldnames=out_fields, delimiter="\t")
        w.writeheader()

        for row in iter_pairs_tsv(args.pairs_tsv):
            qid = row["question_id"].strip()
            subj = row["subject_curie"].strip()
            obj = row["object_curie"].strip()

            payload = build_one_hop_trapi_query(subj, obj)

            try:
                resp = post_trapi(
                    session=session,
                    endpoint=args.endpoint,
                    payload=payload,
                    timeout=args.timeout,
                    retries=3,
                    backoff_sec=2.0,
                )
            except Exception as e:
                print(f"[{qid}] ERROR querying {subj} -> {obj}: {e}", file=sys.stderr)
                # still continue with next pair
                continue

            message = (resp or {}).get("message", {})
            kg = message.get("knowledge_graph", {}) if isinstance(message, dict) else {}
            kg_nodes = kg.get("nodes", {}) if isinstance(kg, dict) else {}
            kg_edges = kg.get("edges", {}) if isinstance(kg, dict) else {}

            # Count edges returned for this pair
            n_edges = len(kg_edges) if isinstance(kg_edges, dict) else 0
            print(f"[{qid}] {subj} -> {obj}: {n_edges} edges")

            if not isinstance(kg_edges, dict):
                if args.sleep:
                    time.sleep(args.sleep)
                continue

            # Write each returned edge as one row
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

                # Fallback to the labels from the input pair if KG doesn't provide them
                if not subj_label:
                    subj_label = row.get("subject_label", "")
                if not obj_label:
                    obj_label = row.get("object_label", "")

                prov = extract_provenance(e)

                w.writerow(
                    {
                        "question_id": qid,
                        "query_edge_key": "e0",     # from the query graph edge id we used
                        "edge_id": edge_id,         # key of KG edge (often looks like uuid:..._e_#)
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
