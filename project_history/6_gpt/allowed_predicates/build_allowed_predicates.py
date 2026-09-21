#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests


def fetch_json(url: str, timeout: int = 60) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout, headers={"accept": "application/json"})
    r.raise_for_status()
    return r.json()


def extract_biolink_and_trapi_versions(openapi: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Translator services may store versions in different places.
    This supports:
      - openapi["x-biolink-version"], openapi["x-trapi-version"]
      - openapi["info"]["x-biolink-version"], openapi["info"]["x-trapi-version"]
      - openapi["info"]["x-translator"]["biolink-version"]   (ROBOKOPKG case)
      - openapi["info"]["x-trapi"]["version"]                 (ROBOKOPKG case)
    """
    info = openapi.get("info", {}) if isinstance(openapi.get("info"), dict) else {}

    # Common patterns
    biolink = openapi.get("x-biolink-version") or info.get("x-biolink-version")
    trapi = openapi.get("x-trapi-version") or info.get("x-trapi-version")

    # Translator patterns (what Automat ROBOKOPKG uses)
    x_translator = info.get("x-translator") if isinstance(info.get("x-translator"), dict) else {}
    if not biolink:
        biolink = x_translator.get("biolink-version")

    x_trapi = info.get("x-trapi") if isinstance(info.get("x-trapi"), dict) else {}
    if not trapi:
        trapi = x_trapi.get("version")

    return biolink, trapi


def extract_predicates_from_meta_kg(meta: Dict[str, Any]) -> Set[str]:
    """
    Supports these common meta_knowledge_graph response shapes:

    1) ROBOKOPKG/Automat style:
       { "nodes": [...], "edges": [ {"predicate": ...}, ... ] }

    2) Wrapped:
       { "meta_knowledge_graph": { "edges": [ ... ] } }

    3) Wrapped under message:
       { "message": { "meta_knowledge_graph": { "edges": [ ... ] } } }

    Also supports edges as either:
      - list[edge_obj]
      - dict[edge_id -> edge_obj]
    """
    preds: Set[str] = set()

    def add_from_edge_obj(e: Any) -> None:
        if not isinstance(e, dict):
            return

        p = e.get("predicate")
        if isinstance(p, str) and p:
            preds.add(p)

        ps = e.get("predicates")
        if isinstance(ps, list):
            for x in ps:
                if isinstance(x, str) and x:
                    preds.add(x)

    def handle_edges(edges: Any) -> bool:
        if isinstance(edges, list):
            for e in edges:
                add_from_edge_obj(e)
            return True
        if isinstance(edges, dict):
            for _, e in edges.items():
                add_from_edge_obj(e)
            return True
        return False

    # Candidate locations in priority order
    candidates: List[Any] = []

    if isinstance(meta, dict):
        # (1) Top-level edges (ROBOKOPKG case)
        if "edges" in meta:
            candidates.append(meta["edges"])

        # (2) meta_knowledge_graph wrapper
        mkg = meta.get("meta_knowledge_graph")
        if isinstance(mkg, dict) and "edges" in mkg:
            candidates.append(mkg["edges"])

        # (3) message wrapper
        msg = meta.get("message")
        if isinstance(msg, dict):
            mkg2 = msg.get("meta_knowledge_graph")
            if isinstance(mkg2, dict) and "edges" in mkg2:
                candidates.append(mkg2["edges"])

    for edges in candidates:
        if handle_edges(edges) and preds:
            return preds

    top_keys = sorted(meta.keys()) if isinstance(meta, dict) else None
    raise ValueError(
        f"Unrecognized /meta_knowledge_graph JSON shape. "
        f"Top-level type={type(meta).__name__}, top_keys={top_keys}"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: build_allowed_predicates.py <BASE_TRAPI_URL>\n"
            "Example: build_allowed_predicates.py https://robokop-automat.apps.renci.org/robokopkg",
            file=sys.stderr,
        )
        return 2

    base = sys.argv[1].rstrip("/")
    meta_url = f"{base}/meta_knowledge_graph"
    openapi_url = f"{base}/openapi.json"

    meta = fetch_json(meta_url)
    preds = sorted(extract_predicates_from_meta_kg(meta))

    biolink_version = None
    trapi_version = None
    service_title = None
    service_version = None
    infores = None

    try:
        openapi = fetch_json(openapi_url)
        biolink_version, trapi_version = extract_biolink_and_trapi_versions(openapi)

        info = openapi.get("info", {}) if isinstance(openapi.get("info"), dict) else {}
        service_title = info.get("title")
        service_version = info.get("version")

        x_translator = info.get("x-translator") if isinstance(info.get("x-translator"), dict) else {}
        infores = x_translator.get("infores")
    except Exception:
        # Not fatal; still produce predicate list from meta_knowledge_graph
        pass

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_trapi_url": base,
        "meta_knowledge_graph_url": meta_url,
        "openapi_url_attempted": openapi_url,
        "service_title": service_title,
        "service_version": service_version,
        "infores": infores,
        "biolink_version": biolink_version,
        "trapi_version": trapi_version,
        "predicate_count": len(preds),
        "predicates": preds,
        "method": "Operational predicate list from TRAPI /meta_knowledge_graph",
    }

    with open("allowed_predicates.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote allowed_predicates.json with {len(preds)} predicates")
    if service_title:
        print(f"Service: {service_title} ({service_version})")
    if infores:
        print(f"Infores: {infores}")
    if biolink_version:
        print(f"Biolink: {biolink_version}")
    if trapi_version:
        print(f"TRAPI: {trapi_version}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
