#!/usr/bin/env python3
import json
import sys
import requests

NAME_RESOLVER_URL = "https://name-resolution-sri.renci.org/lookup"
NODE_NORMALIZER_URL = "https://nodenormalization-sri.renci.org/get_normalized_nodes"

def name_resolve(label, limit=5):
    params = {"string": label, "limit": limit}
    r = requests.get(NAME_RESOLVER_URL, params=params)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        return None
    best = data[0]
    return {
        "curie": best.get("curie"),
        "label": best.get("label"),
        "score": best.get("score")
    }

def normalize_curie(curie):
    if not curie:
        return None
    params = {"curie": curie}
    r = requests.get(NODE_NORMALIZER_URL, params=params)
    r.raise_for_status()
    data = r.json()
    entry = data.get(curie)
    if not entry or entry.get("id") is None:
        return curie
    canonical = entry["id"].get("identifier") or curie
    return canonical

def build_trapi_query(gpt_query):
    trapi = {
        "message": {
            "query_graph": {
                "nodes": {},
                "edges": {}
            }
        }
    }

    qnodes = gpt_query.get("nodes", {})
    qedges = gpt_query.get("edges", {})

    for nid, node in qnodes.items():
        label = node.get("label")
        categories = node.get("categories", [])
        resolved = name_resolve(label) if label else None

        if resolved:
            curie = resolved["curie"]
            canonical = normalize_curie(curie)
            trapi["message"]["query_graph"]["nodes"][nid] = {
                "ids": [canonical],
                "categories": categories
            }
        else:
            trapi["message"]["query_graph"]["nodes"][nid] = {
                "categories": categories
            }

    for eid, edge in qedges.items():
        trapi["message"]["query_graph"]["edges"][eid] = {
            "subject": edge["subject"],
            "object": edge["object"],
            "predicates": edge.get("predicates", [])
        }

    return trapi

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 build_trapi_from_gpt.py INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(in_path, "r", encoding="utf-8") as f:
        gpt_query = json.load(f)

    trapi = build_trapi_query(gpt_query)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trapi, f, indent=2)

    print(f"Wrote TRAPI query to {out_path}")

if __name__ == "__main__":
    main()
