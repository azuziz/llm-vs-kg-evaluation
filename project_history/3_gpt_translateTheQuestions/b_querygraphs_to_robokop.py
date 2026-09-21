#!/usr/bin/env python3
import json
import os
import sys
import glob
import re
import requests

NAME_RESOLVER_URL = "https://name-resolution-sri.renci.org/lookup"
NODE_NORMALIZER_URL = "https://nodenormalization-sri.renci.org/get_normalized_nodes"
ROBOKOP_QUERY_URL = "https://robokop-automat.apps.renci.org/robokopkg/query"

HUMAN_TAXON = "NCBITaxon:9606"

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def normalize_curie(curie):
    if not curie:
        return None
    params = {"curie": curie}
    r = requests.get(NODE_NORMALIZER_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    entry = data.get(curie)
    if not entry or entry.get("id") is None:
        return curie
    return entry["id"].get("identifier") or curie

def _candidate_matches(candidate, desired_categories, require_human_gene=False):
    """
    candidate is an object from name-resolver lookup list.
    desired_categories is list like ["biolink:Gene"] or ["biolink:Disease"].
    """
    c_types = set(_as_list(candidate.get("types")))
    c_taxa = set(_as_list(candidate.get("taxa")))

    # Strong type filtering: if we want a Gene, don't accept Reactome MolecularActivity etc.
    if desired_categories:
        # accept if any desired category appears in candidate types
        if not any(cat in c_types for cat in desired_categories):
            return False

    # If it's a gene and we require human, only accept if taxon includes human.
    # Note: some candidates may omit taxa; in that case, allow but de-prioritize later.
    if require_human_gene:
        if c_taxa and (HUMAN_TAXON not in c_taxa):
            return False

    return True

def name_resolve_best(label, desired_categories, limit=20, prefer_human_gene=False):
    """
    Returns dict {curie,label,score,types,taxa} or None.
    Strategy:
      1) Fetch multiple candidates
      2) Filter by desired biolink types
      3) If gene: prefer/require human taxa
      4) Pick highest score among remaining
    """
    params = {"string": label, "limit": limit}
    r = requests.get(NAME_RESOLVER_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        return None

    # First pass: strict type match; optionally strict human for genes.
    strict = []
    for c in data:
        if not c.get("curie"):
            continue
        if _candidate_matches(
            c,
            desired_categories=desired_categories,
            require_human_gene=(prefer_human_gene and ("biolink:Gene" in desired_categories))
        ):
            strict.append(c)

    # If we required human gene and nothing matched, relax taxon requirement but still keep type match.
    if not strict and ("biolink:Gene" in desired_categories):
        for c in data:
            if not c.get("curie"):
                continue
            if _candidate_matches(c, desired_categories=desired_categories, require_human_gene=False):
                strict.append(c)

    if not strict:
        return None

    # Prefer candidates that explicitly indicate human taxon (if gene), else just best score.
    def rank_key(c):
        score = c.get("score", 0.0)
        taxa = set(_as_list(c.get("taxa")))
        human_bonus = 1 if (HUMAN_TAXON in taxa) else 0
        # Higher is better
        return (human_bonus, score)

    best = max(strict, key=rank_key)
    return {
        "curie": best.get("curie"),
        "label": best.get("label"),
        "score": best.get("score"),
        "types": best.get("types"),
        "taxa": best.get("taxa"),
    }

def resolve_node(label, categories):
    """
    Resolve label -> canonical CURIE, using category-aware logic.
    For genes: enforce Homo sapiens by retrying label + " (Homo sapiens)" if needed.
    """
    desired_categories = categories or []

    is_gene = "biolink:Gene" in desired_categories

    # First attempt: use label as-is, but prefer human if gene
    resolved = name_resolve_best(
        label,
        desired_categories=desired_categories,
        prefer_human_gene=True
    )

    # If gene and resolved is non-human (or missing taxa and later normalizer shows non-human),
    # try again with explicit Homo sapiens suffix.
    if is_gene:
        # quick check: if candidate taxa exists and isn't human, retry
        taxa = set(_as_list(resolved.get("taxa"))) if resolved else set()
        if resolved is None or (taxa and HUMAN_TAXON not in taxa):
            resolved2 = name_resolve_best(
                f"{label} (Homo sapiens)",
                desired_categories=desired_categories,
                prefer_human_gene=True
            )
            if resolved2:
                resolved = resolved2

    if not resolved:
        return None

    canonical = normalize_curie(resolved["curie"])
    return canonical

def should_force_related_to(gpt_query):
    """
    Force biolink:related_to iff question is of the form:
      "Is ... associated with ...?"
    """
    nq = (gpt_query.get("natural_question") or "").strip().lower()
    if not nq:
        return False
    # robust enough: starts with "is" and contains "associated with"
    if nq.startswith("is ") and " associated with " in nq:
        return True
    return False

def build_trapi_query(gpt_query):
    trapi = {"message": {"query_graph": {"nodes": {}, "edges": {}}}}
    qnodes = gpt_query.get("nodes", {})
    qedges = gpt_query.get("edges", {})

    # Build nodes (label -> CURIE) with category-aware selection
    for nid, node in qnodes.items():
        label = node.get("label")
        categories = node.get("categories", [])

        if label:
            canonical = resolve_node(label, categories)
            if canonical:
                trapi["message"]["query_graph"]["nodes"][nid] = {
                    "ids": [canonical],
                    "categories": categories
                }
                continue

        # fallback: category-only node
        trapi["message"]["query_graph"]["nodes"][nid] = {
            "categories": categories
        }

    # Build edges; optionally force predicate to related_to for "Is ... associated with ...?"
    force_related = should_force_related_to(gpt_query)

    for eid, edge in qedges.items():
        preds = edge.get("predicates", [])
        if force_related:
            preds = ["biolink:related_to"]

        trapi["message"]["query_graph"]["edges"][eid] = {
            "subject": edge["subject"],
            "object": edge["object"],
            "predicates": preds
        }

    return trapi

def query_robokop(trapi_query):
    headers = {"Content-Type": "application/json"}
    r = requests.post(ROBOKOP_QUERY_URL, headers=headers, json=trapi_query, timeout=120)
    r.raise_for_status()
    return r.json()

def main():
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print("Usage: python3 b_querygraphs_to_robokop.py [queries_dir]", file=sys.stderr)
        sys.exit(1)

    queries_dir = sys.argv[1] if len(sys.argv) == 2 else "queries"
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    for path in sorted(glob.glob(os.path.join(queries_dir, "*_query.json"))):
        basename = os.path.basename(path)
        qid = basename.split("_")[0]
        print(f"Processing {qid} from {basename}")

        with open(path, "r", encoding="utf-8") as f:
            gpt_query = json.load(f)

        trapi = build_trapi_query(gpt_query)

        trapi_path = os.path.join(out_dir, f"{qid}_trapi.json")
        with open(trapi_path, "w", encoding="utf-8") as f:
            json.dump(trapi, f, indent=2)
        print(f"  -> wrote {trapi_path}")

        try:
            robokop_resp = query_robokop(trapi)
        except requests.HTTPError as e:
            print(f"  !! ROBOKOP query failed for {qid}: {e}", file=sys.stderr)
            continue

        resp_path = os.path.join(out_dir, f"{qid}_robokop.json")
        with open(resp_path, "w", encoding="utf-8") as f:
            json.dump(robokop_resp, f, indent=2)
        print(f"  -> wrote {resp_path}")

if __name__ == "__main__":
    main()
