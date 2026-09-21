import csv
import json
import os
import sys
from glob import glob

def kg_edges_as_dict(kg_edges):
    """
    Normalize knowledge_graph.edges to a dict: edge_id -> edge_obj
    TRAPI implementations vary: dict keyed by id OR list with each edge containing an 'id'.
    """
    if kg_edges is None:
        return {}
    if isinstance(kg_edges, dict):
        return kg_edges
    if isinstance(kg_edges, list):
        out = {}
        for e in kg_edges:
            if isinstance(e, dict):
                eid = e.get("id")
                if eid:
                    out[eid] = e
        return out
    return {}

def kg_nodes_as_dict(kg_nodes):
    if kg_nodes is None:
        return {}
    if isinstance(kg_nodes, dict):
        return kg_nodes
    if isinstance(kg_nodes, list):
        out = {}
        for n in kg_nodes:
            if isinstance(n, dict):
                nid = n.get("id")
                if nid:
                    out[nid] = n
        return out
    return {}

def detect_query_edge_key(trapi: dict) -> str:
    qg_edges = (((trapi.get("message") or {}).get("query_graph") or {}).get("edges") or {})
    if isinstance(qg_edges, dict) and len(qg_edges) >= 1:
        # typically {"e0": {...}}
        return next(iter(qg_edges.keys()))
    # fallback default
    return "e0"

def collect_edge_binding_ids(result: dict, qedge_key: str):
    """
    TRAPI variants:
      - result.edge_bindings[qedge_key] = [{"id": "..."}] or ["..."]
      - result.analyses[*].edge_bindings[qedge_key] similarly
    Return a list of edge IDs.
    """
    edge_ids = []

    def pull_from_binding_obj(binding_val):
        if not binding_val:
            return
        if isinstance(binding_val, list):
            for x in binding_val:
                if isinstance(x, str):
                    edge_ids.append(x)
                elif isinstance(x, dict):
                    if "id" in x:
                        edge_ids.append(x["id"])
        elif isinstance(binding_val, dict):
            # sometimes a dict not list
            if "id" in binding_val:
                edge_ids.append(binding_val["id"])

    # 1) direct edge_bindings
    eb = result.get("edge_bindings") or {}
    if isinstance(eb, dict) and qedge_key in eb:
        pull_from_binding_obj(eb.get(qedge_key))

    # 2) analyses edge_bindings
    analyses = result.get("analyses") or []
    if isinstance(analyses, list):
        for a in analyses:
            if not isinstance(a, dict):
                continue
            aeb = a.get("edge_bindings") or {}
            if isinstance(aeb, dict) and qedge_key in aeb:
                pull_from_binding_obj(aeb.get(qedge_key))

    # de-dup preserve order
    seen = set()
    uniq = []
    for eid in edge_ids:
        if eid and eid not in seen:
            uniq.append(eid)
            seen.add(eid)
    return uniq

def provenance_string(edge: dict) -> str:
    prov = []
    sources = edge.get("sources") or []
    for s in sources:
        if not isinstance(s, dict):
            continue
        rid = s.get("resource_id") or s.get("resource") or s.get("source") or ""
        role = s.get("resource_role") or s.get("role") or ""
        if rid:
            prov.append(f"{rid}({role})" if role else rid)
    # also keep minimal attribute provenance if sources empty
    if not prov:
        attrs = edge.get("attributes") or []
        for a in attrs:
            if isinstance(a, dict):
                src = a.get("attribute_source")
                if src:
                    prov.append(str(src))
    # de-dup
    out = []
    seen = set()
    for x in prov:
        if x not in seen:
            out.append(x); seen.add(x)
    return ";".join(out)

def extract_one(qid: str, trapi: dict):
    msg = trapi.get("message") or {}
    kg = msg.get("knowledge_graph") or {}
    results = msg.get("results") or []

    qedge_key = detect_query_edge_key(trapi)

    kg_edges = kg_edges_as_dict(kg.get("edges"))
    kg_nodes = kg_nodes_as_dict(kg.get("nodes"))

    rows = []
    for res in results:
        if not isinstance(res, dict):
            continue
        eids = collect_edge_binding_ids(res, qedge_key)
        for eid in eids:
            edge = kg_edges.get(eid)
            if not edge:
                # Sometimes eid refers to an auxiliary graph edge not in KG, skip but keep traceable
                continue
            subj = edge.get("subject") or ""
            obj = edge.get("object") or ""
            pred = edge.get("predicate") or ""

            subj_label = (kg_nodes.get(subj) or {}).get("name") or ""
            obj_label = (kg_nodes.get(obj) or {}).get("name") or ""

            rows.append({
                "question_id": qid,
                "query_edge_key": qedge_key,
                "edge_id": eid,
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "subject_label": subj_label,
                "object_label": obj_label,
                "provenance": provenance_string(edge)
            })
    return rows

def main(raw_dir: str, out_triples_tsv: str, out_labels_tsv: str, out_per_q_dir: str):
    os.makedirs(out_per_q_dir, exist_ok=True)

    all_rows = []
    label_rows = []

    files = sorted(glob(os.path.join(raw_dir, "Q*.json")))
    if not files:
        raise SystemExit(f"No Q*.json files found in {raw_dir}")

    for fp in files:
        qid = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, "r", encoding="utf-8") as f:
            trapi = json.load(f)

        rows = extract_one(qid, trapi)

        per_path = os.path.join(out_per_q_dir, f"{qid}.tsv")
        with open(per_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["question_id","query_edge_key","edge_id","subject","predicate","object","subject_label","object_label","provenance"],
                delimiter="\t"
            )
            w.writeheader()
            w.writerows(rows)

        all_rows.extend(rows)

        msg = trapi.get("message") or {}
        results_count = len(msg.get("results") or [])
        unique_edges = len({r["edge_id"] for r in rows})

        # Additional debug counts
        kg_edges = msg.get("knowledge_graph", {}).get("edges")
        kg_edge_count = (len(kg_edges) if isinstance(kg_edges, list) else len(kg_edges) if isinstance(kg_edges, dict) else 0)

        label_rows.append({
            "question_id": qid,
            "kg_answer": "YES" if unique_edges > 0 else "NO",
            "results_count": results_count,
            "unique_edges": unique_edges,
            "kg_edges_total": kg_edge_count,
            "per_question_file": per_path
        })

    with open(out_triples_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["question_id","query_edge_key","edge_id","subject","predicate","object","subject_label","object_label","provenance"],
            delimiter="\t"
        )
        w.writeheader()
        w.writerows(all_rows)

    with open(out_labels_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["question_id","kg_answer","results_count","unique_edges","kg_edges_total","per_question_file"],
            delimiter="\t"
        )
        w.writeheader()
        w.writerows(label_rows)

    print(f"Wrote: {out_triples_tsv}  (rows={len(all_rows)})")
    print(f"Wrote: {out_labels_tsv}   (rows={len(label_rows)})")
    print(f"Wrote per-question TSVs under: {out_per_q_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python step4_extract_triples.py robokop_raw robokop_triples.tsv robokop_labels.tsv robokop_triples_by_question")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])