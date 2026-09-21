#!/usr/bin/env python3
import csv
import glob
import json
import os
import sys

def get_node_name(kg_nodes, node_id):
    n = kg_nodes.get(node_id, {})
    return n.get("name", "")

def get_attr(edge, attr_type_id):
    for a in edge.get("attributes", []):
        if a.get("attribute_type_id") == attr_type_id:
            return a.get("value")
    return None

def get_sources(edge):
    srcs = []
    for s in edge.get("sources", []):
        rid = s.get("resource_id")
        if rid:
            srcs.append(rid)
    return ";".join(sorted(set(srcs)))

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "kg_triples.csv"

    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_robokop.json"))):
        qid = os.path.basename(path).split("_")[0]

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        msg = data.get("message", {})
        kg = msg.get("knowledge_graph", {})
        kg_nodes = kg.get("nodes", {})
        kg_edges = kg.get("edges", {})
        results = msg.get("results", [])

        for r in results:
            node_bindings = r.get("node_bindings", {})
            analyses = r.get("analyses", [])

            # We assume your query always uses n0/n1 and e0 (your pipeline does)
            n0 = node_bindings.get("n0", [{}])[0].get("id")
            n1 = node_bindings.get("n1", [{}])[0].get("id")

            for a in analyses:
                edge_bindings = a.get("edge_bindings", {})
                for eb in edge_bindings.get("e0", []):
                    edge_id = eb.get("id")
                    edge = kg_edges.get(edge_id, {})
                    if not edge:
                        continue

                    subj = edge.get("subject", "")
                    obj = edge.get("object", "")
                    pred = edge.get("predicate", "")

                    rows.append({
                        "question_id": qid,
                        "query_n0": n0 or "",
                        "query_n1": n1 or "",
                        "edge_id": edge_id,
                        "subject": subj,
                        "subject_name": get_node_name(kg_nodes, subj),
                        "predicate": pred,
                        "object": obj,
                        "object_name": get_node_name(kg_nodes, obj),
                        "knowledge_level": get_attr(edge, "biolink:knowledge_level") or "",
                        "agent_type": get_attr(edge, "biolink:agent_type") or "",
                        "sources": get_sources(edge),
                    })

    fieldnames = [
        "question_id","query_n0","query_n1","edge_id",
        "subject","subject_name","predicate","object","object_name",
        "knowledge_level","agent_type","sources"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    main()

