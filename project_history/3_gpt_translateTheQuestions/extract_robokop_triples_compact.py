#!/usr/bin/env python3
import csv, glob, json, os, sys
from collections import defaultdict

# Preference order: you can tune this later
PREDICATE_PRIORITY = [
    "biolink:interacts_with",
    "biolink:genetically_associated_with",
    "biolink:associated_with",
    "biolink:related_to",
    "biolink:affects",
    "biolink:has_phenotype",
    "biolink:contributes_to",
    "biolink:causes",
]

KNOWLEDGE_LEVEL_PRIORITY = {
    "knowledge_assertion": 3,
    "not_provided": 2,
    "logical_entailment": 1,  # often derived/entailed; you may want it lower
    "": 0,
}

AGENT_PRIORITY = {
    "manual_agent": 3,
    "not_provided": 2,
    "automated_agent": 1,
    "text_mining_agent": 0,
    "": 0,
}

def get_node_name(kg_nodes, node_id):
    return kg_nodes.get(node_id, {}).get("name", "")

def get_attr(edge, attr_type_id):
    for a in edge.get("attributes", []):
        if a.get("attribute_type_id") == attr_type_id:
            return a.get("value")
    return ""

def score_edge(edge):
    pred = edge.get("predicate", "")
    kl = get_attr(edge, "biolink:knowledge_level") or ""
    ag = get_attr(edge, "biolink:agent_type") or ""

    pred_score = (len(PREDICATE_PRIORITY) - PREDICATE_PRIORITY.index(pred)) if pred in PREDICATE_PRIORITY else 0
    return (pred_score, KNOWLEDGE_LEVEL_PRIORITY.get(kl, 0), AGENT_PRIORITY.get(ag, 0))

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    out_all = sys.argv[2] if len(sys.argv) > 2 else "kg_triples_compact_all.csv"
    out_best = sys.argv[3] if len(sys.argv) > 3 else "kg_triples_compact_best.csv"

    all_rows = []
    best_by_q = defaultdict(list)

    for path in sorted(glob.glob(os.path.join(results_dir, "*_robokop.json"))):
        qid = os.path.basename(path).split("_")[0]
        data = json.load(open(path, "r", encoding="utf-8"))
        msg = data.get("message", {})
        kg = msg.get("knowledge_graph", {})
        kg_nodes = kg.get("nodes", {})
        kg_edges = kg.get("edges", {})
        results = msg.get("results", [])

        for r in results:
            node_bindings = r.get("node_bindings", {})
            n0 = node_bindings.get("n0", [{}])[0].get("id", "")
            n1 = node_bindings.get("n1", [{}])[0].get("id", "")

            for a in r.get("analyses", []):
                for eb in a.get("edge_bindings", {}).get("e0", []):
                    edge_id = eb.get("id")
                    edge = kg_edges.get(edge_id, {})
                    if not edge:
                        continue

                    row = {
                        "question_id": qid,
                        "query_n0": n0,
                        "query_n1": n1,
                        "subject": edge.get("subject", ""),
                        "subject_name": get_node_name(kg_nodes, edge.get("subject", "")),
                        "predicate": edge.get("predicate", ""),
                        "object": edge.get("object", ""),
                        "object_name": get_node_name(kg_nodes, edge.get("object", "")),
                        "knowledge_level": get_attr(edge, "biolink:knowledge_level"),
                        "agent_type": get_attr(edge, "biolink:agent_type"),
                        "edge_id": edge_id,
                    }
                    all_rows.append(row)
                    best_by_q[qid].append((score_edge(edge), row))

    # write all compact rows
    all_fields = ["question_id","query_n0","query_n1","subject","subject_name","predicate","object","object_name","knowledge_level","agent_type","edge_id"]
    with open(out_all, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        w.writerows(all_rows)

    # pick best per question
    best_rows = []
    for qid, candidates in best_by_q.items():
        if not candidates:
            continue
        best = max(candidates, key=lambda x: x[0])[1]
        best_rows.append(best)

    with open(out_best, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        w.writerows(sorted(best_rows, key=lambda r: r["question_id"]))

    print(f"Wrote ALL compact edges:  {out_all} ({len(all_rows)} rows)")
    print(f"Wrote BEST per question: {out_best} ({len(best_rows)} rows)")

if __name__ == "__main__":
    main()
