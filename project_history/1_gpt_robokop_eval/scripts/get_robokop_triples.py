import requests, json, os

def get_robokop_triples(subject_curie):
    """Query ROBOKOP KG (Automat) for all relationships of a given CURIE."""
    url = "https://automat.renci.org/robokopkg/query"

    payload = {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": [subject_curie]},
                    "n1": {}  # no restriction on object type
                },
                "edges": {
                    "e0": {"subject": "n0", "object": "n1"}
                }
            }
        }
    }

    r = requests.post(url, json=payload)
    print("Status:", r.status_code)
    if r.status_code != 200:
        print("Error:", r.text)
        return []

    try:
        data = r.json()
    except Exception as e:
        print("JSON decode error:", e)
        print("Raw:", r.text[:300])
        return []

    triples = []
    kg = data.get("message", {}).get("knowledge_graph", {})
    for edge in kg.get("edges", {}).values():
        subj = edge.get("subject")
        pred = edge.get("predicate")
        obj = edge.get("object")
        triples.append((subj, pred, obj))

    return triples


if __name__ == "__main__":
    triples = get_robokop_triples("HGNC:1097")  # BRAF example
    os.makedirs("data/robokop_triples", exist_ok=True)
    out_path = "data/robokop_triples/BRAF.json"
    with open(out_path, "w") as f:
        json.dump(triples, f, indent=2)
    print(f"Saved {len(triples)} triples to {out_path}")
