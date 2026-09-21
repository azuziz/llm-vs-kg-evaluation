import csv
import json
import os
import sys
import time
from datetime import datetime

import requests

# We'll try these endpoints in order until one responds successfully.
CANDIDATE_ENDPOINTS = [
    "https://robokop-automat.apps.renci.org/robokopkg/query",
    "https://robokop-automat.apps.renci.org/robokopkg/1.5/query",
    "https://robokop-automat.apps.renci.org/robokopkg/1.4/query",
    "https://automat-u24.apps.renci.org/robokopkg/1.3/query",
]

def build_trapi_onehop(subject_curie: str, object_curie: str) -> dict:
    """
    TRAPI Request: one edge between two pinned nodes.
    We omit predicates so ROBOKOP can return any relation it has between the two nodes.
    TRAPI message/query_graph structure is per ReasonerAPI/TRAPI. 
    """
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": [subject_curie]},
                    "n1": {"ids": [object_curie]},
                },
                "edges": {
                    "e0": {"subject": "n0", "object": "n1"}
                }
            }
        },
        "submitter": "llm-vs-kg-eval"
    }

def pick_working_endpoint() -> str:
    # Use /metadata to sanity check the host if needed; here we just try POST /query with a tiny payload.
    test_query = build_trapi_onehop("MONDO:0018310", "MONDO:0005559")  # from your Q1
    for url in CANDIDATE_ENDPOINTS:
        try:
            r = requests.post(url, json=test_query, timeout=60)
            if r.status_code in (200, 400, 422):  # 400/422 still means endpoint exists (payload issue)
                return url
        except Exception:
            pass
    raise RuntimeError("No working ROBOKOP /query endpoint found. Update CANDIDATE_ENDPOINTS.")

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def main(resolved_tsv: str, out_dir: str, run_log_tsv: str):
    os.makedirs(out_dir, exist_ok=True)

    endpoint = pick_working_endpoint()
    print(f"Using ROBOKOP endpoint: {endpoint}")

    with open(resolved_tsv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    run_log = []
    for row in rows:
        qid = row["question_id"]
        subj_curie = row["subject_curie"].strip()
        obj_curie = row["object_curie"].strip()

        if not subj_curie or not obj_curie:
            run_log.append({
                "question_id": qid,
                "status": "SKIP_MISSING_CURIE",
                "http_status": "",
                "results_count": "",
                "saved_path": "",
                "notes": row.get("notes","")
            })
            continue

        payload = build_trapi_onehop(subj_curie, obj_curie)

        t0 = time.time()
        try:
            resp = requests.post(endpoint, json=payload, timeout=180)
            dt = time.time() - t0

            out_path = os.path.join(out_dir, f"{qid}.json")
            with open(out_path, "w", encoding="utf-8") as out:
                out.write(resp.text)

            # Try to count results if response is JSON TRAPI
            results_count = ""
            try:
                j = resp.json()
                results_count = str(len(j.get("message", {}).get("results", [])))
            except Exception:
                results_count = ""

            run_log.append({
                "question_id": qid,
                "status": "OK" if resp.status_code == 200 else "HTTP_ERROR",
                "http_status": str(resp.status_code),
                "results_count": results_count,
                "saved_path": out_path,
                "notes": row.get("notes","") + f"|elapsed_sec={dt:.2f}"
            })

            print(f"{qid}: HTTP {resp.status_code} results={results_count} saved={out_path}")

        except Exception as e:
            run_log.append({
                "question_id": qid,
                "status": "EXCEPTION",
                "http_status": "",
                "results_count": "",
                "saved_path": "",
                "notes": row.get("notes","") + f"|exception={type(e).__name__}:{e}"
            })
            print(f"{qid}: EXCEPTION {e}")

        # polite delay to avoid hammering
        time.sleep(0.2)

    with open(run_log_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["question_id","status","http_status","results_count","saved_path","notes"],
            delimiter="\t"
        )
        w.writeheader()
        w.writerows(run_log)

    print(f"Wrote: {run_log_tsv}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python step3_robokop_query.py resolved_node_pairs.tsv robokop_raw robokop_run.tsv")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
