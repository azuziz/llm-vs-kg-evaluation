import csv
import json
import os
import sys
import time
import random
from glob import glob
from typing import Dict, List, Set, Tuple, Any

import requests

OPENAI_URL = "https://api.openai.com/v1/responses"

JSON_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "required": ["question_id", "answer", "triples"],
  "properties": {
    "question_id": {"type": "string"},
    "answer": {"type": "string", "enum": ["YES", "NO"]},
    "triples": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["subject", "predicate", "object"],
        "properties": {
          "subject": {"type": "string"},
          "predicate": {"type": "string"},
          "object": {"type": "string"}
        }
      }
    }
  }
}

SYSTEM_PROMPT = (
  "You are an evaluation model that predicts whether a biomedical knowledge graph (ROBOKOP/Translator-style) "
  "contains at least one DIRECT (one-hop) edge between two given nodes.\n"
  "You must output JSON matching the provided schema exactly.\n\n"
  "Rules:\n"
  "1) Use ONLY the provided subject_curie and object_curie as endpoints in triples.\n"
  "2) Use ONLY predicates from allowed_predicates.\n"
  "3) Output ALL distinct direct edges you believe exist (0..N). Do not output duplicates.\n"
  "4) If you output any triple, answer must be YES.\n"
  "5) If you believe no direct edge exists or you are not confident, answer must be NO and triples must be [].\n"
  "6) Do NOT invent other CURIEs, intermediate nodes, or evidence text.\n"
)

def read_resolved_pairs(resolved_tsv: str) -> List[Dict[str, str]]:
    with open(resolved_tsv, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def read_allowed_predicates(robokop_per_q_dir: str, qid: str) -> List[str]:
    """
    Reads robokop_triples_by_question/QX.tsv and returns unique predicates.
    """
    path = os.path.join(robokop_per_q_dir, f"{qid}.tsv")
    if not os.path.exists(path):
        return []
    preds: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            p = (row.get("predicate") or "").strip()
            if p:
                preds.add(p)
    return sorted(preds)

def call_openai(api_key: str, model: str, payload: Dict[str, Any], timeout_s: int = 180) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=timeout_s)
    # For rate limits / transient failures, we handle outside.
    if r.status_code >= 400:
        # Try to include structured error body
        try:
            err = r.json()
        except Exception:
            err = {"error_text": r.text}
        raise requests.HTTPError(f"HTTP {r.status_code}: {err}", response=r)
    return r.json()

def extract_output_text(resp_json: Dict[str, Any]) -> str:
    # Typical Responses API shape
    return resp_json["output"][0]["content"][0]["text"]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def run_one(api_key: str, model: str, user_obj: Dict[str, Any], temperature: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (raw_response_json, parsed_output_json)
    """
    payload = {
        "model": model,
        "temperature": temperature,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)}
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "gpt_edge_decision",
                "strict": True,
                "schema": JSON_SCHEMA
            }
        }
    }

    # Exponential backoff with jitter for transient errors
    backoff = 1.0
    for attempt in range(1, 8):
        try:
            raw = call_openai(api_key, model, payload)
            out_text = extract_output_text(raw)
            parsed = json.loads(out_text)
            return raw, parsed
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            # Retry on typical transient statuses
            if status in (429, 500, 502, 503, 504):
                sleep_s = backoff + random.random() * 0.5
                time.sleep(sleep_s)
                backoff *= 2
                continue
            raise
        except (json.JSONDecodeError, KeyError) as e:
            # If the API responded but parsing failed (should be rare with strict schema),
            # retry a couple times.
            sleep_s = backoff + random.random() * 0.5
            time.sleep(sleep_s)
            backoff *= 2
            if attempt < 7:
                continue
            raise RuntimeError(f"Parse failed after retries: {e}")

def main(resolved_tsv: str, robokop_per_q_dir: str, out_root: str, model: str, runs: int, temperature: float):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")

    pairs = read_resolved_pairs(resolved_tsv)

    ensure_dir(out_root)

    # Run order: Q1..Q8
    for row in pairs:
        qid = row["question_id"].strip()
        subj_curie = row["subject_curie"].strip()
        obj_curie = row["object_curie"].strip()
        subj_label = row.get("subject_label","").strip()
        obj_label = row.get("object_label","").strip()

        if not subj_curie or not obj_curie:
            print(f"{qid}: SKIP (missing CURIE)")
            continue

        allowed_predicates = read_allowed_predicates(robokop_per_q_dir, qid)
        if not allowed_predicates:
            # Fallback predicate if ROBOKOP file missing; keep task well-defined
            allowed_predicates = ["biolink:related_to"]

        q_dir = os.path.join(out_root, qid)
        ensure_dir(q_dir)

        # Resume support: skip runs already present
        existing = set(os.path.basename(p) for p in glob(os.path.join(q_dir, "run_*.json")))
        # Note: we store parsed outputs as run_XXX.json (not raw)
        print(f"{qid}: allowed_predicates={len(allowed_predicates)} existing_runs={len(existing)}")

        for i in range(1, runs + 1):
            run_id = f"run_{i:03d}.json"
            out_path = os.path.join(q_dir, run_id)
            raw_path = os.path.join(q_dir, f"run_{i:03d}_raw.json")

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                continue

            user_obj = {
                "question_id": qid,
                "subject_curie": subj_curie,
                "subject_label": subj_label,
                "object_curie": obj_curie,
                "object_label": obj_label,
                "allowed_predicates": allowed_predicates
            }

            raw, parsed = run_one(api_key, model, user_obj, temperature)

            # Enforce question_id in parsed
            parsed["question_id"] = qid

            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False, indent=2)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)

            triples = parsed.get("triples") or []
            ans = parsed.get("answer")
            print(f"{qid} {i:03d}/{runs}: answer={ans} triples={len(triples)} saved={out_path}")

            # small delay to be polite to the API
            time.sleep(0.15)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python step5_runs_100.py resolved_node_pairs.tsv robokop_triples_by_question gpt_runs gpt-5.2 100 0.7")
        raise SystemExit(2)

    resolved_tsv = sys.argv[1]
    robokop_per_q_dir = sys.argv[2]
    out_root = sys.argv[3]
    model = sys.argv[4]
    runs = int(sys.argv[5])
    temperature = float(sys.argv[6])

    main(resolved_tsv, robokop_per_q_dir, out_root, model, runs, temperature)