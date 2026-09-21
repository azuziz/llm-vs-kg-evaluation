import csv
import json
import os
import sys
import time
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
  "You are a biomedical knowledge extraction system.\n"
  "Task: decide whether there is a DIRECT (one-hop) relationship between the two given nodes.\n"
  "Output MUST match the JSON schema exactly.\n"
  "Rules:\n"
  "1) Use ONLY the provided subject_curie and object_curie as endpoints in triples.\n"
  "2) If you output any triple, answer must be YES.\n"
  "3) If you believe no direct relationship exists, answer must be NO and triples must be [].\n"
  "4) Predicates must be Biolink-style strings like 'biolink:interacts_with', 'biolink:related_to', etc.\n"
  "5) Do NOT invent other CURIEs, genes, diseases, or intermediate nodes."
)

def call_openai(api_key: str, model: str, payload_obj: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    r = requests.post(OPENAI_URL, headers=headers, json=payload_obj, timeout=180)
    r.raise_for_status()
    return r.json()

def extract_output_text(resp_json: dict) -> str:
    # Responses API: output[0].content[0].text is typical
    return resp_json["output"][0]["content"][0]["text"]

def main(resolved_tsv: str, out_dir: str, out_tsv: str, model: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")

    os.makedirs(out_dir, exist_ok=True)

    with open(resolved_tsv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    out_rows = []

    for r in rows:
        qid = r["question_id"]
        subj_curie = r["subject_curie"].strip()
        obj_curie  = r["object_curie"].strip()

        subj_label = r.get("subject_label","").strip()
        obj_label  = r.get("object_label","").strip()

        # Build user content
        user_content = {
            "question_id": qid,
            "subject_curie": subj_curie,
            "subject_label": subj_label,
            "object_curie": obj_curie,
            "object_label": obj_label
        }

        payload = {
            "model": model,
            "temperature": 0,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_content)}
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

        resp = call_openai(api_key, model, payload)

        raw_path = os.path.join(out_dir, f"{qid}.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(resp, f, ensure_ascii=False, indent=2)

        out_text = extract_output_text(resp)
        parsed = json.loads(out_text)

        # basic normalization
        parsed["question_id"] = qid

        # Save summary row
        triples = parsed.get("triples") or []
        out_rows.append({
            "question_id": qid,
            "answer": parsed.get("answer",""),
            "triple_count": str(len(triples)),
            "raw_file": raw_path
        })

        print(f"{qid}: answer={parsed.get('answer')} triples={len(triples)} saved={raw_path}")

        time.sleep(0.2)

    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_id","answer","triple_count","raw_file"], delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote: {out_tsv}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python step5_gpt_query.py resolved_node_pairs.tsv gpt_raw gpt_labels.tsv gpt-5.2")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
