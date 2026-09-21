#!/usr/bin/env python3
import csv
import json
import os
import sys
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-5.1"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

SYSTEM_PROMPT = """You are a biomedical query compiler for the ROBOKOP knowledge graph.

Your task:
- Input: a natural language biomedical question from the user.
- Output: a JSON object describing a TRAPI-style query graph that could be sent to a Translator Reasoner API, but WITHOUT any ids/CURIEs yet.
- Use Biolink Model categories and predicates.

Rules:
1. Output ONLY valid JSON, no markdown, no explanation.
2. JSON schema MUST be:

{
  "question_id": "<string, optional>",
  "natural_question": "<copy of the user question>",
  "nodes": {
    "n0": {
      "label": "<string>",
      "categories": ["<biolink category>"],
      "role": "subject" | "object" | "intermediate"
    },
    "n1": {
      "label": "<string>",
      "categories": ["<biolink category>"],
      "role": "subject" | "object" | "intermediate"
    }
  },
  "edges": {
    "e0": {
      "subject": "n0",
      "object": "n1",
      "predicates": ["<biolink predicate>"]
    }
  }
}

3. DO NOT include ids/CURIEs — only human-readable labels and Biolink categories/predicates.
4. Prefer these categories when appropriate:
   - biolink:Gene
   - biolink:Disease
   - biolink:PhenotypicFeature
   - biolink:Pathway
   - biolink:BiologicalProcessOrActivity
5. Prefer these predicates when appropriate:
   - biolink:gene_associated_with_condition (Gene → Disease)
   - biolink:has_phenotype (Disease → PhenotypicFeature)
   - biolink:participates_in (Gene → Pathway/Process)
   - biolink:related_to (fallback, generic association)
6. Use 1-hop patterns unless the question clearly needs an intermediate node.
7. Use node ids "n0", "n1", "n2", ... and edge ids "e0", "e1", ...
"""

def call_gpt(question_id, question_text):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": question_text
            }
        ],
    }
    r = requests.post(OPENAI_URL, headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    # content is already JSON string thanks to response_format
    qgraph = json.loads(content)
    # add question_id if missing
    if "question_id" not in qgraph or not qgraph["question_id"]:
        qgraph["question_id"] = question_id
    return qgraph

def main():
    if OPENAI_API_KEY is None:
        print("ERROR: OPENAI_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) != 2:
        print("Usage: python a_questions_to_querygraphs.py questions.tsv", file=sys.stderr)
        sys.exit(1)

    questions_path = sys.argv[1]
    out_dir = "queries"
    os.makedirs(out_dir, exist_ok=True)

    with open(questions_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["id"].strip()
            qtext = row["question"].strip()
            print(f"Processing {qid}: {qtext}")
            qgraph = call_gpt(qid, qtext)
            out_path = os.path.join(out_dir, f"{qid}_query.json")
            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump(qgraph, out_f, indent=2)
            print(f"  -> wrote {out_path}")

if __name__ == "__main__":
    main()
