#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

from openai import OpenAI

# ============================================================
# Paths (fixed to your repository layout)
# ============================================================
ROOT = Path(__file__).resolve().parents[1]   # /gpt/7_gpt
DATA = ROOT / "data"
OUT = ROOT / "out"

PAIRS_TSV = DATA / "resolved_node_pairs.tsv"
ALLOWED_PREDICATES_JSON = DATA / "allowed_predicates.json"
OUT_CSV = OUT / "gpt_runs_raw.csv"

# ============================================================
# Run configuration
# ============================================================
RUNS_PER_QUESTION = 20
MODEL = "gpt-5"          # GPT-5: DO NOT set temperature
SLEEP_SECONDS = 0.4
MAX_RETRIES = 3

client = OpenAI()

# ============================================================
# Helpers
# ============================================================
def canonical_edge(curie_a: str, predicate: str, curie_b: str) -> str:
    """Undirected canonical key: min(curie)|predicate|max(curie)"""
    a, b = sorted([curie_a.strip(), curie_b.strip()])
    return f"{a}|{predicate.strip()}|{b}"


def load_allowed_predicates() -> List[str]:
    with open(ALLOWED_PREDICATES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    preds = data.get("predicates", [])
    if not isinstance(preds, list) or not preds:
        raise ValueError("allowed_predicates.json missing non-empty 'predicates'")
    return preds


def load_questions() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(PAIRS_TSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError("resolved_node_pairs.tsv loaded 0 rows")
    return rows


def safe_parse_json(s: str) -> Dict:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


# ============================================================
# GPT call (JSON-only, predicates validated locally)
# ============================================================
def query_gpt_edges(
    subject_curie: str,
    subject_label: str,
    object_curie: str,
    object_label: str,
    allowed_predicates: List[str],
) -> Tuple[List[str], str]:
    """
    Returns:
      (sorted canonical edge keys, raw_model_json_string)

    Expected model output:
      {"edges":[{"predicate":"biolink:..."}]}
      OR {"edges":[]}
    """

    allowed_set = set(allowed_predicates)

    system_msg = (
        "You are a biomedical knowledge graph assistant. "
        "Return ONLY valid JSON. No explanations."
    )

    user_payload = {
        "task": "List plausible biomedical relationships between the two entities.",
        "subject": {"curie": subject_curie, "label": subject_label},
        "object": {"curie": object_curie, "label": object_label},
        "constraints": [
            "Return JSON only.",
            "edges may be empty.",
            "Each edge item must contain only: predicate.",
            "Only use predicates from allowed_predicates.",
            "If no relationship is known, return {\"edges\": []}."
        ],
        "output_schema": {"edges": [{"predicate": "biolink:..."}]},
        "allowed_predicates": allowed_predicates,
    }

    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
                response_format={"type": "json_object"},
            )

            raw = resp.choices[0].message.content or ""
            obj = safe_parse_json(raw)

            edges = obj.get("edges", [])
            if not isinstance(edges, list):
                edges = []

            out: Set[str] = set()
            for e in edges:
                if not isinstance(e, dict):
                    continue
                pred = e.get("predicate")
                if isinstance(pred, str) and pred in allowed_set:
                    out.add(canonical_edge(subject_curie, pred, object_curie))

            return sorted(out), raw

        except Exception as ex:
            last_error = f"{type(ex).__name__}: {ex}"
            if attempt < MAX_RETRIES:
                time.sleep(0.8 * attempt)

    return [], json.dumps({"error": last_error})


# ============================================================
# Main
# ============================================================
def main() -> None:
    OUT.mkdir(exist_ok=True)

    allowed_predicates = load_allowed_predicates()
    questions = load_questions()

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id",
            "run_id",
            "subject_curie",
            "object_curie",
            "edge_key",   # empty string => empty set
            "raw_json",
        ])

        for q in questions:
            qid = q["question_id"]
            subj_curie = q["subject_curie"]
            subj_label = q.get("subject_label", q.get("subject_text", ""))
            obj_curie = q["object_curie"]
            obj_label = q.get("object_label", q.get("object_text", ""))

            for run_id in range(1, RUNS_PER_QUESTION + 1):
                edge_keys, raw_json = query_gpt_edges(
                    subj_curie, subj_label,
                    obj_curie, obj_label,
                    allowed_predicates,
                )

                if not edge_keys:
                    w.writerow([qid, run_id, subj_curie, obj_curie, "", raw_json])
                else:
                    for ek in edge_keys:
                        w.writerow([qid, run_id, subj_curie, obj_curie, ek, raw_json])

                time.sleep(SLEEP_SECONDS)

    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
