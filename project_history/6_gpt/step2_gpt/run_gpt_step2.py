#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

# ----------------------------
# Config (override via env vars)
# ----------------------------
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")
N_RUNS = int(os.environ.get("N_RUNS", "20"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
SLEEP_S = float(os.environ.get("SLEEP_S", "0.2"))

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "gpt_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_PATH = (ROOT.parent / "allowed_predicates" / "allowed_predicates.json").resolve()
QUESTIONS_PATH = (ROOT / "resolved_node_pairs.tsv").resolve()
RUNS_OUT = (ROOT / "gpt_runs.csv").resolve()

client = OpenAI()


def load_allowed_predicates() -> Tuple[List[str], Dict[str, Any]]:
    d = json.loads(ALLOWED_PATH.read_text(encoding="utf-8"))
    preds = d.get("predicates")
    if not isinstance(preds, list) or not preds:
        raise ValueError("allowed_predicates.json has no 'predicates' list")
    preds = [p for p in preds if isinstance(p, str) and p.strip()]
    return sorted(set(preds)), d


def load_questions_tsv() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(QUESTIONS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError("resolved_node_pairs.tsv is empty or unreadable")
    return rows


def build_prompt(q: Dict[str, str], allowed: List[str], meta: Dict[str, Any]) -> Tuple[str, str]:
    allowed_block = "\n".join(allowed)

    # System: strict JSON-only, NO if unsure, predicate constraint
    system = (
        "You are an evaluator for biomedical relations. "
        "Return ONLY valid JSON (no markdown, no extra text). "
        "Task: decide whether there exists a plausible relation between the subject and object. "
        "If you are uncertain, answer NO. "
        "If YES, choose exactly one predicate from the allowed list. "
        "If NO, predicate must be null. "
        "Notes must be <=200 characters."
    )

    # User: provide all needed identifiers; ask for schema compliance
    user = (
        "EVALUATION CONTEXT\n"
        f"- Allowed predicates are derived from TRAPI /meta_knowledge_graph of: {meta.get('base_trapi_url')}\n"
        f"- Biolink version (service-declared): {meta.get('biolink_version')}\n"
        f"- TRAPI version (service-declared): {meta.get('trapi_version')}\n\n"
        "QUESTION\n"
        f"question_id: {q['question_id']}\n"
        f"subject_text: {q.get('subject_text','')}\n"
        f"subject_id: {q['subject_curie']}\n"
        f"subject_label: {q.get('subject_label','')}\n"
        f"object_text: {q.get('object_text','')}\n"
        f"object_id: {q['object_curie']}\n"
        f"object_label: {q.get('object_label','')}\n"
        f"notes: {q.get('notes','')}\n\n"
        "ALLOWED_PREDICATES (choose only from these)\n"
        f"{allowed_block}\n\n"
        "OUTPUT FORMAT (return ONLY this JSON object)\n"
        "{\n"
        '  "question_id": "...",\n'
        '  "answer": "YES" or "NO",\n'
        '  "predicate": "biolink:..." or null,\n'
        '  "subject_id": "CURIE",\n'
        '  "object_id": "CURIE",\n'
        '  "notes": "short justification <=200 chars"\n'
        "}\n\n"
        "CONSTRAINTS\n"
        "- If answer is NO, predicate must be null.\n"
        "- If answer is YES, predicate must be one of ALLOWED_PREDICATES.\n"
    )

    return system, user


def call_model(system: str, user: str) -> str:
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=TEMPERATURE,
    )
    out_text = resp.output_text  # SDK convenience accessor
    if not out_text:
        raise RuntimeError("Empty model output_text")
    return out_text.strip()


def parse_json_or_flag(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_invalid_json": True, "_raw": text}


def validate_record(rec: Dict[str, Any], allowed_set: Set[str], q: Dict[str, str]) -> Dict[str, Any]:
    """
    Enforces schema:
      - answer in {YES, NO}
      - predicate null if NO; allowed if YES
      - subject_id/object_id present
      - notes <=200 chars
    """
    required = ["question_id", "answer", "predicate", "subject_id", "object_id", "notes"]
    for k in required:
        if k not in rec:
            rec["_valid"] = False
            rec["_invalid_reason"] = f"missing_field:{k}"
            return rec

    ans = rec["answer"]
    if ans not in ("YES", "NO"):
        rec["_valid"] = False
        rec["_invalid_reason"] = "bad_answer"
        return rec

    pred = rec["predicate"]
    if ans == "NO":
        if pred is not None:
            rec["_valid"] = False
            rec["_invalid_reason"] = "no_requires_null_predicate"
            return rec
    else:  # YES
        if not isinstance(pred, str) or pred not in allowed_set:
            rec["_valid"] = False
            rec["_invalid_reason"] = "predicate_not_allowed"
            return rec

    # Ensure IDs are present; do not force equality with q curies (but we record both later)
    if not isinstance(rec["subject_id"], str) or not rec["subject_id"]:
        rec["_valid"] = False
        rec["_invalid_reason"] = "bad_subject_id"
        return rec
    if not isinstance(rec["object_id"], str) or not rec["object_id"]:
        rec["_valid"] = False
        rec["_invalid_reason"] = "bad_object_id"
        return rec

    notes = rec["notes"]
    if not isinstance(notes, str) or len(notes) > 200:
        rec["_valid"] = False
        rec["_invalid_reason"] = "bad_notes"
        return rec

    rec["_valid"] = True
    rec["_invalid_reason"] = ""
    return rec


def main() -> None:
    allowed, meta = load_allowed_predicates()
    allowed_set = set(allowed)
    questions = load_questions_tsv()

    with open(RUNS_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id", "run_id",
            "subject_curie_in", "object_curie_in",
            "answer", "predicate", "notes",
            "_valid", "_invalid_reason",
            "raw_file"
        ])

        for q in questions:
            qid = q["question_id"]
            subj_in = q["subject_curie"]
            obj_in = q["object_curie"]
            system, user = build_prompt(q, allowed, meta)

            for run_id in range(1, N_RUNS + 1):
                raw_path = RAW_DIR / f"{qid}_run{run_id:02d}.txt"
                try:
                    out_text = call_model(system, user)
                    raw_path.write_text(out_text, encoding="utf-8")

                    rec = parse_json_or_flag(out_text)
                    if rec.get("_invalid_json"):
                        w.writerow([qid, run_id, subj_in, obj_in, "", "", "", False, "invalid_json", str(raw_path)])
                    else:
                        rec = validate_record(rec, allowed_set, q)
                        w.writerow([
                            qid, run_id, subj_in, obj_in,
                            rec.get("answer", ""),
                            rec.get("predicate", ""),
                            rec.get("notes", ""),
                            rec.get("_valid", False),
                            rec.get("_invalid_reason", ""),
                            str(raw_path),
                        ])

                except Exception as e:
                    raw_path.write_text(json.dumps({"_error": str(e)}, indent=2), encoding="utf-8")
                    w.writerow([qid, run_id, subj_in, obj_in, "", "", "", False, f"exception:{type(e).__name__}", str(raw_path)])

                time.sleep(SLEEP_S)

    print(f"Done.\n- Runs CSV: {RUNS_OUT}\n- Raw outputs: {RAW_DIR}\n- Model: {MODEL}\n- N_RUNS: {N_RUNS}")


if __name__ == "__main__":
    main()
