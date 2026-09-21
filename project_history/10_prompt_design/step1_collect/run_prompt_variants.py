#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]

DATA_PAIRS = ROOT / "data" / "robokop_pairs_filtered_dedup.tsv"
DATA_ALLOWED = ROOT / "data" / "allowed_predicates.json"

CONFIG = ROOT / "config" / "experiment.yaml"
OUT_RUNS = ROOT / "out" / "runs"


@dataclass
class PairRow:
    question_id: str
    subject_curie: str
    object_curie: str


def load_pairs(path: Path) -> List[PairRow]:
    rows: List[PairRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(
                PairRow(
                    question_id=row["question_id"],
                    subject_curie=row["subject_curie"],
                    object_curie=row["object_curie"],
                )
            )
    if not rows:
        raise ValueError(f"No pairs loaded from {path}")
    return rows


def load_allowed_predicates(path: Path) -> List[str]:
    """
    Supports common formats:
      1) ["biolink:...", ...]
      2) {"allowed_predicates":[...]}
      3) {"predicates":[...]}
      4) {"enum":[...]} or {"values":[...]}
      5) {"allowed_predicates":{"enum":[...]}}  (nested)
      6) {"data":[...]} / {"items":[...]} etc (if list of strings)
    """
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)

    def as_list_str(x: Any) -> List[str] | None:
        if isinstance(x, list) and all(isinstance(i, str) for i in x):
            return x
        return None

    got = as_list_str(obj)
    if got is not None:
        return got

    if isinstance(obj, dict):
        candidate_keys = ["allowed_predicates", "predicates", "enum", "values", "data", "items", "result"]
        for k in candidate_keys:
            if k in obj:
                v = obj[k]
                got = as_list_str(v)
                if got is not None:
                    return got
                if isinstance(v, dict):
                    for kk in candidate_keys:
                        if kk in v:
                            got = as_list_str(v[kk])
                            if got is not None:
                                return got

    raise ValueError(
        f"Could not extract allowed predicates as list[str] from {path}. "
        f"Top-level type={type(obj).__name__}, keys={list(obj.keys()) if isinstance(obj, dict) else 'NA'}"
    )


def build_schema(allowed_predicates: List[str]) -> Dict[str, Any]:
    # Not used for enforcement in this SDK path, but kept for compatibility / future use.
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "predicate": {"type": "string", "enum": allowed_predicates},
                        "subject": {"type": "string"},
                        "object": {"type": "string"},
                    },
                    "required": ["predicate", "subject", "object"],
                },
            }
        },
        "required": ["edges"],
    }


def call_gpt_structured(
    client: OpenAI,
    model: str,
    prompt_text: str,
    subject_curie: str,
    object_curie: str,
    allowed_predicates: List[str],
    max_output_tokens: int,
    temperature: float,
    attempt_limit: int = 8,
) -> Dict[str, Any]:
    """
    Returns JSON: {"edges":[{"predicate":..., "subject":..., "object":...}, ...]}

    Notes:
    - openai==2.16.0 + gpt-5.2: use chat.completions + response_format=json_object
    - gpt-5.2: use max_completion_tokens (NOT max_tokens)
    - We do a lightweight validation and auto-filter invalid predicates.
    """
    allowed_str = ", ".join(allowed_predicates)

    user_msg = (
        f"{prompt_text.strip()}\n\n"
        f"Nodes:\n"
        f"- subject: {subject_curie}\n"
        f"- object: {object_curie}\n\n"
        f"Output format (strict JSON object):\n"
        f'{{"edges":[{{"predicate":"<one of allowed_predicates>","subject":"<CURIE>","object":"<CURIE>"}}]}}\n\n'
        f"Constraints:\n"
        f"- Use only these allowed_predicates: [{allowed_str}]\n"
        f"- subject/object must be exactly the provided CURIEs.\n"
        f"- If no relation, return: {{\"edges\":[]}}\n"
    )

    last_err: Exception | None = None

    for attempt in range(1, attempt_limit + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_msg}],
                temperature=temperature,
                max_completion_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            )

            txt = resp.choices[0].message.content or ""
            out = json.loads(txt)

            # Normalize shape
            if not isinstance(out, dict):
                raise ValueError(f"Model returned non-object JSON: {type(out).__name__}")
            edges = out.get("edges", [])
            if edges is None:
                edges = []
            if not isinstance(edges, list):
                raise ValueError("JSON key 'edges' must be a list")

            # Lightweight validation + filtering
            filtered: List[Dict[str, str]] = []
            allowed_set = set(allowed_predicates)

            for e in edges:
                if not isinstance(e, dict):
                    continue
                pred = e.get("predicate")
                subj = e.get("subject")
                obj = e.get("object")
                if not (isinstance(pred, str) and isinstance(subj, str) and isinstance(obj, str)):
                    continue
                if pred not in allowed_set:
                    continue
                # enforce provided CURIEs only (either direction allowed)
                if {subj, obj} != {subject_curie, object_curie}:
                    continue
                filtered.append({"predicate": pred, "subject": subj, "object": obj})

            return {"edges": filtered}

        except TypeError:
            # Programmer error (wrong kwargs) -> do not retry
            raise
        except Exception as e:
            last_err = e
            msg = repr(e)
            # Deterministic "bad request" style errors should not be retried
            if "invalid_request_error" in msg or "Unsupported parameter" in msg:
                raise

            if attempt == attempt_limit:
                raise

            sleep_s = min(60.0, (2 ** (attempt - 1)) + random.random())
            print(f"[RETRY] attempt {attempt}/{attempt_limit} error={msg} sleeping {sleep_s:.1f}s", flush=True)
            time.sleep(sleep_s)

    raise RuntimeError(f"Unreachable; last_err={repr(last_err)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--limit_pairs", type=int, default=0, help="0 = all pairs, else first N pairs")
    ap.add_argument("--limit_runs", type=int, default=0, help="0 = config n_runs_per_prompt, else override runs")
    ap.add_argument("--dry_run", action="store_true", help="Validate prompts/config/data only")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = cfg["model"]["name"]
    temperature = float(cfg["model"]["temperature"])
    max_output_tokens = int(cfg["model"]["max_output_tokens"])
    n_runs_cfg = int(cfg["n_runs_per_prompt"])
    n_runs = args.limit_runs if args.limit_runs > 0 else n_runs_cfg

    prompts = cfg["prompts"]

    # Validate inputs
    assert DATA_PAIRS.exists(), f"Missing {DATA_PAIRS}"
    assert DATA_ALLOWED.exists(), f"Missing {DATA_ALLOWED}"
    for p in prompts:
        pfile = ROOT / p["file"]
        assert pfile.exists(), f"Missing prompt file: {pfile}"

    pairs = load_pairs(DATA_PAIRS)
    if args.limit_pairs > 0:
        pairs = pairs[: args.limit_pairs]

    allowed_predicates = load_allowed_predicates(DATA_ALLOWED)
    _schema = build_schema(allowed_predicates)  # kept for compatibility / future, not used

    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Model: {model} temp={temperature} max_output_tokens={max_output_tokens}")
    print(f"Pairs: {len(pairs)}")
    print(f"Runs per prompt: {n_runs}")
    print(f"Prompts: {[p['id'] for p in prompts]}")

    if args.dry_run:
        print("\nDry-run OK (validated inputs).")
        return

    # OpenAI client (timeout helps prevent “silent hangs”)
    client = OpenAI(timeout=60.0)

    for p in prompts:
        pid = p["id"]
        pfile = ROOT / p["file"]
        prompt_text = pfile.read_text(encoding="utf-8")

        outdir = OUT_RUNS / pid
        outdir.mkdir(parents=True, exist_ok=True)
        out_csv = outdir / "gpt_runs.csv"

        write_header = not out_csv.exists()
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "prompt_version",
                        "question_id",
                        "run_id",
                        "subject_curie",
                        "object_curie",
                        "edges_json",
                        "status",
                        "error",
                    ]
                )

            total = len(pairs) * n_runs
            done = 0

            for run_id in range(1, n_runs + 1):
                for pr in pairs:
                    done += 1
                    tag = f"{pid} run={run_id}/{n_runs} item={done}/{total} q={pr.question_id}"
                    print(f"[CALL] {tag}", flush=True)
                    try:
                        out = call_gpt_structured(
                            client=client,
                            model=model,
                            prompt_text=prompt_text,
                            subject_curie=pr.subject_curie,
                            object_curie=pr.object_curie,
                            allowed_predicates=allowed_predicates,
                            max_output_tokens=max_output_tokens,
                            temperature=temperature,
                        )
                        w.writerow(
                            [
                                pid,
                                pr.question_id,
                                run_id,
                                pr.subject_curie,
                                pr.object_curie,
                                json.dumps(out, ensure_ascii=False),
                                "ok",
                                "",
                            ]
                        )
                        f.flush()
                        print(f"[OK]  {tag}", flush=True)
                    except Exception as e:
                        w.writerow(
                            [
                                pid,
                                pr.question_id,
                                run_id,
                                pr.subject_curie,
                                pr.object_curie,
                                "",
                                "error",
                                repr(e),
                            ]
                        )
                        f.flush()
                        print(f"[ERR] {tag} -> {repr(e)}", flush=True)

        print(f"[DONE] {pid} -> {out_csv}")


if __name__ == "__main__":
    main()
