#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from openai import OpenAI


# ============================================================
# Helpers
# ============================================================

def canonical_edge(curie1: str, predicate: str, curie2: str) -> Tuple[str, str, str]:
    a, b = sorted([curie1, curie2])
    return (a, predicate, b)


def stable_call_id(question_id: str, run_id: int) -> str:
    return hashlib.sha1(f"{question_id}::{run_id}".encode("utf-8")).hexdigest()


def load_allowed_predicates(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    preds = data["predicates"] if isinstance(data, dict) else data
    if not isinstance(preds, list) or not all(isinstance(p, str) for p in preds):
        raise ValueError("allowed_predicates.json must be a list of strings")
    return preds


@dataclass
class Pair:
    question_id: str
    curie1: str
    curie2: str


def load_pairs_tsv(path: Path) -> List[Pair]:
    pairs: List[Pair] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")

        required = {"question_id", "subject_curie", "object_curie"}
        if not required.issubset(set(r.fieldnames or [])):
            raise ValueError(
                f"Pairs TSV must contain columns {sorted(required)}; found {r.fieldnames}"
            )

        for row in r:
            pairs.append(
                Pair(
                    question_id=row["question_id"].strip(),
                    curie1=row["subject_curie"].strip(),
                    curie2=row["object_curie"].strip(),
                )
            )
    return pairs


def load_done_ids(jsonl_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not jsonl_path.exists():
        return done

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "call_id" in obj:
                    done.add(obj["call_id"])
            except json.JSONDecodeError:
                continue
    return done


# ============================================================
# OpenAI call (Responses API + Structured Outputs)
# ============================================================

async def call_model_structured(
    client: OpenAI,
    model: str,
    allowed_predicates: List[str],
    curie1: str,
    curie2: str,
    max_output_tokens: int,
    attempt_limit: int = 8,
) -> Dict[str, Any]:

    json_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "predicate": {
                            "type": "string",
                            "enum": allowed_predicates
                        },
                        "subject": {"type": "string"},
                        "object": {"type": "string"},
                    },
                    "required": ["predicate", "subject", "object"],
                },
            }
        },
        "required": ["edges"],
    }

    system_text = (
        "Extract biomedical relations between exactly two given nodes. "
        "Return only relations that plausibly hold. "
        "If none, return an empty edges list. "
        "Use only the allowed predicates and only the provided CURIEs."
    )

    user_text = f"Node A: {curie1}\nNode B: {curie2}\nReturn 0..N edges."

    base_delay = 0.8

    for attempt in range(1, attempt_limit + 1):
        try:
            resp = await asyncio.to_thread(
                client.responses.create,
                model=model,
                input=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "edge_set",
                        "schema": json_schema,   # ← THIS IS THE CRITICAL FIX
                    }
                },
                max_output_tokens=max_output_tokens,
                store=False,
            )

            obj = json.loads(resp.output_text)
            if "edges" not in obj or not isinstance(obj["edges"], list):
                raise ValueError("Invalid structured output")

            return obj

        except Exception as e:
            msg = str(e).lower()
            retryable = any(x in msg for x in ["rate limit", "429", "timeout", "502", "503", "504"])
            if not retryable or attempt == attempt_limit:
                raise

            await asyncio.sleep(base_delay * (2 ** (attempt - 1)) + random.random() * 0.25)

    raise RuntimeError("Unreachable")


# ============================================================
# Main runner
# ============================================================

async def run_all(
    pairs: List[Pair],
    allowed_predicates: List[str],
    out_jsonl: Path,
    model: str,
    runs_per_question: int,
    concurrency: int,
    max_output_tokens: int,
) -> None:

    client = OpenAI()
    done = load_done_ids(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def one_call(p: Pair, run_id: int) -> None:
        call_id = stable_call_id(p.question_id, run_id)
        if call_id in done:
            return

        async with sem:
            t0 = time.time()

            obj = await call_model_structured(
                client=client,
                model=model,
                allowed_predicates=allowed_predicates,
                curie1=p.curie1,
                curie2=p.curie2,
                max_output_tokens=max_output_tokens,
            )

            keys = sorted({
                canonical_edge(e["subject"], e["predicate"], e["object"])
                for e in obj["edges"]
                if e["subject"] in (p.curie1, p.curie2)
                and e["object"] in (p.curie1, p.curie2)
            })

            rec = {
                "call_id": call_id,
                "question_id": p.question_id,
                "run_id": run_id,
                "curie1": p.curie1,
                "curie2": p.curie2,
                "edge_keys": keys,
                "n_edges": len(keys),
                "model": model,
                "elapsed_s": round(time.time() - t0, 3),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            async with lock:
                with open(out_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                done.add(call_id)

    tasks = [
        asyncio.create_task(one_call(p, run_id))
        for p in pairs
        for run_id in range(1, runs_per_question + 1)
    ]

    completed = 0
    total = len(tasks)

    for fut in asyncio.as_completed(tasks):
        await fut
        completed += 1
        if completed % 200 == 0 or completed == total:
            print(f"Progress: {completed}/{total} ({completed/total:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_tsv", required=True, type=Path)
    ap.add_argument("--allowed_predicates", required=True, type=Path)
    ap.add_argument("--out_jsonl", required=True, type=Path)
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--runs_per_question", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--max_output_tokens", type=int, default=300)
    args = ap.parse_args()

    pairs = load_pairs_tsv(args.pairs_tsv)
    allowed = load_allowed_predicates(args.allowed_predicates)

    print(f"Loaded {len(pairs)} pairs")
    print(f"Loaded {len(allowed)} allowed predicates")
    print(f"Model={args.model}, runs={args.runs_per_question}, concurrency={args.concurrency}")

    asyncio.run(
        run_all(
            pairs=pairs,
            allowed_predicates=allowed,
            out_jsonl=args.out_jsonl,
            model=args.model,
            runs_per_question=args.runs_per_question,
            concurrency=args.concurrency,
            max_output_tokens=args.max_output_tokens,
        )
    )


if __name__ == "__main__":
    main()
