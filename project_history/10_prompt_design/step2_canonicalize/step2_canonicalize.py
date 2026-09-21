#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

# v2/v3/v4 collectors wrote these CSVs
V234_RUNS = [
    ("v2", ROOT / "out" / "runs" / "v2" / "gpt_runs.csv"),
    ("v3", ROOT / "out" / "runs" / "v3" / "gpt_runs.csv"),
    ("v4", ROOT / "out" / "runs" / "v4" / "gpt_runs.csv"),
]

# v1 baseline: put/copy your JSONL here (self-contained)
# If you don't have it yet, locate it with:
#   find .. -maxdepth 5 -type f -name "*163q*x100*.jsonl" -o -name "*.jsonl" | head
V1_JSONL = ROOT / "data" / "gpt_edges_v1_163q_x100.jsonl"

OUT_EDGES = ROOT / "out" / "step2" / "gpt_edges_long.tsv"
OUT_RUNS = ROOT / "out" / "step2" / "gpt_runs_canonical.csv"


def canon_edge_key(predicate: str, subject: str, obj: str) -> str:
    return f"{predicate}|{subject}|{obj}"


def iter_v234_rows(prompt_version: str, path: Path) -> Iterable[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["prompt_version"] = prompt_version
            yield row


def parse_edges_from_edges_json(edges_json: str) -> List[Tuple[str, str, str]]:
    """
    Accepts:
      {"edges":[{"predicate":..., "subject":..., "object":...}, ...]}
    Returns list of (predicate, subject, object).
    """
    if not edges_json:
        return []
    obj = json.loads(edges_json)

    edges = obj.get("edges") if isinstance(obj, dict) else obj
    if edges is None or not isinstance(edges, list):
        return []

    out: List[Tuple[str, str, str]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        pred = e.get("predicate")
        subj = e.get("subject")
        objj = e.get("object")
        if isinstance(pred, str) and isinstance(subj, str) and isinstance(objj, str):
            out.append((pred, subj, objj))
    return out


def iter_v1_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    v1 baseline file format (JSONL), one JSON object per line.
    Expected keys:
      question_id, run_id, curie1, curie2, edge_keys, n_edges
    where edge_keys is either:
      [] OR [[subject, predicate, object], ...]
    """
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
            yield obj


def parse_edges_from_v1_edge_keys(obj: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    edge_keys = obj.get("edge_keys", [])
    if edge_keys is None:
        return []
    if not isinstance(edge_keys, list):
        return []

    out: List[Tuple[str, str, str]] = []
    for triple in edge_keys:
        if not isinstance(triple, (list, tuple)):
            continue
        if len(triple) != 3:
            continue
        subj, pred, objj = triple
        if isinstance(subj, str) and isinstance(pred, str) and isinstance(objj, str):
            out.append((pred, subj, objj))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=ROOT / "out" / "step2")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Validate v2/v3/v4 CSVs
    for pv, p in V234_RUNS:
        if not p.exists():
            raise FileNotFoundError(f"Missing {pv} runs: {p}")

    # Validate v1 JSONL
    if not V1_JSONL.exists():
        raise FileNotFoundError(
            f"Missing v1 jsonl file: {V1_JSONL}\n\n"
            f"Fix: locate your v1 JSONL and copy it here:\n"
            f"  cp /path/to/your/gpt_edges_163q_x100.jsonl {V1_JSONL}\n\n"
            f"To find it, run:\n"
            f"  find .. -maxdepth 6 -type f -name '*.jsonl' | head -n 50\n"
        )

    # Write outputs
    with open(OUT_EDGES, "w", newline="", encoding="utf-8") as f_edges, open(
        OUT_RUNS, "w", newline="", encoding="utf-8"
    ) as f_runs:
        w_edges = csv.writer(f_edges, delimiter="\t")
        w_edges.writerow(["prompt_version", "question_id", "run_id", "predicate", "subject", "object", "edge_key"])

        w_runs = csv.writer(f_runs)
        w_runs.writerow(["prompt_version", "question_id", "run_id", "n_edges"])

        # ---- v2/v3/v4 ----
        for pv, path in V234_RUNS:
            for row in iter_v234_rows(pv, path):
                qid = row["question_id"]
                rid = str(row["run_id"])
                status = row.get("status", "ok")
                if status != "ok":
                    w_runs.writerow([pv, qid, rid, 0])
                    continue

                edges_json = row.get("edges_json", "")
                try:
                    edges = parse_edges_from_edges_json(edges_json)
                except Exception:
                    edges = []

                for pred, subj, objj in edges:
                    ek = canon_edge_key(pred, subj, objj)
                    w_edges.writerow([pv, qid, rid, pred, subj, objj, ek])

                w_runs.writerow([pv, qid, rid, len(edges)])

        # ---- v1 baseline (JSONL) ----
        for obj in iter_v1_jsonl(V1_JSONL):
            pv = "v1"
            qid = str(obj.get("question_id"))
            rid = str(obj.get("run_id"))

            try:
                edges = parse_edges_from_v1_edge_keys(obj)
            except Exception:
                edges = []

            for pred, subj, objj in edges:
                ek = canon_edge_key(pred, subj, objj)
                w_edges.writerow([pv, qid, rid, pred, subj, objj, ek])

            w_runs.writerow([pv, qid, rid, len(edges)])

    print(f"[OK] wrote {OUT_EDGES}")
    print(f"[OK] wrote {OUT_RUNS}")


if __name__ == "__main__":
    main()
