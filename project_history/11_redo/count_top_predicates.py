#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def detect_tsv_predicate_column(fieldnames: List[str]) -> str:
    """
    Tries to find the predicate column in a ROBOKOP triples TSV.
    Common names: predicate, biolink_predicate, predicate_id, edge_predicate, etc.
    """
    candidates = [
        "predicate",
        "biolink_predicate",
        "predicate_id",
        "edge_predicate",
        "pred",
        "relation",
    ]
    lower = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    # fallback: pick the first column that contains 'pred' or 'predicate'
    for c in fieldnames:
        cl = c.lower()
        if "predicate" in cl or cl == "pred" or cl.endswith("_pred"):
            return c
    raise ValueError(f"Could not detect predicate column in TSV header: {fieldnames}")


def read_robokop_predicates(tsv_path: Path) -> Counter:
    """
    Reads a TSV of ROBOKOP triples and counts predicates.
    Expects a header row.
    """
    counts: Counter = Counter()
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise ValueError("TSV appears to have no header / no columns.")
        pred_col = detect_tsv_predicate_column(r.fieldnames)
        for row in r:
            pred = (row.get(pred_col) or "").strip()
            if pred:
                counts[pred] += 1
    return counts


def extract_predicates_from_edge_keys(edge_keys: object) -> Iterable[str]:
    """
    edge_keys is expected to be:
      - [] OR
      - [[subj, predicate, obj], ...]
    """
    if not edge_keys:
        return []
    if not isinstance(edge_keys, list):
        return []
    out: List[str] = []
    for triple in edge_keys:
        if not isinstance(triple, list) or len(triple) < 3:
            continue
        pred = triple[1]
        if isinstance(pred, str) and pred.strip():
            out.append(pred.strip())
    return out


def read_gpt_predicates(jsonl_path: Path) -> Counter:
    """
    Reads GPT runs JSONL where each line is a JSON dict that includes `edge_keys`.
    Example line:
      {"question_id":"Q1", ... , "edge_keys":[["A","biolink:...","B"]], ...}
    """
    counts: Counter = Counter()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i} of {jsonl_path}: {e}") from e
            preds = extract_predicates_from_edge_keys(obj.get("edge_keys"))
            for p in preds:
                counts[p] += 1
    return counts


def write_counts_csv(counts: Counter, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counts.values())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["predicate", "count", "fraction"])
        for pred, c in counts.most_common():
            frac = (c / total) if total else 0.0
            w.writerow([pred, c, f"{frac:.6f}"])


def print_top20(title: str, counts: Counter) -> None:
    total = sum(counts.values())
    print(f"\n=== {title} (top 20) ===")
    print(f"Total predicate occurrences counted: {total}")
    print(f"{'rank':>4}  {'count':>8}  {'fraction':>10}  predicate")
    for idx, (pred, c) in enumerate(counts.most_common(20), start=1):
        frac = (c / total) if total else 0.0
        print(f"{idx:>4}  {c:>8}  {frac:>10.4%}  {pred}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robokop_triples_tsv", required=True, type=Path,
                    help="ROBOKOP triples TSV (with header and predicate column).")
    ap.add_argument("--gpt_runs_jsonl", required=True, type=Path,
                    help="GPT runs JSONL with edge_keys field.")
    ap.add_argument("--outdir", required=True, type=Path,
                    help="Output directory for CSV summaries.")
    args = ap.parse_args()

    robokop_counts = read_robokop_predicates(args.robokop_triples_tsv)
    gpt_counts = read_gpt_predicates(args.gpt_runs_jsonl)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Full distributions
    write_counts_csv(robokop_counts, args.outdir / "predicate_counts_robokop_full.csv")
    write_counts_csv(gpt_counts, args.outdir / "predicate_counts_gpt_full.csv")

    # Top 20 only
    top20_robokop = Counter(dict(robokop_counts.most_common(20)))
    top20_gpt = Counter(dict(gpt_counts.most_common(20)))
    write_counts_csv(top20_robokop, args.outdir / "top20_robokop_predicates.csv")
    write_counts_csv(top20_gpt, args.outdir / "top20_gpt_predicates.csv")

    # Print
    print_top20("ROBOKOP predicate distribution", robokop_counts)
    print_top20("GPT predicate distribution", gpt_counts)

    print(f"\n[OK] Wrote outputs to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
