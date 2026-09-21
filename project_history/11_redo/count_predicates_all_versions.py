#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


# ---------------------------
# ROBOKOP
# ---------------------------

def detect_predicate_column(cols):
    for c in cols:
        cl = c.lower()
        if "predicate" in cl or cl == "pred":
            return c
    raise ValueError(f"No predicate column found in {cols}")


def read_robokop(tsv: Path) -> Counter:
    counts = Counter()
    with tsv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        pred_col = detect_predicate_column(r.fieldnames)
        for row in r:
            p = (row.get(pred_col) or "").strip()
            if p:
                counts[p] += 1
    return counts


# ---------------------------
# GPT v1 (JSONL edge_keys)
# ---------------------------

def read_gpt_v1(jsonl: Path) -> Counter:
    counts = Counter()
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            for ek in obj.get("edge_keys", []):
                if isinstance(ek, list) and len(ek) >= 3:
                    counts[ek[1]] += 1
    return counts


# ---------------------------
# GPT v2 / v3 / v4 (CSV edges_json)
# ---------------------------

def read_gpt_csv(csv_path: Path) -> Counter:
    counts = Counter()
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status") != "ok":
                continue
            raw = row.get("edges_json", "").strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for e in obj.get("edges", []):
                p = e.get("predicate")
                if p:
                    counts[p] += 1
    return counts


# ---------------------------
# Output helpers
# ---------------------------

def write_counts(counter: Counter, out: Path):
    total = sum(counter.values())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["predicate", "count", "fraction"])
        for p, c in counter.most_common():
            w.writerow([p, c, c / total if total else 0.0])


def print_top(counter: Counter, label: str, n=10):
    total = sum(counter.values())
    print(f"\n=== {label} (top {n}) ===")
    for i, (p, c) in enumerate(counter.most_common(n), 1):
        print(f"{i:>2}. {p:45s} {c:>6}  {c/total:6.2%}")


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robokop", required=True, type=Path)
    ap.add_argument("--gpt_v1", type=Path)
    ap.add_argument("--gpt_v2", type=Path)
    ap.add_argument("--gpt_v3", type=Path)
    ap.add_argument("--gpt_v4", type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    rob = read_robokop(args.robokop)
    write_counts(rob, args.outdir / "predicates_robokop.csv")
    print_top(rob, "ROBOKOP")

    if args.gpt_v1:
        c = read_gpt_v1(args.gpt_v1)
        write_counts(c, args.outdir / "predicates_gpt_v1.csv")
        print_top(c, "GPT v1")

    if args.gpt_v2:
        c = read_gpt_csv(args.gpt_v2)
        write_counts(c, args.outdir / "predicates_gpt_v2.csv")
        print_top(c, "GPT v2")

    if args.gpt_v3:
        c = read_gpt_csv(args.gpt_v3)
        write_counts(c, args.outdir / "predicates_gpt_v3.csv")
        print_top(c, "GPT v3")

    if args.gpt_v4:
        c = read_gpt_csv(args.gpt_v4)
        write_counts(c, args.outdir / "predicates_gpt_v4.csv")
        print_top(c, "GPT v4")


if __name__ == "__main__":
    main()
