#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row for row in r]


def parse_distribution(dist: str) -> Counter:
    c = Counter()
    if not dist:
        return c
    s = dist.replace("\t", ";").strip()
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        pred, count_s = part.rsplit(":", 1)
        try:
            c[pred] += int(count_s)
        except ValueError:
            pass
    return c


def plot_yes_rate(rows: List[Dict[str, str]], outdir: Path) -> None:
    qids = [r["question_id"] for r in rows]
    yes_rates = [float(r["gpt_yes_rate_valid"]) if r["gpt_yes_rate_valid"] else 0.0 for r in rows]

    plt.figure(figsize=(8, 4.5))
    plt.bar(qids, yes_rates)
    plt.ylim(0, 1.0)
    plt.ylabel("GPT YES rate (valid runs)")
    plt.xlabel("Question")
    plt.title("GPT YES rate per question")
    plt.tight_layout()
    plt.savefig(outdir / "yes_rate_per_question.png", dpi=300)
    plt.close()


def plot_predicate_stability(rows: List[Dict[str, str]], outdir: Path) -> None:
    qids, stabs = [], []
    for r in rows:
        if r["gpt_mode_answer"] != "YES":
            continue
        qids.append(r["question_id"])
        stabs.append(float(r["gpt_predicate_stability_among_yes"]) if r["gpt_predicate_stability_among_yes"] else 0.0)

    plt.figure(figsize=(8, 4.5))
    plt.bar(qids, stabs)
    plt.ylim(0, 1.0)
    plt.ylabel("Predicate stability among YES runs")
    plt.xlabel("Question (YES only)")
    plt.title("GPT predicate stability among YES runs")
    plt.tight_layout()
    plt.savefig(outdir / "predicate_stability_among_yes.png", dpi=300)
    plt.close()


def plot_predicate_distribution_stacked(rows: List[Dict[str, str]], outdir: Path, top_k: int = 8) -> None:
    per_q = {}
    global_counts = Counter()

    for r in rows:
        qid = r["question_id"]
        c = parse_distribution(r.get("gpt_yes_predicate_distribution", ""))
        per_q[qid] = c
        global_counts.update(c)

    top_preds = [p for p, _ in global_counts.most_common(top_k)]
    qids = [r["question_id"] for r in rows]

    bottoms = [0] * len(qids)
    plt.figure(figsize=(10, 5.5))
    for pred in top_preds:
        vals = [per_q.get(qid, Counter()).get(pred, 0) for qid in qids]
        plt.bar(qids, vals, bottom=bottoms, label=pred)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    plt.ylabel("Count among YES runs (n=20 per question)")
    plt.xlabel("Question")
    plt.title(f"GPT predicate distribution among YES runs (top {top_k} predicates)")
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / "predicate_distribution_stacked.png", dpi=300)
    plt.close()


def plot_overlap_vs_stability(rows: List[Dict[str, str]], outdir: Path) -> None:
    xs, ys, labels = [], [], []
    for r in rows:
        if r["gpt_mode_answer"] != "YES":
            continue
        stab = float(r["gpt_predicate_stability_among_yes"]) if r["gpt_predicate_stability_among_yes"] else 0.0
        any_in = (r.get("any_gpt_predicate_in_robokop") or "").strip()
        if any_in == "":
            continue
        xs.append(stab)
        ys.append(int(any_in))
        labels.append(r["question_id"])

    plt.figure(figsize=(7, 4.5))
    plt.scatter(xs, ys)
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.ylim(-0.1, 1.1)
    plt.xlim(0, 1.0)
    plt.ylabel("Any predicate overlap with ROBOKOP (0/1)")
    plt.xlabel("Predicate stability among YES runs")
    plt.title("Overlap vs stability (YES questions)")
    plt.tight_layout()
    plt.savefig(outdir / "overlap_vs_stability.png", dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 3 plots from step3_per_question.csv")
    ap.add_argument("--per_question", type=Path, required=True, help="Path to step3_per_question.csv")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory for plots")
    ap.add_argument("--top_k", type=int, default=8, help="Top-K predicates for stacked plot")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.per_question)
    rows = sorted(rows, key=lambda r: r["question_id"])

    plot_yes_rate(rows, args.outdir)
    plot_predicate_stability(rows, args.outdir)
    plot_predicate_distribution_stacked(rows, args.outdir, top_k=args.top_k)
    plot_overlap_vs_stability(rows, args.outdir)

    print("Wrote plots to:", args.outdir)


if __name__ == "__main__":
    main()
