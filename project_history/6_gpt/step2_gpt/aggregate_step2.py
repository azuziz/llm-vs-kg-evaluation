#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
RUNS_IN = ROOT / "gpt_runs.csv"
OUT = ROOT / "gpt_aggregated.csv"


def main() -> None:
    rows: List[Dict[str, str]] = []
    with open(RUNS_IN, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    by_q = defaultdict(list)
    for row in rows:
        by_q[row["question_id"]].append(row)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id",
            "n_total", "n_valid", "invalid_rate",
            "yes_rate_valid", "mode_answer", "answer_stability",
            "mode_predicate_among_yes", "predicate_stability_among_yes",
            "yes_predicate_distribution"
        ])

        for qid, items in sorted(by_q.items()):
            n_total = len(items)
            valid = [x for x in items if x["_valid"] == "True"]
            n_valid = len(valid)
            invalid_rate = (n_total - n_valid) / n_total if n_total else 0.0

            if n_valid == 0:
                w.writerow([qid, n_total, 0, invalid_rate, "", "", "", "", "", ""])
                continue

            answers = [x["answer"] for x in valid]
            c_ans = Counter(answers)
            mode_answer, mode_answer_ct = c_ans.most_common(1)[0]
            answer_stability = mode_answer_ct / n_valid

            yes = [x for x in valid if x["answer"] == "YES"]
            yes_rate_valid = len(yes) / n_valid

            if yes:
                preds = [x["predicate"] for x in yes]
                c_pred = Counter(preds)
                mode_pred, mode_pred_ct = c_pred.most_common(1)[0]
                pred_stability = mode_pred_ct / len(yes)

                # compact distribution string
                dist = ";".join([f"{p}:{c_pred[p]}" for p, _ in c_pred.most_common()])
            else:
                mode_pred, pred_stability, dist = "", "", ""

            w.writerow([
                qid,
                n_total, n_valid, f"{invalid_rate:.4f}",
                f"{yes_rate_valid:.4f}", mode_answer, f"{answer_stability:.4f}",
                mode_pred, (f"{pred_stability:.4f}" if yes else ""),
                dist
            ])

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
