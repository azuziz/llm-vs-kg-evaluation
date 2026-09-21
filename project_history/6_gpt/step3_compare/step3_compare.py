#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def read_table(path: Path) -> List[Dict[str, str]]:
    """
    Reads CSV/TSV into list-of-dicts.
    Uses extension when possible; otherwise tries csv.Sniffer().
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)

        if path.suffix.lower() == ".tsv":
            delim = "\t"
        elif path.suffix.lower() == ".csv":
            delim = ","
        else:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
                delim = dialect.delimiter
            except Exception:
                delim = "\t"

        reader = csv.DictReader(f, delimiter=delim)
        rows = [r for r in reader]

    if not rows:
        raise ValueError(f"No rows read from {path}")
    return rows


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_distribution(dist: str) -> Counter:
    """
    Parses gpt yes_predicate_distribution:
      "biolink:foo:12;biolink:bar:8"
    Some environments may contain tabs; normalize to ';'.
    """
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
            continue
    return c


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load_robokop_predicates(robokop_triples: Path) -> Dict[str, Set[str]]:
    """
    Input: single robokop_triples.tsv with columns including question_id, predicate.
    Output: question_id -> set(predicates)
    """
    rows = read_table(robokop_triples)

    missing_cols = [c for c in ("question_id", "predicate") if c not in rows[0]]
    if missing_cols:
        raise ValueError(
            f"ROBOKOP triples file missing columns: {missing_cols}. Found: {list(rows[0].keys())}"
        )

    by_q: Dict[str, Set[str]] = {}
    for r in rows:
        qid = (r.get("question_id") or "").strip()
        pred = (r.get("predicate") or "").strip()
        if not qid:
            continue
        by_q.setdefault(qid, set())
        if pred:
            by_q[qid].add(pred)

    return by_q


def load_robokop_edge_counts(robokop_triples: Path) -> Dict[str, int]:
    rows = read_table(robokop_triples)
    by_q = Counter()
    for r in rows:
        qid = (r.get("question_id") or "").strip()
        if qid:
            by_q[qid] += 1
    return dict(by_q)


def load_gpt_aggregated(gpt_agg_csv: Path) -> Dict[str, Dict[str, str]]:
    rows = read_table(gpt_agg_csv)
    if "question_id" not in rows[0]:
        raise ValueError(f"GPT aggregated file missing 'question_id'. Found: {list(rows[0].keys())}")
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        qid = (r.get("question_id") or "").strip()
        if qid:
            out[qid] = r
    return out


def compute_step3(robokop_triples: Path, gpt_agg_csv: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    robokop_preds = load_robokop_predicates(robokop_triples)
    robokop_counts = load_robokop_edge_counts(robokop_triples)
    gpt = load_gpt_aggregated(gpt_agg_csv)

    qids = sorted(set(robokop_preds.keys()) | set(gpt.keys()))

    per_q_rows: List[Dict[str, str]] = []
    confusion = Counter()

    mode_hit_flags: List[int] = []
    any_overlap_flags: List[int] = []
    jaccs: List[float] = []
    answer_stabilities: List[float] = []
    pred_stabilities_yes: List[float] = []

    for qid in qids:
        r_pred_set = set(sorted(robokop_preds.get(qid, set())))
        r_edge_count = int(robokop_counts.get(qid, 0))
        r_has_edge = 1 if r_edge_count > 0 else 0

        g_row = gpt.get(qid, {})
        n_total = safe_int(g_row.get("n_total")) or 0
        n_valid = safe_int(g_row.get("n_valid")) or 0
        invalid_rate = safe_float(g_row.get("invalid_rate"))
        yes_rate = safe_float(g_row.get("yes_rate_valid"))
        mode_answer = (g_row.get("mode_answer") or "").strip()
        ans_stab = safe_float(g_row.get("answer_stability"))

        mode_pred = (g_row.get("mode_predicate_among_yes") or "").strip()
        pred_stab = safe_float(g_row.get("predicate_stability_among_yes"))
        dist_str = (g_row.get("yes_predicate_distribution") or "").strip()

        g_has_edge = 1 if mode_answer == "YES" else 0

        if r_has_edge == 1 and g_has_edge == 1:
            ex = "TP"
        elif r_has_edge == 0 and g_has_edge == 0:
            ex = "TN"
        elif r_has_edge == 0 and g_has_edge == 1:
            ex = "FP"
        else:
            ex = "FN"
        confusion[ex] += 1

        g_pred_counter = parse_distribution(dist_str)
        g_pred_set = set(g_pred_counter.keys())

        if g_has_edge == 0:
            mode_in = ""
            any_in = ""
        else:
            mode_in = "1" if (mode_pred and mode_pred in r_pred_set) else "0"
            any_in = "1" if ((g_pred_set & r_pred_set) != set()) else "0"

        jac = jaccard(g_pred_set, r_pred_set)

        jaccs.append(jac)
        if ans_stab is not None:
            answer_stabilities.append(ans_stab)

        if g_has_edge == 1:
            if pred_stab is not None:
                pred_stabilities_yes.append(pred_stab)
            if mode_in != "":
                mode_hit_flags.append(int(mode_in))
            if any_in != "":
                any_overlap_flags.append(int(any_in))

        per_q_rows.append({
            "question_id": qid,
            "robokop_edge_count": str(r_edge_count),
            "robokop_predicate_count": str(len(r_pred_set)),
            "robokop_predicates": ";".join(sorted(r_pred_set)),
            "gpt_n_total": str(n_total),
            "gpt_n_valid": str(n_valid),
            "gpt_invalid_rate": "" if invalid_rate is None else f"{invalid_rate:.4f}",
            "gpt_yes_rate_valid": "" if yes_rate is None else f"{yes_rate:.4f}",
            "gpt_mode_answer": mode_answer,
            "gpt_answer_stability": "" if ans_stab is None else f"{ans_stab:.4f}",
            "gpt_mode_predicate": mode_pred,
            "gpt_predicate_stability_among_yes": "" if pred_stab is None else f"{pred_stab:.4f}",
            "gpt_yes_predicate_distribution": dist_str,
            "robokop_has_edge": str(r_has_edge),
            "gpt_has_edge": str(g_has_edge),
            "existence_confusion": ex,
            "mode_predicate_in_robokop": mode_in,
            "any_gpt_predicate_in_robokop": any_in,
            "jaccard_predicates": f"{jac:.4f}",
            "notes": "",
        })

    per_q_path = outdir / "step3_per_question.csv"
    with open(per_q_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_q_rows[0].keys()))
        w.writeheader()
        w.writerows(per_q_rows)

    conf_path = outdir / "step3_confusion_matrix.csv"
    with open(conf_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["TP", "FP", "TN", "FN"])
        w.writerow([confusion["TP"], confusion["FP"], confusion["TN"], confusion["FN"]])

    TP, FP, TN, FN = confusion["TP"], confusion["FP"], confusion["TN"], confusion["FN"]
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total else float("nan")
    precision = TP / (TP + FP) if (TP + FP) else float("nan")
    recall = TP / (TP + FN) if (TP + FN) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")

    predicate_mode_hit_rate = (sum(mode_hit_flags) / len(mode_hit_flags)) if mode_hit_flags else float("nan")
    predicate_any_overlap_rate = (sum(any_overlap_flags) / len(any_overlap_flags)) if any_overlap_flags else float("nan")
    mean_jaccard = (sum(jaccs) / len(jaccs)) if jaccs else float("nan")
    mean_answer_stability = (sum(answer_stabilities) / len(answer_stabilities)) if answer_stabilities else float("nan")
    mean_pred_stability_yes = (sum(pred_stabilities_yes) / len(pred_stabilities_yes)) if pred_stabilities_yes else float("nan")

    summary_path = outdir / "step3_summary_metrics.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "existence_accuracy",
            "existence_precision",
            "existence_recall",
            "existence_f1",
            "predicate_mode_hit_rate",
            "predicate_any_overlap_rate",
            "mean_jaccard_predicates",
            "mean_answer_stability",
            "mean_predicate_stability_among_yes",
        ])
        w.writerow([
            f"{accuracy:.4f}" if not math.isnan(accuracy) else "",
            f"{precision:.4f}" if not math.isnan(precision) else "",
            f"{recall:.4f}" if not math.isnan(recall) else "",
            f"{f1:.4f}" if not math.isnan(f1) else "",
            f"{predicate_mode_hit_rate:.4f}" if not math.isnan(predicate_mode_hit_rate) else "",
            f"{predicate_any_overlap_rate:.4f}" if not math.isnan(predicate_any_overlap_rate) else "",
            f"{mean_jaccard:.4f}" if not math.isnan(mean_jaccard) else "",
            f"{mean_answer_stability:.4f}" if not math.isnan(mean_answer_stability) else "",
            f"{mean_pred_stability_yes:.4f}" if not math.isnan(mean_pred_stability_yes) else "",
        ])

    print("Wrote Step 3 outputs to:", outdir)
    print("-", per_q_path)
    print("-", conf_path)
    print("-", summary_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 3: Compare GPT aggregated outputs vs ROBOKOP triples.")
    ap.add_argument("--robokop", type=Path, required=True, help="Path to robokop_triples.tsv")
    ap.add_argument("--gpt_agg", type=Path, required=True, help="Path to gpt_aggregated.csv")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory (e.g., ~/gpt/6_gpt/statistical_output)")
    args = ap.parse_args()

    compute_step3(args.robokop, args.gpt_agg, args.outdir)


if __name__ == "__main__":
    main()
