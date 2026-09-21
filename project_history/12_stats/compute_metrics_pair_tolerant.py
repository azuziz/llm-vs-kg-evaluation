#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, FrozenSet

# ----------------------------
# Helpers: robust TSV/CSV read
# ----------------------------

def sniff_delimiter(path: str) -> str:
    # simple heuristic: if .tsv -> tab else comma
    if path.lower().endswith(".tsv"):
        return "\t"
    return ","

def read_table(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    delim = sniff_delimiter(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        return reader.fieldnames or [], rows

def ensure_cols(cols: List[str], required: List[str], context: str) -> None:
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(
            f"{context}: missing required columns: {missing}\n"
            f"Found columns: {cols}"
        )

# ----------------------------
# Canonicalization logic
# ----------------------------

def load_inverse_map(path: str) -> Dict[str, str]:
    """
    inverse_predicates.tsv expected columns:
      - predicate
      - inverse_predicate
    (names can vary slightly; we try a few)
    """
    cols, rows = read_table(path)
    # accept alternative column names
    cand_pred = ["predicate", "pred", "p"]
    cand_inv  = ["inverse_predicate", "inverse", "inv", "inverse_pred"]

    pred_col = next((c for c in cand_pred if c in cols), None)
    inv_col  = next((c for c in cand_inv  if c in cols), None)
    if not pred_col or not inv_col:
        raise ValueError(
            f"inverse map file {path}: need columns like "
            f"{cand_pred} and {cand_inv}. Found: {cols}"
        )

    m: Dict[str, str] = {}
    for r in rows:
        p = (r.get(pred_col) or "").strip()
        q = (r.get(inv_col) or "").strip()
        if not p or not q:
            continue
        m[p] = q
    return m

def pred_pair_rep(pred: str, inv_map: Dict[str, str]) -> str:
    """
    Collapse predicate + its inverse into a single representative string.
    We choose lexicographically smallest of {pred, inverse(pred)} if known.
    If inverse isn't known, rep is pred itself.
    """
    pred = pred.strip()
    inv = inv_map.get(pred)
    if not inv:
        # also handle inverse_map defined in the other direction
        # e.g. if pred is inverse of something else
        for k, v in inv_map.items():
            if v == pred:
                inv = k
                break
    if inv:
        return min(pred, inv)
    return pred

def undirected_nodepair(s: str, o: str) -> FrozenSet[str]:
    return frozenset([s.strip(), o.strip()])

EdgeKey = Tuple[FrozenSet[str], str]  # (undirected nodepair, predicate-pair representative)

# ----------------------------
# Metrics
# ----------------------------

@dataclass
class Metrics:
    pred_n: int
    gold_n: int
    tp: int
    precision: float
    recall: float
    jaccard: float
    f1: float

def compute_set_metrics(pred: Set[EdgeKey], gold: Set[EdgeKey]) -> Metrics:
    tp = len(pred & gold)
    pred_n = len(pred)
    gold_n = len(gold)

    precision = tp / pred_n if pred_n else 0.0
    recall    = tp / gold_n if gold_n else 0.0
    union     = len(pred | gold)
    jaccard   = tp / union if union else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return Metrics(pred_n=pred_n, gold_n=gold_n, tp=tp,
                   precision=precision, recall=recall, jaccard=jaccard, f1=f1)

# ----------------------------
# Parse ROBOKOP gold triples
# ----------------------------

def load_robokop_gold(
    robokop_path: str,
    inv_map: Dict[str, str],
) -> Dict[str, Set[EdgeKey]]:
    """
    Expected robokop_triples.tsv to contain at least:
      question_id, subject_curie, predicate, object_curie
    We try to infer column names if slightly different.
    Returns: gold_by_qid[question_id] = set(EdgeKey)
    """
    cols, rows = read_table(robokop_path)

    # candidate names
    qcols = ["question_id", "qid", "question"]
    scols = ["subject_curie", "subject", "subj", "s"]
    ocols = ["object_curie", "object", "obj", "o"]
    pcols = ["predicate", "pred"]

    qcol = next((c for c in qcols if c in cols), None)
    scol = next((c for c in scols if c in cols), None)
    ocol = next((c for c in ocols if c in cols), None)
    pcol = next((c for c in pcols if c in cols), None)

    if not all([qcol, scol, ocol, pcol]):
        raise ValueError(
            f"robokop file {robokop_path}: expected columns like "
            f"{qcols}, {scols}, {pcols}, {ocols}. Found: {cols}"
        )

    gold_by_qid: Dict[str, Set[EdgeKey]] = defaultdict(set)
    for r in rows:
        qid = (r.get(qcol) or "").strip()
        s   = (r.get(scol) or "").strip()
        o   = (r.get(ocol) or "").strip()
        p   = (r.get(pcol) or "").strip()
        if not qid or not s or not o or not p:
            continue

        rep = pred_pair_rep(p, inv_map)
        key: EdgeKey = (undirected_nodepair(s, o), rep)
        gold_by_qid[qid].add(key)

    return gold_by_qid

# ----------------------------
# Parse GPT runs (CSV)
# ----------------------------

def extract_pred_edges_from_edges_json(
    edges_json_str: str,
    inv_map: Dict[str, str],
) -> Set[EdgeKey]:
    """
    edges_json looks like:
      {"edges":[{"predicate": "...", "subject": "...", "object": "..."}, ...]}
    We ignore direction by undirected nodepair().
    We collapse predicate into pair rep via inverse map.
    """
    out: Set[EdgeKey] = set()
    if not edges_json_str:
        return out

    try:
        obj = json.loads(edges_json_str)
    except Exception:
        return out

    edges = obj.get("edges", [])
    if not isinstance(edges, list):
        return out

    for e in edges:
        if not isinstance(e, dict):
            continue
        p = (e.get("predicate") or "").strip()
        s = (e.get("subject") or "").strip()
        o = (e.get("object") or "").strip()
        if not p or not s or not o:
            continue
        rep = pred_pair_rep(p, inv_map)
        out.add((undirected_nodepair(s, o), rep))
    return out

def iter_prompt_runs(prompt_csv: str) -> Iterable[Dict[str, str]]:
    cols, rows = read_table(prompt_csv)
    ensure_cols(cols, ["question_id", "run_id", "edges_json", "status"], f"gpt runs {prompt_csv}")
    for r in rows:
        yield r

# ----------------------------
# Aggregations
# ----------------------------

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def write_csv(path: str, header: List[str], rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing v2/v3/v4 subfolders")
    ap.add_argument("--robokop", required=True, help="robokop_triples.tsv")
    ap.add_argument("--prompts", nargs="+", required=True, help="prompt versions e.g. v2 v3 v4")
    ap.add_argument("--inverse_map", required=True, help="inverse_predicates.tsv")
    ap.add_argument("--outdir", required=True, help="output directory")
    args = ap.parse_args()

    inv_map = load_inverse_map(args.inverse_map)
    gold_by_qid = load_robokop_gold(args.robokop, inv_map)

    run_rows: List[Dict[str, object]] = []

    # compute per-run metrics
    for pv in args.prompts:
        prompt_csv = os.path.join(args.root, pv, "gpt_runs.csv")
        if not os.path.exists(prompt_csv):
            raise FileNotFoundError(f"Missing: {prompt_csv}")

        for r in iter_prompt_runs(prompt_csv):
            status = (r.get("status") or "").strip()
            if status != "ok":
                continue

            qid = (r.get("question_id") or "").strip()
            run_id = (r.get("run_id") or "").strip()
            edges_json_str = r.get("edges_json") or ""

            gold = gold_by_qid.get(qid, set())
            pred = extract_pred_edges_from_edges_json(edges_json_str, inv_map)

            m = compute_set_metrics(pred, gold)
            run_rows.append({
                "prompt_version": pv,
                "question_id": qid,
                "run_id": run_id,
                "pred_n": m.pred_n,
                "gold_n": m.gold_n,
                "tp": m.tp,
                "precision": m.precision,
                "recall": m.recall,
                "jaccard": m.jaccard,
                "f1": m.f1,
            })

    out_run = os.path.join(args.outdir, "run_metrics.csv")
    write_csv(out_run,
              ["prompt_version","question_id","run_id","pred_n","gold_n","tp","precision","recall","jaccard","f1"],
              run_rows)

    # per-question summary: average over runs (25 expected)
    by_pq: Dict[Tuple[str,str], List[Dict[str, object]]] = defaultdict(list)
    for rr in run_rows:
        by_pq[(rr["prompt_version"], rr["question_id"])].append(rr)

    pq_rows: List[Dict[str, object]] = []
    for (pv, qid), rows in sorted(by_pq.items()):
        pq_rows.append({
            "prompt_version": pv,
            "question_id": qid,
            "n_runs": len(rows),
            "mean_precision": mean([float(x["precision"]) for x in rows]),
            "mean_recall": mean([float(x["recall"]) for x in rows]),
            "mean_jaccard": mean([float(x["jaccard"]) for x in rows]),
            "mean_f1": mean([float(x["f1"]) for x in rows]),
            "mean_pred_n": mean([float(x["pred_n"]) for x in rows]),
            "gold_n": int(rows[0]["gold_n"]) if rows else 0,
        })

    out_pq = os.path.join(args.outdir, "per_question_summary.csv")
    write_csv(out_pq,
              ["prompt_version","question_id","n_runs","mean_precision","mean_recall","mean_jaccard","mean_f1","mean_pred_n","gold_n"],
              pq_rows)

    # per-prompt summary: average over questions (macro-average over question means)
    by_p: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in pq_rows:
        by_p[str(r["prompt_version"])].append(r)

    pp_rows: List[Dict[str, object]] = []
    for pv, rows in sorted(by_p.items()):
        pp_rows.append({
            "prompt_version": pv,
            "n_questions": len(rows),
            "mean_of_mean_precision": mean([float(x["mean_precision"]) for x in rows]),
            "mean_of_mean_recall": mean([float(x["mean_recall"]) for x in rows]),
            "mean_of_mean_jaccard": mean([float(x["mean_jaccard"]) for x in rows]),
            "mean_of_mean_f1": mean([float(x["mean_f1"]) for x in rows]),
            "mean_of_mean_pred_n": mean([float(x["mean_pred_n"]) for x in rows]),
        })

    out_pp = os.path.join(args.outdir, "per_prompt_summary.csv")
    write_csv(out_pp,
              ["prompt_version","n_questions","mean_of_mean_precision","mean_of_mean_recall","mean_of_mean_jaccard","mean_of_mean_f1","mean_of_mean_pred_n"],
              pp_rows)

    print(f"[OK] wrote {out_run}")
    print(f"[OK] wrote {out_pq}")
    print(f"[OK] wrote {out_pp}")
    print("\nPer-prompt summary (macro over questions):")
    for r in pp_rows:
        print(f"  {r['prompt_version']}: meanF1={float(r['mean_of_mean_f1']):.6f} "
              f"meanJ={float(r['mean_of_mean_jaccard']):.6f} "
              f"meanP={float(r['mean_of_mean_precision']):.6f} "
              f"meanR={float(r['mean_of_mean_recall']):.6f}")

if __name__ == "__main__":
    main()
