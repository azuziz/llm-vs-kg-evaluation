#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# -------------------------
# Paths (repo layout)
# -------------------------
ROOT = Path(__file__).resolve().parents[1]  # /gpt/7_gpt
DATA = ROOT / "data"
OUT = ROOT / "out"

PAIRS_TSV = DATA / "resolved_node_pairs.tsv"
ROBOKOP_TSV = DATA / "robokop_triples.tsv"
GPT_RUNS_CSV = OUT / "gpt_runs_raw.csv"

OUT_PER_RUN = OUT / "step2_metrics_per_run.csv"

# -------------------------
# Canonicalization helpers
# -------------------------
def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    a = (a or "").strip()
    b = (b or "").strip()
    return tuple(sorted([a, b]))  # type: ignore

def canonical_edge(curie_a: str, predicate: str, curie_b: str) -> str:
    a, b = canonical_pair(curie_a, curie_b)
    return f"{a}|{predicate.strip()}|{b}"

# -------------------------
# Loading questions
# -------------------------
def load_questions() -> Tuple[Dict[str, Tuple[str, str]], Dict[Tuple[str, str], str]]:
    """
    Returns:
      qid_to_pair: question_id -> (subject_curie, object_curie)
      pair_to_qid: (min_curie, max_curie) -> question_id
    """
    qid_to_pair: Dict[str, Tuple[str, str]] = {}
    pair_to_qid: Dict[Tuple[str, str], str] = {}

    with open(PAIRS_TSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            qid = row["question_id"]
            s = row["subject_curie"]
            o = row["object_curie"]
            qid_to_pair[qid] = (s, o)
            pair_to_qid[canonical_pair(s, o)] = qid

    if not qid_to_pair:
        raise ValueError("resolved_node_pairs.tsv loaded 0 rows")

    return qid_to_pair, pair_to_qid

# -------------------------
# Loading GPT sets (per run)
# -------------------------
def load_gpt_sets() -> Dict[Tuple[str, int], Set[str]]:
    """
    Reads out/gpt_runs_raw.csv and returns:
      (question_id, run_id) -> set(edge_key)
    Empty sets are represented by a row with edge_key == "".
    """
    gpt_sets: Dict[Tuple[str, int], Set[str]] = defaultdict(set)

    with open(GPT_RUNS_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"question_id", "run_id", "edge_key"}
        if not required.issubset(r.fieldnames or []):
            raise ValueError(f"gpt_runs_raw.csv missing columns: {required}")

        for row in r:
            qid = row["question_id"].strip()
            run_id = int(row["run_id"])
            ek = (row.get("edge_key") or "").strip()
            if ek:
                gpt_sets[(qid, run_id)].add(ek)
            else:
                # ensure empty set exists even if no edges
                _ = gpt_sets[(qid, run_id)]

    if not gpt_sets:
        raise ValueError("No GPT runs found in out/gpt_runs_raw.csv")

    return gpt_sets

# -------------------------
# Robust ROBOKOP triples loading
# -------------------------
def pick_first(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def load_robokop_sets(pair_to_qid: Dict[Tuple[str, str], str]) -> Dict[str, Set[str]]:
    """
    Loads robokop_triples.tsv into:
      question_id -> set(canonical_edge)
    Supports two formats:
      A) robokop_triples.tsv has a question_id column
      B) no question_id column: we assign to a question via (subject_curie, object_curie) matching resolved_node_pairs
    """
    by_qid: Dict[str, Set[str]] = defaultdict(set)

    with open(ROBOKOP_TSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        fns = r.fieldnames or []
        if not fns:
            raise ValueError("robokop_triples.tsv has no header")

        qid_col = pick_first(fns, ["question_id", "qid", "question"])
        subj_col = pick_first(fns, ["subject_curie", "subject", "subj", "subject_id", "subject_curie_id"])
        obj_col  = pick_first(fns, ["object_curie", "object", "obj", "object_id", "object_curie_id"])
        pred_col = pick_first(fns, ["predicate", "pred", "edge_predicate", "relation"])

        if not (subj_col and obj_col and pred_col):
            raise ValueError(
                "robokop_triples.tsv must contain subject/object/predicate columns.\n"
                f"Found columns: {fns}\n"
                "Tried subject in {subject_curie,subject,subj,subject_id}, "
                "object in {object_curie,object,obj,object_id}, "
                "predicate in {predicate,pred,edge_predicate,relation}."
            )

        for row in r:
            s = (row.get(subj_col) or "").strip()
            o = (row.get(obj_col) or "").strip()
            p = (row.get(pred_col) or "").strip()
            if not (s and o and p):
                continue

            if qid_col:
                qid = (row.get(qid_col) or "").strip()
            else:
                qid = pair_to_qid.get(canonical_pair(s, o), "")

            if not qid:
                # triple doesn't map to any of the 8 questions; ignore
                continue

            by_qid[qid].add(canonical_edge(s, p, o))

    if not by_qid:
        raise ValueError(
            "No ROBOKOP triples mapped to your questions. "
            "Either robokop_triples.tsv has different CURIEs, or it lacks subject/object columns we can read."
        )

    return by_qid

# -------------------------
# Metrics
# -------------------------
def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def precision_like(a_gpt: Set[str], b_ref: Set[str]) -> float:
    # |∩| / |GPT|
    if not a_gpt:
        return 1.0 if not b_ref else 0.0
    return len(a_gpt & b_ref) / len(a_gpt)

def recall_like(a_gpt: Set[str], b_ref: Set[str]) -> float:
    # |∩| / |ROBOKOP|
    if not b_ref:
        return 1.0 if not a_gpt else 0.0
    return len(a_gpt & b_ref) / len(b_ref)

def f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)

# -------------------------
# Main
# -------------------------
def main() -> None:
    OUT.mkdir(exist_ok=True)

    qid_to_pair, pair_to_qid = load_questions()
    gpt_sets = load_gpt_sets()
    robokop_sets = load_robokop_sets(pair_to_qid)

    # Ensure ROBOKOP empty sets exist for all questions (if a question has no triples)
    for qid in qid_to_pair:
        _ = robokop_sets.setdefault(qid, set())

    with open(OUT_PER_RUN, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id",
            "run_id",
            "n_gpt",
            "n_robokop",
            "n_intersection",
            "jaccard",
            "precision_like",
            "recall_like",
            "f1",
        ])

        for (qid, run_id), gset in sorted(gpt_sets.items(), key=lambda x: (x[0][0], x[0][1])):
            rset = robokop_sets.get(qid, set())
            inter = len(gset & rset)
            jac = jaccard(gset, rset)
            prec = precision_like(gset, rset)
            rec = recall_like(gset, rset)
            w.writerow([
                qid,
                run_id,
                len(gset),
                len(rset),
                inter,
                f"{jac:.6f}",
                f"{prec:.6f}",
                f"{rec:.6f}",
                f"{f1(prec, rec):.6f}",
            ])

    print(f"Wrote: {OUT_PER_RUN}")

if __name__ == "__main__":
    main()
