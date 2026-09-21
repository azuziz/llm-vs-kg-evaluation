import csv
import json
import math
import os
import sys
from glob import glob
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, FrozenSet

QUESTIONS = [f"Q{i}" for i in range(1,9)]

def read_truth(robokop_per_q_dir: str, qid: str):
    """
    Returns truth sets under three granularities:
      A: endpoints-only (undirected)
      B: endpoints+predicate (undirected endpoints)
      C: directed triple
    """
    path = os.path.join(robokop_per_q_dir, f"{qid}.tsv")
    TA, TB, TC = set(), set(), set()
    if not os.path.exists(path):
        return TA, TB, TC

    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            s = (row.get("subject") or "").strip()
            o = (row.get("object") or "").strip()
            p = (row.get("predicate") or "").strip()
            if not s or not o:
                continue
            endpoints = frozenset([s, o])
            TA.add(endpoints)
            if p:
                TB.add((endpoints, p))
                TC.add((s, p, o))
    return TA, TB, TC

def read_pred_runs(q_dir: str, qid: str):
    """
    Reads parsed run files: run_001.json ... run_100.json (NOT *_raw.json)
    Returns list of dicts: {"answer":..., "triples":[...]}
    """
    files = sorted(glob(os.path.join(q_dir, qid, "run_[0-9][0-9][0-9].json")))
    runs = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs

def triple_sets(parsed: dict):
    triples = parsed.get("triples") or []
    GA, GB, GC = set(), set(), set()
    for t in triples:
        s = (t.get("subject") or "").strip()
        o = (t.get("object") or "").strip()
        p = (t.get("predicate") or "").strip()
        if not s or not o:
            continue
        endpoints = frozenset([s, o])
        GA.add(endpoints)
        if p:
            GB.add((endpoints, p))
            GC.add((s, p, o))
    return GA, GB, GC

def entropy_from_counts(counts: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p, 2)
    return ent

def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def score_question(qid: str, TA, TB, TC, runs: List[dict]):
    """
    Computes metrics per granularity A/B/C, plus stability.
    """
    n = len(runs)
    if n == 0:
        return None

    # Helper to compute per-granularity hit/hall/precision/recall
    def compute_metrics(T: Set, G_list: List[Set]):
        hits = 0
        yes_ans = 0
        empty = 0
        hall_yes = 0

        inter_sum = 0
        pred_sum = 0
        truth_size = len(T)

        hall_sum = 0

        for parsed, G in zip(runs, G_list):
            ans = (parsed.get("answer") or "").strip().upper()
            if ans == "YES":
                yes_ans += 1
            if len(G) == 0:
                empty += 1

            inter = len(G & T)
            if inter > 0:
                hits += 1

            # Hallucinated YES: claims yes but no overlap
            if ans == "YES" and inter == 0:
                hall_yes += 1

            inter_sum += inter
            pred_sum += len(G)
            hall_sum += len(G - T)

        hit_rate = hits / n
        empty_rate = empty / n
        yes_rate = yes_ans / n

        # Micro precision/recall
        micro_precision = (inter_sum / pred_sum) if pred_sum > 0 else float("nan")
        micro_recall = (inter_sum / (truth_size * n)) if truth_size > 0 else float("nan")  # average recall over runs in micro form
        # Note: micro_recall here is normalized by truth_size*n, i.e. mean(|inter|/|T|)

        hall_mass = (hall_sum / pred_sum) if pred_sum > 0 else float("nan")
        hall_yes_rate = hall_yes / n

        return {
            "hit_rate": hit_rate,
            "empty_rate": empty_rate,
            "yes_rate": yes_rate,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "hall_mass": hall_mass,
            "hall_yes_rate": hall_yes_rate,
            "truth_size": truth_size,
            "pred_total": pred_sum,
            "inter_total": inter_sum,
            "hall_total": hall_sum
        }

    # Build predicted sets for each run
    GA_list, GB_list, GC_list = [], [], []
    for r in runs:
        GA, GB, GC = triple_sets(r)
        GA_list.append(GA)
        GB_list.append(GB)
        GC_list.append(GC)

    A = compute_metrics(TA, GA_list)
    B = compute_metrics(TB, GB_list)
    C = compute_metrics(TC, GC_list)

    # Stability on A signatures (endpoints-only) and on B (predicate-aware)
    def stability(G_list: List[Set]):
        # Frequency of individual edges across runs (count presence per run)
        freq = Counter()
        for G in G_list:
            for e in G:
                freq[e] += 1
        support = len(freq)
        mode_count = max(freq.values()) if freq else 0
        mode_freq = mode_count / n if n else 0.0
        ent = entropy_from_counts(freq, n)  # treat each run as a trial where edge may appear
        # Mean pairwise Jaccard (O(n^2) but n=100 is fine)
        js = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i+1, n):
                js += jaccard(G_list[i], G_list[j])
                pairs += 1
        mean_j = (js / pairs) if pairs else 0.0
        return {"support": support, "mode_freq": mode_freq, "entropy": ent, "mean_jaccard": mean_j}

    stab_A = stability(GA_list)
    stab_B = stability(GB_list)

    return {"A": A, "B": B, "C": C, "stab_A": stab_A, "stab_B": stab_B, "runs": n}

def main(robokop_per_q_dir: str, gpt_runs_dir: str, out_per_q: str, out_overall: str):
    per_q_rows = []
    overall_acc = defaultdict(list)

    for qid in QUESTIONS:
        TA, TB, TC = read_truth(robokop_per_q_dir, qid)
        runs = read_pred_runs(gpt_runs_dir, qid)
        scored = score_question(qid, TA, TB, TC, runs)
        if not scored:
            continue

        row = {
            "question_id": qid,
            "runs": scored["runs"],

            # Truth sizes
            "truth_A": scored["A"]["truth_size"],
            "truth_B": scored["B"]["truth_size"],
            "truth_C": scored["C"]["truth_size"],

            # Primary: A endpoints-only
            "hit_rate_A": scored["A"]["hit_rate"],
            "empty_rate_A": scored["A"]["empty_rate"],
            "yes_rate_A": scored["A"]["yes_rate"],
            "precision_A": scored["A"]["micro_precision"],
            "recall_A": scored["A"]["micro_recall"],
            "hall_mass_A": scored["A"]["hall_mass"],
            "hall_yes_A": scored["A"]["hall_yes_rate"],

            # Predicate-aware: B
            "hit_rate_B": scored["B"]["hit_rate"],
            "precision_B": scored["B"]["micro_precision"],
            "recall_B": scored["B"]["micro_recall"],
            "hall_mass_B": scored["B"]["hall_mass"],
            "hall_yes_B": scored["B"]["hall_yes_rate"],

            # Strict: C
            "hit_rate_C": scored["C"]["hit_rate"],
            "precision_C": scored["C"]["micro_precision"],
            "recall_C": scored["C"]["micro_recall"],
            "hall_mass_C": scored["C"]["hall_mass"],
            "hall_yes_C": scored["C"]["hall_yes_rate"],

            # Stability (A and B)
            "support_A": scored["stab_A"]["support"],
            "mode_freq_A": scored["stab_A"]["mode_freq"],
            "entropy_A": scored["stab_A"]["entropy"],
            "mean_jaccard_A": scored["stab_A"]["mean_jaccard"],

            "support_B": scored["stab_B"]["support"],
            "mode_freq_B": scored["stab_B"]["mode_freq"],
            "entropy_B": scored["stab_B"]["entropy"],
            "mean_jaccard_B": scored["stab_B"]["mean_jaccard"],
        }
        per_q_rows.append(row)

        # Collect for overall summaries
        overall_acc["hit_rate_A"].append(row["hit_rate_A"])
        overall_acc["hall_mass_A"].append(row["hall_mass_A"])
        overall_acc["empty_rate_A"].append(row["empty_rate_A"])
        overall_acc["mode_freq_A"].append(row["mode_freq_A"])
        overall_acc["mean_jaccard_A"].append(row["mean_jaccard_A"])

    # Write per-question table
    fields = list(per_q_rows[0].keys()) if per_q_rows else []
    with open(out_per_q, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(per_q_rows)

    # Write overall summary (means)
    def mean(xs):
        xs2 = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return sum(xs2)/len(xs2) if xs2 else float("nan")

    overall = {
        "n_questions": len(per_q_rows),
        "mean_hit_rate_A": mean(overall_acc["hit_rate_A"]),
        "mean_empty_rate_A": mean(overall_acc["empty_rate_A"]),
        "mean_hall_mass_A": mean(overall_acc["hall_mass_A"]),
        "mean_mode_freq_A": mean(overall_acc["mode_freq_A"]),
        "mean_mean_jaccard_A": mean(overall_acc["mean_jaccard_A"]),
    }

    with open(out_overall, "w", encoding="utf-8") as f:
        for k, v in overall.items():
            f.write(f"{k}\t{v}\n")

    print(f"Wrote: {out_per_q}")
    print(f"Wrote: {out_overall}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python step6_score.py robokop_triples_by_question gpt_runs stats_per_question.tsv stats_overall.tsv")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
