import argparse
from pathlib import Path
import pandas as pd
from collections import Counter
from itertools import combinations

from io_kg import load_kg_triples_by_question
from io_gpt import load_gpt_runs_jsonl, group_runs_by_question
from metrics import compute_run_metrics

def canonical_set_key(triples_set):
    # stable representation for counting identical outputs
    return " || ".join(sorted([f"{s}|{p}|{o}" for (s,p,o) in triples_set]))

def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = a.union(b)
    if not u:
        return 1.0
    return len(a.intersection(b)) / len(u)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kg_by_q = load_kg_triples_by_question(args.kg)
    runs = load_gpt_runs_jsonl(args.gpt)
    runs_by_q = group_runs_by_question(runs)

    rows = []

    # For global binary-ish accounting
    total_questions = 0
    kg_zero_questions = 0

    for qid, kg_set in kg_by_q.items():
        total_questions += 1
        if len(kg_set) == 0:
            kg_zero_questions += 1

        run_list = runs_by_q.get(qid, [])
        g_sets = [r.edges for r in run_list]

        # Silence
        n_runs = len(run_list)
        n_nonempty = sum(1 for s in g_sets if len(s) > 0)
        silence_rate = 1.0 - (n_nonempty / n_runs) if n_runs > 0 else 1.0

        # Conditional precision among non-empty runs
        prec_vals = []
        rec_vals = []
        f1_vals = []
        hit_vals = []
        for s in g_sets:
            m = compute_run_metrics(s, kg_set)
            if len(s) > 0:
                prec_vals.append(m.precision)
                rec_vals.append(m.recall)
                f1_vals.append(m.f1)
            hit_vals.append(1 if m.matched_edges > 0 else 0)

        prec_out = float(sum(prec_vals) / len(prec_vals)) if prec_vals else 0.0
        rec_out = float(sum(rec_vals) / len(rec_vals)) if rec_vals else 0.0
        f1_out  = float(sum(f1_vals)  / len(f1_vals))  if f1_vals else 0.0

        match_rate = float(sum(hit_vals) / len(hit_vals)) if hit_vals else 0.0

        # Union recall (capability across 100 runs)
        union_set = set()
        for s in g_sets:
            union_set |= s
        union_tp = len(union_set.intersection(kg_set))
        union_recall = (union_tp / len(kg_set)) if len(kg_set) > 0 else 0.0
        union_precision = (union_tp / len(union_set)) if len(union_set) > 0 else 0.0

        # Stability: mode share
        keys = [canonical_set_key(s) for s in g_sets]
        mode_share = 0.0
        mode_size = 0
        if keys:
            c = Counter(keys)
            mode_key, mode_count = c.most_common(1)[0]
            mode_share = mode_count / len(keys)
            # infer mode size from key
            mode_size = 0 if mode_key == "" else mode_key.count(" || ") + 1

        # Stability: mean pairwise Jaccard (skip if too few runs)
        mean_pairwise_jaccard = 0.0
        if len(g_sets) >= 2:
            js = []
            # 4950 pairs at most for 100 runs → ok
            for i, j in combinations(range(len(g_sets)), 2):
                js.append(jaccard(g_sets[i], g_sets[j]))
            mean_pairwise_jaccard = float(sum(js) / len(js))

        rows.append({
            "question_id": qid,
            "kg_edges": len(kg_set),
            "n_runs": n_runs,
            "silence_rate": silence_rate,
            "match_rate": match_rate,                 # any match per run
            "precision_cond_on_output": prec_out,     # among runs where GPT output >=1 edge
            "recall_cond_on_output": rec_out,
            "f1_cond_on_output": f1_out,
            "union_recall": union_recall,             # across all runs union
            "union_precision": union_precision,
            "mode_share": mode_share,                 # stability
            "mode_set_size": mode_size,
            "mean_pairwise_jaccard": mean_pairwise_jaccard,
            "union_set_size": len(union_set),
            "union_tp": union_tp,
        })

    df = pd.DataFrame(rows)

    # Summary across questions (macro)
    summary = pd.DataFrame([{
        "questions_total": int(total_questions),
        "kg_zero_questions": int(kg_zero_questions),
        "mean_silence_rate": float(df["silence_rate"].mean()),
        "mean_match_rate": float(df["match_rate"].mean()),
        "mean_precision_cond_on_output": float(df["precision_cond_on_output"].mean()),
        "mean_recall_cond_on_output": float(df["recall_cond_on_output"].mean()),
        "mean_f1_cond_on_output": float(df["f1_cond_on_output"].mean()),
        "mean_union_recall": float(df["union_recall"].mean()),
        "mean_union_precision": float(df["union_precision"].mean()),
        "mean_mode_share": float(df["mode_share"].mean()),
        "mean_pairwise_jaccard": float(df["mean_pairwise_jaccard"].mean()),
    }])

    df.to_csv(out_dir / "per_question_extended_diagnostics.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "extended_summary.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()
