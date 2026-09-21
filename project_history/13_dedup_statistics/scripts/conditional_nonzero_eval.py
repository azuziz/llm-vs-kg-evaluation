import argparse
from pathlib import Path
import pandas as pd

from io_kg import load_kg_triples_by_question
from io_gpt import load_gpt_runs_jsonl, group_runs_by_question
from metrics import compute_run_metrics, f1_from_pr

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

    # Per-run conditional (nonzero GPT edges)
    rows = []
    for qid, run_list in runs_by_q.items():
        kg = kg_by_q.get(qid, set())
        for r in run_list:
            if len(r.edges) == 0:
                continue
            m = compute_run_metrics(r.edges, kg)
            rows.append({
                "question_id": qid,
                "run_id": r.run_id,
                "gpt_edges": len(r.edges),
                "kg_edges": len(kg),
                "matched_edges": m.matched_edges,
                "tp": m.tp, "fp": m.fp, "fn": m.fn,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
            })

    df_runs = pd.DataFrame(rows)
    df_runs.to_csv(out_dir / "nonzero_runs_per_run_metrics.tsv", sep="\t", index=False)

    # Per-question conditional averages
    if len(df_runs) > 0:
        df_q = (
            df_runs.groupby("question_id", dropna=False)
            .agg(
                n_nonzero_runs=("run_id", "count"),
                mean_gpt_edges=("gpt_edges", "mean"),
                kg_edges=("kg_edges", "max"),
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                mean_f1=("f1", "mean"),
                match_rate_nonzero=("matched_edges", lambda x: float((x > 0).mean())),
            )
            .reset_index()
        )
    else:
        df_q = pd.DataFrame(columns=[
            "question_id","n_nonzero_runs","mean_gpt_edges","kg_edges",
            "mean_precision","mean_recall","mean_f1","match_rate_nonzero"
        ])

    # Ensure all questions are present (fill zeros if GPT never produced output)
    all_qids = sorted(kg_by_q.keys())
    have = set(df_q["question_id"].tolist())
    missing = [q for q in all_qids if q not in have]
    if missing:
        filler = pd.DataFrame([{
            "question_id": q,
            "n_nonzero_runs": 0,
            "mean_gpt_edges": 0.0,
            "kg_edges": float(len(kg_by_q.get(q, set()))),
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f1": 0.0,
            "match_rate_nonzero": 0.0,
        } for q in missing])
        df_q = pd.concat([df_q, filler], ignore_index=True)

    df_q.to_csv(out_dir / "nonzero_runs_per_question_metrics.tsv", sep="\t", index=False)

    # Summary (macro across questions) + micro across nonzero runs
    macro_precision = float(df_q["mean_precision"].mean()) if len(df_q) else 0.0
    macro_recall = float(df_q["mean_recall"].mean()) if len(df_q) else 0.0
    macro_f1 = float(df_q["mean_f1"].mean()) if len(df_q) else 0.0

    tp_sum = int(df_runs["tp"].sum()) if len(df_runs) else 0
    fp_sum = int(df_runs["fp"].sum()) if len(df_runs) else 0
    fn_sum = int(df_runs["fn"].sum()) if len(df_runs) else 0
    micro_precision = (tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
    micro_recall = (tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = f1_from_pr(micro_precision, micro_recall)

    summary = pd.DataFrame([{
        "questions_total": int(df_q["question_id"].nunique()),
        "nonzero_runs_total": int(len(df_runs)),
        "macro_precision_nonzero": macro_precision,
        "macro_recall_nonzero": macro_recall,
        "macro_f1_nonzero": macro_f1,
        "micro_precision_nonzero": micro_precision,
        "micro_recall_nonzero": micro_recall,
        "micro_f1_nonzero": micro_f1,
        "mean_nonzero_runs_per_question": float(df_q["n_nonzero_runs"].mean()) if len(df_q) else 0.0,
    }])

    summary.to_csv(out_dir / "nonzero_runs_summary.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()
