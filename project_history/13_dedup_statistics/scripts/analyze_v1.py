import pandas as pd
from typing import Dict, List, Tuple, Set
from io_kg import load_kg_triples_by_question
from io_gpt import load_gpt_runs_jsonl, group_runs_by_question, GptRun
from metrics import compute_run_metrics, f1_from_pr

Triple = Tuple[str, str, str]

def analyze(
    kg_tsv: str,
    gpt_jsonl: str,
) -> Dict[str, pd.DataFrame]:
    kg_by_q: Dict[str, Set[Triple]] = load_kg_triples_by_question(kg_tsv)
    runs: List[GptRun] = load_gpt_runs_jsonl(gpt_jsonl)
    runs_by_q = group_runs_by_question(runs)

    # --- per-run rows ---
    run_rows = []
    for qid, run_list in runs_by_q.items():
        kg_edges = kg_by_q.get(qid, set())
        for r in run_list:
            m = compute_run_metrics(r.edges, kg_edges)
            run_rows.append({
                "question_id": qid,
                "run_id": r.run_id,
                "curie1": r.curie1,
                "curie2": r.curie2,
                "gpt_edges": m.gpt_edges,
                "kg_edges": m.kg_edges,
                "matched_edges": m.matched_edges,
                "tp": m.tp, "fp": m.fp, "fn": m.fn,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "has_any_gpt_edge": int(m.gpt_edges > 0),
                "has_any_match": int(m.matched_edges > 0),
            })

    df_runs = pd.DataFrame(run_rows)

    # Ensure we include questions that exist in KG but maybe missing in GPT runs (shouldn't happen, but safe)
    all_qids = sorted(set(kg_by_q.keys()).union(set(runs_by_q.keys())))

    # --- per-question aggregation (mean over runs) ---
    if len(df_runs) > 0:
        df_q = (
            df_runs.groupby("question_id", dropna=False)
            .agg(
                n_runs=("run_id", "count"),
                mean_gpt_edges=("gpt_edges", "mean"),
                mean_matched_edges=("matched_edges", "mean"),
                kg_edges=("kg_edges", "max"),
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                mean_f1=("f1", "mean"),
                match_rate=("has_any_match", "mean"),      # fraction of runs with >=1 match
                gpt_edge_rate=("has_any_gpt_edge", "mean") # fraction of runs with >=1 GPT edge
            )
            .reset_index()
        )
    else:
        df_q = pd.DataFrame(columns=[
            "question_id","n_runs","mean_gpt_edges","mean_matched_edges","kg_edges",
            "mean_precision","mean_recall","mean_f1","match_rate","gpt_edge_rate"
        ])

    # add missing qids (if any) with zeros
    have = set(df_q["question_id"].tolist())
    for qid in all_qids:
        if qid not in have:
            df_q = pd.concat([df_q, pd.DataFrame([{
                "question_id": qid,
                "n_runs": 0,
                "mean_gpt_edges": 0.0,
                "mean_matched_edges": 0.0,
                "kg_edges": float(len(kg_by_q.get(qid, set()))),
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "mean_f1": 0.0,
                "match_rate": 0.0,
                "gpt_edge_rate": 0.0,
            }])], ignore_index=True)

    # --- overall summary (macro over questions) ---
    # macro averages of mean_* across questions
    df_q2 = df_q.copy()
    df_q2["kg_edges"] = df_q2["kg_edges"].astype(float)

    macro_precision = float(df_q2["mean_precision"].mean()) if len(df_q2) else 0.0
    macro_recall = float(df_q2["mean_recall"].mean()) if len(df_q2) else 0.0
    macro_f1 = float(df_q2["mean_f1"].mean()) if len(df_q2) else 0.0

    zero_f1_rate = float((df_q2["mean_f1"] == 0.0).mean()) if len(df_q2) else 0.0

    # micro (optional, useful)
    tp_sum = int(df_runs["tp"].sum()) if len(df_runs) else 0
    fp_sum = int(df_runs["fp"].sum()) if len(df_runs) else 0
    fn_sum = int(df_runs["fn"].sum()) if len(df_runs) else 0
    micro_precision = (tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
    micro_recall = (tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = f1_from_pr(micro_precision, micro_recall)

    summary = pd.DataFrame([{
        "questions_total": int(df_q2["question_id"].nunique()),
        "runs_total": int(len(df_runs)),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "zero_f1_rate": zero_f1_rate,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "mean_kg_edges_per_question": float(df_q2["kg_edges"].mean()) if len(df_q2) else 0.0,
        "mean_gpt_edges_per_run": float(df_runs["gpt_edges"].mean()) if len(df_runs) else 0.0,
        "mean_match_rate_per_question": float(df_q2["match_rate"].mean()) if len(df_q2) else 0.0,
        "mean_gpt_edge_rate_per_question": float(df_q2["gpt_edge_rate"].mean()) if len(df_q2) else 0.0,
    }])

    # --- top 15 questions by mean_f1 (ties broken by mean_recall then mean_precision) ---
    top15 = (
        df_q2.sort_values(["mean_f1", "mean_recall", "mean_precision"], ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    return {
        "per_run": df_runs,
        "per_question": df_q2,
        "summary": summary,
        "top15_questions": top15,
    }
