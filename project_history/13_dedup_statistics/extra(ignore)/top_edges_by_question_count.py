#!/usr/bin/env python3
import pandas as pd

INPUT = "robokop_triples_deduplicated.tsv"
TOPN = 15

def main():
    df = pd.read_csv(INPUT, sep="\t", dtype=str).fillna("")

    df["edge"] = (
        df["subject"] + " | " +
        df["predicate"] + " | " +
        df["object"]
    )

    # count in how many questions each edge appears
    edge_q_counts = (
        df.groupby("edge")["question_id"]
        .nunique()
        .rename("questions_count")
        .reset_index()
        .sort_values("questions_count", ascending=False)
    )

    top = edge_q_counts.head(TOPN).copy()

    top.to_csv(
        "robokop_top15_edges_by_question_count.tsv",
        sep="\t",
        index=False
    )

    print(top.to_string(index=False))
    print("\nSaved: robokop_top15_edges_by_question_count.tsv")

if __name__ == "__main__":
    main()
