#!/usr/bin/env python3
import pandas as pd

INPUT = "robokop_triples_deduplicated.tsv"

def main():
    df = pd.read_csv(INPUT, sep="\t", dtype=str).fillna("")

    df["edge"] = (
        df["subject"] + " | " +
        df["predicate"] + " | " +
        df["object"]
    )

    edge_q_counts = (
        df.groupby("edge")["question_id"]
        .nunique()
        .rename("questions_count")
        .reset_index()
        .sort_values("questions_count", ascending=False)
    )

    # Save full sorted table
    edge_q_counts.to_csv(
        "robokop_edges_by_question_count_ALL.tsv",
        sep="\t",
        index=False
    )

    # Top 20
    edge_q_counts.head(20).to_csv(
        "robokop_top20_edges.tsv",
        sep="\t",
        index=False
    )

    # Top 50
    edge_q_counts.head(50).to_csv(
        "robokop_top50_edges.tsv",
        sep="\t",
        index=False
    )

    print("\nTop 20 edges:")
    print(edge_q_counts.head(20).to_string(index=False))

    print("\nTop 50 edges saved.")
    print("\nFiles written:")
    print("  robokop_edges_by_question_count_ALL.tsv")
    print("  robokop_top20_edges.tsv")
    print("  robokop_top50_edges.tsv")


if __name__ == "__main__":
    main()
