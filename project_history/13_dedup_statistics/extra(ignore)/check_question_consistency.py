#!/usr/bin/env python3
import pandas as pd

PAIRS_FILE = "robokop_pairs_filtered_dedup.tsv"
TRIPLES_FILE = "robokop_triples_deduplicated.tsv"


def main():
    pairs = pd.read_csv(PAIRS_FILE, sep="\t", dtype=str).fillna("")
    triples = pd.read_csv(TRIPLES_FILE, sep="\t", dtype=str).fillna("")

    print("\n--- BASIC COUNTS ---")
    print("Pairs questions:", pairs["question_id"].nunique())
    print("Triples questions:", triples["question_id"].nunique())

    # Extract canonical node pair per question from triples
    # We assume the original query nodes are the pair that matches subject_curie/object_curie
    # But triples can contain reversed direction, so we check both directions.

    triple_pairs = (
        triples.groupby("question_id")[["subject", "object"]]
        .first()
        .reset_index()
    )

    # Merge with pairs file
    merged = pairs.merge(triple_pairs, on="question_id", how="left")

    print("\n--- MISSING QUESTIONS ---")

    missing_in_triples = set(pairs["question_id"]) - set(triples["question_id"])
    missing_in_pairs = set(triples["question_id"]) - set(pairs["question_id"])

    print("Missing in triples:", len(missing_in_triples))
    print("Missing in pairs:", len(missing_in_pairs))

    if missing_in_triples:
        print("Examples:", list(missing_in_triples)[:5])

    # Check node consistency
    print("\n--- NODE CONSISTENCY CHECK ---")

    def nodes_match(row):
        s_pair = row["subject_curie"]
        o_pair = row["object_curie"]
        s_triple = row["subject"]
        o_triple = row["object"]

        if pd.isna(s_triple):
            return False

        # Allow direction reversal
        return (
            (s_pair == s_triple and o_pair == o_triple) or
            (s_pair == o_triple and o_pair == s_triple)
        )

    merged["match"] = merged.apply(nodes_match, axis=1)

    mismatches = merged[merged["match"] == False]

    print("Total questions:", len(merged))
    print("Matching node pairs:", merged["match"].sum())
    print("Mismatching node pairs:", len(mismatches))

    if len(mismatches) > 0:
        print("\nExamples of mismatches:")
        print(mismatches[[
            "question_id",
            "subject_curie",
            "object_curie",
            "subject",
            "object"
        ]].head())

    print("\nConsistency check complete.")


if __name__ == "__main__":
    main()
