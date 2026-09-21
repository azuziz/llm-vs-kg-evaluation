#!/usr/bin/env python3
import pandas as pd

PAIRS = "robokop_pairs_filtered_dedup.tsv"
TRIPLES = "robokop_triples_deduplicated.tsv"

pairs = pd.read_csv(PAIRS, sep="\t", dtype=str).fillna("")
triples = pd.read_csv(TRIPLES, sep="\t", dtype=str).fillna("")

missing_query_edge = []

for _, row in pairs.iterrows():
    qid = row["question_id"]
    s = row["subject_curie"]
    o = row["object_curie"]

    t = triples[triples["question_id"] == qid]

    found = (
        ((t["subject"] == s) & (t["object"] == o)) |
        ((t["subject"] == o) & (t["object"] == s))
    ).any()

    if not found:
        missing_query_edge.append(qid)

print("Questions checked:", len(pairs))
print("Questions where original node pair edge exists:",
      len(pairs) - len(missing_query_edge))
print("Questions WITHOUT direct edge:",
      len(missing_query_edge))

if missing_query_edge:
    print("\nExamples:")
    print(missing_query_edge[:10])
