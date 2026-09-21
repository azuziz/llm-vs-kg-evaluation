#!/usr/bin/env python3
import pandas as pd

INPUT = "robokop_triples.tsv"
OUTPUT = "robokop_triples_deduplicated.tsv"

def main():
    df = pd.read_csv(INPUT, sep="\t", dtype=str).fillna("")

    before = len(df)

    df_dedup = df.drop_duplicates(
        subset=["question_id", "subject", "predicate", "object"]
    ).copy()

    after = len(df_dedup)

    df_dedup.to_csv(OUTPUT, sep="\t", index=False)

    print(f"Original rows:      {before}")
    print(f"After dedup:        {after}")
    print(f"Rows removed:       {before - after}")
    print(f"Removed percent:    {round((before - after)/before*100, 2)}%")
    print(f"\nSaved: {OUTPUT}")

if __name__ == "__main__":
    main()
