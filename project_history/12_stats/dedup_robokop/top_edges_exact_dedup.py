#!/usr/bin/env python3
import pandas as pd

INPUT = "robokop_triples.tsv"
TOPN = 15


def vc_table(series: pd.Series, count_col: str) -> pd.DataFrame:
    s = series.value_counts(dropna=False)
    out = s.rename(count_col).to_frame().reset_index()
    out.columns = ["edge", count_col]
    return out


def main():
    df = pd.read_csv(INPUT, sep="\t", dtype=str).fillna("")

    # Directed edge string
    df["edge"] = df["subject"] + " | " + df["predicate"] + " | " + df["object"]

    # BEFORE dedup (raw rows)
    before = vc_table(df["edge"], "count_before_dedup")

    # AFTER exact dedup (per question, keep direction)
    df_dedup = df.drop_duplicates(subset=["question_id", "subject", "predicate", "object"]).copy()
    after = vc_table(df_dedup["edge"], "count_after_exact_dedup")

    # Merge counts
    merged = before.merge(after, on="edge", how="outer").fillna(0)
    merged["count_before_dedup"] = merged["count_before_dedup"].astype(int)
    merged["count_after_exact_dedup"] = merged["count_after_exact_dedup"].astype(int)

    # How many rows removed for each edge
    merged["rows_removed"] = merged["count_before_dedup"] - merged["count_after_exact_dedup"]

    # Sort by most frequent before dedup
    merged = merged.sort_values(
        ["count_before_dedup", "rows_removed", "count_after_exact_dedup"],
        ascending=[False, False, False],
    )

    top = merged.head(TOPN).copy()

    out_top = "robokop_top15_edges_exact_dedup.tsv"
    out_all = "robokop_all_edges_exact_dedup.tsv"

    top.to_csv(out_top, sep="\t", index=False)
    merged.to_csv(out_all, sep="\t", index=False)

    print(top.to_string(index=False))
    print(f"\nSaved:\n  {out_top}\n  {out_all}")


if __name__ == "__main__":
    main()
