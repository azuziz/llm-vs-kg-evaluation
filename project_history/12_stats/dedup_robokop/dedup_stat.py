#!/usr/bin/env python3
import pandas as pd

INPUT = "robokop_triples.tsv"
TOPN = 15


def vc_table(series: pd.Series, count_col: str) -> pd.DataFrame:
    """
    Return a 2-col dataframe: edge, <count_col>
    Robust across pandas versions.
    """
    s = series.value_counts(dropna=False)
    out = s.rename(count_col).to_frame().reset_index()
    out.columns = ["edge", count_col]
    return out


def main():
    df = pd.read_csv(INPUT, sep="\t", dtype=str).fillna("")

    # ---- RAW edge key (directional) ----
    df["edge_raw"] = df["subject"] + " | " + df["predicate"] + " | " + df["object"]

    raw_counts = vc_table(df["edge_raw"], "count_before_dedup")

    # ---- EXACT dedup (within question) ----
    df_exact = df.drop_duplicates(subset=["question_id", "subject", "predicate", "object"])
    exact_counts = vc_table(df_exact["edge_raw"], "count_exact_dedup")

    # ---- Directionless normalization (within question) ----
    # canonicalize node order: (min(subject, object), max(subject, object))
    a = df["subject"]
    b = df["object"]
    df["subj_dl"] = a.where(a <= b, b)
    df["obj_dl"] = b.where(a <= b, a)

    df["edge_directionless"] = df["subj_dl"] + " | " + df["predicate"] + " | " + df["obj_dl"]

    df_dir = df.drop_duplicates(subset=["question_id", "subj_dl", "predicate", "obj_dl"])
    dir_counts = vc_table(df_dir["edge_directionless"], "count_directionless")

    # ---- Merge ----
    merged = raw_counts.merge(exact_counts, on="edge", how="outer")
    merged = merged.merge(dir_counts, on="edge", how="outer")
    merged = merged.fillna(0)

    for c in ["count_before_dedup", "count_exact_dedup", "count_directionless"]:
        merged[c] = merged[c].astype(int)

    merged = merged.sort_values(["count_before_dedup", "count_exact_dedup", "count_directionless"],
                                ascending=False)

    top = merged.head(TOPN).copy()

    out_file = "robokop_top15_edges_dedup_comparison.tsv"
    top.to_csv(out_file, sep="\t", index=False)

    print(top.to_string(index=False))
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
