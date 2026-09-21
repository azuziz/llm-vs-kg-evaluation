#!/usr/bin/env python3
"""
robokop_dedup_report.py

Usage:
  python robokop_dedup_report.py \
    --in_tsv robokop_triples.tsv \
    --out_prefix robokop_dedup

Outputs:
  robokop_dedup.per_question_counts.tsv
  robokop_dedup.overall_summary.tsv
  robokop_dedup.affected_questions.tsv
  robokop_dedup.duplicate_rows_sample.tsv

Optional (inverse predicate collapsing):
  Provide a TSV with columns: predicate   inverse_predicate
  Example row:
    biolink:causes   biolink:caused_by

  Then run:
    python robokop_dedup_report.py --in_tsv robokop_triples.tsv --out_prefix robokop_dedup \
      --inverse_map predicate_inverses.tsv
"""

import argparse
import pandas as pd
from pathlib import Path


REQUIRED_COLS = [
    "question_id",
    "subject",
    "predicate",
    "object",
]


def load_inverse_map(path: str | None) -> dict[str, str]:
    """
    Loads predicate->canonical_predicate mapping using inverse pairs.
    We canonicalize each inverse-pair to a stable representative:
      canonical = min(predicate, inverse_predicate) (lexicographically)
    """
    if not path:
        return {}

    inv_df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if not {"predicate", "inverse_predicate"}.issubset(inv_df.columns):
        raise ValueError(
            f"Inverse map file must have columns: predicate, inverse_predicate. Found: {list(inv_df.columns)}"
        )

    mapping: dict[str, str] = {}
    for _, row in inv_df.iterrows():
        p = row["predicate"].strip()
        inv = row["inverse_predicate"].strip()
        if not p or not inv:
            continue
        canon = min(p, inv)
        mapping[p] = canon
        mapping[inv] = canon

    return mapping


def directionless_key(subject: str, obj: str) -> tuple[str, str]:
    """Canonicalize node order for directionless normalization."""
    return (subject, obj) if subject <= obj else (obj, subject)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True, help="Input robokop_triples.tsv")
    ap.add_argument("--out_prefix", required=True, help="Prefix for output TSV files")
    ap.add_argument(
        "--inverse_map",
        default=None,
        help="Optional TSV with columns: predicate<TAB>inverse_predicate",
    )
    ap.add_argument(
        "--sample_n",
        type=int,
        default=200,
        help="How many duplicate rows to sample into duplicate_rows_sample.tsv",
    )
    args = ap.parse_args()

    in_path = Path(args.in_tsv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path, sep="\t", dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}\nFound: {list(df.columns)}")

    inv_map = load_inverse_map(args.inverse_map)

    # ---- Normalization columns ----
    # 1) Exact edge identity (directional)
    df["key_exact"] = (
        df["question_id"].astype(str)
        + "||"
        + df["subject"].astype(str)
        + "||"
        + df["predicate"].astype(str)
        + "||"
        + df["object"].astype(str)
    )

    # 2) Directionless normalization (ignore direction, keep predicate as-is)
    #    (subject, object) sorted
    so = df.apply(lambda r: directionless_key(r["subject"], r["object"]), axis=1, result_type="expand")
    df["subj_dl"] = so[0]
    df["obj_dl"] = so[1]
    df["key_directionless"] = (
        df["question_id"].astype(str)
        + "||"
        + df["subj_dl"].astype(str)
        + "||"
        + df["predicate"].astype(str)
        + "||"
        + df["obj_dl"].astype(str)
    )

    # 3) Directionless + inverse-collapsed predicate (optional)
    if inv_map:
        df["predicate_canon"] = df["predicate"].map(lambda p: inv_map.get(p, p))
    else:
        df["predicate_canon"] = df["predicate"]

    df["key_directionless_inv"] = (
        df["question_id"].astype(str)
        + "||"
        + df["subj_dl"].astype(str)
        + "||"
        + df["predicate_canon"].astype(str)
        + "||"
        + df["obj_dl"].astype(str)
    )

    # ---- Per-question counts ----
    g = df.groupby("question_id", dropna=False)

    per_q = pd.DataFrame(
        {
            "question_id": g.size().index,
            "n_rows_total": g.size().values,
            "n_unique_exact": g["key_exact"].nunique().values,
            "n_unique_directionless": g["key_directionless"].nunique().values,
            "n_unique_directionless_inv": g["key_directionless_inv"].nunique().values,
        }
    )

    per_q["dup_rows_exact"] = per_q["n_rows_total"] - per_q["n_unique_exact"]
    per_q["dup_rows_directionless"] = per_q["n_rows_total"] - per_q["n_unique_directionless"]
    per_q["dup_rows_directionless_inv"] = per_q["n_rows_total"] - per_q["n_unique_directionless_inv"]

    # How many questions affected by duplicates (by exact identity)
    affected = per_q[per_q["dup_rows_exact"] > 0].copy()
    affected = affected.sort_values(["dup_rows_exact", "n_rows_total"], ascending=False)

    # ---- Overall summary ----
    overall = pd.DataFrame(
        [
            {
                "metric": "questions_total",
                "value": int(per_q.shape[0]),
            },
            {
                "metric": "questions_with_exact_duplicates",
                "value": int((per_q["dup_rows_exact"] > 0).sum()),
            },
            {
                "metric": "rows_total",
                "value": int(df.shape[0]),
            },
            {
                "metric": "rows_unique_exact_total",
                "value": int(df["key_exact"].nunique()),
            },
            {
                "metric": "rows_unique_directionless_total",
                "value": int(df["key_directionless"].nunique()),
            },
            {
                "metric": "rows_unique_directionless_inv_total",
                "value": int(df["key_directionless_inv"].nunique()),
            },
            {
                "metric": "rows_removed_by_exact_dedup",
                "value": int(df.shape[0] - df["key_exact"].nunique()),
            },
            {
                "metric": "rows_removed_by_directionless",
                "value": int(df.shape[0] - df["key_directionless"].nunique()),
            },
            {
                "metric": "rows_removed_by_directionless_inv",
                "value": int(df.shape[0] - df["key_directionless_inv"].nunique()),
            },
        ]
    )

    # ---- Sample duplicate rows for inspection ----
    # Duplicate by exact key within question
    dup_mask = df.duplicated(subset=["key_exact"], keep=False)
    dup_rows = df.loc[dup_mask].copy()

    # Keep a compact set of columns (include originals if present)
    keep_cols = []
    for c in [
        "question_id",
        "query_edge_key",
        "edge_id",
        "subject",
        "predicate",
        "object",
        "subject_label",
        "object_label",
        "provenance",
        "key_exact",
    ]:
        if c in dup_rows.columns:
            keep_cols.append(c)

    dup_rows = dup_rows[keep_cols].sort_values(["question_id", "key_exact"])
    if args.sample_n and dup_rows.shape[0] > args.sample_n:
        dup_rows = dup_rows.head(args.sample_n)

    # ---- Write outputs ----
    out_prefix = Path(args.out_prefix)

    per_q_out = out_prefix.with_suffix(".per_question_counts.tsv")
    overall_out = out_prefix.with_suffix(".overall_summary.tsv")
    affected_out = out_prefix.with_suffix(".affected_questions.tsv")
    dup_sample_out = out_prefix.with_suffix(".duplicate_rows_sample.tsv")

    per_q.to_csv(per_q_out, sep="\t", index=False)
    overall.to_csv(overall_out, sep="\t", index=False)
    affected.to_csv(affected_out, sep="\t", index=False)
    dup_rows.to_csv(dup_sample_out, sep="\t", index=False)

    print("Wrote:")
    print(f"  {per_q_out}")
    print(f"  {overall_out}")
    print(f"  {affected_out}")
    print(f"  {dup_sample_out}")


if __name__ == "__main__":
    main()
