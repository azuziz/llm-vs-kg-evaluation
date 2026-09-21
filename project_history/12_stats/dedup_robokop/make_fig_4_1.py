#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

IN_DIR = os.path.expanduser("~/gpt/12_stats/dedup_robokop")
IN_FILE = os.path.join(IN_DIR, "robokop_dedup.per_question_counts.tsv")

OUT_DIR = os.path.join(IN_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG = os.path.join(OUT_DIR, "fig_4_1_kg_edge_count_distribution.png")

df = pd.read_csv(IN_FILE, sep="\t")

# --- Try to infer column names robustly ---
# Common patterns you might have:
# - raw_count / raw_edges / n_edges_raw
# - dedup_count / dedup_edges / n_edges_dedup / unique_edges
cols_lower = {c.lower(): c for c in df.columns}

def find_col(candidates):
    for cand in candidates:
        for c_lower, c_orig in cols_lower.items():
            if cand in c_lower:
                return c_orig
    return None

raw_col = find_col(["raw", "before"])
dedup_col = find_col(["dedup", "after", "unique"])

if dedup_col is None:
    raise ValueError(
        f"Could not infer a deduplicated edge-count column from columns: {list(df.columns)}.\n"
        "Open the TSV and set dedup_col manually."
    )

# Prefer plotting deduplicated counts (evaluation baseline)
dedup_counts = df[dedup_col].astype(int)

# If counts are guaranteed 1..10, set bins accordingly; otherwise infer a safe range
min_c, max_c = int(dedup_counts.min()), int(dedup_counts.max())
bins = range(min_c, max_c + 2)  # integer bins

plt.figure(figsize=(7, 4.2))
plt.hist(dedup_counts, bins=bins, edgecolor="black", align="left")
plt.xlabel("Number of deduplicated ROBOKOP edges per entity pair")
plt.ylabel("Number of entity pairs")
plt.title("Distribution of baseline KG edge counts (deduplicated)")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()

print(f"[OK] Saved Figure 4.1 to: {OUT_FIG}")

# Optional: also export a raw histogram if available
if raw_col is not None:
    OUT_FIG_RAW = os.path.join(OUT_DIR, "fig_4_1b_kg_edge_count_distribution_raw.png")
    raw_counts = df[raw_col].astype(int)
    min_r, max_r = int(raw_counts.min()), int(raw_counts.max())
    bins_r = range(min_r, max_r + 2)

    plt.figure(figsize=(7, 4.2))
    plt.hist(raw_counts, bins=bins_r, edgecolor="black", align="left")
    plt.xlabel("Number of raw ROBOKOP edges per entity pair")
    plt.ylabel("Number of entity pairs")
    plt.title("Distribution of baseline KG edge counts (raw, pre-deduplication)")
    plt.tight_layout()
    plt.savefig(OUT_FIG_RAW, dpi=300)
    plt.close()
    print(f"[OK] Saved raw distribution (optional) to: {OUT_FIG_RAW}")
