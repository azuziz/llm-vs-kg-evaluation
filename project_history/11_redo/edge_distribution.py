import pandas as pd

# Input file
input_file = "robokop_edges_per_question.tsv"
output_file = "edge_count_distribution.tsv"

# Load TSV
df = pd.read_csv(input_file, sep="\t")

# Count edge frequencies (0–9)
dist = (
    df["robokop_n_edges"]
    .value_counts()
    .reindex(range(10), fill_value=0)  # ensures 0–9 included
    .sort_index()
    .rename_axis("edge_count")
    .reset_index(name="num_questions")
)

# Save result
dist.to_csv(output_file, sep="\t", index=False)

print("Saved:", output_file)
print(dist)
