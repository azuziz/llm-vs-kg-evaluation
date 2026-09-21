import pandas as pd
import os

f = os.path.expanduser("~/gpt/13_dedup_statistics/out/v1/per_run_metrics.tsv")
df = pd.read_csv(f, sep="\t")
print(df.columns)

# Look for common flags
for col in ["is_valid", "valid", "parse_ok", "schema_valid", "json_valid", "rejected", "error_type"]:
    if col in df.columns:
        print(col, df[col].value_counts(dropna=False).head(10))
