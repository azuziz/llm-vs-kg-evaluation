import argparse
from pathlib import Path
import pandas as pd

def load_summary(path, label):
    df = pd.read_csv(path, sep="\t")
    df["prompt_version"] = label
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True,
                    help="Pairs: label=path, e.g. v1=out/v1/summary.tsv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = []
    for item in args.summaries:
        label, path = item.split("=", 1)
        frames.append(load_summary(path, label))

    df = pd.concat(frames, ignore_index=True)

    cols = [
        "prompt_version",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "zero_f1_rate",
        "mean_gpt_edges_per_run",
        "mean_match_rate_per_question"
    ]

    df[cols].to_csv(args.out, sep="\t", index=False)

if __name__ == "__main__":
    main()
