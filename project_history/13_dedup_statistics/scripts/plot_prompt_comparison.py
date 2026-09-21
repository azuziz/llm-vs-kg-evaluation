import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["macro_precision", "macro_recall", "macro_f1", "zero_f1_rate"]

    for m in metrics:
        plt.figure()
        plt.bar(df["prompt_version"], df[m])
        plt.ylabel(m)
        plt.tight_layout()
        plt.savefig(out_dir / f"{m}.png")
        plt.close()

if __name__ == "__main__":
    main()
