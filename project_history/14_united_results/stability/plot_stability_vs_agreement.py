#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-tsv", nargs="+", required=True,
                    help="One or more *_stability_with_agreement.tsv files.")
    ap.add_argument("--x", default="predicate_set_stability")
    ap.add_argument("--y", default="mean_f1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dfs = []
    for p in args.merged_tsv:
        df = pd.read_csv(p, sep="\t")
        df["prompt"] = Path(p).name.split("_")[0]  # v1/v2/v3/v4 from filename prefix
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    if args.x not in data.columns:
        raise SystemExit(f"Missing x='{args.x}'. Available columns: {list(data.columns)}")
    if args.y not in data.columns:
        raise SystemExit(f"Missing y='{args.y}'. Available columns: {list(data.columns)}")

    x = pd.to_numeric(data[args.x], errors="coerce")
    y = pd.to_numeric(data[args.y], errors="coerce")

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title("Stability vs agreement across questions")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()