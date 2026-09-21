#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, type=Path)
    ap.add_argument("--out_tsv", required=True, type=Path)
    args = ap.parse_args()

    with args.in_csv.open("r", encoding="utf-8", newline="") as fin, \
         args.out_tsv.open("w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout, delimiter="\t", lineterminator="\n")

        for row in reader:
            writer.writerow(row)

    print(f"[OK] wrote {args.out_tsv}")


if __name__ == "__main__":
    main()
