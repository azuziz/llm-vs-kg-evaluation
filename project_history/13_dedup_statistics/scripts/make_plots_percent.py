import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def is_rate_series(s: pd.Series) -> bool:
    s2 = s.dropna()
    if s2.empty:
        return False
    return (s2.min() >= 0.0) and (s2.max() <= 1.0)

def save_hist_percent(df: pd.DataFrame, col: str, outpath: Path, bins=20):
    if col not in df.columns:
        return
    data = pd.to_numeric(df[col], errors="coerce").dropna()
    if data.empty:
        return

    weights = [100.0 / len(data)] * len(data)

    plt.figure()
    plt.hist(data, bins=bins, weights=weights)
    plt.xlabel(col + (" (%)" if is_rate_series(data) else ""))
    plt.ylabel("% of questions")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def save_scatter(df: pd.DataFrame, x: str, y: str, outpath: Path):
    if x not in df.columns or y not in df.columns:
        return
    xx = pd.to_numeric(df[x], errors="coerce")
    yy = pd.to_numeric(df[y], errors="coerce")
    m = xx.notna() & yy.notna()
    xx = xx[m]
    yy = yy[m]
    if len(xx) == 0:
        return

    plt.figure()
    plt.scatter(xx, yy)
    plt.xlabel(x + (" (%)" if is_rate_series(xx) else ""))
    plt.ylabel(y + (" (%)" if is_rate_series(yy) else ""))
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_question", required=True, help="TSV with per-question metrics")
    ap.add_argument("--extended", required=False, help="Optional TSV with extended diagnostics")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hist_cols", default=None,
                    help="Comma-separated list of columns to histogram (overrides defaults)")
    ap.add_argument("--scatter_pairs", default=None,
                    help="Comma-separated list like x1:y1,x2:y2 (overrides defaults)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q = pd.read_csv(args.per_question, sep="\t")

    # Defaults for the original per_question_metrics.tsv
    default_hist = ["mean_f1", "match_rate", "gpt_edge_rate", "kg_edges"]
    default_scatter = [("kg_edges", "mean_f1"), ("gpt_edge_rate", "match_rate")]

    # If user passes explicit columns
    if args.hist_cols:
        hist_cols = [c.strip() for c in args.hist_cols.split(",") if c.strip()]
    else:
        hist_cols = default_hist

    if args.scatter_pairs:
        pairs = []
        for part in args.scatter_pairs.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                continue
            x, y = part.split(":", 1)
            pairs.append((x.strip(), y.strip()))
        scatter_pairs = pairs
    else:
        scatter_pairs = default_scatter

    # Make histograms
    for col in hist_cols:
        save_hist_percent(q, col, out_dir / f"hist_{col}.png")

    # Make scatters
    for x, y in scatter_pairs:
        save_scatter(q, x, y, out_dir / f"scatter_{x}_vs_{y}.png")

    # Extended optional
    if args.extended:
        e = pd.read_csv(args.extended, sep="\t")
        # These exist in per_question_extended_diagnostics.tsv
        for col in ["union_recall", "mode_share", "mean_pairwise_jaccard", "silence_rate",
                    "precision_cond_on_output", "recall_cond_on_output", "f1_cond_on_output"]:
            save_hist_percent(e, col, out_dir / f"hist_ext_{col}.png")
        save_scatter(e, "mode_share", "union_recall", out_dir / "scatter_mode_share_vs_union_recall.png")

if __name__ == "__main__":
    main()
