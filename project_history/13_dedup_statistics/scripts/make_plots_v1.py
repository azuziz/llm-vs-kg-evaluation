import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def save_hist(df, col, outpath, bins=20):
    plt.figure()
    df[col].hist(bins=bins)
    plt.xlabel(col)
    plt.ylabel("count of questions")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def save_boxplot(df, cols, outpath, ylabel, title, ylim=None, scale=1.0):
    data = [(df[c].astype(float) * scale).values for c in cols]

    plt.figure()
    plt.boxplot(data, tick_labels=cols, showfliers=True)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def save_scatter(df, x, y, outpath):
    plt.figure()
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def save_boxplot(df, cols, outpath, ylabel, title, ylim=None, scale=1.0):
    """
    Generic boxplot saver.
    scale=100.0 is useful for percent plots.
    """
    data = [(df[c].astype(float) * scale).values for c in cols]

    plt.figure()
    plt.boxplot(data, tick_labels=cols, showfliers=True)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_question", required=True)
    ap.add_argument("--extended", required=False)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q = pd.read_csv(args.per_question, sep="\t")

    # Boxplots across questions (rates/scores in [0,1])
    save_boxplot(
        q,
        cols=["mean_precision", "mean_recall", "mean_f1"],
        outpath=out_dir / "box_mean_PRF1.png",
        ylabel="score",
        title="Distributions across questions (mean over runs)",
        ylim=(0, 1),
    )

    save_boxplot(
        q,
        cols=["match_rate", "gpt_edge_rate"],
        outpath=out_dir / "box_match_and_edge_rate.png",
        ylabel="rate",
        title="Distributions across questions (rates)",
        ylim=(0, 1),
    )

    # Optional: percent versions (often more readable)
    save_boxplot(
        q,
        cols=["mean_precision", "mean_recall", "mean_f1"],
        outpath=out_dir / "box_mean_PRF1_percent.png",
        ylabel="score (%)",
        title="Distributions across questions (mean over runs, %)",
        ylim=(0, 100),
        scale=100.0,
    )

    save_boxplot(
        q,
        cols=["match_rate", "gpt_edge_rate"],
        outpath=out_dir / "box_match_and_edge_rate_percent.png",
        ylabel="rate (%)",
        title="Distributions across questions (rates, %)",
        ylim=(0, 100),
        scale=100.0,
    )

    # Counts: keep separate axis (not bounded to [0,1])
    save_boxplot(
        q,
        cols=["kg_edges", "mean_gpt_edges", "mean_matched_edges"],
        outpath=out_dir / "box_edge_counts.png",
        ylabel="count",
        title="Distributions across questions (edge counts)",
        ylim=None,
    )

    # Boxplots across questions (per-question metrics)
    save_boxplot(
        q,
        cols=["mean_f1", "match_rate", "gpt_edge_rate"],
        outpath=out_dir / "box_v1_core_metrics.png",
        ylabel="value",
        title="Distributions across questions (v1 core metrics)",
        ylim=(0, 1),
    )

    # Separate boxplot for counts
    save_boxplot(
        q,
        cols=["kg_edges"],
        outpath=out_dir / "box_v1_kg_edges.png",
        ylabel="count",
        title="Distribution across questions (KG edges per pair)",
        ylim=None,
    )

    # Optional: mean_f1 in percent
    save_boxplot(
        q,
        cols=["mean_f1"],
        outpath=out_dir / "box_v1_mean_f1_percent.png",
        ylabel="F1 (%)",
        title="Distribution across questions (mean F1, %)",
        ylim=(0, 100),
        scale=100.0,
    )
	
    save_hist(q, "mean_precision", out_dir / "hist_mean_precision.png")
    save_hist(q, "mean_recall", out_dir / "hist_mean_recall.png")
    save_hist(q, "mean_f1", out_dir / "hist_mean_f1.png")
    save_hist(q, "match_rate", out_dir / "hist_match_rate.png")
    save_hist(q, "gpt_edge_rate", out_dir / "hist_gpt_edge_rate.png")
    save_hist(q, "kg_edges", out_dir / "hist_kg_edges.png")

    save_scatter(q, "kg_edges", "mean_f1", out_dir / "scatter_kg_edges_vs_mean_f1.png")
    save_scatter(q, "gpt_edge_rate", "match_rate", out_dir / "scatter_gpt_edge_rate_vs_match_rate.png")

    if args.extended:
        e = pd.read_csv(args.extended, sep="\t")

        # Boxplots across questions (extended metrics)
        save_boxplot(
            e,
            cols=["union_recall", "mode_share"],
            outpath=out_dir / "box_v1_extended_metrics.png",
            ylabel="value",
            title="Distributions across questions (v1 extended metrics)",
            ylim=(0, 1),
        )

        # Percent version
        save_boxplot(
            e,
            cols=["union_recall", "mode_share"],
            outpath=out_dir / "box_v1_extended_metrics_percent.png",
            ylabel="value (%)",
            title="Distributions across questions (v1 extended metrics, %)",
            ylim=(0, 100),
            scale=100.0,
        )

        save_hist(e, "union_recall", out_dir / "hist_union_recall.png")
        save_hist(e, "mode_share", out_dir / "hist_mode_share.png")
        save_scatter(e, "mode_share", "union_recall", out_dir / "scatter_mode_share_vs_union_recall.png")

if __name__ == "__main__":
    main()