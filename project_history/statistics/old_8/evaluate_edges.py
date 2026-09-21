import os, json, glob
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats

ROOT = "."
GT_DIR = "robokop_triples_by_question"
PRED_DIR = "gpt_runs"

# -----------------------
# Helpers
# -----------------------

def canon_edge(a, pred, b):
    """Direction-agnostic canonical edge"""
    return tuple(sorted([a, b]) + [pred])

def load_gt(question_id):
    path = os.path.join(GT_DIR, f"{question_id}.tsv")
    df = pd.read_csv(path, sep="\t")
    edges = set()
    for _, r in df.iterrows():
        edges.add(canon_edge(r["subject"], r["predicate"], r["object"]))
    return edges, df

def load_predictions(question_id):
    qdir = os.path.join(PRED_DIR, question_id)
    runs = []
    for f in glob.glob(os.path.join(qdir, "*.json")):
        with open(f) as fh:
            data = json.load(fh)
        preds = set()
        for t in data.get("triples", []):
            preds.add(canon_edge(t["subject"], t["predicate"], t["object"]))
        runs.append(preds)
    return runs

# -----------------------
# Main evaluation
# -----------------------

all_question_stats = []
all_tp = all_fp = all_fn = 0

edge_recovery = defaultdict(Counter)

for q in sorted(os.listdir(PRED_DIR)):
    print("Processing", q)

    gt_edges, gt_df = load_gt(q)
    pred_runs = load_predictions(q)

    precs, recs, f1s = [], [], []

    for run_edges in pred_runs:
        tp = len(run_edges & gt_edges)
        fp = len(run_edges - gt_edges)
        fn = len(gt_edges - run_edges)

        all_tp += tp
        all_fp += fp
        all_fn += fn

        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2*p*r/(p+r) if p+r else 0

        precs.append(p)
        recs.append(r)
        f1s.append(f1)

        for e in run_edges & gt_edges:
            edge_recovery[q][e] += 1

    stats_q = {
        "question": q,
        "precision_mean": sum(precs)/len(precs),
        "recall_mean": sum(recs)/len(recs),
        "f1_mean": sum(f1s)/len(f1s),
        "f1_std": pd.Series(f1s).std()
    }

    all_question_stats.append(stats_q)

# -----------------------
# Summary tables
# -----------------------

df_stats = pd.DataFrame(all_question_stats)
df_stats.to_csv("per_question_stats.csv", index=False)

micro_p = all_tp / (all_tp + all_fp)
micro_r = all_tp / (all_tp + all_fn)
micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r)

macro_f1 = df_stats["f1_mean"].mean()

print("\n=== OVERALL ===")
print("Micro P/R/F1:", micro_p, micro_r, micro_f1)
print("Macro F1:", macro_f1)

# -----------------------
# Plots
# -----------------------

# Bar plot of mean F1
plt.figure()
plt.bar(df_stats["question"], df_stats["f1_mean"])
plt.ylabel("Mean F1")
plt.title("Mean F1 per Question")
plt.savefig("f1_bar_per_question.png", dpi=200)
plt.close()

# Boxplots
box_data = []
labels = []
for q in sorted(os.listdir(PRED_DIR)):
    gt_edges, _ = load_gt(q)
    runs = load_predictions(q)
    f1s = []
    for run_edges in runs:
        tp = len(run_edges & gt_edges)
        fp = len(run_edges - gt_edges)
        fn = len(gt_edges - run_edges)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        f1s.append(f1)
    box_data.append(f1s)
    labels.append(q)

plt.figure(figsize=(10,5))
sns.boxplot(data=box_data)
plt.xticks(range(len(labels)), labels)
plt.ylabel("F1")
plt.title("F1 Distribution Across 100 Runs")
plt.savefig("f1_boxplots.png", dpi=200)
plt.close()

# -----------------------
# Edge recovery plots
# -----------------------

for q, counter in edge_recovery.items():
    labels = [str(i) for i in range(len(counter))]
    vals = [v/100 for v in counter.values()]

    plt.figure()
    plt.bar(labels, vals)
    plt.ylim(0,1)
    plt.ylabel("Recovery Rate")
    plt.title(f"GT Edge Recovery Rates — {q}")
    plt.savefig(f"edge_recovery_{q}.png", dpi=200)
    plt.close()

# -----------------------
# Network diagrams
# -----------------------

for q in sorted(os.listdir(PRED_DIR)):
    gt_edges, gt_df = load_gt(q)
    runs = load_predictions(q)

    freq = Counter()
    for r in runs:
        for e in r:
            freq[e] += 1

    G = nx.Graph()

    for (a, b, pred), c in freq.items():
        G.add_edge(a, b, weight=c, label=pred)

    plt.figure(figsize=(6,6))
    pos = nx.spring_layout(G, seed=3)
    widths = [G[u][v]["weight"]/10 for u,v in G.edges()]
    nx.draw(G, pos, with_labels=False, width=widths, node_size=800)

    nx.draw_networkx_labels(G, pos, font_size=7)

    plt.title(f"Predicted Edge Frequencies — {q}")
    plt.savefig(f"network_pred_{q}.png", dpi=200)
    plt.close()

print("\nSaved outputs:")
print("- per_question_stats.csv")
print("- f1_bar_per_question.png")
print("- f1_boxplots.png")
print("- edge_recovery_Q*.png")
print("- network_pred_Q*.png")
