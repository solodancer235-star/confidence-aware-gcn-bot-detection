"""Synthetic social graph: correctness check (RQ1) and Fig. 1.

Usage: python scripts/run_synthetic.py
"""
import json
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from confgcn.data import synthetic_graph  # noqa: E402
from confgcn.train import cv_splits, train_run  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
OUT = "results/synthetic"
os.makedirs(OUT, exist_ok=True)

G, emb, y, ei, stats = synthetic_graph(seed=42)
print(stats)
x = torch.as_tensor(emb)

# single stratified 60/20/20 split (paper Table III)
idx = np.arange(len(y))
trva, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
tr, va = train_test_split(trva, test_size=0.25, stratify=y[trva], random_state=42)
single = {}
for m in ["gcn", "confgcn"]:
    r = train_run(m, x, ei, y, np.sort(tr), np.sort(va), np.sort(te), seed=42)
    single[m] = dict(test=r.test, loss=r.loss, conf=r.mean_conf)
    print(m, r.test)

# repeated CV for all models (sanity: all should saturate)
rows = []
for seed, fold, tr_, va_, te_ in cv_splits(y):
    for m in ["mlp", "gcn", "sage", "gat", "confgcn"]:
        r = train_run(m, x, ei, y, tr_, va_, te_, seed=seed)
        rows.append(dict(model=m, seed=seed, fold=fold, **r.test))
cv = pd.DataFrame(rows)
cv.to_csv(f"{OUT}/cv_runs.csv", index=False)
print(cv.groupby("model")[["acc", "prec", "rec", "f1"]].agg(["mean", "std"]).round(4))

with open(f"{OUT}/synthetic_results.json", "w") as f:
    json.dump(dict(stats=stats, single=single), f, indent=1)

# Fig. 1: graph coloured by Conf-GCN prediction (all nodes)
r = train_run("confgcn", x, ei, y, np.sort(tr), np.sort(va), np.sort(te), seed=42)
pred = (r.p_all >= 0.5).astype(int)
pos = nx.spring_layout(G, seed=7, k=0.08, iterations=200)
cmap = {(0, 0): "#2ca02c", (1, 1): "#d62728", (0, 1): "#ff7f0e", (1, 0): "#e6c229"}
col = [cmap[(int(t), int(p))] for t, p in zip(y, pred)]
fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=200)
nx.draw_networkx_edges(G, pos, ax=ax, width=0.15, alpha=0.15, edge_color="#888888")
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=9, node_color=col, linewidths=0)
for c, lab in [("#2ca02c", "human, correct"), ("#d62728", "bot, correct"),
               ("#ff7f0e", "false positive"), ("#e6c229", "false negative")]:
    ax.scatter([], [], c=c, s=20, label=lab)
ax.legend(loc="lower left", frameon=False, fontsize=7)
ax.set_axis_off()
plt.tight_layout()
plt.savefig("paper/figures/fig1_synthetic_graph.png", bbox_inches="tight")
print("saved fig1; misclassified:", int((pred != y).sum()))
