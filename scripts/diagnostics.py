"""Mechanism diagnostics (Section VII-F of the paper).

For Conf-GCN trained on every (seed, fold) of seed 0 we measure
  * edge homophily of the k-NN graphs,
  * how far the confidence weights move away from uniform (mean) aggregation:
    the relative spread  deg(i) * std_j(w_{j->i})  and max/min weight ratio
    inside each neighbourhood,
  * whether confidence is informative: AUROC of c_j for predicting that node j
    is classified correctly (test nodes), and the ECE of the classifier.
Output: results/diagnostics.json
"""
import json
import os
import sys
import warnings

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from confgcn.data import load_processed  # noqa: E402
from confgcn.layers import ConfidenceWeightedConv  # noqa: E402
from confgcn.train import cv_splits, set_seed  # noqa: E402
from confgcn.models import ConfidenceAwareGCN  # noqa: E402
import torch.nn.functional as F  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
hp = json.load(open("results/selected_hparams.json"))["confgcn"]
out = {}
for kind in ("desc", "feat"):
    g = load_processed("data/processed/twitter_5000.npz", 10, kind)
    s, d = g.edge_index.numpy()
    res = dict(edges=int(g.edge_index.shape[1]), edge_homophily=float((g.y[s] == g.y[d]).mean()),
               runs=[])
    for seed, fold, tr, va, te in cv_splits(g.y, seeds=(0,)):
        set_seed(seed)
        x = g.features(tr)
        m = ConfidenceAwareGCN(x.size(1))
        opt = torch.optim.Adam(m.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        y = torch.as_tensor(g.y)
        for _ in range(100):
            m.train(); opt.zero_grad()
            loss = F.cross_entropy(m(x, g.edge_index)[tr], y[tr]); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                m.update_confidence(m(x, g.edge_index))
        with torch.no_grad():
            logits = m(x, g.edge_index)
        c = m.confidence
        pred = logits.argmax(1).numpy()
        correct = (pred == g.y)
        w = ConfidenceWeightedConv.edge_weights(g.edge_index, c, len(g.y)).numpy()
        deg = np.bincount(d, minlength=len(g.y))
        # relative deviation from uniform 1/deg
        rel = np.abs(w * deg[d] - 1.0)
        order = np.argsort(d, kind="stable")
        starts = np.r_[0, np.cumsum(deg)[:-1]]
        ratio = np.maximum.reduceat(w[order], starts) / np.minimum.reduceat(w[order], starts)
        res["runs"].append(dict(
            fold=fold, mean_conf=float(c.mean()), conf_q10=float(np.quantile(c, 0.1)),
            conf_q90=float(np.quantile(c, 0.9)),
            mean_abs_rel_dev_from_uniform=float(rel.mean()),
            median_max_min_ratio=float(np.median(ratio)), p95_max_min_ratio=float(np.quantile(ratio, 0.95)),
            auroc_conf_vs_correct_test=float(roc_auc_score(correct[te], c.numpy()[te])),
            acc_test=float(correct[te].mean())))
        print(kind, res["runs"][-1], flush=True)
    agg = {k: float(np.mean([r[k] for r in res["runs"]])) for k in res["runs"][0] if k != "fold"}
    res["mean"] = agg
    out[kind] = res
json.dump(out, open("results/diagnostics.json", "w"), indent=1)
print(json.dumps({k: dict(h=v["edge_homophily"], **v["mean"]) for k, v in out.items()}, indent=1))
