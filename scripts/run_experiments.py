"""Run the experimental protocol of the paper (Section VI).

Groups
  select    hyper-parameter selection on the validation folds of seed 0
  main      MLP / GCN / GraphSAGE / GAT / Conf-GCN, 5-fold CV x 5 seeds
  ablation  uniform / shuffled / inverted confidence, alpha, self-loop
  density   k in {5, 20} for GCN and Conf-GCN (k = 10 is in `main`)
  graph     k-NN graph built on description + metadata instead of description only
  v2        future-work extensions (learnable alpha, AC-SAGE, agreement weights, self-training)
  meta      metadata-only k-NN graph
  meta2     remaining models and extensions on the metadata graph
  labels    label-scarce regime (2/5/10 % labelled nodes), feature graph

Every run is appended to results/runs_<tag>.csv and its validation/test
probabilities to results/probs/<variant>/<seed>_<fold>.npz, so the script is
resumable and the analysis (scripts/analyze.py) never re-trains anything.

Usage: python scripts/run_experiments.py --group select main ablation density
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from confgcn.data import load_processed  # noqa: E402
from confgcn.train import cv_splits, train_run  # noqa: E402

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--group", nargs="+", default=["select", "main", "ablation", "density"])
ap.add_argument("--data", default="data/processed/twitter_5000.npz")
ap.add_argument("--out", default="results")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--tag", default="0", help="worker tag; each worker appends to its own runs_<tag>.csv")
ap.add_argument("--only", nargs="*", default=None, help="restrict to these variant names")
a = ap.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(f"{a.out}/probs", exist_ok=True)
RUNS = f"{a.out}/runs_{a.tag}.csv"
HP_FILE = f"{a.out}/selected_hparams.json"

LR_GRID, WD_GRID, HEADS_GRID = [0.01, 0.005], [5e-4, 5e-5], [1, 4, 8]
BASE = {"mlp": {}, "gcn": {}, "sage": {}, "gat": {}, "confgcn": {}}


def all_runs() -> pd.DataFrame:
    files = sorted(glob.glob(f"{a.out}/runs_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files]) if files else pd.DataFrame(
        columns=["variant", "seed", "fold", "group"])


def done_keys():
    d = all_runs()
    return set(zip(d.variant, d.seed, d.fold))


def record(row: dict):
    pd.DataFrame([row]).to_csv(RUNS, mode="a", header=not os.path.exists(RUNS), index=False)


_graphs = {}


def graph(k, kind="desc"):
    if (k, kind) not in _graphs:
        _graphs[(k, kind)] = load_processed(a.data, k, kind)
    return _graphs[(k, kind)]


def label_splits(y, frac, reps=10):
    """Label-scarce splits: fixed 20 % test and 20 % validation, and a stratified
    training set of ``frac`` x |V| labelled nodes drawn from the rest."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    for rep in range(reps):
        rest, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=rep)
        rest2, va = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=rep)
        tr, _ = train_test_split(rest2, train_size=int(round(frac * len(y))), stratify=y[rest2],
                                 random_state=rep)
        yield rep, rep, np.sort(tr), np.sort(va), np.sort(te)


def run_variant(variant, group, model, k=10, seeds=None, save_probs=True, graph_kind="desc",
                label_frac=None, **kw):
    if a.only and variant not in a.only:
        return
    have = done_keys()
    g = graph(k, graph_kind)
    splits = (label_splits(g.y, label_frac) if label_frac is not None
              else cv_splits(g.y, seeds=tuple(seeds or a.seeds)))
    for seed, fold, tr, va, te in splits:
        if (variant, seed, fold) in have:
            continue
        t0 = time.time()
        r = train_run(model, g.features(tr), g.edge_index, g.y, tr, va, te, seed=seed,
                      epochs=a.epochs, **kw)
        row = dict(group=group, variant=variant, model=model, k=k, graph=graph_kind, seed=seed, fold=fold,
                   hparams=json.dumps(kw, sort_keys=True), secs=round(time.time() - t0, 2),
                   **{f"val_{m}": v for m, v in r.val.items()},
                   **{f"test_{m}": v for m, v in r.test.items()},
                   final_loss=r.loss[-1], loss_curve=json.dumps([round(v, 5) for v in r.loss]),
                   conf_curve=json.dumps([round(v, 5) for v in r.mean_conf]),
                   alpha_final=json.dumps(r.alpha[-1] if r.alpha else None),
                   label_frac=label_frac)
        record(row)
        if save_probs:
            d = f"{a.out}/probs/{variant}"
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(f"{d}/{seed}_{fold}.npz", y_val=g.y[va], p_val=r.p_val,
                                y_test=g.y[te], p_test=r.p_test, idx_test=te)
        print(f"[{group}] {variant} seed={seed} fold={fold} f1={r.test['f1']:.4f} "
              f"prec={r.test['prec']:.4f} rec={r.test['rec']:.4f} ({row['secs']}s)", flush=True)


def selected():
    with open(HP_FILE) as f:
        return json.load(f)


if "select" in a.group:
    for model in BASE:
        grid = [dict(lr=lr, weight_decay=wd) for lr, wd in itertools.product(LR_GRID, WD_GRID)]
        if model == "gat":
            grid = [dict(g, heads=h) for g in grid for h in HEADS_GRID]
        for hp in grid:
            name = f"sel_{model}_" + "_".join(f"{k}{v}" for k, v in sorted(hp.items()))
            run_variant(name, "select", model, seeds=[0], save_probs=False, **hp)
    d = all_runs()
    d = d[d.group == "select"]
    best = {}
    for model in BASE:
        s = d[d.model == model].groupby("hparams").val_f1.mean().sort_values(ascending=False)
        if a.only or s.empty:
            continue
        best[model] = json.loads(s.index[0])
        print(model, "best", best[model], f"val F1={s.iloc[0]:.4f}")
    if best:
        with open(HP_FILE, "w") as f:
            json.dump(best, f, indent=2)

if "main" in a.group:
    hp = selected()
    for model in ["mlp", "gcn", "sage", "gat", "confgcn"]:
        run_variant(model, "main", model, **hp[model])

if "ablation" in a.group:
    hp = selected()["confgcn"]
    for name, kw in [("conf_uniform", dict(mode="uniform")),
                     ("conf_shuffle", dict(mode="shuffle")),
                     ("conf_inverted", dict(mode="inverted")),
                     ("conf_alpha0.5", dict(alpha=0.5)),
                     ("conf_alpha2", dict(alpha=2.0)),
                     ("conf_alpha4", dict(alpha=4.0)),
                     ("conf_selfloop", dict(self_loop=True)),
                     ("conf_uniform_selfloop", dict(mode="uniform", self_loop=True))]:
        run_variant(name, "ablation", "confgcn", **hp, **kw)

if "density" in a.group:
    hp = selected()
    for k in (5, 20):
        for model in ("gcn", "confgcn"):
            run_variant(f"{model}_k{k}", "density", model, k=k, **hp[model])

if "graph" in a.group:
    hp = selected()
    for model in ("gcn", "confgcn", "sage", "gat"):
        run_variant(f"{model}_featgraph", "graph", model, graph_kind="feat", **hp[model])

# ---------------------------------------------------------------- extensions (future work)
V2 = [  # name, model, hp-source, extra kwargs
    ("acgcn", "acgcn", "gcn", {}),                       # learnable alpha on GCN normalisation
    ("acsage", "acsage", "sage", {}),                    # learnable alpha on SAGE mean
    ("cagree", "cagree", "gcn", {}),                     # edge weights from endpoint agreement
    ("gcn_st", "gcn", "gcn", dict(self_train=True)),     # pseudo-label self-training
    ("sage_st", "sage", "sage", dict(self_train=True)),
    ("acgcn_st", "acgcn", "gcn", dict(self_train=True)),
    ("acsage_st", "acsage", "sage", dict(self_train=True)),
]
if "v2" in a.group:
    hp = selected()
    for name, model, src, kw in V2:
        run_variant(f"{name}_featgraph", "v2", model, graph_kind="feat", **hp[src], **kw)
    for name, model, src, kw in V2[:3]:
        run_variant(f"{name}", "v2", model, graph_kind="desc", **hp[src], **kw)

if "meta" in a.group:
    hp = selected()
    for model in ("gcn", "sage"):
        run_variant(f"{model}_metagraph", "meta", model, graph_kind="meta", **hp[model])

LABEL_MODELS = [("mlp", "mlp", "mlp", {}), ("gcn", "gcn", "gcn", {}), ("sage", "sage", "sage", {}),
                ("acgcn", "acgcn", "gcn", {}), ("acsage", "acsage", "sage", {}),
                ("gcn_st", "gcn", "gcn", dict(self_train=True)),
                ("sage_st", "sage", "sage", dict(self_train=True))]
if "labels" in a.group:
    hp = selected()
    for frac in (0.02, 0.05, 0.10):
        for name, model, src, kw in LABEL_MODELS:
            run_variant(f"{name}_lab{frac}", "labels", model, graph_kind="feat", label_frac=frac,
                        **hp[src], **kw)

if "meta2" in a.group:  # complete the metadata-graph column and extensions on the best graph
    hp = selected()
    for model in ("gat", "confgcn"):
        run_variant(f"{model}_metagraph", "meta", model, graph_kind="meta", **hp[model])
    for name, model, src, kw in [V2[0], V2[1], V2[4]]:  # acgcn, acsage, sage_st
        run_variant(f"{name}_metagraph", "meta", model, graph_kind="meta", **hp[src], **kw)
