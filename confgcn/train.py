"""Training, evaluation, splits and statistics."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .models import AdaptiveConfidenceGNN, ConfidenceAwareGCN, build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cv_splits(y: np.ndarray, seeds=(0, 1, 2, 3, 4), n_folds=5, val_frac=0.25):
    """Repeated stratified k-fold; the training part of each fold is split again
    into train/val (val_frac=0.25 of 80% -> 60/20/20 overall)."""
    idx = np.arange(len(y))
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (trval, test) in enumerate(skf.split(idx, y)):
            tr, va = train_test_split(trval, test_size=val_frac, stratify=y[trval], random_state=seed)
            yield seed, fold, np.sort(tr), np.sort(va), np.sort(test)


def metrics(y_true: np.ndarray, p_bot: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p_bot >= thr).astype(int)
    out = dict(acc=accuracy_score(y_true, pred),
               prec=precision_score(y_true, pred, zero_division=0),
               rec=recall_score(y_true, pred, zero_division=0),
               f1=f1_score(y_true, pred, zero_division=0))
    if len(np.unique(y_true)) == 2:
        out["auroc"] = roc_auc_score(y_true, p_bot)
        out["aucpr"] = average_precision_score(y_true, p_bot)
    return out


@dataclass
class RunResult:
    val: dict
    test: dict
    p_val: np.ndarray
    p_test: np.ndarray
    loss: list = field(default_factory=list)
    mean_conf: list = field(default_factory=list)
    alpha: list = field(default_factory=list)
    p_all: np.ndarray | None = None


def train_run(model_name: str, x: torch.Tensor, edge_index: torch.Tensor, y: np.ndarray,
              tr: np.ndarray, va: np.ndarray, te: np.ndarray, seed: int, epochs: int = 100,
              lr: float = 0.01, weight_decay: float = 5e-4, self_train: bool = False,
              st_tau: float = 0.9, st_lambda: float = 0.5, st_warmup: int = 20,
              **model_kw) -> RunResult:
    """Train one model. With ``self_train`` the confident predictions (max
    softmax >= st_tau) of the previous epoch on *unlabelled* nodes (all nodes
    outside ``tr``; their true labels are never used) are added as
    pseudo-labels with weight st_lambda after st_warmup epochs."""
    set_seed(seed)
    model = build_model(model_name, x.size(1), **model_kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    yt = torch.as_tensor(y)
    tr_t = torch.as_tensor(tr)
    gen = torch.Generator().manual_seed(seed)
    is_conf = isinstance(model, (ConfidenceAwareGCN, AdaptiveConfidenceGNN))
    losses, confs, alphas = [], [], []
    unl = torch.ones(len(y), dtype=torch.bool)
    unl[tr_t] = False
    prev_prob = None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[tr_t], yt[tr_t])
        if self_train and prev_prob is not None and epoch >= st_warmup:
            conf_u, pseudo = prev_prob.max(dim=1)
            m = unl & (conf_u >= st_tau)
            if m.any():
                loss = loss + st_lambda * F.cross_entropy(out[m], pseudo[m])
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if is_conf or self_train:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index)
            prev_prob = F.softmax(logits, dim=-1)
            if is_conf:
                raw = model.update_confidence(logits, generator=gen)
                confs.append(float(raw.mean()))
            if isinstance(model, AdaptiveConfidenceGNN):
                alphas.append([round(float(a), 4) for a in model.alpha])
    model.eval()
    with torch.no_grad():
        p = F.softmax(model(x, edge_index), dim=-1)[:, 1].numpy()
    return RunResult(val=metrics(y[va], p[va]), test=metrics(y[te], p[te]),
                     p_val=p[va], p_test=p[te], loss=losses, mean_conf=confs, p_all=p, alpha=alphas)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down adjustment."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adj[k] = running
    return adj


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def recall_matched_threshold(y_val: np.ndarray, p_val: np.ndarray, target_recall: float) -> float:
    """Largest threshold whose validation recall is >= target_recall."""
    pos = np.sort(p_val[y_val == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = int(np.ceil(target_recall * len(pos)))
    k = min(max(k, 1), len(pos))
    return float(pos[k - 1])
