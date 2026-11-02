"""Models compared in the paper.

All graph models use 3 layers, hidden size 64 and dropout 0.5 so that
capacity is matched. ``ConfidenceAwareGCN`` differs from the GCN baseline only
in the convolution (``ConfidenceWeightedConv`` instead of ``GCNConv``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

import math

from .layers import ConfidenceGCNConv, ConfidenceSAGEConv, ConfidenceWeightedConv

CONF_MODES = ("conf", "uniform", "shuffle", "inverted")


class _Stack(nn.Module):
    """Generic L-layer stack: conv -> ReLU -> dropout, last layer outputs logits."""

    uses_graph = True

    def __init__(self, convs: list[nn.Module], dropout: float):
        super().__init__()
        self.convs = nn.ModuleList(convs)
        self.dropout = dropout

    def _conv(self, conv, x, edge_index):
        return conv(x, edge_index)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self._conv(conv, x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


def _dims(in_dim, hidden, out_dim, layers):
    return [in_dim] + [hidden] * (layers - 1) + [out_dim]


class MLP(_Stack):
    uses_graph = False

    def __init__(self, in_dim, hidden=64, out_dim=2, layers=3, dropout=0.5):
        d = _dims(in_dim, hidden, out_dim, layers)
        super().__init__([nn.Linear(d[i], d[i + 1]) for i in range(layers)], dropout)

    def _conv(self, conv, x, edge_index):
        return conv(x)


class GCN(_Stack):
    def __init__(self, in_dim, hidden=64, out_dim=2, layers=3, dropout=0.5):
        d = _dims(in_dim, hidden, out_dim, layers)
        super().__init__([GCNConv(d[i], d[i + 1]) for i in range(layers)], dropout)


class GraphSAGE(_Stack):
    def __init__(self, in_dim, hidden=64, out_dim=2, layers=3, dropout=0.5):
        d = _dims(in_dim, hidden, out_dim, layers)
        super().__init__([SAGEConv(d[i], d[i + 1], aggr="mean") for i in range(layers)], dropout)


class GAT(_Stack):
    """GAT with total hidden width 64 (heads x 64/heads), single-head output layer."""

    def __init__(self, in_dim, hidden=64, out_dim=2, layers=3, dropout=0.5, heads=4):
        assert hidden % heads == 0
        convs, d_in = [], in_dim
        for _ in range(layers - 1):
            convs.append(GATConv(d_in, hidden // heads, heads=heads, concat=True, dropout=dropout))
            d_in = hidden
        convs.append(GATConv(d_in, out_dim, heads=1, concat=False, dropout=dropout))
        super().__init__(convs, dropout)


class ConfidenceAwareGCN(_Stack):
    """Confidence-Aware GCN.

    mode:
      * ``conf``     - s_j = c_j ** alpha (the proposed method, alpha=1 by default)
      * ``uniform``  - s_j = 1 (same layer, mean aggregation; normalisation control)
      * ``shuffle``  - confidences randomly permuted across nodes each epoch
      * ``inverted`` - s_j = 1 - c_j + 0.5 (favours low-confidence neighbours)

    Confidence is a plain tensor attribute (not a parameter / not in the
    autograd graph). It is initialised lazily to ones and refreshed by
    :meth:`update_confidence` from a no-grad eval-mode forward pass.
    """

    def __init__(self, in_dim, hidden=64, out_dim=2, layers=3, dropout=0.5,
                 mode="conf", alpha=1.0, self_loop=False):
        assert mode in CONF_MODES
        d = _dims(in_dim, hidden, out_dim, layers)
        super().__init__([ConfidenceWeightedConv(d[i], d[i + 1], self_loop=self_loop)
                          for i in range(layers)], dropout)
        self.mode, self.alpha = mode, float(alpha)
        self.confidence: Tensor | None = None

    def source_weight(self, n: int, device) -> Tensor:
        if self.confidence is None or self.confidence.numel() != n:
            self.confidence = torch.ones(n, device=device)
        c = self.confidence
        if self.mode == "uniform":
            return torch.ones_like(c)
        if self.mode == "inverted":
            return 1.0 - c + 0.5
        return c.pow(self.alpha)  # conf and shuffle (shuffle permutes c itself)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        s = self.source_weight(x.size(0), x.device)
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, s)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

    @torch.no_grad()
    def update_confidence(self, logits: Tensor, generator: torch.Generator | None = None) -> Tensor:
        c = F.softmax(logits, dim=-1).max(dim=-1).values.detach()
        raw = c.clone()
        if self.mode == "shuffle":
            perm = torch.randperm(c.numel(), generator=generator)
            c = c[perm.to(c.device)]
        self.confidence = c
        return raw


class AdaptiveConfidenceGNN(_Stack):
    """Confidence-aware GNN v2 (``base='gcn'`` -> AC-GCN, ``base='sage'`` -> AC-SAGE).

    * confidence modulates the *baseline's own* normalisation (GCN: symmetric
      with self-loops; SAGE: neighbour mean next to a root transform), so with
      alpha = 0 the model is exactly the baseline;
    * s_j = c_j ** alpha_l with a learnable, layer-wise exponent
      alpha_l = softplus(rho_l) >= 0 (3 scalar parameters). Large alpha removes
      the factor-2 bound on neighbour influence of the fixed alpha = 1 model.
    """

    def __init__(self, in_dim, hidden=64, out_dim=2, layers=3, dropout=0.5, base="gcn",
                 alpha_init=1.0, learn_alpha=True, signal="node"):
        assert signal in ("node", "agree")
        assert not (signal == "agree" and base != "gcn")
        self_signal = signal
        d = _dims(in_dim, hidden, out_dim, layers)
        conv = {"gcn": ConfidenceGCNConv, "sage": ConfidenceSAGEConv}[base]
        super().__init__([conv(d[i], d[i + 1]) for i in range(layers)], dropout)
        rho = math.log(math.expm1(alpha_init)) if alpha_init > 0 else -20.0
        init = torch.full((layers,), rho)
        if learn_alpha:
            self.rho = nn.Parameter(init)
        else:
            self.register_buffer("rho", init)
        self.signal = self_signal
        self.confidence: Tensor | None = None
        self.prob: Tensor | None = None

    @property
    def alpha(self) -> Tensor:
        return F.softplus(self.rho)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n = x.size(0)
        if self.confidence is None or self.confidence.numel() != n:
            self.confidence = torch.ones(n, device=x.device)
        logc = self.confidence.clamp_min(1e-6).log()
        alpha = self.alpha
        prob = self.prob if self.prob is not None and self.prob.size(0) == n else None
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.signal == "agree":
                if prob is None:
                    sw = torch.ones(n, device=x.device)
                else:
                    a_i = alpha[i]
                    sw = lambda src, dst, a_i=a_i: torch.exp(  # noqa: E731
                        a_i * (prob[src] * prob[dst]).sum(-1).clamp_min(1e-6).log())
            else:
                sw = torch.exp(alpha[i] * logc)
            x = conv(x, edge_index, sw)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

    @torch.no_grad()
    def update_confidence(self, logits: Tensor, generator=None) -> Tensor:
        p = F.softmax(logits, dim=-1).detach()
        c = p.max(dim=-1).values
        self.confidence, self.prob = c, p
        return c.clone()


def build_model(name: str, in_dim: int, **kw) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLP(in_dim)
    if name == "gcn":
        return GCN(in_dim)
    if name == "sage":
        return GraphSAGE(in_dim)
    if name == "gat":
        return GAT(in_dim, heads=kw.get("heads", 4))
    if name == "confgcn":
        return ConfidenceAwareGCN(in_dim, mode=kw.get("mode", "conf"), alpha=kw.get("alpha", 1.0),
                                  self_loop=kw.get("self_loop", False))
    if name in ("acgcn", "acsage", "cagree"):
        base = "sage" if name == "acsage" else "gcn"
        return AdaptiveConfidenceGNN(in_dim, base=base, alpha_init=kw.get("alpha_init", 1.0),
                                     learn_alpha=kw.get("learn_alpha", True),
                                     signal="agree" if name == "cagree" else "node")
    raise ValueError(name)
