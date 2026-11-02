"""Confidence-weighted message passing layer.

For every directed edge j -> i the message W h_j is scaled by

    w_{j->i} = s_j / sum_{k in N(i)} s_k ,

where s_j is a non-negative per-node *source weight*. In the full
Confidence-Aware GCN s_j = c_j ** alpha with c_j the maximum softmax
probability of node j from the previous epoch. With s_j == 1 the layer is
exactly mean aggregation over N(i) (see tests/test_layer.py).

The layer has the same learnable parameters as GCNConv (one weight matrix
and one bias); the reweighting itself adds no parameters.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter


class ConfidenceWeightedConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, self_loop: bool = False):
        super().__init__(aggr="add")
        self.self_loop = self_loop
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer="glorot")
        self.bias = Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        torch.nn.init.zeros_(self.bias)

    @staticmethod
    def edge_weights(edge_index: Tensor, source_weight: Tensor, num_nodes: int) -> Tensor:
        """Per-edge weights renormalised over the incoming edges of each target."""
        src, dst = edge_index
        w = source_weight[src]
        denom = scatter(w, dst, dim=0, dim_size=num_nodes, reduce="sum")
        return w / denom[dst].clamp_min(1e-12)

    def prepare_edges(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        edge_index, _ = remove_self_loops(edge_index)
        if self.self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        return edge_index

    def forward(self, x: Tensor, edge_index: Tensor, source_weight: Tensor) -> Tensor:
        n = x.size(0)
        edge_index = self.prepare_edges(edge_index, n)
        w = self.edge_weights(edge_index, source_weight.detach(), n)
        x = self.lin(x)
        return self.propagate(edge_index, x=x, w=w) + self.bias

    def message(self, x_j: Tensor, w: Tensor) -> Tensor:
        return w.view(-1, 1) * x_j


class ConfidenceGCNConv(MessagePassing):
    """GCN convolution (symmetric normalisation + self-loops) modulated by confidence.

    Edge weights are
        a_{j->i} = 1 / sqrt(d_i d_j)                  (exactly GCNConv)
        w_{j->i} = a_{j->i} * s_j * (sum_k a_{k->i}) / (sum_k a_{k->i} s_k)
    i.e. the confidence only redistributes the GCN mass of each neighbourhood
    and keeps its row sum. With constant s the layer *equals* GCNConv, so any
    difference to the GCN baseline is due to the confidence signal alone.
    ``source_weight`` is not detached so that a learnable exponent alpha
    (s = c ** alpha) receives gradients.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer="glorot")
        self.bias = Parameter(torch.zeros(out_channels))
        self._cache = None

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        torch.nn.init.zeros_(self.bias)

    def _gcn_norm(self, edge_index: Tensor, n: int):
        if self._cache is not None and self._cache[0] is edge_index and self._cache[1] == n:
            return self._cache[2], self._cache[3]
        ei, _ = remove_self_loops(edge_index)
        ei, _ = add_self_loops(ei, num_nodes=n)
        src, dst = ei
        deg = scatter(torch.ones_like(src, dtype=torch.float), dst, dim=0, dim_size=n, reduce="sum")
        a = deg[src].rsqrt() * deg[dst].rsqrt()
        self._cache = (edge_index, n, ei, a)
        return ei, a

    def edge_weights(self, edge_index: Tensor, source_weight, n: int):
        """``source_weight``: per-node tensor s_j, or a callable (src, dst) -> per-edge factor."""
        ei, a = self._gcn_norm(edge_index, n)
        src, dst = ei
        f = source_weight(src, dst) if callable(source_weight) else source_weight[src]
        num = a * f
        row_a = scatter(a, dst, dim=0, dim_size=n, reduce="sum")
        row_num = scatter(num, dst, dim=0, dim_size=n, reduce="sum")
        return ei, num * (row_a / row_num.clamp_min(1e-12))[dst]

    def forward(self, x: Tensor, edge_index: Tensor, source_weight) -> Tensor:
        n = x.size(0)
        ei, w = self.edge_weights(edge_index, source_weight, n)
        return self.propagate(ei, x=self.lin(x), w=w) + self.bias

    def message(self, x_j: Tensor, w: Tensor) -> Tensor:
        return w.view(-1, 1) * x_j


class ConfidenceSAGEConv(MessagePassing):
    """GraphSAGE (mean) convolution whose neighbour mean is confidence-weighted:
        h_i' = W_r h_i + W_n * sum_j w_{j->i} h_j ,  w_{j->i} = s_j / sum_k s_k .
    With constant s it equals SAGEConv(aggr='mean')."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin_n = Linear(in_channels, out_channels, bias=True, weight_initializer="glorot")
        self.lin_r = Linear(in_channels, out_channels, bias=False, weight_initializer="glorot")

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_n.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, source_weight: Tensor) -> Tensor:
        n = x.size(0)
        ei, _ = remove_self_loops(edge_index)
        w = ConfidenceWeightedConv.edge_weights(ei, source_weight, n)
        agg = self.propagate(ei, x=x, w=w)
        return self.lin_n(agg) + self.lin_r(x)

    def message(self, x_j: Tensor, w: Tensor) -> Tensor:
        return w.view(-1, 1) * x_j
