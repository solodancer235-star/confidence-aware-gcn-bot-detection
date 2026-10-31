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
