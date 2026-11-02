"""Correctness tests for the confidence-weighted layer (RQ1)."""
import torch
from torch_geometric.utils import scatter

from confgcn.layers import ConfidenceWeightedConv
from confgcn.models import ConfidenceAwareGCN, GCN


def _graph(n=30, e=120, seed=0):
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, n, (2, e), generator=g)
    ei = ei[:, ei[0] != ei[1]]
    ei = torch.cat([ei, ei.flip(0)], 1)
    return torch.randn(n, 8, generator=g), ei


def test_weights_sum_to_one():
    x, ei = _graph()
    c = torch.rand(x.size(0)) * 0.5 + 0.5
    w = ConfidenceWeightedConv.edge_weights(ei, c, x.size(0))
    s = scatter(w, ei[1], dim=0, dim_size=x.size(0), reduce="sum")
    has_in = scatter(torch.ones_like(w), ei[1], dim=0, dim_size=x.size(0), reduce="sum") > 0
    assert torch.allclose(s[has_in], torch.ones(int(has_in.sum())), atol=1e-6)


def test_uniform_confidence_equals_mean_aggregation():
    x, ei = _graph()
    conv = ConfidenceWeightedConv(8, 4)
    out = conv(x, ei, torch.ones(x.size(0)))
    h = conv.lin(x)
    mean = scatter(h[ei[0]], ei[1], dim=0, dim_size=x.size(0), reduce="mean")
    assert torch.allclose(out, mean + conv.bias, atol=1e-5)


def test_scale_invariance_of_confidence():
    """Multiplying all confidences by a constant leaves the output unchanged."""
    x, ei = _graph()
    conv = ConfidenceWeightedConv(8, 4)
    c = torch.rand(x.size(0)) * 0.5 + 0.5
    assert torch.allclose(conv(x, ei, c), conv(x, ei, 3.0 * c), atol=1e-5)


def test_no_extra_parameters():
    a = sum(p.numel() for p in ConfidenceAwareGCN(16).parameters())
    b = sum(p.numel() for p in GCN(16).parameters())
    assert a == b


def test_confidence_not_a_parameter_and_no_grad():
    x, ei = _graph()
    m = ConfidenceAwareGCN(8)
    out = m(x, ei)
    m.update_confidence(out)
    assert not m.confidence.requires_grad
    assert all(p is not m.confidence for p in m.parameters())
    assert float(m.confidence.min()) >= 0.5 - 1e-6 and float(m.confidence.max()) <= 1 + 1e-6


def test_confidence_gcnconv_equals_gcnconv_for_constant_confidence():
    from torch_geometric.nn import GCNConv
    from confgcn.layers import ConfidenceGCNConv
    x, ei = _graph()
    ref = GCNConv(8, 4)
    conv = ConfidenceGCNConv(8, 4)
    conv.lin.weight.data.copy_(ref.lin.weight.data)
    conv.bias.data.copy_(ref.bias.data)
    for s in (torch.ones(x.size(0)), 0.7 * torch.ones(x.size(0))):
        assert torch.allclose(conv(x, ei, s), ref(x, ei), atol=1e-5)


def test_confidence_gcnconv_preserves_row_mass():
    from confgcn.layers import ConfidenceGCNConv
    x, ei = _graph()
    conv = ConfidenceGCNConv(8, 4)
    c = torch.rand(x.size(0)) * 0.5 + 0.5
    e1, w1 = conv.edge_weights(ei, torch.ones_like(c), x.size(0))
    e2, w2 = conv.edge_weights(ei, c, x.size(0))
    r1 = scatter(w1, e1[1], dim=0, dim_size=x.size(0), reduce="sum")
    r2 = scatter(w2, e2[1], dim=0, dim_size=x.size(0), reduce="sum")
    assert torch.allclose(r1, r2, atol=1e-5)


def test_adaptive_alpha_zero_is_baseline_and_alpha_gets_gradient():
    from confgcn.models import AdaptiveConfidenceGNN
    x, ei = _graph()
    m = AdaptiveConfidenceGNN(8, alpha_init=1.0)
    m.eval()
    m.confidence = torch.rand(x.size(0)) * 0.5 + 0.5
    out = m(x, ei)
    out.sum().backward()
    assert m.rho.grad is not None and torch.isfinite(m.rho.grad).all()
    m0 = AdaptiveConfidenceGNN(8, alpha_init=0.0, learn_alpha=False)
    m0.eval()
    a = m0(x, ei)
    m0.confidence = torch.rand(x.size(0)) * 0.5 + 0.5
    assert torch.allclose(a, m0(x, ei), atol=1e-5)
