"""Confidence-Aware Graph Convolutional Networks for social bot detection."""
from .layers import ConfidenceWeightedConv
from .models import ConfidenceAwareGCN, build_model

__all__ = ["ConfidenceWeightedConv", "ConfidenceAwareGCN", "build_model"]
__version__ = "1.0.0"
