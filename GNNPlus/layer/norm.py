"""Pluggable normalization layers for the GNN+ course project.

This is the only new component introduced by our CS5491 project.  The official
GNN+ implementation hard-codes ``nn.BatchNorm1d`` inside every convolution
layer, so experiments can only ask "normalization: yes or no?".  Exposing the
*type* of normalization lets us ask "which normalization, and does it interact
with depth?".

Design constraints
------------------
* ``build_norm('batch', d)`` must be numerically identical to the original
  ``nn.BatchNorm1d(d)`` so that the published configuration is reproducible.
* Every normalization shares the call signature ``forward(x, batch)``, where
  ``batch`` is the PyG graph-index vector (``data.batch``).  Module types that
  do not need it simply ignore it.
* The module depends only on PyTorch; ``torch_scatter`` is used for GraphNorm's
  per-graph mean when available, with a global-mean fallback otherwise.

References
----------
[1] Cai et al. GraphNorm: A Principled Approach to Accelerating Graph Neural
    Network Training. ICML 2021.
[2] Zhao & Akoglu. PairNorm: Tackling Oversmoothing in GNNs. ICLR 2020.
"""

import torch
import torch.nn as nn

try:  # GNN+ already depends on torch_scatter; keep a fallback for portability.
    from torch_scatter import scatter
    _HAS_SCATTER = True
except ImportError:  # pragma: no cover - exercised only on minimal installs
    scatter = None
    _HAS_SCATTER = False


class IdentityNorm(nn.Module):
    """No-op normalization, used for the ``none`` control condition."""

    def forward(self, x, batch=None):
        return x


class PairNorm(nn.Module):
    """PairNorm: center all node features, then rescale to constant variance.

    Originally proposed to fight oversmoothing in deep GNNs.  It has no
    learnable parameters, which makes it a cheap baseline against BatchNorm.
    """

    def __init__(self, dim, scale=1.0, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.eps = eps

    def forward(self, x, batch=None):
        x = x - x.mean(dim=0, keepdim=True)
        rms = x.pow(2).sum(dim=1).mean().clamp(min=self.eps).sqrt()
        return self.scale * x / (rms + self.eps)


class GraphNorm(nn.Module):
    """GraphNorm with a learnable global/per-graph mean mixture (alpha).

    ``alpha = 0`` reduces to a global normalization; positive alpha lets the
    module adapt to the statistics of individual graphs in the batch, which is
    the property we expect to matter at larger depths.
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, batch=None):
        if batch is not None and _HAS_SCATTER:
            mean = scatter(x, batch, dim=0, reduce='mean')
            mean = self.alpha * mean + (1.0 - self.alpha) * mean.mean(
                dim=0, keepdim=True)
            mean = mean[batch]
        else:
            mean = x.mean(dim=0, keepdim=True)
        x = x - mean
        std = x.std(dim=0, unbiased=False, keepdim=True)
        return x / (std + self.eps) * self.weight + self.bias


class NormWrapper(nn.Module):
    """Uniform ``forward(x, batch)`` interface for every normalization type."""

    def __init__(self, norm, name):
        super().__init__()
        self.norm = norm
        self.name = name

    def forward(self, x, batch=None):
        if isinstance(self.norm, (GraphNorm, PairNorm, IdentityNorm)):
            return self.norm(x, batch)
        return self.norm(x)  # BatchNorm1d / LayerNorm ignore graph indices

    def extra_repr(self):
        return 'type={}'.format(self.name)


def build_norm(name, dim):
    """Create a normalization module by name.

    Args:
        name: one of ``batch``, ``layer``, ``graph``, ``pair``, ``none``.
        dim: feature dimension of the input tensor.

    Returns:
        An ``nn.Module`` callable as ``module(x, batch)``.
    """
    key = str(name or 'none').lower()
    if key in ('batch', 'batchnorm', 'batch_norm'):
        module = nn.BatchNorm1d(dim)
    elif key in ('layer', 'layernorm', 'layer_norm'):
        module = nn.LayerNorm(dim)
    elif key in ('graph', 'graphnorm', 'graph_norm'):
        module = GraphNorm(dim)
    elif key in ('pair', 'pairnorm', 'pair_norm'):
        module = PairNorm(dim)
    elif key in ('none', 'identity', 'no', 'false', ''):
        module = IdentityNorm()
    else:
        raise ValueError(
            "Unknown normalization {!r}. Choose from: "
            "batch, layer, graph, pair, none".format(name))
    return NormWrapper(module, key)
