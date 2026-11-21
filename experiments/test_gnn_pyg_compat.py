#!/usr/bin/env python3
"""Regression test for the PyG-version-independent GCN-with-edges layer.

Upstream GNN+ called ``self.propagate(..., edge_attr=..., edge_weight=...)``.
With the ``torch_scatter`` fallback present (as on Colab), PyG generates a
jittable ``propagate`` template that rejects those keyword arguments, so the
very first ZINC training cell failed with::

    TypeError: propagate() got an unexpected keyword argument 'edge_attr'

``GCNConvWithEdges.forward`` now implements the same sum-aggregation with
plain PyTorch and never calls ``propagate``.  This test loads the real class
in exactly that environment (fallback injected before PyG is imported) and
checks:

1. forward succeeds -- this alone reproduces and guards the original bug;
2. outputs match an independently written scatter-add ground truth;
3. gradients reach the linear weights;
4. ``propagate`` is never called.

Usage::

    python experiments/test_gnn_pyg_compat.py
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_conv_class():
    """Import the real GCNConvWithEdges with a minimal package skeleton."""
    if 'torch_scatter' not in sys.modules:
        try:
            import torch_scatter  # noqa: F401
        except ImportError:
            spec = importlib.util.spec_from_file_location(
                'torch_scatter',
                ROOT / 'experiments' / 'torch_scatter_fallback.py')
            shim = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shim)
            sys.modules['torch_scatter'] = shim

    for name, path in (('GNNPlus', ROOT / 'GNNPlus'),
                       ('GNNPlus.layer', ROOT / 'GNNPlus' / 'layer')):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg

    spec = importlib.util.spec_from_file_location(
        'GNNPlus.layer.gcn_conv_layer_e',
        ROOT / 'GNNPlus' / 'layer' / 'gcn_conv_layer_e.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.GCNConvWithEdges


def ground_truth(weight, bias, x, edge_index, edge_attr):
    """Independent reference: sum over relu(x_j + edge_attr), then add bias."""
    h = torch.nn.functional.linear(x, weight)
    src, dst = edge_index[0], edge_index[1]
    x_j = h.index_select(0, src)
    msg = x_j.relu() if edge_attr is None else (x_j + edge_attr).relu()
    out = torch.zeros_like(h).scatter_add(
        0, dst.view(-1, 1).expand_as(msg), msg)
    return out + bias


def main():
    try:
        Conv = load_conv_class()
    except ImportError as exc:
        print('SKIP: torch_geometric is not available here ({})'.format(exc))
        return 0

    torch.manual_seed(0)
    conv = Conv(8, 8, 8, bias=True)
    # If anyone restores the propagate() call, this turns it into a hard fail.
    def _no_propagate(*_args, **_kwargs):
        raise AssertionError(
            'GCNConvWithEdges.forward must not call propagate(): '
            'jittable PyG templates reject custom keyword arguments.')
    conv.propagate = _no_propagate

    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    edge_attr = torch.randn(5, 8)

    for label, ea in (('with edge_attr', edge_attr),
                      ('edge_attr=None', None)):
        ours = conv(x, edge_index, ea)
        expected = ground_truth(conv.lin.weight, conv.bias,
                                x, edge_index, ea)
        assert ours.shape == x.shape, label
        assert torch.isfinite(ours).all(), label
        assert torch.allclose(ours, expected, atol=1e-6), (
            '{}: output diverged from ground truth'.format(label))
        print('{:16s} -> shape={} max_diff={:.2e}'.format(
            label, tuple(ours.shape), (ours - expected).abs().max().item()))

    conv(x, edge_index, edge_attr).sum().backward()
    assert conv.lin.weight.grad is not None
    assert torch.isfinite(conv.lin.weight.grad).all()
    print('backward pass -> finite gradients')

    print('\nPyG-compatibility regression test passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
