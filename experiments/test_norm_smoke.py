#!/usr/bin/env python3
"""Self-contained smoke test for the pluggable normalization module.

Requires only PyTorch (no PyG / dataset download).  Verifies that the default
'batch' option is numerically identical to upstream and that all new options
produce finite, correctly shaped outputs.
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
# Load norm.py directly so the smoke test does not require torch_geometric.
_spec = importlib.util.spec_from_file_location(
    'gnnplus_norm', ROOT / 'GNNPlus' / 'layer' / 'norm.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
build_norm = _module.build_norm


def main():
    torch.manual_seed(0)
    x = torch.randn(24, 8)
    batch = torch.tensor([0] * 12 + [1] * 12)

    # Default must be exactly the upstream nn.BatchNorm1d.
    wrapper = build_norm('batch', 8)
    assert isinstance(wrapper.norm, nn.BatchNorm1d)
    wrapper.train()
    ref = nn.BatchNorm1d(8)
    ref.load_state_dict(wrapper.norm.state_dict())
    assert torch.allclose(wrapper(x, batch), ref(x), atol=1e-6)

    for name in ('batch', 'layer', 'graph', 'pair', 'none', 'identity'):
        module = build_norm(name, 8).eval()
        out = module(x, batch)
        assert out.shape == x.shape, (name, out.shape)
        assert torch.isfinite(out).all(), name
        print('{:8s} -> shape={} mean={:.4f} std={:.4f}'.format(
            name, tuple(out.shape), out.mean().item(), out.std().item()))

    try:
        build_norm('bogus', 8)
    except ValueError as exc:
        print('invalid name correctly rejected: {}'.format(exc))
    else:
        raise AssertionError('unknown normalization was accepted')

    # No graph-index vector: GraphNorm must still work (global fallback).
    out = build_norm('graph', 8).eval()(x, None)
    assert torch.isfinite(out).all()
    print('graph norm without batch index: OK')

    print('\nAll normalization smoke tests passed.')


if __name__ == '__main__':
    main()
