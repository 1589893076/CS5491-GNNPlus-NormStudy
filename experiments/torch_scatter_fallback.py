#!/usr/bin/env python3
"""Pure-PyTorch fallback for ``torch_scatter.scatter`` (CS5491 portability).

Upstream GNN+ imports ``torch_scatter`` in several layers.  That package is a
compiled C++/CUDA extension, and on Colab runtimes whose PyTorch version has
no matching pre-built wheel it cannot be installed; the same is true on a
plain Windows/CPU machine.  This module implements the small subset of the API
that GNN+ actually uses, with the same signature and reduction semantics, via
``torch.scatter_reduce`` (available since PyTorch 1.12).

It is NOT imported by default.  ``colab/run_norm_study.ipynb`` copies this file
to the repository root as ``torch_scatter.py`` and puts that directory on
``PYTHONPATH`` only when the real extension is missing.  When the real
extension is installed it takes precedence, so the published baseline path is
unchanged and no experimental number is affected.

Self-test:  python experiments/torch_scatter_fallback.py
"""

import torch

# torch_scatter reduction name -> torch.scatter_reduce reduction name.
_REDUCTIONS = {
    'sum': 'sum', 'add': 'sum',
    'mean': 'mean',
    'max': 'amax', 'amax': 'amax',
    'min': 'amin', 'amin': 'amin',
    'mul': 'prod', 'prod': 'prod',
}


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce='sum'):
    """Match ``torch_scatter.scatter`` for the reductions GNN+ needs.

    Args:
        src: tensor of values to aggregate.
        index: long tensor of group ids, broadcastable to ``src``'s shape.
        dim: axis to reduce along; negative values wrap (default -1, matching
            torch_scatter).
        out: optional preallocated output tensor (accepted by torch_scatter).
        dim_size: output length along ``dim``; inferred from ``index`` if None.
        reduce: one of sum / add / mean / max / min / mul.

    Returns:
        Aggregated tensor, or ``out`` if one was supplied.
    """
    src = torch.as_tensor(src)
    index = torch.as_tensor(index, dtype=torch.long, device=src.device)
    if dim < 0:
        dim += src.dim()
    if index.dim() == 0:
        index = index.reshape(1)
    # A 1-D graph-index vector against a [num_nodes, channels] source is the
    # common case: expand trailing singleton dimensions to match ``src``.
    while index.dim() < src.dim():
        index = index.unsqueeze(-1)
    if tuple(index.shape) != tuple(src.shape):
        index = index.expand(src.shape)

    size = list(src.shape)
    if dim_size is not None:
        size[dim] = int(dim_size)
    elif out is not None:
        size[dim] = out.shape[dim]
    else:
        size[dim] = int(index.max().item()) + 1 if index.numel() else 0

    op = _REDUCTIONS.get(str(reduce).lower())
    if op is None:
        raise ValueError('Unsupported reduce={!r}; expected one of {}'.format(
            reduce, sorted(_REDUCTIONS)))
    # include_self=False makes the initial zeros act as an identity, which is
    # required for mean (no phantom zero observation) and prod (identity 1).
    result = torch.zeros(size, dtype=src.dtype, device=src.device)
    result = result.scatter_reduce(dim, index, src, reduce=op,
                                   include_self=False)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _self_test():
    """Check every reduction used by GNN+ against hand-computed values."""
    src = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    idx = torch.tensor([0, 0, 1])
    assert torch.allclose(scatter(src, idx, dim=0, reduce='sum'),
                          torch.tensor([[4., 6.], [5., 6.]]))
    assert torch.allclose(scatter(src, idx, dim=0, reduce='mean'),
                          torch.tensor([[2., 3.], [5., 6.]]))
    assert torch.allclose(scatter(src, idx, dim=0, reduce='max'),
                          torch.tensor([[3., 4.], [5., 6.]]))
    # Positional signature used by GNNPlus/layer/gatedgcn_layer.py:
    # scatter(src, index, 0, None, dim_size, reduce='mean').
    sized = scatter(src, idx, 0, None, 3, 'mean')
    assert sized.shape == (3, 2)
    assert torch.allclose(sized[:2], torch.tensor([[2., 3.], [5., 6.]]))
    # 'mul' is used by GNNPlus/utils.py.
    values = torch.tensor([2., 3., 4.])
    assert torch.allclose(scatter(values, torch.tensor([0, 0, 1]), dim=0,
                                  reduce='mul'),
                          torch.tensor([6., 4.]))
    # dim=-1 default for 1-D inputs (torch_scatter convention).
    assert torch.allclose(scatter(values, torch.tensor([0, 0, 1]),
                                  reduce='sum'),
                          torch.tensor([5., 4.]))
    print('torch_scatter fallback self-test: OK')


if __name__ == '__main__':
    _self_test()
