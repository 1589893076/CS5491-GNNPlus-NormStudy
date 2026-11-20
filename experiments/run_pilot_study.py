#!/usr/bin/env python3
"""CPU-only pilot study for the normalization module.

This is NOT the ZINC experiment.  It trains a small GNN+ style model on a
synthetic graph-regression task using the *same* ``build_norm`` factory as the
real experiment.  Its purpose is to produce genuine preliminary evidence that
the patch has the expected qualitative behaviour (depth x normalization
interaction) on a machine without a GPU or PyG.  Do not present these numbers
as ZINC results.

Usage:
    python experiments/run_pilot_study.py --epochs 150 --seeds 3
"""

import argparse
import csv
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'gnnplus_norm', ROOT / 'GNNPlus' / 'layer' / 'norm.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
build_norm = _module.build_norm

HIDDEN = 32
NORMS = ('batch', 'layer', 'graph', 'pair', 'none')
DEPTHS = (4, 8, 12, 20)


def make_dataset(n_graphs=240, seed=1234):
    g = torch.Generator().manual_seed(seed)
    xs, es, edge_indices, batches, ys = [], [], [], [], []
    node_offset = 0
    true_w = torch.randn(8, 1, generator=g)
    for graph_id in range(n_graphs):
        n = int(torch.randint(10, 19, (1,), generator=g).item())
        x = torch.randn(n, 8, generator=g)
        src, dst = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand(1, generator=g).item() < 0.25:
                    src.extend([i, j])
                    dst.extend([j, i])
        if not src:
            src, dst = [0, 1], [1, 0]
        # One feature row per directed edge (matches GNN+'s edge encoder).
        e = torch.randn(len(src), 8, generator=g)
        edge_index = torch.tensor([src, dst], dtype=torch.long) + node_offset
        xs.append(x)
        es.append(e)
        edge_indices.append(edge_index)
        batches.append(torch.full((n,), graph_id, dtype=torch.long))
        # Learnable but nontrivial graph-level target.
        y = math.tanh(float((x @ true_w).sum().item()) / 12.0) \
            + 0.2 * float(e.mean().item()) \
            + 0.05 * float(torch.randn(1, generator=g).item())
        ys.append(y)
        node_offset += n
    return (torch.cat(xs), torch.cat(es), torch.cat(edge_indices, dim=1),
            torch.cat(batches), torch.tensor(ys).unsqueeze(-1),
            node_offset, n_graphs)


def scatter_mean(src, index, dim_size):
    out = src.new_zeros(dim_size, src.size(1))
    cnt = src.new_zeros(dim_size, 1)
    out.index_add_(0, index, src)
    cnt.index_add_(0, index, torch.ones_like(src[:, :1]))
    return out / cnt.clamp(min=1.0)


class TinyGNNPlus(nn.Module):
    def __init__(self, depth, norm_name):
        super().__init__()
        self.depth = depth
        self.node_in = nn.Linear(8, HIDDEN)
        self.edge_in = nn.Linear(8, HIDDEN)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Module()
            layer.lin = nn.Linear(HIDDEN, HIDDEN)
            layer.norm = build_norm(norm_name, HIDDEN)
            layer.ff1 = nn.Linear(HIDDEN, 2 * HIDDEN)
            layer.ff2 = nn.Linear(2 * HIDDEN, HIDDEN)
            layer.norm_ff = build_norm(norm_name, HIDDEN)
            self.layers.append(layer)
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                  nn.Linear(HIDDEN, 1))

    def forward(self, x, e, edge_index, batch, n_graphs):
        x = F.relu(self.node_in(x))
        e = F.relu(self.edge_in(e))
        b_idx = batch
        for layer in self.layers:
            src, dst = edge_index[0], edge_index[1]
            msg = F.relu(x[src] + e)
            agg = scatter_mean(msg, dst, x.size(0))
            h = F.relu(layer.lin(agg))
            h = layer.norm(h, b_idx)
            h = x + h
            h = layer.norm_ff(h, b_idx)
            h = h + layer.ff2(F.relu(layer.ff1(h)))
            x = h
        pooled = scatter_mean(x, batch, n_graphs)
        return self.head(pooled)


def run_cell(norm_name, depth, epochs, seed, data):
    torch.manual_seed(seed)
    x, e, edge_index, batch, y, n_nodes, n_graphs = data
    n_train = int(0.6 * n_graphs)
    n_val = int(0.2 * n_graphs)
    train_idx = torch.arange(n_train)
    val_idx = torch.arange(n_train, n_train + n_val)
    test_idx = torch.arange(n_train + n_val, n_graphs)

    model = TinyGNNPlus(depth, norm_name)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    def full_forward(indices):
        # Build a subgraph containing only the requested graphs.
        keep = torch.isin(batch, indices)
        node_map = torch.full((n_nodes,), -1, dtype=torch.long)
        node_map[keep] = torch.arange(int(keep.sum()))
        mask_edges = keep[edge_index[0]] & keep[edge_index[1]]
        ei = node_map[edge_index[:, mask_edges]]
        # Map original graph ids to local indices 0..len(indices)-1.
        graph_map = torch.full((n_graphs,), -1, dtype=torch.long)
        graph_map[indices] = torch.arange(len(indices))
        b = graph_map[batch[keep]]
        # b follows the order of `indices`, so targets align with y[indices].
        target = y[indices]
        pred = model(x[keep], e[mask_edges], ei, b, len(indices))
        return pred, target

    best_val, best_epoch = float('inf'), 0
    for epoch in range(epochs):
        model.train()
        pred, target = full_forward(train_idx)
        loss = F.l1_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vp, vt = full_forward(val_idx)
            val_mae = F.l1_loss(vp, vt).item()
        if val_mae < best_val:
            best_val, best_epoch = val_mae, epoch
    model.eval()
    with torch.no_grad():
        tp, tt = full_forward(test_idx)
        test_mae = F.l1_loss(tp, tt).item()
    return test_mae, best_epoch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--norms', nargs='+', default=list(NORMS))
    p.add_argument('--depths', nargs='+', type=int, default=list(DEPTHS))
    args = p.parse_args()

    data = make_dataset()
    print('Synthetic pilot: {} graphs, {} nodes, {} edges'.format(
        data[6], data[5], data[2].shape[1]))
    results = []
    for norm_name in args.norms:
        for depth in args.depths:
            vals = []
            for seed in range(args.seeds):
                mae, ep = run_cell(norm_name, depth, args.epochs, seed, data)
                vals.append(mae)
                print('  {:6s} d={:2d} seed={} test_mae={:.4f} '
                      '(best epoch {})'.format(norm_name, depth, seed, mae, ep),
                      flush=True)
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            results.append((norm_name, depth, mean, std, vals))
            print('=> {:6s} d={:2d}  mean={:.4f} std={:.4f}\n'.format(
                norm_name, depth, mean, std), flush=True)

    out = ROOT / 'results' / 'pilot_norm_study.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['norm', 'depth', 'seeds', 'test_mae_mean',
                         'test_mae_std', 'all'])
        for norm_name, depth, mean, std, vals in results:
            writer.writerow([norm_name, depth, len(vals),
                             round(mean, 4), round(std, 4),
                             '|'.join('{:.4f}'.format(v) for v in vals)])
    print('Saved', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
