# Which Normalization Makes Classic GNNs Strong?

**CS5491 course project — normalization × depth study in GNN+**

Group: Xinsen Yuan (72656002), Zanqun Li (72655183)

This repository is a **modified fork** of the official GNN+ code released with
the ICML 2025 paper *"Can Classic GNNs Be Strong Baselines for Graph-level
Tasks?"* by Luo, Shi & Wu ([paper](https://arxiv.org/abs/2502.09263),
[upstream repository](https://github.com/LUOyk1999/GNNPlus)). We gratefully
acknowledge the authors; all original code remains under its original LICENSE.
Our contribution is a small, clearly marked patch plus experiment scripts and
reports (see [`MYPROJECT.md`](MYPROJECT.md) and
[`MYPROJECT_CHANGES.diff`](MYPROJECT_CHANGES.diff)).

---

## 1. Research question

GNN+ wraps classic GNNs with six ingredients: edge features, normalization,
dropout, residual connections, feed-forward networks and positional encoding.
In the released implementation **normalization is hard-coded** —
`self.batch_norm = True` inside every convolution layer — and only BatchNorm is
ever evaluated. Yet the paper's headline claim (residuals allow classic GNNs to
reach 20 layers) depends on training stability at depth, which normalization
governs.

We ask: **does the *choice* of normalization matter, and does it interact with
depth?** We make normalization a first-class configuration option and compare
BatchNorm, LayerNorm, GraphNorm, PairNorm and a `none` control across depths
4 / 8 / 12 / 20 on ZINC.

---

## 2. What we changed

| File | Change |
| --- | --- |
| `GNNPlus/layer/norm.py` | **New.** `build_norm(name, dim)` factory: `batch`, `layer`, `graph`, `pair`, `none`. Self-contained GraphNorm / PairNorm implementations, uniform `forward(x, batch)` signature. |
| `GNNPlus/config/custom_gnn_config.py` | Registers `cfg.gnn.norm = 'batch'` (default preserves published behavior). |
| `GNNPlus/layer/gcn_conv_layer.py`, `gcn_conv_layer_e.py` | Replaces hard-coded `nn.BatchNorm1d` with `build_norm(cfg.gnn.norm, ...)`. `gcn_conv_layer_e.py` is the ZINC backbone. |
| `GNNPlus/layer/gine_conv_layer.py`, `gatedgcn_layer.py` | Same knob for the GIN+ / GatedGCN+ backbones (secondary check). |
| `configs/gcn/zinc_norm_*.yaml` | Five example configs differing only in `gnn.norm`. |
| `experiments/run_norm_study.py` | Generates one YAML per (norm, depth) cell, launches GraphGym with 3 seeds, writes a manifest. |
| `experiments/aggregate_norm_study.py` | Aggregates best-epoch test MAE into `summary.csv` and `summary.md`. |
| `experiments/test_norm_smoke.py` | PyTorch-only test that the default equals upstream BatchNorm and all options are finite. |

The patch is surgical: default `cfg.gnn.norm='batch'` reproduces the original
model exactly, so the paper's numbers remain reachable with the unmodified
`configs/gcn/zinc.yaml`.

---

## 3. Environment setup

The upstream code was tested with Python 3.10, PyTorch 2.2.0 and PyG 2.3.1.
Use conda (or Colab with the same versions):

```bash
conda create -n GNNPlus python=3.10 -y
conda activate GNNPlus

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

pip install scikit-learn==1.4.0 fsspec rdkit pytorch-lightning yacs \
  torchmetrics networkx tensorboardX ogb wandb
```

`requirements.txt` lists the same packages. No private API keys, external
servers or credentials are required: ZINC is downloaded automatically by PyG
on first use, and `wandb.use` is `False` in all study configs.

---

## 4. Reproduction commands

All commands are run from the repository root.

```bash
# 0) Sanity check of the new normalization module (CPU only, no dataset).
python experiments/test_norm_smoke.py

# 1) Reproduce the paper's original setting (unchanged default = BatchNorm).
python main.py --cfg configs/gcn/zinc.yaml --repeat 3 seed 0

# 2) Run the full normalization x depth grid (20 cells x 3 seeds).
#    --epochs 300 keeps the grid affordable; the paper uses 2000.
python experiments/run_norm_study.py --repeat 3 --epochs 300

# 3) Aggregate into results/norm_study_summary.{csv,md}.
python experiments/aggregate_norm_study.py

# 4) (Optional) re-run key cells at the paper's full 2000 epochs.
python experiments/run_norm_study.py --norms batch graph --depths 12 20 \
  --epochs 2000 --repeat 3
```

A single cell can also be run directly, e.g.:

```bash
python main.py --cfg configs/generated/zinc_norm_graph_d20.yaml --repeat 3 seed 0
```

Results are written to
`results/norm_study/<config-stem>/<seed>/{train,val,test}/stats.json`, matching
GraphGym's standard layout. Logs are in `logs/norm_study/`. The sweep is
resumable with `--skip-existing`.

---

## 5. Repository layout

```
GNNPlus/                 original code + our norm.py patch
configs/gcn/             original configs + five zinc_norm_*.yaml
configs/generated/       per-cell YAMLs created by the sweep script
experiments/             sweep, aggregation, smoke test, manifest
results/                 training output (not committed)
logs/                    sweep logs (not committed)
docs/                    milestone and final report drafts (PDF source)
main.py                  unchanged GraphGym entry point
```

Every new module and function contains docstrings explaining its role; the
modified layers have inline `# CS5491:` comments marking our changes, so the
data-preprocessing / model / train-eval pipeline remains traceable.

---

## 6. Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{luo2025can,
  title={Can Classic {GNN}s Be Strong Baselines for Graph-level Tasks? Simple Architectures Meet Excellence},
  author={Luo, Yuankai and Shi, Lei and Wu, Xiao-Ming},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=ZH7YgIZ3DF}
}
```
