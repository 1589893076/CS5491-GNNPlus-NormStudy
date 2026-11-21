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

### Running on Colab instead

`colab/run_norm_study.ipynb` is a single T4 notebook: install dependencies →
**preflight the ZINC download** → run the grid → aggregate → rebuild both
PDFs → download the results. The preflight matters: it checks that the
dataset is reachable *before* a multi-hour sweep begins.

Compiled PyG extensions are optional on Colab. If no `torch_scatter` wheel
matches the runtime's PyTorch version, the notebook activates
`experiments/torch_scatter_fallback.py`, a pure-PyTorch implementation of the
`scatter` reductions GNN+ uses (`python experiments/torch_scatter_fallback.py`
runs its self-test). When the real extension is installed it takes
precedence, so this fallback never changes an experimental number.

**PyG API compatibility.** Newer `torch_geometric` releases generate a
jittable `MessagePassing.propagate` template that rejects the custom keyword
arguments used by upstream `GCNConvWithEdges.forward`
(`TypeError: propagate() got an unexpected keyword argument 'edge_attr'`),
which aborted the first ZINC training cell before any `stats.json` existed.
That layer now performs the identical sum aggregation with plain PyTorch
(`relu(x_j + edge_attr)` followed by `index_add_`) and never calls
`propagate`, so the sweep runs on any PyG version. Equivalence is checked by
`python experiments/test_gnn_pyg_compat.py`: it loads the real layer with the
fallback active, compares its output against an independent scatter-add
ground truth (max difference 0), verifies gradients and asserts that
`propagate` is never called. `MYPROJECT_CHANGES.diff` lists the upstream-file
edits; this compatibility patch is described here and in the layer's comment.

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

# 5) Rebuild both report PDFs from the result CSVs.
python docs/build_reports.py
#    On Colab, write both PDFs into one folder instead:
python docs/build_reports.py --out-dir reports
```

`docs/build_reports.py` is data-driven. When
`results/norm_study_summary.csv` exists it fills Table 3, redraws Figure 1
from those numbers and derives the Section 4 conclusions (best option at the
deepest depth, BatchNorm's depth trend, the `none` ablation) by computing
them from the CSV. When it is absent, the report keeps an explicit *pending*
marker and never substitutes a hand-written number.

### One-command finalization

After copying the aggregated CSV back into `results/`, run:

```bash
python docs/finish_project.py            # origin already points at the public repo
python docs/finish_project.py --skip-git # rebuild the reports only
python docs/finish_project.py --push     # also push (needs your own credentials)
```

This validates the result grid, rebuilds both PDFs, checks page counts and
Table 3 / Figure 1 / conclusions, then commits the results. `--repo-url` is
only needed if `origin` still points elsewhere. The public repository for this
project is `https://github.com/1589893076/CS5491-GNNPlus-NormStudy`.

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
experiments/             sweep, aggregation, smoke test, manifest,
                         torch_scatter_fallback.py (extension-free fallback)
colab/                   T4 GPU notebook for the full ZINC sweep
results/                 pilot CSV and the aggregated ZINC summary (committed,
                         since every report number comes from them);
                         per-run stats.json files are ignored
logs/                    sweep logs (not committed)
docs/                    data-driven report generator (PDFs are written to the
                         course stage folders, or to --out-dir)
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
