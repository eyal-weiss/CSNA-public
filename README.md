# CSNA: Cost-Sensitive Neighborhood Aggregation for Heterophilous Graphs — When Does Per-Edge Routing Help?

This repository contains the official implementation of the paper:

> **Cost-Sensitive Neighborhood Aggregation for Heterophilous Graphs: When Does Per-Edge Routing Help?**
> Eyal Weiss, Technion -- Israel Institute of Technology
> arXiv preprint, 2026

Graph Neural Networks (GNNs) learn from data organized as networks — social graphs, molecules, citation networks. They work by aggregating information from neighboring nodes, but this fails when connected nodes tend to belong to *different* classes (a property called *heterophily*). CSNA addresses this by measuring how different each pair of connected nodes is and routing their messages through separate processing channels depending on whether they appear similar or dissimilar.

Recent work distinguishes two heterophily regimes: *adversarial*, where cross-class edges harm classification, and *informative*, where they carry useful structural signal. We introduce CSNA, a method that computes pairwise distance in a learned projection and uses it to soft-route messages through **concordant** and **discordant** channels with independent transformations. CSNA's per-edge routing succeeds on adversarial-heterophily datasets but fails on informative-heterophily datasets — a pattern that operationalizes the regime distinction and reveals when per-edge routing helps.

![Toy example: uniform vs per-edge routing](paper/figures/toy_example.png)

**Left:** ACM-GNN applies a uniform high-pass filter to all cross-class edges. **Right:** CSNA assigns individualized routing weights, upweighting helpful cross-class edges and downweighting harmful ones.

## Main Results

Node classification accuracy (%) on heterophily benchmarks. All methods tuned over the same hyperparameter grid. Best in **bold**, second-best in *italics*.

| Method | Texas (H=0.09) | Wisconsin (H=0.19) | Cornell (H=0.13) | Actor (H=0.22) | Chameleon (H=0.23) | Squirrel (H=0.22) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| MLP | *77.3 ± 4.6* | *83.7 ± 4.8* | 72.2 ± 3.6 | *35.0 ± 1.4* | 51.9 ± 1.8 | 34.8 ± 1.4 |
| GCN | 55.7 ± 9.9 | 50.6 ± 8.5 | 47.0 ± 8.7 | 27.3 ± 1.4 | **67.3 ± 1.7** | **53.4 ± 0.8** |
| GAT | 50.8 ± 9.8 | 51.4 ± 7.8 | 47.3 ± 5.8 | 28.0 ± 1.4 | *65.7 ± 2.2* | *50.4 ± 1.4* |
| GraphSAGE | 76.5 ± 6.8 | 75.9 ± 5.8 | 66.2 ± 7.7 | 34.1 ± 0.6 | 63.9 ± 2.1 | 45.8 ± 1.4 |
| H2GCN | 75.9 ± 5.0 | 76.1 ± 6.1 | 67.8 ± 7.7 | 31.3 ± 1.7 | 51.5 ± 2.8 | 37.3 ± 2.8 |
| GPRGNN | **78.1 ± 8.0** | 78.6 ± 5.1 | 70.0 ± 6.7 | 34.2 ± 0.7 | 58.5 ± 2.8 | 38.4 ± 2.3 |
| ACM-GNN | 75.9 ± 6.3 | 80.0 ± 6.1 | 69.5 ± 5.1 | 33.5 ± 1.6 | 59.3 ± 2.0 | 41.6 ± 1.3 |
| **CSNA (ours)** | 77.0 ± 8.3 | 79.6 ± 6.2 | **72.7 ± 4.3** | **35.7 ± 1.2** | 54.6 ± 2.7 | 37.8 ± 1.9 |

CSNA is competitive on adversarial-heterophily datasets (Texas, Wisconsin, Cornell, Actor) while underperforming on informative-heterophily datasets (Chameleon, Squirrel) — see the paper for detailed analysis.

## Installation

```bash
pip install torch torch_geometric numpy matplotlib
```

See [PyTorch Geometric installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for platform-specific instructions.

## Usage

### Quick Start

```python
import sys; sys.path.insert(0, 'src')  # run from repo root
from csna import CSNA

# Initialize model
model = CSNA(
    in_channels=1703,       # Input feature dimension
    hidden_channels=64,     # Hidden layer dimension
    out_channels=5,         # Number of classes
    num_layers=2,           # Number of CSNA layers
    tau=1.0,                # Temperature for concordance routing
    dropout=0.5,
)

# Forward pass (standard PyG interface)
out = model(data.x, data.edge_index)  # [num_nodes, num_classes]
```

### Running Experiments

```bash
cd src

# Run all models on all datasets
python run_experiments.py

# Run specific models/datasets
python run_experiments.py --datasets Texas Wisconsin --models CSNA GCN MLP

# Quick test with fewer splits
python run_experiments.py --num_splits 3 --epochs 100

# Save results to a specific file
python run_experiments.py --output ../results/my_results.json
```

### Extended Variant (g + h)

The extended variant adds a learned cost component h_ij beyond the observed divergence g_ij. In our experiments, this did not provide a consistent advantage over the default (g_ij only) and risks overfitting on small datasets, but it may be useful in other settings:

```python
from csna import CSNAExt  # assumes sys.path includes src/

model = CSNAExt(
    in_channels=1703,
    hidden_channels=64,
    out_channels=5,
    tau=1.0,
)

# Training with calibration regularization
out = model(data.x, data.edge_index)
loss = F.cross_entropy(out[train_mask], data.y[train_mask])
loss += 0.1 * model.calibration_loss(data.y, train_mask=train_mask)
loss += 0.01 * model.consistency_loss()
```

## Citation

```bibtex
@article{weiss2026csna,
  title={Cost-Sensitive Neighborhood Aggregation for Heterophilous Graphs: When Does Per-Edge Routing Help?},
  author={Weiss, Eyal},
  journal={arXiv preprint},
  year={2026}
}
```

## Note on Softmax Normalization

CSNA normalizes concordance weights per source node (`edge_index[0]`), meaning each node distributes its outgoing influence uniformly. This differs from the more common per-destination normalization (`edge_index[1]`). We tested both variants with full hyperparameter tuning on Texas, Wisconsin, and Cornell and found no consistent accuracy difference across datasets (differences within standard deviation and typical tuning variance). The code includes a comment indicating how to switch to per-destination normalization if desired.

## Note on Baselines

The H2GCN, GPRGNN, and ACM-GNN implementations in this repository are simplified in-house versions, not official reproductions. Results may differ slightly from those reported in the original papers due to implementation differences. All models are tuned over the same hyperparameter grid for fair comparison.

## Author

**Eyal Weiss**
Computer Science Faculty, Technion -- Israel Institute of Technology
[eweiss@campus.technion.ac.il](mailto:eweiss@campus.technion.ac.il)

## License

MIT License. See [LICENSE](LICENSE) for details.
