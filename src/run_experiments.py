"""
Experiment runner for CSNA heterophily benchmarks.

Runs all baselines and CSNA on six heterophily datasets with fair hyperparameter
tuning. All methods share the same tuning grid; CSNA additionally tunes temperature.

Usage:
    # Run all models on all datasets (full experiment)
    python run_experiments.py

    # Run specific models/datasets
    python run_experiments.py --datasets Texas Wisconsin --models CSNA GCN MLP

    # Quick test with fewer splits
    python run_experiments.py --num_splits 3 --epochs 100
"""

import os
import sys
import json
import copy
import time
import argparse
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor as ActorDataset
from torch_geometric.utils import to_undirected, homophily

from csna import CSNA
from baselines import GCN, GAT, GraphSAGE, H2GCN, GPRGNN, ACM_GNN, MLP

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset(name, root='../data'):
    """Load a heterophily benchmark dataset.

    Args:
        name: One of Texas, Wisconsin, Cornell, Actor, Chameleon, Squirrel.
        root: Root directory for dataset download/cache.

    Returns:
        data: PyG Data object.
        info: Dictionary with dataset statistics.
    """
    if name in ['Texas', 'Wisconsin', 'Cornell']:
        dataset = WebKB(root=root, name=name)
    elif name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root=root, name=name.lower())
    elif name == 'Actor':
        dataset = ActorDataset(root=root)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)

    h = homophily(data.edge_index, data.y, method='edge')

    info = {
        'name': name,
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.size(1) // 2,
        'num_features': data.num_features,
        'num_classes': dataset.num_classes,
        'homophily': float(h),
    }
    return data, info


def generate_splits(data, num_splits=10, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Generate random train/val/test splits.

    Args:
        data: PyG Data object.
        num_splits: Number of random splits to generate.
        train_ratio: Fraction of nodes for training.
        val_ratio: Fraction of nodes for validation.
        seed: Random seed for reproducibility.

    Returns:
        List of (train_mask, val_mask, test_mask) tuples.
    """
    rng = np.random.RandomState(seed)
    n = data.num_nodes
    splits = []
    for _ in range(num_splits):
        perm = rng.permutation(n)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[perm[:train_end]] = True
        val_mask[perm[train_end:val_end]] = True
        test_mask[perm[val_end:]] = True
        splits.append((train_mask, val_mask, test_mask))
    return splits


def build_model(model_name, in_channels, hidden_channels, out_channels,
                dropout=0.5, num_layers=2, tau=1.0):
    """Instantiate a model by name.

    Args:
        model_name: One of MLP, GCN, GAT, GraphSAGE, H2GCN, GPRGNN, ACM-GNN, CSNA.
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer dimension.
        out_channels: Number of classes.
        dropout: Dropout probability.
        num_layers: Number of layers.
        tau: Temperature for CSNA concordance scoring.

    Returns:
        Model instance.
    """
    if model_name == 'CSNA':
        return CSNA(in_channels, hidden_channels, out_channels,
                     num_layers=num_layers, tau=tau, dropout=dropout)
    elif model_name == 'GCN':
        return GCN(in_channels, hidden_channels, out_channels,
                    num_layers=num_layers, dropout=dropout)
    elif model_name == 'GAT':
        return GAT(in_channels, hidden_channels, out_channels,
                    num_layers=num_layers, dropout=dropout)
    elif model_name == 'GraphSAGE':
        return GraphSAGE(in_channels, hidden_channels, out_channels,
                         num_layers=num_layers, dropout=dropout)
    elif model_name == 'H2GCN':
        return H2GCN(in_channels, hidden_channels, out_channels,
                      num_layers=num_layers, dropout=dropout)
    elif model_name == 'GPRGNN':
        return GPRGNN(in_channels, hidden_channels, out_channels,
                       K=10, dropout=dropout)
    elif model_name == 'ACM-GNN':
        return ACM_GNN(in_channels, hidden_channels, out_channels,
                        num_layers=num_layers, dropout=dropout)
    elif model_name == 'MLP':
        return MLP(in_channels, hidden_channels, out_channels,
                    num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluate accuracy on a subset of nodes."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    correct = (pred == data.y[mask]).sum().item()
    return correct / mask.sum().item()


def train_and_eval(model_name, data, train_mask, val_mask, test_mask,
                   in_ch, n_cls, hidden, lr, dropout=0.5, tau=1.0,
                   epochs=300, patience=50):
    """Train one model on one split with proper checkpointing.

    Saves the checkpoint with best validation accuracy and evaluates it
    once on the test set.

    Returns:
        best_val: Best validation accuracy.
        test_acc: Test accuracy of the best-val checkpoint.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = build_model(model_name, in_ch, hidden, n_cls,
                        dropout=dropout, tau=tau).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val = -float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, data, val_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Reload best model and evaluate on test once
    model.load_state_dict(best_state)
    test_acc = evaluate(model, data, test_mask)
    return best_val, test_acc


def tune_model(model_name, data, info, splits, is_large=False):
    """Tune hyperparameters for a model.

    All models share the same base grid (lr, hidden). CSNA additionally tunes tau.

    Args:
        model_name: Model name string.
        data: PyG Data object.
        info: Dataset info dictionary.
        splits: List of (train_mask, val_mask, test_mask) tuples.
        is_large: If True, use a smaller tuning grid.

    Returns:
        Dictionary with best hyperparameters.
    """
    in_ch = info['num_features']
    n_cls = info['num_classes']

    if is_large:
        common_grid = list(itertools.product([0.01, 0.005], [64]))
        tune_splits = 2
        tune_epochs = 150
    else:
        common_grid = list(itertools.product([0.01, 0.005], [64, 128]))
        tune_splits = 3
        tune_epochs = 200

    tau_values = [0.1, 0.5, 1.0, 2.0] if model_name == 'CSNA' else [1.0]

    best_val = -1
    best_cfg = None

    for lr, hidden in common_grid:
        for tau in tau_values:
            vals = []
            for train_mask, val_mask, test_mask in splits[:tune_splits]:
                bv, _ = train_and_eval(
                    model_name, data, train_mask, val_mask, test_mask,
                    in_ch, n_cls, hidden, lr, tau=tau,
                    epochs=tune_epochs, patience=30)
                vals.append(bv)

            mean_val = np.mean(vals)
            if mean_val > best_val:
                best_val = mean_val
                best_cfg = {'lr': lr, 'hidden': int(hidden), 'tau': tau,
                            'dropout': 0.5, 'num_layers': 2}

    return best_cfg


def main():
    parser = argparse.ArgumentParser(
        description='CSNA heterophily benchmark experiments')
    parser.add_argument('--datasets', nargs='+',
                        default=['Texas', 'Wisconsin', 'Cornell',
                                 'Actor', 'Chameleon', 'Squirrel'],
                        help='Datasets to evaluate on')
    parser.add_argument('--models', nargs='+',
                        default=['MLP', 'GCN', 'GAT', 'GraphSAGE',
                                 'H2GCN', 'GPRGNN', 'ACM-GNN', 'CSNA'],
                        help='Models to evaluate')
    parser.add_argument('--num_splits', type=int, default=10,
                        help='Number of random train/val/test splits')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--output', type=str, default='../results/results.json',
                        help='Output file for results')
    parser.add_argument('--data_root', type=str, default='../data',
                        help='Root directory for dataset download')
    args = parser.parse_args()

    large_datasets = {'Actor', 'Chameleon', 'Squirrel'}

    all_results = {}
    dataset_infos = {}

    for dname in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dname}")
        print(f"{'=' * 60}")

        data, info = load_dataset(dname, root=args.data_root)
        data = data.to(DEVICE)
        dataset_infos[dname] = info
        is_large = dname in large_datasets

        print(f"  Nodes: {info['num_nodes']}, Edges: {info['num_edges']}, "
              f"H: {info['homophily']:.3f}")

        splits = generate_splits(data, num_splits=args.num_splits)
        all_results[dname] = {}

        for mname in args.models:
            print(f"\n  --- {mname} ---")

            # Step 1: Tune
            print(f"  Tuning...")
            t0 = time.time()
            cfg = tune_model(mname, data, info, splits, is_large=is_large)
            tune_time = time.time() - t0
            print(f"    Best config: lr={cfg['lr']}, hidden={cfg['hidden']}"
                  + (f", tau={cfg['tau']}" if mname == 'CSNA' else '')
                  + f" ({tune_time:.0f}s)")

            # Step 2: Evaluate on all splits with best config
            print(f"  Evaluating ({args.num_splits} splits)...")
            t0 = time.time()
            accs = []
            for train_mask, val_mask, test_mask in splits:
                _, test_acc = train_and_eval(
                    mname, data, train_mask, val_mask, test_mask,
                    info['num_features'], info['num_classes'],
                    cfg['hidden'], cfg['lr'], tau=cfg['tau'],
                    epochs=args.epochs, patience=args.patience)
                accs.append(test_acc)
            eval_time = time.time() - t0

            m_acc = np.mean(accs) * 100
            s_acc = np.std(accs) * 100
            all_results[dname][mname] = {
                'mean': float(m_acc),
                'std': float(s_acc),
                'accs': [float(a) for a in accs],
                'config': cfg,
            }
            print(f"    {mname:12s}: {m_acc:.2f} +/- {s_acc:.2f}  "
                  f"(tune={tune_time:.0f}s, eval={eval_time:.0f}s)")

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output_data = {
        'results': all_results,
        'dataset_info': dataset_infos,
    }
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print(f"\n{'=' * 100}")
    print("SUMMARY (Accuracy %)")
    print(f"{'=' * 100}")
    header = f"{'Dataset':12s} | {'H':>5s}"
    for mname in args.models:
        header += f" | {mname:>12s}"
    print(header)
    print("-" * len(header))
    for dname in args.datasets:
        if dname not in all_results:
            continue
        h = dataset_infos[dname]['homophily']
        row = f"{dname:12s} | {h:5.3f}"
        for mname in args.models:
            if mname in all_results[dname]:
                r = all_results[dname][mname]
                row += f" | {r['mean']:5.2f}+/-{r['std']:4.2f}"
            else:
                row += f" | {'--':>12s}"
        print(row)


if __name__ == '__main__':
    main()
