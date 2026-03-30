"""
Main experiment runner for CSNA heterophily benchmarks.
Runs all models on all datasets with multiple random splits.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor
from torch_geometric.utils import to_undirected, homophily

# Add parent dir
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.csna import CSNA
from models.baselines import GCN, GAT, GraphSAGE, H2GCN, GPRGNN, ACM_GNN, MLP


DEVICE = torch.device('cpu')


def load_dataset(name, root='../../data'):
    """Load a heterophily benchmark dataset."""
    if name in ['Texas', 'Wisconsin', 'Cornell']:
        dataset = WebKB(root=root, name=name)
    elif name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root=root, name=name.lower())
    elif name == 'Actor':
        dataset = Actor(root=root)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)

    # Compute homophily ratio
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
    """Generate random train/val/test splits."""
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


def build_model(model_name, in_channels, hidden_channels, out_channels, **kwargs):
    """Instantiate a model by name."""
    dropout = kwargs.get('dropout', 0.5)
    num_layers = kwargs.get('num_layers', 2)

    if model_name == 'CSNA':
        return CSNA(in_channels, hidden_channels, out_channels,
                     num_layers=num_layers, tau=kwargs.get('tau', 1.0),
                     dropout=dropout)
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
                      k=kwargs.get('k', 2), dropout=dropout)
    elif model_name == 'GPRGNN':
        return GPRGNN(in_channels, hidden_channels, out_channels,
                       K=10, dropout=dropout)
    elif model_name == 'ACM_GNN' or model_name == 'ACM-GNN':
        return ACM_GNN(in_channels, hidden_channels, out_channels,
                        num_layers=num_layers, dropout=dropout)
    elif model_name == 'MLP':
        return MLP(in_channels, hidden_channels, out_channels,
                    num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model, data, train_mask, optimizer, model_name='CSNA',
                lambda_adm=0.1, lambda_cons=0.01):
    """Single training epoch."""
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])

    # Add CSNA regularization losses (only use training labels for admissibility)
    if model_name == 'CSNA':
        loss_adm = model.admissibility_loss(data.y, train_mask=train_mask)
        loss_cons = model.consistency_loss()
        loss = loss + lambda_adm * loss_adm + lambda_cons * loss_cons

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluate accuracy on a mask."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    correct = (pred == data.y[mask]).sum().item()
    return correct / mask.sum().item()


def run_single(model_name, data, splits, in_ch, num_classes,
               hidden=64, lr=0.01, weight_decay=5e-4, epochs=200,
               patience=50, **model_kwargs):
    """Run a model across all splits, return list of test accuracies."""
    test_accs = []

    for split_idx, (train_mask, val_mask, test_mask) in enumerate(splits):
        model = build_model(model_name, in_ch, hidden, num_classes, **model_kwargs)
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val = 0
        best_test = 0
        no_improve = 0

        for epoch in range(epochs):
            train_epoch(model, data, train_mask, optimizer, model_name=model_name)
            val_acc = evaluate(model, data, val_mask)
            test_acc = evaluate(model, data, test_mask)

            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        test_accs.append(best_test)

    return test_accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['Texas', 'Wisconsin', 'Cornell', 'Actor', 'Chameleon', 'Squirrel'])
    parser.add_argument('--models', nargs='+',
                        default=['MLP', 'GCN', 'GAT', 'GraphSAGE', 'H2GCN', 'GPRGNN', 'CSNA'])
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--num_splits', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--output', type=str, default='../../results/results.json')
    args = parser.parse_args()

    data_root = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    results = {}
    dataset_infos = {}

    for dname in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dname}")
        print(f"{'='*60}")

        data, info = load_dataset(dname, root=data_root)
        data = data.to(DEVICE)
        dataset_infos[dname] = info
        print(f"  Nodes: {info['num_nodes']}, Edges: {info['num_edges']}, "
              f"Features: {info['num_features']}, Classes: {info['num_classes']}, "
              f"Homophily: {info['homophily']:.3f}")

        splits = generate_splits(data, num_splits=args.num_splits)
        results[dname] = {}

        for mname in args.models:
            t0 = time.time()
            accs = run_single(
                mname, data, splits,
                in_ch=info['num_features'],
                num_classes=info['num_classes'],
                hidden=args.hidden,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                num_layers=args.num_layers,
                dropout=args.dropout,
                tau=args.tau,
            )
            elapsed = time.time() - t0
            mean_acc = np.mean(accs) * 100
            std_acc = np.std(accs) * 100
            results[dname][mname] = {
                'mean': float(mean_acc),
                'std': float(std_acc),
                'accs': [float(a) for a in accs],
                'time': float(elapsed),
            }
            print(f"  {mname:12s}: {mean_acc:.2f} ± {std_acc:.2f}  ({elapsed:.1f}s)")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {'results': results, 'dataset_info': dataset_infos}
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE (Accuracy %)")
    print(f"{'='*80}")
    header = f"{'Dataset':12s}"
    for mname in args.models:
        header += f" | {mname:>12s}"
    print(header)
    print("-" * len(header))
    for dname in args.datasets:
        row = f"{dname:12s}"
        for mname in args.models:
            r = results[dname][mname]
            row += f" | {r['mean']:5.2f}±{r['std']:4.2f}"
        print(row)


if __name__ == '__main__':
    main()
