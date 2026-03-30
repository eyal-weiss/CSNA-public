"""
Re-run the three baselines that were replaced with faithful implementations:
H2GCN, GPRGNN, ACM-GNN.

Uses the exact same fair tuning protocol as run_fair_experiments.py:
  - lr:     {0.01, 0.005}
  - hidden: {64, 128} (small datasets) or {64} (large datasets)
  - Tune on 3 splits (small) or 2 splits (large)
  - Evaluate on all 10 splits with best config
  - Proper model checkpointing (save best-val, eval test once)
  - Early stopping patience=50, max 300 epochs

Results are merged into results_v2.json (overwriting only the three re-run models).
"""

import os, sys, json, time, itertools, copy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.baselines import H2GCN, GPRGNN, ACM_GNN
from experiments.run_experiments import load_dataset, evaluate
from experiments.run_fair_experiments import generate_splits

DEVICE = torch.device('cpu')


def build_model(model_name, in_channels, hidden_channels, out_channels, dropout=0.5):
    if model_name == 'H2GCN':
        return H2GCN(in_channels, hidden_channels, out_channels,
                      k=2, dropout=dropout)
    elif model_name == 'GPRGNN':
        return GPRGNN(in_channels, hidden_channels, out_channels,
                       K=10, alpha=0.1, dropout=dropout, dprate=0.5)
    elif model_name == 'ACM-GNN':
        return ACM_GNN(in_channels, hidden_channels, out_channels,
                        num_layers=2, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, data, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def run_single_split(model_name, data, train_mask, val_mask, test_mask,
                     in_ch, n_cls, hidden, lr, dropout=0.5,
                     epochs=300, patience=50):
    model = build_model(model_name, in_ch, hidden, n_cls, dropout=dropout)
    model = model.to(DEVICE)

    # GPRGNN: zero weight decay on propagation coefficients (per official repo)
    if model_name == 'GPRGNN':
        optimizer = torch.optim.Adam([
            {'params': model.lin1.parameters(), 'weight_decay': 5e-4},
            {'params': model.lin2.parameters(), 'weight_decay': 5e-4},
            {'params': model.prop.parameters(), 'weight_decay': 0.0},
        ], lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        train_one_epoch(model, data, train_mask, optimizer)
        val_acc = evaluate(model, data, val_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    test_acc = evaluate(model, data, test_mask)
    return best_val, test_acc


def tune_and_evaluate(model_name, data, info, splits, is_large=False):
    in_ch = info['num_features']
    n_cls = info['num_classes']

    if is_large:
        grid = list(itertools.product([0.01, 0.005], [64]))
        tune_splits, tune_epochs = 2, 150
    else:
        grid = list(itertools.product([0.01, 0.005], [64, 128]))
        tune_splits, tune_epochs = 3, 200

    # Tune
    best_val = -1
    best_cfg = None
    for lr, hidden in grid:
        vals = []
        for train_mask, val_mask, test_mask in splits[:tune_splits]:
            bv, _ = run_single_split(model_name, data, train_mask, val_mask, test_mask,
                                     in_ch, n_cls, hidden, lr, epochs=tune_epochs, patience=30)
            vals.append(bv)
        mean_val = np.mean(vals)
        if mean_val > best_val:
            best_val = mean_val
            best_cfg = {'lr': lr, 'hidden': int(hidden), 'dropout': 0.5}

    # Evaluate on all 10 splits
    accs = []
    for train_mask, val_mask, test_mask in splits:
        _, test_acc = run_single_split(model_name, data, train_mask, val_mask, test_mask,
                                       in_ch, n_cls, best_cfg['hidden'], best_cfg['lr'],
                                       epochs=300, patience=50)
        accs.append(test_acc)

    return accs, best_cfg


def main():
    data_root = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    datasets = ['Texas', 'Wisconsin', 'Cornell', 'Actor', 'Chameleon', 'Squirrel']
    large_datasets = {'Actor', 'Chameleon', 'Squirrel'}
    models_to_run = ['H2GCN', 'GPRGNN', 'ACM-GNN']

    results = {}

    for dname in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dname}")
        print(f"{'='*60}")

        data, info = load_dataset(dname, root=data_root)
        data = data.to(DEVICE)
        is_large = dname in large_datasets
        print(f"  Nodes: {info['num_nodes']}, Edges: {info['num_edges']}, H: {info['homophily']:.3f}")

        splits = generate_splits(data, num_splits=10)
        results[dname] = {}

        for mname in models_to_run:
            print(f"\n  --- {mname} ---")
            t0 = time.time()
            accs, cfg = tune_and_evaluate(mname, data, info, splits, is_large=is_large)
            elapsed = time.time() - t0

            m_acc = np.mean(accs) * 100
            s_acc = np.std(accs) * 100
            results[dname][mname] = {
                'mean': float(m_acc), 'std': float(s_acc),
                'accs': [float(a) for a in accs],
                'config': cfg,
                'time': float(elapsed),
            }
            print(f"    {mname:12s}: {m_acc:.1f} +/- {s_acc:.1f}  "
                  f"(cfg: lr={cfg['lr']}, h={cfg['hidden']}, {elapsed:.0f}s)")

        # Save incremental
        with open(os.path.join(results_dir, 'faithful_baselines.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("FAITHFUL BASELINES RESULTS (Accuracy %)")
    print(f"{'='*80}")
    header = f"{'Dataset':12s}"
    for m in models_to_run:
        header += f" | {m:>12s}"
    print(header)
    print("-" * len(header))
    for dname in datasets:
        row = f"{dname:12s}"
        for m in models_to_run:
            r = results[dname][m]
            row += f" | {r['mean']:5.1f}+/-{r['std']:4.1f}"
        print(row)

    # Merge into results_v2.json
    v2_path = os.path.join(results_dir, 'results_v2.json')
    if os.path.exists(v2_path):
        with open(v2_path) as f:
            v2 = json.load(f)
        for dname in datasets:
            for mname in models_to_run:
                # Map ACM-GNN key for consistency
                key = 'ACM-GNN' if mname == 'ACM-GNN' else mname
                v2['results'][dname][key] = results[dname][mname]
        with open(v2_path, 'w') as f:
            json.dump(v2, f, indent=2)
        print(f"\nMerged into {v2_path}")

    print(f"\nStandalone results saved to {results_dir}/faithful_baselines.json")


if __name__ == '__main__':
    main()
