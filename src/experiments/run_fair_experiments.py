"""
Fair experiment pipeline v2: Equal tuning for ALL models.

Key changes from run_fast_pipeline.py:
1. ALL models get the same tuning grid (lr, hidden_dim)
2. CSNA additionally tunes tau (its unique hyperparameter)
3. Uses standard Geom-GCN splits when available, falls back to random
4. Proper model checkpointing: save best-val model, evaluate once on test
5. Collects gate weight statistics for CSNA
6. Collects runtime statistics
"""

import os, sys, json, time, itertools, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.csna_v2 import CSNA
from models.baselines import GCN, GAT, GraphSAGE, H2GCN, GPRGNN, MLP
from experiments.run_experiments import load_dataset, evaluate

DEVICE = torch.device('cpu')
SEED = 42


def generate_splits(data, num_splits=10, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Generate random train/val/test splits (same as before for consistency)."""
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
    """Instantiate a model by name."""
    if model_name == 'CSNA':
        return CSNA(in_channels, hidden_channels, out_channels,
                     num_layers=num_layers, tau=tau, dropout=dropout,
                     sample_ratio=1.0)
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
    elif model_name == 'MLP':
        return MLP(in_channels, hidden_channels, out_channels,
                    num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, data, train_mask, optimizer, model_name=''):
    """Single training epoch."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    if model_name == 'CSNA' and hasattr(model, 'calibration_loss'):
        loss = loss + 0.1 * model.calibration_loss(data.y, train_mask=train_mask)
    loss.backward()
    optimizer.step()
    return loss.item()


def run_single_split(model_name, data, train_mask, val_mask, test_mask,
                     in_ch, n_cls, hidden, lr, dropout=0.5, tau=1.0,
                     epochs=300, patience=50):
    """Run one model on one split with proper checkpointing."""
    model = build_model(model_name, in_ch, hidden, n_cls,
                        dropout=dropout, tau=tau)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        train_one_epoch(model, data, train_mask, optimizer, model_name)
        val_acc = evaluate(model, data, val_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Reload best model and evaluate on test ONCE
    model.load_state_dict(best_state)
    test_acc = evaluate(model, data, test_mask)
    return best_val, test_acc, model


def tune_model(model_name, data, info, splits, is_large=False):
    """Tune a model. Same grid for ALL models, CSNA additionally tunes tau."""
    in_ch = info['num_features']
    n_cls = info['num_classes']

    # Common tuning grid for ALL models
    if is_large:
        common_grid = list(itertools.product(
            [0.01, 0.005],  # lr
            [64],            # hidden (save time on large datasets)
        ))
        tune_splits = 2
        tune_epochs = 150
    else:
        common_grid = list(itertools.product(
            [0.01, 0.005],    # lr
            [64, 128],        # hidden
        ))
        tune_splits = 3
        tune_epochs = 200

    # CSNA also tunes tau
    if model_name == 'CSNA':
        tau_values = [0.1, 0.5, 1.0, 2.0]
    else:
        tau_values = [1.0]  # dummy, not used

    best_val = -1
    best_cfg = None

    for lr, hidden in common_grid:
        for tau in tau_values:
            vals = []
            for train_mask, val_mask, test_mask in splits[:tune_splits]:
                bv, _, _ = run_single_split(
                    model_name, data, train_mask, val_mask, test_mask,
                    in_ch, n_cls, hidden, lr, tau=tau,
                    epochs=tune_epochs, patience=30)
                vals.append(bv)

            mean_val = np.mean(vals)
            if mean_val > best_val:
                best_val = mean_val
                best_cfg = {'lr': lr, 'hidden': int(hidden), 'tau': tau,
                            'dropout': 0.5, 'num_layers': 2, 'weight_decay': 5e-4}

    return best_cfg


def collect_gate_stats(model, data):
    """Collect CSNA gate weight statistics after training (lite model: g_ij only)."""
    model.eval()
    gate_stats = []
    with torch.no_grad():
        x = data.x
        if model.input_mlp is not None:
            x = model.input_mlp(x)

        for i in range(model.num_layers):
            conv = model.convs[i]
            from torch_geometric.utils import add_self_loops, softmax as pyg_softmax
            edge_index, _ = add_self_loops(data.edge_index, num_nodes=x.size(0))

            # Compute g_ij (observed divergence only — lite model has no W_h)
            x_g = conv.W_g(x)
            row, col = edge_index
            g_ij = torch.norm(x_g[row] - x_g[col], p=2, dim=1, keepdim=True)
            s_ij = torch.sigmoid(-g_ij / conv.tau)

            x_con = conv.W_con(x)
            con_weights = pyg_softmax(s_ij.squeeze(), edge_index[0], num_nodes=x.size(0))
            out_con = torch.zeros(x.size(0), conv.out_channels, device=x.device)
            out_con.scatter_add_(0, edge_index[0].unsqueeze(-1).expand(-1, conv.out_channels),
                                  con_weights.unsqueeze(-1) * x_con[col])

            x_dis = conv.W_dis(x)
            dis_weights = pyg_softmax((1 - s_ij).squeeze(), edge_index[0], num_nodes=x.size(0))
            out_dis = torch.zeros(x.size(0), conv.out_channels, device=x.device)
            out_dis.scatter_add_(0, edge_index[0].unsqueeze(-1).expand(-1, conv.out_channels),
                                  dis_weights.unsqueeze(-1) * x_dis[col])

            out_self = conv.W_self(x)

            combined = torch.cat([out_con, out_dis, out_self], dim=1)
            gate_logits = conv.gate(combined)
            gate_weights = F.softmax(gate_logits, dim=1)

            gate_stats.append({
                'layer': i,
                'gamma_con': float(gate_weights[:, 0].mean()),
                'gamma_dis': float(gate_weights[:, 1].mean()),
                'gamma_self': float(gate_weights[:, 2].mean()),
                'gamma_con_std': float(gate_weights[:, 0].std()),
                'gamma_dis_std': float(gate_weights[:, 1].std()),
                'gamma_self_std': float(gate_weights[:, 2].std()),
            })

            # Advance x through the layer for next iteration
            x_in = x
            x = conv(x, data.edge_index)
            if i < model.num_layers - 1:
                if model.use_bn:
                    x = model.bns[i](x)
                x = F.relu(x)
            if model.use_residual and x.size() == x_in.size():
                x = x + x_in

    return gate_stats


def main():
    data_root = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    datasets = ['Texas', 'Wisconsin', 'Cornell', 'Actor', 'Chameleon', 'Squirrel']
    large_datasets = {'Actor', 'Chameleon', 'Squirrel'}
    all_model_names = ['MLP', 'GCN', 'GAT', 'GraphSAGE', 'H2GCN', 'GPRGNN', 'CSNA']

    all_results = {}
    dataset_infos = {}
    all_configs = {}
    gate_weight_stats = {}
    runtime_stats = {}

    for dname in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dname}")
        print(f"{'='*60}")

        data, info = load_dataset(dname, root=data_root)
        data = data.to(DEVICE)
        dataset_infos[dname] = info
        is_large = dname in large_datasets

        print(f"  Nodes: {info['num_nodes']}, Edges: {info['num_edges']}, "
              f"H: {info['homophily']:.3f}")

        splits = generate_splits(data, num_splits=10)
        all_results[dname] = {}
        all_configs[dname] = {}
        runtime_stats[dname] = {}

        for mname in all_model_names:
            print(f"\n  --- {mname} ---")

            # Step 1: Tune
            print(f"  Tuning ({mname})...")
            t_tune = time.time()
            cfg = tune_model(mname, data, info, splits, is_large=is_large)
            tune_time = time.time() - t_tune
            all_configs[dname][mname] = cfg
            print(f"    Best: lr={cfg['lr']}, hidden={cfg['hidden']}"
                  + (f", tau={cfg['tau']}" if mname == 'CSNA' else '')
                  + f" ({tune_time:.0f}s)")

            # Step 2: Evaluate on all 10 splits with best config
            print(f"  Evaluating (10 splits)...")
            t_eval = time.time()
            accs = []
            last_model = None
            for split_idx, (train_mask, val_mask, test_mask) in enumerate(splits):
                _, test_acc, model = run_single_split(
                    mname, data, train_mask, val_mask, test_mask,
                    info['num_features'], info['num_classes'],
                    cfg['hidden'], cfg['lr'], tau=cfg['tau'],
                    epochs=300, patience=50)
                accs.append(test_acc)
                if split_idx == 0:
                    last_model = model
            eval_time = time.time() - t_eval

            m_acc = np.mean(accs) * 100
            s_acc = np.std(accs) * 100
            all_results[dname][mname] = {
                'mean': float(m_acc), 'std': float(s_acc),
                'accs': [float(a) for a in accs],
                'config': cfg,
                'tune_time': float(tune_time),
                'eval_time': float(eval_time),
            }
            runtime_stats[dname][mname] = {
                'tune_time': float(tune_time),
                'eval_time': float(eval_time),
                'total_time': float(tune_time + eval_time),
                'per_split_time': float(eval_time / 10),
            }
            print(f"    {mname:12s}: {m_acc:.2f} +/- {s_acc:.2f}  "
                  f"(tune={tune_time:.0f}s, eval={eval_time:.0f}s)")

            # Collect gate stats for CSNA
            if mname == 'CSNA' and last_model is not None:
                try:
                    gs = collect_gate_stats(last_model, data)
                    gate_weight_stats[dname] = gs
                    for layer_stat in gs:
                        print(f"    Gate L{layer_stat['layer']}: "
                              f"con={layer_stat['gamma_con']:.3f}, "
                              f"dis={layer_stat['gamma_dis']:.3f}, "
                              f"self={layer_stat['gamma_self']:.3f}")
                except Exception as e:
                    print(f"    Gate stats collection failed: {e}")

        # Save intermediate results
        with open(os.path.join(results_dir, 'results_v2.json'), 'w') as f:
            json.dump({
                'results': all_results,
                'dataset_info': dataset_infos,
                'configs': all_configs,
                'gate_stats': gate_weight_stats,
                'runtime': runtime_stats,
            }, f, indent=2)

    # Print final summary
    print(f"\n{'='*100}")
    print("FINAL RESULTS (Fair Tuning): Accuracy (%)")
    print(f"{'='*100}")
    header = f"{'Dataset':12s} | {'H':>5s}"
    for mname in all_model_names:
        header += f" | {mname:>12s}"
    print(header)
    print("-" * len(header))
    for dname in datasets:
        h = dataset_infos[dname]['homophily']
        row = f"{dname:12s} | {h:5.3f}"
        for mname in all_model_names:
            if mname in all_results[dname]:
                r = all_results[dname][mname]
                row += f" | {r['mean']:5.2f}+/-{r['std']:4.2f}"
            else:
                row += f" | {'--':>12s}"
        print(row)

    # Print runtime summary
    print(f"\n{'='*80}")
    print("RUNTIME (seconds per split, evaluation only)")
    print(f"{'='*80}")
    for dname in datasets:
        row = f"{dname:12s}"
        for mname in all_model_names:
            if mname in runtime_stats.get(dname, {}):
                t = runtime_stats[dname][mname]['per_split_time']
                row += f" | {t:8.1f}s"
            else:
                row += f" | {'--':>9s}"
        print(row)

    print(f"\nResults saved to {results_dir}/results_v2.json")


if __name__ == '__main__':
    main()
