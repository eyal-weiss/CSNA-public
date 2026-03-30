"""
Generate paper figures:
1. Performance vs homophily ratio (the "killer figure")
2. Cost distribution analysis (concordance scores for same-class vs different-class edges)
3. Ablation bar chart
4. Gate weight analysis
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from experiments.run_experiments import load_dataset, generate_splits
from models.csna import CSNA

FIGURE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'paper', 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def fig1_performance_vs_homophily(results_path):
    """Plot accuracy vs homophily ratio for all methods."""
    with open(results_path) as f:
        data = json.load(f)

    results = data['results']
    infos = data['dataset_info']

    models = ['MLP', 'GCN', 'GAT', 'GraphSAGE', 'H2GCN', 'GPRGNN', 'CSNA']
    colors = {
        'MLP': '#888888', 'GCN': '#1f77b4', 'GAT': '#ff7f0e',
        'GraphSAGE': '#2ca02c', 'H2GCN': '#d62728', 'GPRGNN': '#9467bd',
        'CSNA': '#e41a1c'
    }
    markers = {
        'MLP': 'x', 'GCN': 'o', 'GAT': 's', 'GraphSAGE': '^',
        'H2GCN': 'D', 'GPRGNN': 'v', 'CSNA': '*'
    }
    linestyles = {
        'MLP': ':', 'GCN': '--', 'GAT': '--', 'GraphSAGE': '--',
        'H2GCN': '-.', 'GPRGNN': '-.', 'CSNA': '-'
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    datasets = list(results.keys())
    homophily_ratios = [infos[d]['homophily'] for d in datasets]
    sorted_idx = np.argsort(homophily_ratios)

    for mname in models:
        if mname not in list(results.values())[0]:
            continue
        h_vals = [homophily_ratios[i] for i in sorted_idx]
        acc_vals = [results[datasets[i]][mname]['mean'] for i in sorted_idx]
        std_vals = [results[datasets[i]][mname]['std'] for i in sorted_idx]

        ax.errorbar(h_vals, acc_vals, yerr=std_vals,
                     label=mname, color=colors.get(mname, 'black'),
                     marker=markers.get(mname, 'o'),
                     markersize=9 if mname == 'CSNA' else 6,
                     linewidth=2.5 if mname == 'CSNA' else 1.2,
                     linestyle=linestyles.get(mname, '-'),
                     capsize=3, alpha=0.85 if mname != 'CSNA' else 1.0,
                     zorder=10 if mname == 'CSNA' else 5)

    # Add dataset name annotations
    for i in sorted_idx:
        d = datasets[i]
        h = homophily_ratios[i]
        # Place label at the bottom
        ax.annotate(d, (h, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 20),
                     textcoords="offset points", xytext=(0, 5),
                     ha='center', fontsize=7, color='gray', rotation=45)

    ax.set_xlabel('Edge Homophily Ratio $\\mathcal{H}$')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Node Classification Accuracy vs. Homophily')
    ax.legend(loc='best', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'performance_vs_homophily.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {path}")


def fig2_cost_distribution(configs=None):
    """Analyze concordance score distributions for same-class vs different-class edges."""
    data_root = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    datasets_to_plot = ['Texas', 'Chameleon', 'Squirrel']

    for ax_idx, dname in enumerate(datasets_to_plot):
        data, info = load_dataset(dname, root=data_root)
        data = data.to(torch.device('cpu'))
        splits = generate_splits(data, num_splits=10)

        # Load best config
        cfg = {'tau': 1.0, 'lr': 0.01, 'hidden': 64, 'dropout': 0.5, 'num_layers': 2, 'weight_decay': 5e-4}
        if configs and dname in configs:
            cfg = configs[dname]

        # Train a model
        model = CSNA(info['num_features'], cfg['hidden'], info['num_classes'],
                      num_layers=cfg['num_layers'], tau=cfg['tau'], dropout=cfg['dropout'])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        train_mask, val_mask, test_mask = splits[0]
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])
            loss = loss + 0.1 * model.admissibility_loss(data.y)
            loss.backward()
            optimizer.step()

        # Extract concordance scores
        model.eval()
        with torch.no_grad():
            _ = model(data.x, data.edge_index)

        conv = model.convs[0]
        if hasattr(conv, '_h_ij'):
            edge_index = conv._edge_index
            h_ij = conv._h_ij.squeeze()
            x_g = conv.W_g(data.x)
            row, col = edge_index
            g_ij = torch.norm(x_g[row] - x_g[col], p=2, dim=1)
            f_ij = g_ij + h_ij
            s_ij = torch.sigmoid(-f_ij / cfg['tau']).numpy()

            # Separate same-class and different-class
            mask_valid = (row < data.y.size(0)) & (col < data.y.size(0))
            same_class = (data.y[row[mask_valid]] == data.y[col[mask_valid]]).numpy()
            scores = s_ij[mask_valid.numpy()]

            ax = axes[ax_idx]
            ax.hist(scores[same_class], bins=30, alpha=0.6, color='#2ca02c',
                     label='Same class', density=True)
            ax.hist(scores[~same_class], bins=30, alpha=0.6, color='#d62728',
                     label='Diff. class', density=True)
            ax.set_xlabel('Concordance score $s_{ij}$')
            ax.set_ylabel('Density')
            ax.set_title(f'{dname} ($\\mathcal{{H}}$={info["homophily"]:.2f})')
            ax.legend(fontsize=8)
            ax.set_xlim(0, 1)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'cost_distribution.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {path}")


def fig3_ablation_chart(ablation_path):
    """Bar chart of ablation results."""
    with open(ablation_path) as f:
        ablation = json.load(f)

    datasets = list(ablation.keys())
    variants = list(ablation[datasets[0]].keys())

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 3.5))
    if len(datasets) == 1:
        axes = [axes]

    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    short_names = {
        'CSNA-full': 'Full',
        'CSNA-no-dual': 'No Dual',
        'CSNA-no-adm': 'No Adm.',
        'CSNA-no-heuristic': 'No Heur.',
    }

    for ax_idx, dname in enumerate(datasets):
        ax = axes[ax_idx]
        vals = [ablation[dname][v]['mean'] for v in variants]
        stds = [ablation[dname][v]['std'] for v in variants]
        names = [short_names.get(v, v) for v in variants]

        bars = ax.bar(range(len(variants)), vals, yerr=stds, capsize=4,
                       color=colors[:len(variants)], edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(dname)

        # Highlight best
        best_idx = np.argmax(vals)
        bars[best_idx].set_edgecolor('#e41a1c')
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'ablation.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {path}")


def fig4_gate_weights(configs=None):
    """Visualize gate weight distributions across datasets."""
    data_root = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    datasets_to_plot = ['Texas', 'Chameleon', 'Squirrel']

    for ax_idx, dname in enumerate(datasets_to_plot):
        data, info = load_dataset(dname, root=data_root)
        data = data.to(torch.device('cpu'))
        splits = generate_splits(data, num_splits=10)

        cfg = {'tau': 1.0, 'lr': 0.01, 'hidden': 64, 'dropout': 0.5, 'num_layers': 2, 'weight_decay': 5e-4}
        if configs and dname in configs:
            cfg = configs[dname]

        model = CSNA(info['num_features'], cfg['hidden'], info['num_classes'],
                      num_layers=cfg['num_layers'], tau=cfg['tau'], dropout=cfg['dropout'])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        train_mask = splits[0][0]
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])
            loss = loss + 0.1 * model.admissibility_loss(data.y)
            loss.backward()
            optimizer.step()

        # Extract gate weights from first layer
        model.eval()
        with torch.no_grad():
            # Manual forward to capture gate weights
            x = data.x
            conv = model.convs[0]
            out = conv(x, data.edge_index)

            # Recompute gate weights
            edge_index_sl, _ = torch._C._VariableFunctions.frobenius_norm if False else (None, None)
            # Just re-run forward and hook into the gate
            from torch_geometric.utils import add_self_loops, softmax as pyg_softmax
            edge_index_sl, _ = add_self_loops(data.edge_index, num_nodes=x.size(0))
            x_g = conv.W_g(x)
            row, col = edge_index_sl
            g_ij = torch.norm(x_g[row] - x_g[col], p=2, dim=1, keepdim=True)
            h_ij = F.softplus(conv.W_h(torch.cat([x_g[row], x_g[col]], dim=1)))
            f_ij = g_ij + h_ij
            s_ij = torch.sigmoid(-f_ij / cfg['tau'])

            x_con = conv.W_con(x)
            con_w = pyg_softmax(s_ij.squeeze(), edge_index_sl[0], num_nodes=x.size(0))
            out_con = torch.zeros(x.size(0), x_con.size(1))
            out_con.scatter_add_(0, col.unsqueeze(-1).expand(-1, x_con.size(1)),
                                  (con_w.unsqueeze(-1) * x_con[row]))

            x_dis = conv.W_dis(x)
            dis_w = pyg_softmax((1 - s_ij).squeeze(), edge_index_sl[0], num_nodes=x.size(0))
            out_dis = torch.zeros(x.size(0), x_dis.size(1))
            out_dis.scatter_add_(0, col.unsqueeze(-1).expand(-1, x_dis.size(1)),
                                  (dis_w.unsqueeze(-1) * x_dis[row]))

            out_self = conv.W_self(x)

            combined = torch.cat([out_con, out_dis, out_self], dim=1)
            gate_weights = F.softmax(conv.gate(combined), dim=1).numpy()

        ax = axes[ax_idx]
        labels = ['Concordant', 'Discordant', 'Self']
        means = gate_weights.mean(axis=0)
        stds = gate_weights.std(axis=0)

        bars = ax.bar(range(3), means, yerr=stds, capsize=4,
                       color=['#2ca02c', '#d62728', '#1f77b4'],
                       edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Gate Weight')
        ax.set_title(f'{dname} ($\\mathcal{{H}}$={info["homophily"]:.2f})')
        ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'gate_weights.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {path}")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

    # Load configs
    cfg_path = os.path.join(results_dir, 'best_csna_configs.json')
    configs = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            configs = json.load(f)

    # Figure 1: Performance vs homophily
    results_path = os.path.join(results_dir, 'results.json')
    if os.path.exists(results_path):
        print("Generating Figure 1: Performance vs Homophily...")
        fig1_performance_vs_homophily(results_path)

    # Figure 2: Cost distribution
    print("Generating Figure 2: Cost Distribution...")
    fig2_cost_distribution(configs)

    # Figure 3: Ablation
    ablation_path = os.path.join(results_dir, 'ablation_results.json')
    if os.path.exists(ablation_path):
        print("Generating Figure 3: Ablation...")
        fig3_ablation_chart(ablation_path)

    # Figure 4: Gate weights
    print("Generating Figure 4: Gate Weights...")
    fig4_gate_weights(configs)


if __name__ == '__main__':
    main()
