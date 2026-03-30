"""
Generate publication-quality figures from results_v2.json.
Fixes from v1:
- Performance plot uses scatter (not connected lines) since datasets are discrete
- Better label placement for overlapping H values
"""

import os, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def gen_performance_scatter(results, dataset_info, outdir):
    """Performance scatter plot: accuracy vs homophily ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['MLP', 'GCN', 'GAT', 'GraphSAGE', 'H2GCN', 'GPRGNN', 'CSNA']
    markers = ['x', 'o', 's', '^', 'D', 'v', '*']
    colors = ['gray', '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#E03131']
    sizes = [80, 60, 60, 60, 60, 60, 120]

    datasets = list(results.keys())
    homophily = [dataset_info[d]['homophily'] for d in datasets]

    for mi, mname in enumerate(models):
        accs = []
        hs = []
        stds = []
        for d in datasets:
            if mname in results[d]:
                accs.append(results[d][mname]['mean'])
                stds.append(results[d][mname]['std'])
                hs.append(dataset_info[d]['homophily'])

        zorder = 10 if mname == 'CSNA' else 5
        ax.scatter(hs, accs, marker=markers[mi], c=colors[mi],
                   s=sizes[mi], label=mname, zorder=zorder,
                   edgecolors='black' if mname == 'CSNA' else 'none',
                   linewidths=0.5 if mname == 'CSNA' else 0)

    # Add dataset labels with offset to avoid overlap
    label_offsets = {
        'Texas': (0, 5),
        'Cornell': (0, -8),
        'Wisconsin': (0, 3),
        'Actor': (-0.008, -8),
        'Chameleon': (0.005, 3),
        'Squirrel': (-0.005, -8),
    }
    for d in datasets:
        h = dataset_info[d]['homophily']
        # Find max accuracy for this dataset to place label
        max_acc = max(results[d][m]['mean'] for m in results[d])
        off = label_offsets.get(d, (0, 5))
        ax.annotate(d, (h + off[0], max_acc + off[1]),
                    fontsize=8, ha='center', style='italic')

    ax.set_xlabel('Edge Homophily Ratio $\\mathcal{H}$', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(outdir, f'performance_vs_homophily.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved performance_vs_homophily.pdf/png")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    fig_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'paper', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Try v2 results first, fall back to v1
    v2_path = os.path.join(results_dir, 'results_v2.json')
    v1_path = os.path.join(results_dir, 'results.json')

    if os.path.exists(v2_path):
        with open(v2_path) as f:
            data = json.load(f)
        results = data['results']
        dataset_info = data['dataset_info']
        print("Using results_v2.json (fair tuning)")
    elif os.path.exists(v1_path):
        with open(v1_path) as f:
            data = json.load(f)
        results = data['results']
        dataset_info = data['dataset_info']
        print("Using results.json (v1)")
    else:
        print("No results file found!")
        sys.exit(1)

    gen_performance_scatter(results, dataset_info, fig_dir)


if __name__ == '__main__':
    main()
