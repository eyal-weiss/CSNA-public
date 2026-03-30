"""Generate Figure 2: Cost distribution analysis."""
import os, sys, json, torch, torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.csna import CSNA
from experiments.run_experiments import load_dataset, generate_splits

plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'figure.dpi': 300,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})

with open(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'best_csna_configs.json')) as f:
    configs = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax_idx, dname in enumerate(['Texas', 'Cornell', 'Actor']):
    data, info = load_dataset(dname, root=os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    data = data.to(torch.device('cpu'))
    splits = generate_splits(data, num_splits=10)
    cfg = configs.get(dname, {'tau': 1.0, 'lr': 0.01, 'hidden': 64, 'dropout': 0.5,
                               'num_layers': 2, 'weight_decay': 5e-4})

    model = CSNA(info['num_features'], cfg['hidden'], info['num_classes'],
                  num_layers=cfg['num_layers'], tau=cfg['tau'], dropout=cfg['dropout'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    train_mask = splits[0][0]
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss = loss + 0.1 * model.admissibility_loss(data.y, train_mask=train_mask)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _ = model(data.x, data.edge_index)

    conv = model.convs[0]
    edge_index = conv._edge_index
    h_ij = conv._h_ij.squeeze()
    inp = model.input_mlp(data.x) if model.input_mlp else data.x
    x_g = conv.W_g(inp)
    row, col = edge_index
    g_ij = torch.norm(x_g[row] - x_g[col], p=2, dim=1)
    f_ij = g_ij + h_ij
    s_ij = torch.sigmoid(-f_ij / cfg['tau']).detach().numpy()

    mask_valid = (row < data.y.size(0)) & (col < data.y.size(0))
    same = (data.y[row[mask_valid]] == data.y[col[mask_valid]]).detach().numpy()
    scores = s_ij[mask_valid.detach().numpy()]

    ax = axes[ax_idx]
    ax.hist(scores[same], bins=30, alpha=0.6, color='#2ca02c', label='Same class', density=True)
    ax.hist(scores[~same], bins=30, alpha=0.6, color='#d62728', label='Diff. class', density=True)
    ax.set_xlabel(r'Concordance score $s_{ij}$')
    if ax_idx == 0:
        ax.set_ylabel('Density')
    h_val = info['homophily']
    ax.set_title(f'{dname} ($\\mathcal{{H}}$={h_val:.2f})')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

plt.tight_layout()
fig_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'paper', 'figures')
plt.savefig(os.path.join(fig_dir, 'cost_distribution.pdf'))
plt.savefig(os.path.join(fig_dir, 'cost_distribution.png'))
print('Figure 2 saved')
