"""Count trainable parameters for all models across datasets."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from experiments.run_fair_experiments import build_model

datasets_info = {
    'Texas':     {'num_features': 1703, 'num_classes': 5},
    'Wisconsin': {'num_features': 1703, 'num_classes': 5},
    'Cornell':   {'num_features': 1703, 'num_classes': 5},
    'Actor':     {'num_features': 932,  'num_classes': 5},
    'Chameleon': {'num_features': 2325, 'num_classes': 5},
    'Squirrel':  {'num_features': 2089, 'num_classes': 5},
}

# Load tuned configs
results_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'results_v2.json')
configs = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        data = json.load(f)
    configs = data.get('configs', {})

models = ['MLP', 'GCN', 'GAT', 'GraphSAGE', 'H2GCN', 'GPRGNN', 'CSNA']

print(f"{'Dataset':12s}", end="")
for m in models:
    print(f" | {m:>10s}", end="")
print()
print("-" * 100)

for dname, info in datasets_info.items():
    print(f"{dname:12s}", end="")
    for mname in models:
        # Get hidden dim from tuned config if available
        if dname in configs and mname in configs[dname]:
            hidden = configs[dname][mname]['hidden']
            tau = configs[dname][mname].get('tau', 1.0)
        else:
            hidden = 64
            tau = 1.0

        model = build_model(mname, info['num_features'], hidden, info['num_classes'],
                           tau=tau)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if n_params > 1_000_000:
            print(f" | {n_params/1e6:8.1f}M", end="")
        else:
            print(f" | {n_params:8d}K"[:-1], end="")

    print()

# Also print for a representative dataset in a cleaner format
print("\n\nDetailed for Texas (hidden from tuned config):")
dname = 'Texas'
info = datasets_info[dname]
for mname in models:
    if dname in configs and mname in configs[dname]:
        hidden = configs[dname][mname]['hidden']
        tau = configs[dname][mname].get('tau', 1.0)
    else:
        hidden = 64
        tau = 1.0
    model = build_model(mname, info['num_features'], hidden, info['num_classes'],
                       tau=tau)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {mname:12s}: {n_params:>8,d} params (hidden={hidden})")
