"""
Baseline models for heterophily benchmarks.

Includes: GCN, GAT, GraphSAGE, H2GCN, GPRGNN, ACM-GNN, MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


# --- GCN -------------------------------------------------------------------

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# --- GAT -------------------------------------------------------------------

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, heads=8, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads,
                                  heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads,
                                      heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=1,
                                  concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# --- GraphSAGE -------------------------------------------------------------

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# --- H2GCN -----------------------------------------------------------------
# Zhu et al., "Beyond Homophily in Graph Neural Networks" (NeurIPS 2020)

class H2GCN(nn.Module):
    """Simplified H2GCN: separate ego/neighbor/2-hop embeddings, concatenate."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.lin_feat = nn.Linear(in_channels, hidden_channels)

        self.round_lins = nn.ModuleList()
        self.round_bns = nn.ModuleList()
        for _ in range(num_layers):
            self.round_lins.append(nn.Linear(hidden_channels * 3, hidden_channels))
            self.round_bns.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        from torch_geometric.utils import degree

        num_nodes = x.size(0)
        row, col = edge_index

        deg = degree(row, num_nodes, dtype=x.dtype).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.lin_feat(x)
        x = F.relu(x)

        for i in range(self.num_layers):
            # 1-hop aggregation
            agg1 = torch.zeros_like(x)
            agg1.scatter_add_(0, col.unsqueeze(-1).expand(-1, x.size(1)),
                              norm.unsqueeze(-1) * x[row])

            # 2-hop aggregation
            agg2 = torch.zeros_like(x)
            agg2.scatter_add_(0, col.unsqueeze(-1).expand(-1, x.size(1)),
                              norm.unsqueeze(-1) * agg1[row])

            combined = torch.cat([x, agg1, agg2], dim=1)
            x = self.round_lins[i](combined)
            x = self.round_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)


# --- GPRGNN ----------------------------------------------------------------
# Chien et al., "Adaptive Universal Generalized PageRank GNN" (ICLR 2021)

class GPRGNN(nn.Module):
    """Generalized PageRank GNN: learns polynomial filter coefficients."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 K=10, dropout=0.5, alpha=0.1):
        super().__init__()
        self.dropout = dropout
        self.K = K

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.bn = nn.BatchNorm1d(hidden_channels)

        init_gamma = alpha * (1 - alpha) ** torch.arange(K + 1).float()
        init_gamma[-1] = (1 - alpha) ** K
        self.gamma = nn.Parameter(init_gamma)

    def forward(self, x, edge_index):
        from torch_geometric.utils import degree

        num_nodes = x.size(0)
        row, col = edge_index

        deg = degree(row, num_nodes, dtype=x.dtype).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.lin1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        hidden = x * self.gamma[0]
        h = x
        for k in range(1, self.K + 1):
            h_new = torch.zeros_like(h)
            h_new.scatter_add_(0, col.unsqueeze(-1).expand(-1, h.size(1)),
                               norm.unsqueeze(-1) * h[row])
            h = h_new
            hidden = hidden + self.gamma[k] * h

        return hidden


# --- ACM-GNN ---------------------------------------------------------------
# Luan et al., "Revisiting Heterophily For Graph Neural Networks" (NeurIPS 2022)

class ACMConv(nn.Module):
    """ACM-GNN convolutional layer with three filter channels:
    low-pass (GCN), high-pass (I - A_low), identity (MLP)."""

    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        self.W_low = nn.Linear(in_channels, out_channels, bias=True)
        self.W_high = nn.Linear(in_channels, out_channels, bias=True)
        self.W_id = nn.Linear(in_channels, out_channels, bias=True)
        self.gate = nn.Linear(in_channels, 3, bias=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        from torch_geometric.utils import add_self_loops, degree

        num_nodes = x.size(0)

        # Low-pass: A_low = D_hat^{-1/2} (I + A) D_hat^{-1/2}
        edge_index_hat, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_hat
        deg = degree(row, num_nodes, dtype=x.dtype).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x_low = self.W_low(x)
        agg_low = torch.zeros_like(x_low)
        agg_low.scatter_add_(0, col.unsqueeze(-1).expand(-1, x_low.size(1)),
                             norm.unsqueeze(-1) * x_low[row])

        # High-pass: I - A_low
        x_high = self.W_high(x)
        agg_high_low = torch.zeros_like(x_high)
        agg_high_low.scatter_add_(0, col.unsqueeze(-1).expand(-1, x_high.size(1)),
                                  norm.unsqueeze(-1) * x_high[row])
        agg_high = x_high - agg_high_low

        # Identity (MLP)
        x_id = self.W_id(x)

        # Per-node gating
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=1)

        out = (gate_weights[:, 0:1] * agg_low +
               gate_weights[:, 1:2] * agg_high +
               gate_weights[:, 2:3] * x_id)

        return out


class ACM_GNN(nn.Module):
    """ACM-GNN model for node classification."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(ACMConv(in_channels, hidden_channels, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(ACMConv(hidden_channels, hidden_channels, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(ACMConv(hidden_channels, out_channels, dropout=dropout))

    def forward(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# --- MLP -------------------------------------------------------------------

class MLP(nn.Module):
    """Simple MLP baseline (no graph structure)."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index=None):
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
