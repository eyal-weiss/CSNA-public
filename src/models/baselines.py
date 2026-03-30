"""
Baseline models for heterophily benchmarks.

Standard baselines (GCN, GAT, GraphSAGE, MLP) use PyG layers with uniform design.

Heterophily-specific baselines are faithful to the original papers/repos:
  - H2GCN: Zhu et al., NeurIPS 2020.  Based on GitEventhandler/H2GCN-PyTorch.
  - GPRGNN: Chien et al., ICLR 2021.  Based on jianhao2016/GPRGNN.
  - ACM-GNN: Luan et al., NeurIPS 2022.  Based on SitaoLuan/ACM-GNN.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, degree, to_scipy_sparse_matrix
import numpy as np


# ─── GCN ──────────────────────────────────────────────

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
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


# ─── GAT ──────────────────────────────────────────────

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 heads=8, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ─── GraphSAGE ────────────────────────────────────────

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
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


# ─── H2GCN (faithful) ────────────────────────────────
# Zhu et al., "Beyond Homophily in Graph Neural Networks" (NeurIPS 2020)
# Based on: https://github.com/GitEventhandler/H2GCN-PyTorch
#
# Key design choices from the paper:
#   1. Ego/neighbor separation: ego embedding (r^0) is separate from neighbor aggs
#   2. Exact non-ego neighborhoods: A1 = adj without self-loops,
#      A2 = indicator(A^2 - A - I) for true 2-hop-only neighbors
#   3. No learned transforms during aggregation rounds — just sparse matmul + concat
#   4. Final representation = concat of ALL intermediate [r^0, r^1, ..., r^K]
#   5. Only two learnable matrices: W_embed and W_classify

class H2GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 k=2, dropout=0.5, use_relu=True, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.k = k
        self.act = F.relu if use_relu else lambda x: x

        self.w_embed = nn.Parameter(torch.empty(in_channels, hidden_channels))
        # After K rounds: r^0 is h-dim, each r^i doubles (concat of A1*r, A2*r)
        # Total: h + 2h + 4h + ... + 2^K * h = (2^(K+1) - 1) * h
        classify_dim = (2 ** (k + 1) - 1) * hidden_channels
        self.w_classify = nn.Parameter(torch.empty(classify_dim, out_channels))

        self._initialized = False
        self._a1 = None
        self._a2 = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor):
        """Binarize sparse tensor: all positive values become 1."""
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1.0, 0.0),
            size=csp.size(), dtype=torch.float
        )

    @staticmethod
    def _sym_normalize(adj):
        """Symmetric normalization D^{-1/2} A D^{-1/2} for sparse tensor."""
        n = adj.size(0)
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.0
        indices = torch.arange(n, device=adj.device)
        D = torch.sparse_coo_tensor(
            torch.stack([indices, indices]), d_inv_sqrt, (n, n)
        )
        return torch.sparse.mm(torch.sparse.mm(D, adj), D)

    def _prepare_prop(self, edge_index, num_nodes, device):
        """Precompute A1 (1-hop, no self-loops) and A2 (exact 2-hop non-ego)."""
        # Build adjacency as sparse tensor (with self-loops from input)
        row, col = edge_index
        values = torch.ones(row.size(0), dtype=torch.float, device=device)
        adj = torch.sparse_coo_tensor(
            torch.stack([row, col]), values, (num_nodes, num_nodes)
        ).coalesce()

        # Identity matrix
        idx = torch.arange(num_nodes, device=device)
        eye = torch.sparse_coo_tensor(
            torch.stack([idx, idx]),
            torch.ones(num_nodes, dtype=torch.float, device=device),
            (num_nodes, num_nodes)
        )

        # A1 = indicator(adj - I): adjacency without self-loops
        a1 = self._indicator(adj - eye)
        # A2 = indicator(A^2 - A - I): exact 2-hop, excluding 1-hop and self
        a_sq = torch.sparse.mm(adj, adj)
        a2 = self._indicator(a_sq - adj - eye)

        # Symmetric normalization
        self._a1 = self._sym_normalize(a1)
        self._a2 = self._sym_normalize(a2)
        self._initialized = True

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        if not self._initialized:
            self._prepare_prop(edge_index, num_nodes, x.device)

        # Initial ego embedding
        rs = [self.act(torch.mm(x, self.w_embed))]

        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.sparse.mm(self._a1, r_last)
            r2 = torch.sparse.mm(self._a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))

        # Concatenate ALL intermediate representations
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        return torch.mm(r_final, self.w_classify)


# ─── GPRGNN (faithful) ───────────────────────────────
# Chien et al., "Adaptive Universal Generalized PageRank GNN" (ICLR 2021)
# Based on: https://github.com/jianhao2016/GPRGNN
#
# Key design choices from the paper / official code:
#   1. Propagation uses gcn_norm (self-loop-augmented symmetric normalization)
#   2. Learnable polynomial coefficients gamma (multiple init modes)
#   3. Separate dprate (propagation dropout) from dropout (MLP dropout)
#   4. Zero weight decay on gamma coefficients (handled in optimizer setup)

class GPR_prop(MessagePassing):
    """Propagation class for GPRGNN. Faithful to official implementation."""

    def __init__(self, K, alpha, Init='PPR', **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        if Init == 'SGC':
            TEMP = np.zeros(K + 1)
            TEMP[int(alpha)] = 1.0
        elif Init == 'PPR':
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            TEMP = alpha ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        else:
            raise ValueError(f"Unknown Init: {Init}")

        self.temp = Parameter(torch.tensor(TEMP, dtype=torch.float32))

    def reset_parameters(self):
        if self.Init == 'PPR':
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
            self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        # gcn_norm adds self-loops and computes D^{-1/2}(A+I)D^{-1/2}
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            hidden = hidden + self.temp[k + 1] * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 K=10, alpha=0.1, Init='PPR', dropout=0.5, dprate=0.5, **kwargs):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = GPR_prop(K, alpha, Init)
        self.dropout = dropout
        self.dprate = dprate

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate > 0:
            x = F.dropout(x, p=self.dprate, training=self.training)

        x = self.prop(x, edge_index)
        return x


# ─── ACM-GNN (faithful) ──────────────────────────────
# Luan et al., "Revisiting Heterophily For Graph Neural Networks" (NeurIPS 2022)
# Based on: https://github.com/SitaoLuan/ACM-GNN
#
# Key design choices from the paper / official code:
#   1. Three channels: low-pass (A_low = row-norm(I+A)), high-pass (I - A_low), identity
#   2. Separate weight matrices per channel
#   3. Attention: per-channel projection → sigmoid → [3,3] mixing matrix / T=3 → softmax
#   4. Output multiplied by 3 (number of channels)
#   5. variant=False (default): filter then nonlinearity (ACMII in paper)
#      variant=True: nonlinearity then filter (ACM in paper)

class ACMGraphConv(nn.Module):
    """Single ACM-GNN layer. Faithful to official GraphConvolution class."""

    def __init__(self, in_features, out_features, variant=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.variant = variant

        self.weight_low = Parameter(torch.empty(in_features, out_features))
        self.weight_high = Parameter(torch.empty(in_features, out_features))
        self.weight_mlp = Parameter(torch.empty(in_features, out_features))

        self.att_vec_low = Parameter(torch.empty(out_features, 1))
        self.att_vec_high = Parameter(torch.empty(out_features, 1))
        self.att_vec_mlp = Parameter(torch.empty(out_features, 1))

        self.att_vec = Parameter(torch.empty(3, 3))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))
        std_att = 1.0 / math.sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)

        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

    def attention(self, output_low, output_high, output_mlp):
        T = 3
        logits = torch.mm(
            torch.sigmoid(torch.cat([
                torch.mm(output_low, self.att_vec_low),
                torch.mm(output_high, self.att_vec_high),
                torch.mm(output_mlp, self.att_vec_mlp),
            ], dim=1)),
            self.att_vec
        ) / T
        att = torch.softmax(logits, dim=1)
        return att[:, 0:1], att[:, 1:2], att[:, 2:3]

    def forward(self, x, adj_low, adj_high):
        if self.variant:
            # ACM: nonlinearity then filter
            output_low = torch.spmm(adj_low, F.relu(torch.mm(x, self.weight_low)))
            output_high = torch.spmm(adj_high, F.relu(torch.mm(x, self.weight_high)))
            output_mlp = F.relu(torch.mm(x, self.weight_mlp))
        else:
            # ACMII (default): filter then nonlinearity
            output_low = F.relu(torch.spmm(adj_low, torch.mm(x, self.weight_low)))
            output_high = F.relu(torch.spmm(adj_high, torch.mm(x, self.weight_high)))
            output_mlp = F.relu(torch.mm(x, self.weight_mlp))

        att_low, att_high, att_mlp = self.attention(output_low, output_high, output_mlp)
        return 3 * (att_low * output_low + att_high * output_high + att_mlp * output_mlp)


class ACM_GNN(nn.Module):
    """ACM-GNN model for node classification.

    Faithful to official repo (SitaoLuan/ACM-GNN, 'acmgcn' variant).
    Precomputes adj_low = row_norm(I+A) and adj_high = I - adj_low from edge_index.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, variant=False, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self._adj_low = None
        self._adj_high = None
        self._initialized = False

        self.convs = nn.ModuleList()
        self.convs.append(ACMGraphConv(in_channels, hidden_channels, variant=variant))
        for _ in range(num_layers - 2):
            self.convs.append(ACMGraphConv(hidden_channels, hidden_channels, variant=variant))
        self.convs.append(ACMGraphConv(hidden_channels, out_channels, variant=variant))

    def _precompute_adj(self, edge_index, num_nodes, device):
        """Compute adj_low = row_norm(I+A) and adj_high = I - adj_low."""
        import scipy.sparse as sp

        adj_scipy = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
        adj_with_self = sp.eye(num_nodes) + adj_scipy

        # Row normalization: D^{-1}(I + A)
        rowsum = np.array(adj_with_self.sum(1)).flatten()
        d_inv = np.power(rowsum, -1.0)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)
        adj_low_scipy = d_mat.dot(adj_with_self)

        # Convert to torch sparse
        adj_low_scipy = adj_low_scipy.tocoo()
        indices = torch.tensor(
            np.vstack([adj_low_scipy.row, adj_low_scipy.col]),
            dtype=torch.long, device=device
        )
        values = torch.tensor(adj_low_scipy.data, dtype=torch.float32, device=device)
        self._adj_low = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

        # adj_high = I - adj_low
        idx = torch.arange(num_nodes, device=device)
        eye = torch.sparse_coo_tensor(
            torch.stack([idx, idx]),
            torch.ones(num_nodes, dtype=torch.float32, device=device),
            (num_nodes, num_nodes)
        )
        self._adj_high = (eye - self._adj_low).coalesce()
        self._initialized = True

    def forward(self, x, edge_index):
        if not self._initialized:
            self._precompute_adj(edge_index, x.size(0), x.device)

        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, self._adj_low, self._adj_high)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, self._adj_low, self._adj_high)
        return x


# ─── MLP baseline ─────────────────────────────────────

class MLP(nn.Module):
    """Simple MLP baseline (no graph structure)."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, **kwargs):
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
