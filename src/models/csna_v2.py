"""
CSNA v2: Cost-Sensitive Neighborhood Aggregation.

The main model uses only observed divergence g_ij (no learned h_ij),
with stochastic edge sampling as default regularization.

Variants:
  - CSNA: g_ij routing + dual channels + edge sampling (main method)
  - CSNA-cached: same but with cost caching every k epochs (for scaling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class CSNAConv(MessagePassing):
    """CSNA convolutional layer: dual-channel routing via observed divergence g_ij.

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        tau: Temperature for concordance scoring (lower = sharper routing).
        dropout: Dropout probability on aggregation weights.
        sample_ratio: Fraction of edges to keep per forward pass (training only).
                      Acts as stochastic regularization. Default 0.5.
    """

    def __init__(self, in_channels, out_channels, tau=1.0, dropout=0.5,
                 sample_ratio=0.5):
        super().__init__(aggr=None)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tau = tau
        self.dropout = dropout
        self.sample_ratio = sample_ratio

        # Projection for cost: representation divergence in projected space
        self.W_g = nn.Linear(in_channels, out_channels, bias=False)

        # Concordant channel (low-cost / likely same-class neighbors)
        self.W_con = nn.Linear(in_channels, out_channels, bias=False)

        # Discordant channel (high-cost / likely different-class neighbors)
        self.W_dis = nn.Linear(in_channels, out_channels, bias=False)

        # Ego transform
        self.W_self = nn.Linear(in_channels, out_channels, bias=True)

        # Gating: per-node channel combination
        self.gate = nn.Linear(3 * out_channels, 3, bias=True)

        # Cache for cost caching variant
        self._cached_s_ij = None
        self._cached_edge_index = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.xavier_uniform_(self.W_con.weight)
        nn.init.xavier_uniform_(self.W_dis.weight)
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.zeros_(self.W_self.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        self.gate.bias.data = torch.tensor([0.0, 0.0, 1.0])

    def invalidate_cache(self):
        self._cached_s_ij = None
        self._cached_edge_index = None

    def _sample_edges(self, edge_index, num_nodes):
        """Sample edges during training. Always keep self-loops."""
        if self.sample_ratio >= 1.0 or not self.training:
            return edge_index

        is_self_loop = (edge_index[0] == edge_index[1])
        non_self_idx = (~is_self_loop).nonzero(as_tuple=True)[0]
        self_loop_idx = is_self_loop.nonzero(as_tuple=True)[0]

        num_keep = max(1, int(non_self_idx.size(0) * self.sample_ratio))
        perm = torch.randperm(non_self_idx.size(0), device=edge_index.device)[:num_keep]
        sampled = non_self_idx[perm]

        keep = torch.cat([self_loop_idx, sampled]).sort()[0]
        return edge_index[:, keep]

    def forward(self, x, edge_index, use_cache=False):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if use_cache and self._cached_s_ij is not None and self._cached_edge_index is not None:
            # Reuse cached costs AND the edge set they were computed on
            s_ij = self._cached_s_ij
            edge_index = self._cached_edge_index
        else:
            # When recomputing: apply edge sampling (if enabled) to get a new subset,
            # then compute costs on that subset and cache both.
            # This means sampling + caching work together: a new random edge sample
            # is drawn every k epochs, and reused for k-1 epochs in between.
            if self.sample_ratio < 1.0 and self.training:
                edge_index = self._sample_edges(edge_index, x.size(0))

            # Compute observed divergence
            x_g = self.W_g(x)
            row, col = edge_index
            g_ij = torch.norm(x_g[row] - x_g[col], p=2, dim=1, keepdim=True)

            # Concordance score
            s_ij = torch.sigmoid(-g_ij / self.tau)

            # Store g_ij for calibration regularization
            self._g_ij = g_ij
            self._edge_index = edge_index

            # Cache scores + edge set for reuse on subsequent epochs
            self._cached_s_ij = s_ij.detach()
            self._cached_edge_index = edge_index

        # Concordant aggregation
        x_con = self.W_con(x)
        con_w = softmax(s_ij.squeeze(), edge_index[0], num_nodes=x.size(0))
        con_w = F.dropout(con_w, p=self.dropout, training=self.training)
        out_con = self.propagate(edge_index, x=x_con, weights=con_w)

        # Discordant aggregation
        x_dis = self.W_dis(x)
        dis_w = softmax((1 - s_ij).squeeze(), edge_index[0], num_nodes=x.size(0))
        dis_w = F.dropout(dis_w, p=self.dropout, training=self.training)
        out_dis = self.propagate(edge_index, x=x_dis, weights=dis_w)

        # Ego
        out_self = self.W_self(x)

        # Gated combination
        combined = torch.cat([out_con, out_dis, out_self], dim=1)
        gate_w = F.softmax(self.gate(combined), dim=1)

        return (gate_w[:, 0:1] * out_con +
                gate_w[:, 1:2] * out_dis +
                gate_w[:, 2:3] * out_self)

    def message(self, x_j, weights):
        return weights.unsqueeze(-1) * x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch.zeros(dim_size, inputs.size(1), device=inputs.device).scatter_add_(
            0, index.unsqueeze(-1).expand_as(inputs), inputs)


class CSNA(nn.Module):
    """CSNA model for node classification.

    Cost-Sensitive Neighborhood Aggregation with:
    - Observed divergence g_ij for edge routing (no learned h_ij)
    - Dual concordant/discordant channels
    - Per-node gating
    - Stochastic edge sampling (default 50%)
    - Optional cost caching for scaling

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer dimension.
        out_channels: Number of classes.
        num_layers: Number of CSNA layers.
        tau: Temperature for concordance scoring.
        dropout: Dropout probability.
        sample_ratio: Edge sampling ratio (default 0.5).
        cache_k: Recompute costs every k epochs. 1 = no caching.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 tau=1.0, dropout=0.5, use_bn=True, use_residual=True,
                 input_mlp=True, sample_ratio=0.5, cache_k=1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.cache_k = cache_k
        self._epoch_counter = 0

        # Input MLP
        self.input_mlp = None
        if input_mlp:
            self.input_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            in_channels = hidden_channels

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(CSNAConv(in_channels, hidden_channels, tau=tau,
                                    dropout=dropout, sample_ratio=sample_ratio))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(CSNAConv(hidden_channels, hidden_channels, tau=tau,
                                        dropout=dropout, sample_ratio=sample_ratio))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        if num_layers > 1:
            self.convs.append(CSNAConv(hidden_channels, hidden_channels, tau=tau,
                                        dropout=dropout, sample_ratio=sample_ratio))
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def step_epoch(self):
        """Call at start of each epoch. Manages cost cache invalidation."""
        self._epoch_counter += 1
        if self.cache_k > 1 and self._epoch_counter % self.cache_k == 0:
            for conv in self.convs:
                conv.invalidate_cache()

    def reset_epoch_counter(self):
        self._epoch_counter = 0
        for conv in self.convs:
            conv.invalidate_cache()

    def forward(self, x, edge_index):
        use_cache = self.cache_k > 1

        if self.input_mlp is not None:
            x = self.input_mlp(x)

        for i in range(self.num_layers):
            x_in = x
            x = self.convs[i](x, edge_index, use_cache=use_cache)
            if i < self.num_layers - 1:
                if self.use_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual and x.size() == x_in.size():
                x = x + x_in

        return self.classifier(x)

    def calibration_loss(self, y, train_mask=None):
        """Calibration regularization per Eq (8): penalize g_ij > 1[y_i != y_j].

        Encourages lower costs (higher concordance) for same-class edges.
        """
        total_loss = 0.0
        count = 0
        for conv in self.convs:
            if hasattr(conv, '_g_ij') and conv._g_ij is not None:
                edge_index = conv._edge_index
                g_ij = conv._g_ij.squeeze()
                row, col = edge_index

                mask = (row < y.size(0)) & (col < y.size(0))
                if train_mask is not None:
                    mask = mask & train_mask[row] & train_mask[col]

                if mask.any():
                    true_div = (y[row[mask]] != y[col[mask]]).float()
                    violation = F.relu(g_ij[mask] - true_div)
                    total_loss += (violation ** 2).mean()
                    count += 1

        return total_loss / max(count, 1)
