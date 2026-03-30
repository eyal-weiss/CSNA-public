"""
Cost-Sensitive Neighborhood Aggregation (CSNA) for Heterophily-Aware Graph Learning.

The key insight: in heuristic search, information propagation across a graph is governed by
cost semantics — edges have propagation costs, and admissibility/consistency constraints
ensure reliable information flow. We import this discipline into GNN aggregation.

Instead of learning attention (correlation-based, like GAT), we learn a *propagation cost*
for each edge, decomposed into:
  - g_ij: observed representation divergence (analogous to known path cost)
  - h_ij: learned reliability estimate (analogous to heuristic estimate)

Edges are then soft-routed into concordant (low-cost, homophilous) and discordant
(high-cost, heterophilous) channels, each with its own transformation. This dual-channel
design means heterophilous neighbors are not suppressed — they are *processed differently*.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax


class CSNAConv(MessagePassing):
    """Cost-Sensitive Neighborhood Aggregation convolutional layer.

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        tau: Temperature for concordance scoring (lower = sharper routing).
        dropout: Dropout probability on aggregation weights.
    """

    def __init__(self, in_channels, out_channels, tau=1.0, dropout=0.5):
        super().__init__(aggr=None)  # custom aggregation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tau = tau
        self.dropout = dropout

        # Projection for observed cost (g): representation divergence
        self.W_g = nn.Linear(in_channels, out_channels, bias=False)

        # Learned heuristic cost (h): edge reliability estimator
        self.W_h = nn.Linear(2 * out_channels, 1, bias=True)

        # Concordant channel (low-cost / homophilous neighbors)
        self.W_con = nn.Linear(in_channels, out_channels, bias=False)

        # Discordant channel (high-cost / heterophilous neighbors)
        self.W_dis = nn.Linear(in_channels, out_channels, bias=False)

        # Self-loop (ego) transform
        self.W_self = nn.Linear(in_channels, out_channels, bias=True)

        # Learnable combination weights (initialized to favor self-loop)
        self.gate = nn.Linear(3 * out_channels, 3, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.xavier_uniform_(self.W_con.weight)
        nn.init.xavier_uniform_(self.W_dis.weight)
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.zeros_(self.W_self.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        # Initialize gate bias to favor self-loop channel (index 2)
        # This helps on heterophilous graphs where graph structure can hurt
        self.gate.bias.data = torch.tensor([0.0, 0.0, 1.0])
        nn.init.zeros_(self.W_h.bias)

    def forward(self, x, edge_index):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Project for cost computation
        x_g = self.W_g(x)  # [N, out_channels]

        # Compute edge costs
        row, col = edge_index
        # g_ij: observed divergence (L2 distance in projected space)
        g_ij = torch.norm(x_g[row] - x_g[col], p=2, dim=1, keepdim=True)  # [E, 1]

        # h_ij: learned heuristic cost
        h_input = torch.cat([x_g[row], x_g[col]], dim=1)  # [E, 2*out]
        h_ij = F.softplus(self.W_h(h_input))  # [E, 1], non-negative

        # Total propagation cost: f_ij = g_ij + h_ij
        f_ij = g_ij + h_ij  # [E, 1]

        # Concordance score: sigmoid(-f/tau) — low cost = high concordance
        s_ij = torch.sigmoid(-f_ij / self.tau)  # [E, 1] in (0, 1)

        # Store costs for regularization
        self._h_ij = h_ij
        self._edge_index = edge_index

        # Concordant aggregation: weighted by s_ij
        x_con = self.W_con(x)  # [N, out]
        con_weights = softmax(s_ij.squeeze(), edge_index[0], num_nodes=x.size(0))
        con_weights = F.dropout(con_weights, p=self.dropout, training=self.training)
        out_con = self.propagate(edge_index, x=x_con, weights=con_weights)

        # Discordant aggregation: weighted by (1 - s_ij)
        x_dis = self.W_dis(x)  # [N, out]
        dis_weights = softmax((1 - s_ij).squeeze(), edge_index[0], num_nodes=x.size(0))
        dis_weights = F.dropout(dis_weights, p=self.dropout, training=self.training)
        out_dis = self.propagate(edge_index, x=x_dis, weights=dis_weights)

        # Self-loop (ego)
        out_self = self.W_self(x)  # [N, out]

        # Gated combination
        combined = torch.cat([out_con, out_dis, out_self], dim=1)  # [N, 3*out]
        gate_logits = self.gate(combined)  # [N, 3]
        gate_weights = F.softmax(gate_logits, dim=1)  # [N, 3]

        out = (gate_weights[:, 0:1] * out_con +
               gate_weights[:, 1:2] * out_dis +
               gate_weights[:, 2:3] * out_self)

        return out

    def message(self, x_j, weights):
        return weights.unsqueeze(-1) * x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch.zeros(dim_size, inputs.size(1), device=inputs.device).scatter_add_(
            0, index.unsqueeze(-1).expand_as(inputs), inputs
        )


class CSNA(nn.Module):
    """Full CSNA model for node classification.

    Architecture:
        Optional input MLP → [CSNAConv + BN + ReLU + Dropout] × (L-1) → CSNAConv

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer dimension.
        out_channels: Number of output classes.
        num_layers: Number of CSNA layers.
        tau: Temperature for concordance scoring.
        dropout: Dropout probability.
        use_bn: Whether to use batch normalization.
        use_residual: Whether to use residual connections.
        input_mlp: Whether to add an input MLP for feature transformation.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 tau=1.0, dropout=0.5, use_bn=True, use_residual=True, input_mlp=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual

        # Optional input MLP (helps on small heterophily datasets with high-dim features)
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

        # First layer
        self.convs.append(CSNAConv(in_channels, hidden_channels, tau=tau, dropout=dropout))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(CSNAConv(hidden_channels, hidden_channels, tau=tau, dropout=dropout))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Last CSNA layer outputs hidden_channels, then a linear classifier
        if num_layers > 1:
            self.convs.append(CSNAConv(hidden_channels, hidden_channels, tau=tau, dropout=dropout))
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Optional input MLP
        if self.input_mlp is not None:
            x = self.input_mlp(x)

        for i in range(self.num_layers):
            x_in = x
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                if self.use_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # Residual connection (if dims match)
            if self.use_residual and x.size() == x_in.size():
                x = x + x_in

        x = self.classifier(x)
        return x

    def admissibility_loss(self, y, train_mask=None):
        """
        Admissibility regularization: h_ij should not overestimate true label divergence.

        In search: h(n) <= h*(n) ensures optimality.
        Here: h_ij <= 1_{y_i != y_j} encourages calibrated cost estimates.

        Args:
            y: Node labels.
            train_mask: If provided, only use edges between training nodes.
        """
        total_loss = 0.0
        count = 0
        for conv in self.convs:
            if hasattr(conv, '_h_ij') and conv._h_ij is not None:
                edge_index = conv._edge_index
                h_ij = conv._h_ij.squeeze()
                row, col = edge_index

                # Only use edges where both endpoints have valid labels
                mask = (row < y.size(0)) & (col < y.size(0))

                # If train_mask provided, further restrict to training nodes
                if train_mask is not None:
                    mask = mask & train_mask[row] & train_mask[col]

                if mask.any():
                    true_div = (y[row[mask]] != y[col[mask]]).float()
                    # Admissibility: penalize h_ij > true_div
                    violation = F.relu(h_ij[mask] - true_div)
                    total_loss += violation.mean()
                    count += 1

        return total_loss / max(count, 1)

    def consistency_loss(self):
        """
        Consistency regularization: encourage smooth heuristic cost estimates.

        In search: h(n) <= c(n,n') + h(n') for consistency.
        We approximate this with a variance penalty on h_ij values.
        """
        total_loss = 0.0
        count = 0
        for conv in self.convs:
            if hasattr(conv, '_h_ij') and conv._h_ij is not None:
                h_ij = conv._h_ij.squeeze()
                total_loss += h_ij.var()
                count += 1

        return total_loss / max(count, 1)
