"""
Cost-Sensitive Neighborhood Aggregation (CSNA) for heterophilous graphs.

CSNA computes pairwise distance in a learned projection and uses it to
soft-route messages through two channels -- concordant (low cost, likely
same-class) and discordant (high cost, likely different-class) -- each with
an independent learned transformation. A per-node gating mechanism combines
the two channels with an ego term.

Two variants are provided:
  - CSNAConv / CSNA: default (lite) version using only observed divergence g_ij.
  - CSNAExtConv / CSNAExt: extended version adding a learned component h_ij,
    so the total edge cost is f_ij = g_ij + h_ij (analogous to A*'s f = g + h).

Reference:
    Weiss, E. "Cost-Sensitive Neighborhood Aggregation for Heterophilous Graphs:
    When Does Per-Edge Routing Help?" arXiv preprint, 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, softmax


# ---------------------------------------------------------------------------
# Default variant: observed divergence g_ij only
# ---------------------------------------------------------------------------

class CSNAConv(MessagePassing):
    """CSNA convolutional layer with dual-channel routing via observed divergence.

    For each edge (i, j), a concordance score s_ij = sigmoid(-g_ij / tau) is
    computed from the L2 distance g_ij in a learned projection space. Messages
    are routed through concordant (weighted by s_ij) and discordant (weighted
    by 1 - s_ij) channels, each with independent weight matrices.

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        tau: Temperature for concordance scoring (lower = sharper routing).
        dropout: Dropout probability on aggregation weights.
        sample_ratio: Fraction of edges to keep per forward pass (training only).
                      Acts as stochastic regularization. Default 1.0 (no sampling).
    """

    def __init__(self, in_channels, out_channels, tau=1.0, dropout=0.5,
                 sample_ratio=1.0):
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

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.xavier_uniform_(self.W_con.weight)
        nn.init.xavier_uniform_(self.W_dis.weight)
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.zeros_(self.W_self.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        # Bias toward ego channel so the model starts near MLP-like behavior
        self.gate.bias.data = torch.tensor([0.0, 0.0, 1.0])

    def _sample_edges(self, edge_index, num_nodes):
        """Stochastic edge sampling during training. Always keeps self-loops."""
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

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

        # Optional edge sampling
        if self.sample_ratio < 1.0 and self.training:
            edge_index = self._sample_edges(edge_index, x.size(0))

        # Compute observed divergence g_ij
        x_g = self.W_g(x)
        row, col = edge_index
        g_ij = torch.norm(x_g[row] - x_g[col], p=2, dim=1, keepdim=True)

        # Concordance score: sigmoid(-g/tau) in (0, 1)
        s_ij = torch.sigmoid(-g_ij / self.tau)

        # Concordant aggregation
        # NOTE: We normalize per source node (edge_index[0]), meaning each node
        # distributes its outgoing influence uniformly. This differs from the more
        # common per-destination normalization (edge_index[1]). We tested both and
        # found no consistent accuracy difference across datasets (within std).
        # To switch to per-destination normalization, change edge_index[0] to
        # edge_index[1] in the two softmax calls below.
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
    """CSNA model for node classification (default / lite variant).

    Uses only observed divergence g_ij for edge routing. Architecture:
        [Input MLP] -> [CSNAConv + BN + ReLU + Dropout] x (L-1) -> CSNAConv -> Linear

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer dimension.
        out_channels: Number of classes.
        num_layers: Number of CSNA layers.
        tau: Temperature for concordance scoring.
        dropout: Dropout probability.
        use_bn: Whether to use batch normalization.
        use_residual: Whether to use residual connections.
        input_mlp: Whether to add an input MLP for feature transformation.
        sample_ratio: Edge sampling ratio (1.0 = no sampling).
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 tau=1.0, dropout=0.5, use_bn=True, use_residual=True,
                 input_mlp=True, sample_ratio=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual

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

    def forward(self, x, edge_index):
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
            if self.use_residual and x.size() == x_in.size():
                x = x + x_in

        return self.classifier(x)


# ---------------------------------------------------------------------------
# Extended variant: g_ij + h_ij (analogous to A*'s f = g + h)
# ---------------------------------------------------------------------------

class CSNAExtConv(MessagePassing):
    """CSNA convolutional layer with both observed (g) and learned (h) costs.

    The total edge cost is f_ij = g_ij + h_ij, where:
      - g_ij = ||W_g x_i - W_g x_j||_2  (observed divergence)
      - h_ij = softplus(a^T [W_g x_i || W_g x_j])  (learned estimate)

    This mirrors the f = g + h decomposition in A* search.

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        tau: Temperature for concordance scoring.
        dropout: Dropout probability on aggregation weights.
    """

    def __init__(self, in_channels, out_channels, tau=1.0, dropout=0.5):
        super().__init__(aggr=None)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tau = tau
        self.dropout = dropout

        # Projection for observed cost g_ij
        self.W_g = nn.Linear(in_channels, out_channels, bias=False)

        # Learned heuristic cost h_ij
        self.W_h = nn.Linear(2 * out_channels, 1, bias=True)

        # Concordant channel
        self.W_con = nn.Linear(in_channels, out_channels, bias=False)

        # Discordant channel
        self.W_dis = nn.Linear(in_channels, out_channels, bias=False)

        # Ego transform
        self.W_self = nn.Linear(in_channels, out_channels, bias=True)

        # Per-node gating
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
        self.gate.bias.data = torch.tensor([0.0, 0.0, 1.0])
        nn.init.zeros_(self.W_h.bias)

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

        # Project for cost computation
        x_g = self.W_g(x)
        row, col = edge_index

        # g_ij: observed divergence
        g_ij = torch.norm(x_g[row] - x_g[col], p=2, dim=1, keepdim=True)

        # h_ij: learned heuristic cost
        h_input = torch.cat([x_g[row], x_g[col]], dim=1)
        h_ij = F.softplus(self.W_h(h_input))

        # Total cost: f_ij = g_ij + h_ij
        f_ij = g_ij + h_ij

        # Concordance score
        s_ij = torch.sigmoid(-f_ij / self.tau)

        # Store for regularization
        self._h_ij = h_ij
        self._edge_index = edge_index

        # Concordant aggregation (per-source normalization; see note in CSNAConv)
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


class CSNAExt(nn.Module):
    """CSNA model with extended cost function (g + h variant).

    Uses both observed divergence g_ij and learned heuristic h_ij for routing.
    Includes calibration and consistency regularization losses.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer dimension.
        out_channels: Number of classes.
        num_layers: Number of CSNA layers.
        tau: Temperature for concordance scoring.
        dropout: Dropout probability.
        use_bn: Whether to use batch normalization.
        use_residual: Whether to use residual connections.
        input_mlp: Whether to add an input MLP.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 tau=1.0, dropout=0.5, use_bn=True, use_residual=True, input_mlp=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual

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

        self.convs.append(CSNAExtConv(in_channels, hidden_channels, tau=tau, dropout=dropout))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(CSNAExtConv(hidden_channels, hidden_channels, tau=tau, dropout=dropout))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        if num_layers > 1:
            self.convs.append(CSNAExtConv(hidden_channels, hidden_channels, tau=tau, dropout=dropout))
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
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
            if self.use_residual and x.size() == x_in.size():
                x = x + x_in

        return self.classifier(x)

    def calibration_loss(self, y, train_mask=None):
        """Calibration regularization: penalizes h_ij > 1{y_i != y_j}.

        Encourages the learned heuristic to not overestimate true label divergence.

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

                mask = (row < y.size(0)) & (col < y.size(0))
                if train_mask is not None:
                    mask = mask & train_mask[row] & train_mask[col]

                if mask.any():
                    true_div = (y[row[mask]] != y[col[mask]]).float()
                    violation = F.relu(h_ij[mask] - true_div)
                    total_loss += violation.mean()
                    count += 1

        return total_loss / max(count, 1)

    def consistency_loss(self):
        """Consistency regularization: encourages smooth heuristic estimates."""
        total_loss = 0.0
        count = 0
        for conv in self.convs:
            if hasattr(conv, '_h_ij') and conv._h_ij is not None:
                h_ij = conv._h_ij.squeeze()
                total_loss += h_ij.var()
                count += 1

        return total_loss / max(count, 1)
