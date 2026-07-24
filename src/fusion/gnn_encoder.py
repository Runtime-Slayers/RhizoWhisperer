"""
PyG 2.0 GNN Encoder for Root Topology Graphs
================================================

Encodes root network graphs into fixed-size graph embedding vectors using
Graph Attention Networks (GAT) or Graph Convolutional Networks (GCN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class RootGNNEncoder(nn.Module):
    """
    GNN Encoder for root topological graphs.

    Encodes graph structure (nodes=junctions/tips, edges=segments) into
    a dense root embedding vector for multi-modal fusion.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 64,
        out_channels: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        gnn_type: str = "GAT",
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout

        if not HAS_PYG:
            # Fallback MLP if PyG is not installed
            self.fallback_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ELU(inplace=True),
                nn.Linear(hidden_channels, out_channels),
            )
            return

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        prev_dim = in_channels
        for i in range(num_layers):
            is_last = i == num_layers - 1
            out_dim = out_channels if is_last else hidden_channels

            if gnn_type == "GAT":
                num_heads = 1 if is_last else heads
                conv = GATConv(prev_dim, out_dim // num_heads, heads=num_heads, concat=True)
            else:
                conv = GCNConv(prev_dim, out_dim)

            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(out_dim))
            prev_dim = out_dim

    def forward(self, x, edge_index, batch=None):
        if not HAS_PYG:
            # Aggregate node features via simple mean
            if batch is not None:
                num_graphs = batch.max().item() + 1
                pooled = torch.zeros(num_graphs, x.size(1), device=x.device)
                pooled.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
                counts = torch.zeros(num_graphs, 1, device=x.device).scatter_add_(
                    0, batch.unsqueeze(1), torch.ones_like(batch.unsqueeze(1), dtype=torch.float)
                )
                x = pooled / (counts + 1e-8)
            else:
                x = x.mean(dim=0, keepdim=True)
            return self.fallback_mlp(x)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        mean_pooled = global_mean_pool(x, batch)
        max_pooled = global_max_pool(x, batch)

        return mean_pooled + max_pooled
