"""Edge-Augmented Graph Attention Network (Edge-GAT) for Document Token Classification.

Attends over planar Delaunay visibility graphs, conditioning attention distributions and
message passing on 16-D explicit spatial relation vectors (Geometry, Containment, LoS, Typography).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGATLayer(nn.Module):
    """Multi-Head Graph Attention Layer with Edge Feature Conditioning."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha_leaky: float = 0.2,
    ):
        super().__init__()
        assert out_dim % num_heads == 0, f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        # Linear projections for source nodes, target nodes, and edge features
        self.lin_src = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim > 0 else None

        # Value projections
        self.lin_val_node = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_val_edge = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim > 0 else None

        # Attention score vector
        self.attn_vec = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(alpha_leaky)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

        # Residual projection if dimensions differ
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape (N, in_dim).
            edge_index: Graph connectivity of shape (2, E).
            edge_attr: Edge features of shape (E, edge_dim).

        Returns:
            Updated node features of shape (N, out_dim).
        """
        N = x.size(0)
        E = edge_index.size(1)
        if E == 0:
            return self.layer_norm(self.residual_proj(x))

        src, dst = edge_index[0], edge_index[1]

        # Project nodes: (N, H, D_head)
        h_src = self.lin_src(x).view(N, self.num_heads, self.head_dim)
        h_dst = self.lin_dst(x).view(N, self.num_heads, self.head_dim)

        # Edge features projection: (E, H, D_head)
        if self.lin_edge is not None and edge_attr is not None and edge_attr.size(0) == E:
            e_proj = self.lin_edge(edge_attr).view(E, self.num_heads, self.head_dim)
        else:
            e_proj = 0.0

        # Compute unnormalized attention logits: (E, H, D_head)
        # alpha_raw = LeakyReLU(h_src[src] + h_dst[dst] + e_proj)
        edge_logits = self.leaky_relu(h_src[src] + h_dst[dst] + e_proj)
        # Dot product with attention vector: (E, H)
        alpha = (edge_logits * self.attn_vec.unsqueeze(0)).sum(dim=-1)

        # Softmax over incoming edges per target node
        # Using numerical-stability max-subtraction scatter
        alpha_max = torch.zeros((N, self.num_heads), device=x.device, dtype=alpha.dtype).fill_(-1e9)
        alpha_max = alpha_max.scatter_reduce(0, dst.unsqueeze(1).expand(-1, self.num_heads), alpha, reduce="amax", include_self=False)
        alpha_exp = torch.exp(alpha - alpha_max[dst])

        alpha_denom = torch.zeros((N, self.num_heads), device=x.device, dtype=alpha.dtype)
        alpha_denom = alpha_denom.scatter_add(0, dst.unsqueeze(1).expand(-1, self.num_heads), alpha_exp) + 1e-12
        attn_weights = self.dropout(alpha_exp / alpha_denom[dst])  # (E, H)

        # Compute messages from neighbor nodes and edges
        v_node = self.lin_val_node(x).view(N, self.num_heads, self.head_dim)
        if self.lin_val_edge is not None and edge_attr is not None and edge_attr.size(0) == E:
            v_edge = self.lin_val_edge(edge_attr).view(E, self.num_heads, self.head_dim)
        else:
            v_edge = 0.0

        messages = (v_node[src] + v_edge) * attn_weights.unsqueeze(-1)  # (E, H, D_head)

        # Aggregate messages at destination nodes: (N, H, D_head)
        out = torch.zeros((N, self.num_heads, self.head_dim), device=x.device, dtype=x.dtype)
        out = out.scatter_add(0, dst.view(E, 1, 1).expand(-1, self.num_heads, self.head_dim), messages)
        out = out.view(N, self.out_dim)

        # Residual connection + LayerNorm
        out = self.layer_norm(out + self.residual_proj(x))
        return F.gelu(out)


class EdgeGATTokenClassifier(nn.Module):
    """Document Token Classifier using Relational Edge-GAT Layers."""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        num_labels: int = 7,
        num_layers: int = 2,
        edge_dim: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        self.layers = nn.ModuleList([
            EdgeGATLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                edge_dim=edge_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict token class logits.

        Args:
            x: Node feature embeddings (N, in_dim).
            edge_index: Graph connectivity (2, E).
            edge_attr: 16-D structural edge attributes (E, 16).

        Returns:
            Logits of shape (N, num_labels).
        """
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.classifier(h)
