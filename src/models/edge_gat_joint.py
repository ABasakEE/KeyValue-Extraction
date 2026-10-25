"""Dual-Head Edge-Augmented Graph Attention Network (Edge-GAT Joint Extractor).

Performs joint prediction over planar Delaunay visibility graphs:
1. Node Head: Predicts structural roles (KEY, VALUE, HEADER, OTHER) or typed categories.
2. Edge Head: Predicts directed relation links P(Key -> Value) conditioned on asymmetric 16-D spatial vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.edge_gat import EdgeGATLayer


class EdgeGATJointExtractor(nn.Module):
    """Joint Node Role & Edge Link Extractor using Relational Edge-GAT."""

    def __init__(
        self,
        in_dim: int = 776,       # 768-D LiLT + 8-D one-hot NER type (or projected)
        hidden_dim: int = 256,
        num_node_labels: int = 4, # 0: O, 1: KEY, 2: VALUE, 3: HEADER (or typed extension)
        num_layers: int = 3,
        edge_dim: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_node_labels = num_node_labels
        self.num_layers = num_layers
        self.edge_dim = edge_dim

        # Project input to hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        # Multi-layer relational Edge-GAT message passing
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

        # Node classification head: predicts role of each token/phrase
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_node_labels),
        )

        # Edge relation head: predicts directed semantic link P(u -> v)
        # Conditioned on [h_src || h_dst || e_attr_16d]
        edge_in_dim = (2 * hidden_dim) + edge_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Node embeddings (N, in_dim).
            edge_index: Delaunay graph edges (2, E).
            edge_attr: 16-D spatial relative vectors (E, 16).

        Returns:
            node_logits: (N, num_node_labels)
            edge_logits: (E,)
        """
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        node_logits = self.node_classifier(h)

        E = edge_index.size(1) if (edge_index is not None and edge_index.numel() > 0) else 0
        if E > 0:
            src, dst = edge_index[0], edge_index[1]
            if edge_attr is not None and edge_attr.size(0) == E and self.edge_dim > 0:
                edge_feat = torch.cat([h[src], h[dst], edge_attr], dim=-1)
            else:
                dummy_e = torch.zeros((E, self.edge_dim), device=x.device, dtype=x.dtype)
                edge_feat = torch.cat([h[src], h[dst], dummy_e], dim=-1)
            edge_logits = self.edge_classifier(edge_feat).squeeze(-1)
        else:
            edge_logits = torch.zeros((0,), device=x.device, dtype=x.dtype)

        return node_logits, edge_logits


def focal_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Focal Loss for binary edge classification to counter the 95:5 class imbalance."""
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_factor = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = alpha_factor * modulating_factor * bce
    return loss.mean()
