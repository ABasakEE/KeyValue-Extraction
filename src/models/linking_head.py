"""Biaffine Key-Value Relation Extraction (Linking) Head.

In visually rich document understanding (VRDU / FUNSD / RFUND), key-value extraction
consists of two complementary tasks:
1. Entity Tagging (Token Classification): Identifying field types (QUESTION/KEY vs ANSWER/VALUE vs HEADER).
2. Key-Value Linking (Relation Extraction): Identifying directed semantic relations (k -> v)
   connecting each question/key entity to its corresponding answer/value entity.

This module implements a Spatial Biaffine Relation Classifier that predicts the adjacency
matrix over entity spans. Given entity contextual representations and bounding boxes, it
computes pairwise link scores by fusing:
- Deep semantic features via bilinear and linear projections of head (key) and tail (value) spans.
- Explicit 2D geometric spatial features (horizontal/vertical displacement, Euclidean centroid
  distance, relative scale, and orientation angle).

The predicted link sets directly interface with ``evaluate.linking_f1``, enabling strict
exact-match evaluation under zero-same-template holdouts.

References:
- Teakgyu Hong et al. BROS: A Pre-trained Language Model Focusing on Text and Layout for
  Better Key Information Extraction from Documents, AAAI 2022.
- Chuwei Luo et al. GeoLayoutLM: Geometric Pre-training for Visual Information Extraction, CVPR 2023.
- Timothy Dozat and Christopher D. Manning. Deep Biaffine Attention for Neural Dependency Parsing,
  ICLR 2017.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_spatial_features(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    norm_scale: float = 1000.0,
) -> torch.Tensor:
    """Compute pairwise geometric spatial edge features between two sets of bounding boxes.

    Args:
        boxes_a: Tensor of shape (M, 4) with coordinates [x0, y0, x1, y1] in range [0, norm_scale].
        boxes_b: Tensor of shape (N, 4) with coordinates [x0, y0, x1, y1] in range [0, norm_scale].
        norm_scale: Coordinate normalization denominator (default: 1000.0).

    Returns:
        Tensor of shape (M, N, 10) containing normalized spatial features:
        - [0] Normalized horizontal gap: (x0_b - x1_a) / scale
        - [1] Normalized vertical gap:   (y0_b - y1_a) / scale
        - [2] Centroid x-delta:          (cx_b - cx_a) / scale
        - [3] Centroid y-delta:          (cy_b - cy_a) / scale
        - [4] Euclidean centroid dist:   sqrt(dx^2 + dy^2)
        - [5] Direction angle cosine:    cos(theta)
        - [6] Direction angle sine:      sin(theta)
        - [7] Log width ratio:           log((w_b + 1) / (w_a + 1))
        - [8] Log height ratio:          log((h_b + 1) / (h_a + 1))
        - [9] Log area ratio:            log((area_b + 1) / (area_a + 1))
    """
    # boxes_a: (M, 1, 4), boxes_b: (1, N, 4)
    a = boxes_a.unsqueeze(1).float()
    b = boxes_b.unsqueeze(0).float()

    x0_a, y0_a, x1_a, y1_a = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x0_b, y0_b, x1_b, y1_b = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w_a = torch.clamp(x1_a - x0_a, min=1.0)
    h_a = torch.clamp(y1_a - y0_a, min=1.0)
    w_b = torch.clamp(x1_b - x0_b, min=1.0)
    h_b = torch.clamp(y1_b - y0_b, min=1.0)

    area_a = w_a * h_a
    area_b = w_b * h_b

    cx_a = (x0_a + x1_a) / 2.0
    cy_a = (y0_a + y1_a) / 2.0
    cx_b = (x0_b + x1_b) / 2.0
    cy_b = (y0_b + y1_b) / 2.0

    dx_gap = (x0_b - x1_a) / norm_scale
    dy_gap = (y0_b - y1_a) / norm_scale

    dx_c = (cx_b - cx_a) / norm_scale
    dy_c = (cy_b - cy_a) / norm_scale

    dist_c = torch.sqrt(dx_c**2 + dy_c**2 + 1e-8)
    theta = torch.atan2(dy_c, dx_c)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    log_w_ratio = torch.log(w_b / w_a)
    log_h_ratio = torch.log(h_b / h_a)
    log_area_ratio = torch.log(area_b / area_a)

    features = torch.stack(
        [
            dx_gap,
            dy_gap,
            dx_c,
            dy_c,
            dist_c,
            cos_theta,
            sin_theta,
            log_w_ratio,
            log_h_ratio,
            log_area_ratio,
        ],
        dim=-1,
    )
    return features


class SpatialEdgeEncoder(nn.Module):
    """Encodes pairwise geometric features into a dense representation vector."""

    def __init__(self, in_features: int = 10, geo_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, geo_dim),
            nn.LayerNorm(geo_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(geo_dim, geo_dim),
            nn.LayerNorm(geo_dim),
        )

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """Args: spatial_features of shape (..., in_features). Returns: (..., geo_dim)."""
        return self.net(spatial_features)


class BiaffineRelationClassifier(nn.Module):
    """Biaffine relation extraction head with spatial geometric inductive bias.

    Computes pairwise scores S(i -> j) indicating the confidence of a directed key-value
    link from entity span i to entity span j:
        S(i -> j) = u_i^T W v_j + U^T [u_i; v_j; g_{ij}] + b
    """

    def __init__(
        self,
        hidden_dim: int,
        proj_dim: int = 128,
        geo_dim: int = 64,
        dropout: float = 0.1,
        use_spatial: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.geo_dim = geo_dim if use_spatial else 0
        self.use_spatial = use_spatial

        # Head (Key / Subject) and Tail (Value / Object) projections
        self.head_mlp = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tail_mlp = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.use_spatial:
            self.spatial_encoder = SpatialEdgeEncoder(in_features=10, geo_dim=geo_dim, dropout=dropout)

        # Bilinear weight matrix: (proj_dim, proj_dim)
        self.bilinear = nn.Bilinear(proj_dim, proj_dim, 1, bias=False)

        # Linear projection combining head, tail, and spatial features
        linear_in = 2 * proj_dim + (geo_dim if use_spatial else 0)
        self.linear = nn.Linear(linear_in, 1, bias=True)

    def forward(
        self,
        entity_reps: torch.Tensor,
        entity_boxes: torch.Tensor,
        type_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute relation logits for all entity pairs.

        Args:
            entity_reps: Tensor of shape (M, hidden_dim) representing M entity embeddings.
            entity_boxes: Tensor of shape (M, 4) with bounding box coordinates [x0, y0, x1, y1].
            type_mask: Optional boolean tensor of shape (M, M) where True allows a link
                       and False forbids it (e.g. self-loops or incompatible entity types).

        Returns:
            Tensor of shape (M, M) with unnormalized relation logits.
        """
        num_entities = entity_reps.size(0)
        if num_entities == 0:
            return torch.zeros((0, 0), device=entity_reps.device, dtype=entity_reps.dtype)

        # Project to head and tail representations: (M, proj_dim)
        heads = self.head_mlp(entity_reps)
        tails = self.tail_mlp(entity_reps)

        # Pairwise expansion:
        # heads_exp: (M, M, proj_dim), tails_exp: (M, M, proj_dim)
        heads_exp = heads.unsqueeze(1).expand(-1, num_entities, -1)
        tails_exp = tails.unsqueeze(0).expand(num_entities, -1, -1)

        # 1. Bilinear score: (M, M)
        # Reshape to (M*M, proj_dim) for nn.Bilinear
        bilinear_score = self.bilinear(
            heads_exp.reshape(-1, self.proj_dim),
            tails_exp.reshape(-1, self.proj_dim),
        ).view(num_entities, num_entities)

        # 2. Linear combination of features
        if self.use_spatial:
            spatial_feats = compute_spatial_features(entity_boxes, entity_boxes)  # (M, M, 10)
            spatial_emb = self.spatial_encoder(spatial_feats)                     # (M, M, geo_dim)
            combined_feats = torch.cat([heads_exp, tails_exp, spatial_emb], dim=-1)
        else:
            combined_feats = torch.cat([heads_exp, tails_exp], dim=-1)

        linear_score = self.linear(combined_feats).squeeze(-1)  # (M, M)

        logits = bilinear_score + linear_score

        # Disallow self-loops (an entity cannot be linked to itself)
        eye_mask = torch.eye(num_entities, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(eye_mask, -1e9)

        # Apply domain/type compatibility mask if provided
        if type_mask is not None:
            logits = logits.masked_fill(~type_mask, -1e9)

        return logits

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        gold_links: set[tuple[int, int]] | list[tuple[int, int]],
        pos_weight: float = 10.0,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Binary Cross-Entropy loss with positive class weighting.

        Link matrices in forms are heavily sparse (< 2% positive links). A positive weight
        prevents the model from collapsing to predicting no links.

        Args:
            logits: Predicted logits of shape (M, M).
            gold_links: Ground truth directed links as pairs (src_idx, dst_idx).
            pos_weight: Loss weight multiplier for positive links.
            valid_mask: Optional boolean mask of shape (M, M) indicating pairs to include in loss.
        """
        num_entities = logits.size(0)
        if num_entities == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        target = torch.zeros_like(logits)
        for s, d in gold_links:
            if 0 <= s < num_entities and 0 <= d < num_entities and s != d:
                target[s, d] = 1.0

        weight_tensor = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor, reduction="none")
        loss_matrix = criterion(logits, target)

        # Exclude self-loops
        eye_mask = torch.eye(num_entities, dtype=torch.bool, device=logits.device)
        loss_mask = ~eye_mask
        if valid_mask is not None:
            loss_mask = loss_mask & valid_mask

        denom = loss_mask.sum().float()
        if denom == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return (loss_matrix * loss_mask.float()).sum() / denom

    @staticmethod
    def decode_predictions(
        logits: torch.Tensor,
        threshold: float = 0.5,
        one_to_one: bool = False,
    ) -> set[tuple[int, int]]:
        """Decode predicted logits into a set of directed (head, tail) link pairs.

        Args:
            logits: (M, M) tensor of link logits.
            threshold: Probability decision threshold (default: 0.5).
            one_to_one: If True, each tail (value) entity can be linked to at most one
                        highest-scoring head (key).

        Returns:
            Set of (head_idx, tail_idx) tuples.
        """
        probs = torch.sigmoid(logits)
        num_entities = logits.size(0)
        predicted: set[tuple[int, int]] = set()

        if one_to_one:
            # For each column (value j), find the best row (key i)
            for j in range(num_entities):
                best_val, best_i = probs[:, j].max(dim=0)
                if best_val.item() >= threshold and best_i.item() != j:
                    predicted.add((best_i.item(), j))
        else:
            indices = (probs >= threshold).nonzero(as_tuple=False)
            for pair in indices:
                i, j = pair[0].item(), pair[1].item()
                if i != j:
                    predicted.add((i, j))

        return predicted


def pool_entity_representations(
    token_hidden: torch.Tensor,
    token_boxes: torch.Tensor,
    entity_token_indices: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool word/subword token vectors into entity-level span representations.

    Args:
        token_hidden: Contextual token representations of shape (L, hidden_dim).
        token_boxes: Token bounding boxes of shape (L, 4).
        entity_token_indices: List of token index lists, one per entity span.

    Returns:
        entity_reps: (M, hidden_dim) pooled entity embeddings.
        entity_boxes: (M, 4) bounding box per entity: [min x0, min y0, max x1, max y1].
    """
    device = token_hidden.device
    num_entities = len(entity_token_indices)
    hidden_dim = token_hidden.size(-1)

    if num_entities == 0:
        return (
            torch.zeros((0, hidden_dim), device=device, dtype=token_hidden.dtype),
            torch.zeros((0, 4), device=device, dtype=token_boxes.dtype),
        )

    entity_reps = []
    entity_boxes = []

    for indices in entity_token_indices:
        valid_indices = [idx for idx in indices if 0 <= idx < token_hidden.size(0)]
        if not valid_indices:
            # Fallback for degenerate empty entity
            entity_reps.append(torch.zeros(hidden_dim, device=device, dtype=token_hidden.dtype))
            entity_boxes.append(torch.tensor([0, 0, 0, 0], device=device, dtype=token_boxes.dtype))
            continue

        idx_tensor = torch.tensor(valid_indices, device=device, dtype=torch.long)
        span_tokens = token_hidden.index_select(0, idx_tensor)
        span_rep = span_tokens.mean(dim=0)
        entity_reps.append(span_rep)

        span_boxes = token_boxes.index_select(0, idx_tensor)
        x0 = span_boxes[:, 0].min()
        y0 = span_boxes[:, 1].min()
        x1 = span_boxes[:, 2].max()
        y1 = span_boxes[:, 3].max()
        entity_boxes.append(torch.stack([x0, y0, x1, y1]))

    return torch.stack(entity_reps, dim=0), torch.stack(entity_boxes, dim=0)


def build_type_compatibility_mask(
    entity_types: list[str],
    allowed_relations: tuple[tuple[str, str], ...] = (
        ("question", "answer"),
        ("header", "question"),
        ("header", "answer"),
    ),
) -> torch.Tensor:
    """Build a boolean mask enforcing domain entity typing rules on relation pairs.

    Forms have rigid semantic typing constraints:
    - A question links to an answer.
    - A header can link to a question or answer.
    - An answer NEVER originates a link to a question.

    Args:
        entity_types: List of entity types (e.g. ['question', 'answer', 'header']).
        allowed_relations: Permitted (head_type, tail_type) tuples.

    Returns:
        Boolean tensor of shape (M, M) where True allows relation evaluation.
    """
    num_entities = len(entity_types)
    mask = torch.zeros((num_entities, num_entities), dtype=torch.bool)
    allowed_set = {
        (head.lower(), tail.lower()) for head, tail in allowed_relations
    }

    for i, t_i in enumerate(entity_types):
        for j, t_j in enumerate(entity_types):
            if i != j and (t_i.lower(), t_j.lower()) in allowed_set:
                mask[i, j] = True

    return mask
