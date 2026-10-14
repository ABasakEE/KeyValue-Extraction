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


def compute_line_of_sight(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Line-of-Sight (LOS) visibility ray-casting between entity bounding boxes.

    Simulates horizontal and vertical reading corridors. A ray between box i and box j
    is unobstructed if no intermediate bounding box k blocks the visual corridor.

    Args:
        boxes: Tensor of shape (M, 4) with coordinates [x0, y0, x1, y1].

    Returns:
        los_h: Tensor of shape (M, M), 1.0 if horizontal path is unobstructed, 0.0 otherwise.
        los_v: Tensor of shape (M, M), 1.0 if vertical path is unobstructed, 0.0 otherwise.
    """
    M = boxes.size(0)
    if M <= 1:
        ones = torch.ones((M, M), device=boxes.device, dtype=boxes.dtype)
        return ones, ones

    x0 = boxes[:, 0].float()
    y0 = boxes[:, 1].float()
    x1 = boxes[:, 2].float()
    y1 = boxes[:, 3].float()

    x0_i, x0_j = x0.unsqueeze(1), x0.unsqueeze(0)
    y0_i, y0_j = y0.unsqueeze(1), y0.unsqueeze(0)
    x1_i, x1_j = x1.unsqueeze(1), x1.unsqueeze(0)
    y1_i, y1_j = y1.unsqueeze(1), y1.unsqueeze(0)

    # 1. Horizontal Corridor: Overlap in Y, gap in X
    y_corr_low = torch.maximum(y0_i, y0_j)
    y_corr_high = torch.minimum(y1_i, y1_j)
    has_h_overlap = (y_corr_high > y_corr_low)

    x_gap_left = torch.minimum(x1_i, x1_j)
    x_gap_right = torch.maximum(x0_i, x0_j)
    has_h_gap = (x_gap_right > x_gap_left)

    eye = torch.eye(M, dtype=torch.bool, device=boxes.device)
    mask_k = ~eye.unsqueeze(1) & ~eye.unsqueeze(0)  # shape (M, M, M), k != i and k != j

    x0_k = x0.view(1, 1, M)
    x1_k = x1.view(1, 1, M)
    y0_k = y0.view(1, 1, M)
    y1_k = y1.view(1, 1, M)

    k_in_x_gap = (torch.minimum(x1_k, x_gap_right.unsqueeze(2)) > torch.maximum(x0_k, x_gap_left.unsqueeze(2)))
    k_in_y_corr = (torch.minimum(y1_k, y_corr_high.unsqueeze(2)) > torch.maximum(y0_k, y_corr_low.unsqueeze(2)))
    k_blocks_h = k_in_x_gap & k_in_y_corr & mask_k

    any_block_h = k_blocks_h.any(dim=2)
    los_h = torch.where(has_h_overlap & (~has_h_gap | ~any_block_h), 1.0, 0.0)
    los_h.fill_diagonal_(1.0)

    # 2. Vertical Corridor: Overlap in X, gap in Y
    x_corr_left = torch.maximum(x0_i, x0_j)
    x_corr_right = torch.minimum(x1_i, x1_j)
    has_v_overlap = (x_corr_right > x_corr_left)

    y_gap_low = torch.minimum(y1_i, y1_j)
    y_gap_high = torch.maximum(y0_i, y0_j)
    has_v_gap = (y_gap_high > y_gap_low)

    k_in_y_gap = (torch.minimum(y1_k, y_gap_high.unsqueeze(2)) > torch.maximum(y0_k, y_gap_low.unsqueeze(2)))
    k_in_x_corr = (torch.minimum(x1_k, x_corr_right.unsqueeze(2)) > torch.maximum(x0_k, x_corr_left.unsqueeze(2)))
    k_blocks_v = k_in_y_gap & k_in_x_corr & mask_k

    any_block_v = k_blocks_v.any(dim=2)
    los_v = torch.where(has_v_overlap & (~has_v_gap | ~any_block_v), 1.0, 0.0)
    los_v.fill_diagonal_(1.0)

    return los_h.to(dtype=boxes.dtype), los_v.to(dtype=boxes.dtype)


def compute_spatial_features(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    norm_scale: float = 1000.0,
    feature_dim: int = 16,
    indices_a: torch.Tensor | None = None,
    indices_b: torch.Tensor | None = None,
    use_geometry: bool = True,
    use_containment: bool = True,
    use_line_of_sight: bool = True,
    use_typography: bool = True,
) -> torch.Tensor:
    """Compute pairwise geometric and structural spatial edge features between two sets of bounding boxes.

    Supports:
      - Standard 10-D and 12-D legacy representations (for backwards compatibility).
      - Full 16-D structural layout vector (Continuous Geometry + Overlap-over-Minimum + LOSG + Typography).
      - Modular ablation subsets via boolean feature group toggles.
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

    # Continuous Geometry features
    dx_c = (cx_b - cx_a) / norm_scale
    dy_c = (cy_b - cy_a) / norm_scale
    dist_c = torch.sqrt(dx_c**2 + dy_c**2 + 1e-8)
    theta = torch.atan2(dy_c, dx_c)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Legacy 10-D / 12-D handling
    if feature_dim in (10, 12):
        dx_gap = (x0_b - x1_a) / norm_scale
        dy_gap = (y0_b - y1_a) / norm_scale
        log_w_ratio = torch.log(w_b / w_a)
        log_h_ratio = torch.log(h_b / h_a)
        log_area_ratio = torch.log(area_b / area_a)
        feat_list = [
            dx_gap, dy_gap, dx_c, dy_c, dist_c,
            cos_theta, sin_theta, log_w_ratio, log_h_ratio, log_area_ratio,
        ]
        if feature_dim == 12:
            inter_y = torch.clamp(torch.min(y1_a, y1_b) - torch.max(y0_a, y0_b), min=0.0)
            iou_y = inter_y / torch.min(h_a, h_b)
            if indices_a is not None and indices_b is not None:
                delta_order = torch.tanh((indices_b.unsqueeze(0) - indices_a.unsqueeze(1)).float() / 10.0)
            else:
                idx_a = torch.arange(boxes_a.size(0), device=boxes_a.device).unsqueeze(1)
                idx_b = torch.arange(boxes_b.size(0), device=boxes_b.device).unsqueeze(0)
                delta_order = torch.tanh((idx_b - idx_a).float() / 10.0)
            feat_list.extend([iou_y, delta_order])
        return torch.stack(feat_list, dim=-1)

    # =========================================================================
    # UPGRADED 16-D STRUCTURAL FEATURE GROUPS
    # =========================================================================
    active_feats = []

    # Group 1: Continuous Relative Geometry (5 dims)
    if use_geometry:
        active_feats.extend([dx_c, dy_c, dist_c, cos_theta, sin_theta])

    # Group 2: Axial Containment Overlap-over-Minimum (5 dims)
    if use_containment:
        inter_x = torch.clamp(torch.min(x1_a, x1_b) - torch.max(x0_a, x0_b), min=0.0)
        min_w = torch.min(w_a, w_b)
        overlap_ratio_x = inter_x / min_w

        inter_y = torch.clamp(torch.min(y1_a, y1_b) - torch.max(y0_a, y0_b), min=0.0)
        min_h = torch.min(h_a, h_b)
        overlap_ratio_y = inter_y / min_h

        is_disjoint_x = (overlap_ratio_x == 0.0).float()
        is_disjoint_y = (overlap_ratio_y == 0.0).float()
        is_downward = ((y0_b >= y1_a) & (overlap_ratio_x > 0.3)).float()
        active_feats.extend([overlap_ratio_x, overlap_ratio_y, is_disjoint_x, is_disjoint_y, is_downward])

    # Group 3: Line-of-Sight Visibility (2 dims)
    if use_line_of_sight:
        if boxes_a.size(0) == boxes_b.size(0) and torch.equal(boxes_a, boxes_b):
            los_h, los_v = compute_line_of_sight(boxes_a)
        else:
            inter_h = (torch.minimum(y1_a, y1_b) > torch.maximum(y0_a, y0_b)).float()
            inter_v = (torch.minimum(x1_a, x1_b) > torch.maximum(x0_a, x0_b)).float()
            los_h, los_v = inter_h, inter_v
        active_feats.extend([los_h, los_v])

    # Group 4: Typographical Scale Hierarchy (4 dims)
    if use_typography:
        delta_height = torch.log(h_b / h_a)
        delta_area = torch.log(area_b / area_a)
        h_all = (boxes_a[:, 3] - boxes_a[:, 1]).float()
        mean_h = torch.mean(h_all)
        std_h = torch.std(h_all, unbiased=False) + 1e-5
        z_prominence_a = ((h_a - mean_h) / std_h).expand(-1, boxes_b.size(0))
        z_prominence_b = ((h_b - mean_h) / std_h).expand(boxes_a.size(0), -1)
        active_feats.extend([delta_height, delta_area, z_prominence_a, z_prominence_b])

    if not active_feats:
        # Fallback to geometry
        active_feats = [dx_c, dy_c, dist_c, cos_theta, sin_theta]

    return torch.stack(active_feats, dim=-1)


class SpatialEdgeEncoder(nn.Module):
    """Encodes pairwise geometric features into a dense representation vector."""

    def __init__(self, in_features: int = 16, geo_dim: int = 64, dropout: float = 0.1):
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
        spatial_in_features: int = 16,
        dropout: float = 0.1,
        use_spatial: bool = True,
        use_geometry: bool = True,
        use_containment: bool = True,
        use_line_of_sight: bool = True,
        use_typography: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.geo_dim = geo_dim if use_spatial else 0
        self.use_spatial = use_spatial
        self.use_geometry = use_geometry
        self.use_containment = use_containment
        self.use_line_of_sight = use_line_of_sight
        self.use_typography = use_typography

        # Determine spatial feature dimension dynamically if in 16-D mode
        if spatial_in_features not in (10, 12):
            dim = 0
            if use_geometry: dim += 5
            if use_containment: dim += 5
            if use_line_of_sight: dim += 2
            if use_typography: dim += 4
            self.spatial_in_features = dim if dim > 0 else 5
        else:
            self.spatial_in_features = spatial_in_features

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
            self.spatial_encoder = SpatialEdgeEncoder(
                in_features=self.spatial_in_features, geo_dim=geo_dim, dropout=dropout
            )

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
        soft_priors: torch.Tensor | None = None,
        spatial_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute relation logits for all entity pairs."""
        num_entities = entity_reps.size(0)
        if num_entities == 0:
            return torch.zeros((0, 0), device=entity_reps.device, dtype=entity_reps.dtype)

        # Project to head and tail representations: (M, proj_dim)
        heads = self.head_mlp(entity_reps)
        tails = self.tail_mlp(entity_reps)

        heads_exp = heads.unsqueeze(1).expand(-1, num_entities, -1)
        tails_exp = tails.unsqueeze(0).expand(num_entities, -1, -1)

        # 1. Bilinear score: (M, M)
        bilinear_score = self.bilinear(
            heads_exp.reshape(-1, self.proj_dim),
            tails_exp.reshape(-1, self.proj_dim),
        ).view(num_entities, num_entities)

        # 2. Linear combination with spatial features
        if self.use_spatial:
            spatial_feats = compute_spatial_features(
                entity_boxes,
                entity_boxes,
                feature_dim=self.spatial_in_features,
                indices_a=spatial_indices,
                indices_b=spatial_indices,
                use_geometry=self.use_geometry,
                use_containment=self.use_containment,
                use_line_of_sight=self.use_line_of_sight,
                use_typography=self.use_typography,
            )
            spatial_emb = self.spatial_encoder(spatial_feats)
            combined_feats = torch.cat([heads_exp, tails_exp, spatial_emb], dim=-1)
        else:
            combined_feats = torch.cat([heads_exp, tails_exp], dim=-1)

        linear_score = self.linear(combined_feats).squeeze(-1)  # (M, M)

        logits = bilinear_score + linear_score

        # Disallow self-loops (an entity cannot be linked to itself)
        eye_mask = torch.eye(num_entities, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(eye_mask, -1e9)

        # Apply soft log-priors if provided: S_ij_final = S_ij + log P(y_i = Q) + log P(y_j = A)
        if soft_priors is not None:
            logits = logits + soft_priors

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
        loss_type: str = "focal",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
    ) -> torch.Tensor:
        """Compute Binary Cross-Entropy or Sigmoid Focal loss.

        Link matrices in forms are heavily sparse (< 2% positive links). Focal loss
        dynamically downweights easy negative pairs, concentrating gradients on hard links.

        Args:
            logits: Predicted logits of shape (M, M).
            gold_links: Ground truth directed links as pairs (src_idx, dst_idx).
            pos_weight: Loss weight multiplier for positive links when loss_type='bce'.
            valid_mask: Optional boolean mask of shape (M, M) indicating pairs to include in loss.
            loss_type: 'focal' for Sigmoid Focal Loss (default) or 'bce' for Weighted BCE.
            focal_gamma: Focusing parameter gamma for focal loss (default: 2.0).
            focal_alpha: Balancing parameter alpha for positive links (default: 0.75).
        """
        num_entities = logits.size(0)
        if num_entities == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        target = torch.zeros_like(logits)
        for s, d in gold_links:
            if 0 <= s < num_entities and 0 <= d < num_entities and s != d:
                target[s, d] = 1.0

        # Exclude self-loops
        eye_mask = torch.eye(num_entities, dtype=torch.bool, device=logits.device)
        loss_mask = ~eye_mask
        if valid_mask is not None:
            loss_mask = loss_mask & valid_mask

        if loss_type == "focal":
            # Numerically stable Sigmoid Focal Loss
            bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            probs = torch.sigmoid(logits)
            p_t = target * probs + (1.0 - target) * (1.0 - probs)
            alpha_t = target * focal_alpha + (1.0 - target) * (1.0 - focal_alpha)
            focal_weight = alpha_t * ((1.0 - p_t) ** focal_gamma)
            loss_matrix = focal_weight * bce_loss
            # Canonical Focal Loss (Lin et al., RetinaNet): normalize by positive links (N_pos)
            # to prevent easy negatives from diluting positive link gradients.
            n_pos = (target * loss_mask.float()).sum()
            denom = torch.clamp(n_pos, min=1.0)
            return (loss_matrix * loss_mask.float()).sum() / denom
        else:
            denom = loss_mask.sum().float()
            if denom == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            weight_tensor = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
            criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor, reduction="none")
            loss_matrix = criterion(logits, target)
            return (loss_matrix * loss_mask.float()).sum() / denom

    @staticmethod
    def decode_predictions(
        logits: torch.Tensor,
        threshold: float = 0.5,
        one_to_one: bool = False,
        fallback_top_k: int = 0,
    ) -> set[tuple[int, int]]:
        """Decode predicted logits into a set of directed (head, tail) link pairs.

        Args:
            logits: (M, M) tensor of link logits.
            threshold: Probability decision threshold (default: 0.5).
            one_to_one: If True, each tail (value) entity can be linked to at most one
                        highest-scoring head (key).
            fallback_top_k: If > 0 and no pairs pass threshold, fallback to top-k unmasked scores
                            for diagnostic introspection.

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

        # Diagnostic fallback: if no link passes threshold, retain top-k unmasked scores
        if len(predicted) == 0 and fallback_top_k > 0 and num_entities > 1:
            flat_probs = probs.clone()
            eye_mask = torch.eye(num_entities, dtype=torch.bool, device=logits.device)
            flat_probs.masked_fill_(eye_mask, 0.0)
            k_eff = min(fallback_top_k, num_entities * (num_entities - 1))
            flat_vals, flat_indices = torch.topk(flat_probs.view(-1), k=k_eff)
            for val, idx in zip(flat_vals, flat_indices):
                if val.item() > 0.05:  # filter negligible noise floor
                    i = (idx // num_entities).item()
                    j = (idx % num_entities).item()
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


def build_soft_type_priors(
    entity_probs: torch.Tensor,
    id2label: dict[int, str],
    eps: float = 1e-4,
) -> torch.Tensor:
    """Build soft log-prior matrix for question -> answer compatibility.

    Computes:
        log_prior(i, j) = log(P(y_i in QUESTION) + eps) + log(P(y_j in ANSWER) + eps)

    Args:
        entity_probs: Tensor of shape (M, num_labels) with predicted class probabilities.
        id2label: Mapping from class ID to label name.
        eps: Epsilon smoothing factor to avoid log(0).

    Returns:
        Tensor of shape (M, M) with log prior offsets to add to relation logits.
    """
    q_indices = [idx for idx, lab in id2label.items() if "QUESTION" in lab.upper()]
    a_indices = [idx for idx, lab in id2label.items() if "ANSWER" in lab.upper()]

    p_q = entity_probs[:, q_indices].sum(dim=-1).clamp(min=eps, max=1.0)
    p_a = entity_probs[:, a_indices].sum(dim=-1).clamp(min=eps, max=1.0)

    log_p_q = torch.log(p_q)
    log_p_a = torch.log(p_a)

    return log_p_q.unsqueeze(1) + log_p_a.unsqueeze(0)


def build_soft_type_priors_from_labels(
    entity_types: list[str],
    penalty_mismatch: float = -5.0,
) -> torch.Tensor:
    """Build soft log-prior matrix from categorical labels with finite penalty for invalid pairs.

    Unlike hard -1e9 masking, this applies a finite log-space penalty (-5.0), allowing high-confidence
    biaffine links to survive slight label noise.
    """
    num_entities = len(entity_types)
    priors = torch.zeros((num_entities, num_entities), dtype=torch.float32)
    allowed = {("question", "answer"), ("header", "question"), ("header", "answer")}

    for i, t_i in enumerate(entity_types):
        for j, t_j in enumerate(entity_types):
            if i != j:
                pair = (t_i.lower(), t_j.lower())
                if pair not in allowed:
                    priors[i, j] = penalty_mismatch
    return priors


def build_knn_spatial_candidate_mask(
    boxes: torch.Tensor,
    k_neighbors: int = 8,
    gold_links: set[tuple[int, int]] | list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Construct a candidate mask restricting candidates to top-K nearest spatial neighbors.

    Filters the O(N^2) combination space down by ~85%, drastically balancing positive
    and negative candidate pairs. All gold links are preserved in the mask.

    Args:
        boxes: Tensor of shape (M, 4) with entity bounding boxes.
        k_neighbors: Number of nearest spatial neighbors to retain for each entity.
        gold_links: Optional ground-truth link pairs to ensure 100% positive retention.

    Returns:
        Boolean tensor of shape (M, M) where True indicates a pair to evaluate.
    """
    num_entities = boxes.size(0)
    if num_entities <= k_neighbors or k_neighbors <= 0:
        mask = torch.ones((num_entities, num_entities), dtype=torch.bool, device=boxes.device)
    else:
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        dx = cx.unsqueeze(0) - cx.unsqueeze(1)
        dy = cy.unsqueeze(0) - cy.unsqueeze(1)
        dist = torch.sqrt(dx**2 + dy**2 + 1e-6)

        dist.fill_diagonal_(float("inf"))

        _, topk_indices = torch.topk(dist, k=min(k_neighbors, num_entities - 1), largest=False, dim=-1)
        mask = torch.zeros((num_entities, num_entities), dtype=torch.bool, device=boxes.device)
        mask.scatter_(dim=1, index=topk_indices, value=True)

    if gold_links:
        for s, d in gold_links:
            if 0 <= s < num_entities and 0 <= d < num_entities:
                mask[s, d] = True

    return mask
