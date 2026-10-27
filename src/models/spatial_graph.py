"""Phase 1 & 2: Visual Span Merging and Direction-Agnostic Beta-Skeleton Graph Construction.

Implements:
1. Phase 1 (Visual Phrase / BIO Span Merging & Scale-Invariant Dimension Extraction):
   - Merges collinear, horizontally adjacent OCR word fragments (e.g. ['Oct', '25,', '2026'])
     into composite visual nodes before computing topological graph edges.
   - Extracts scale-invariant dimensions normalized to page width/height in [0, 1]:
     centroid c_i = (x_i, y_i), height h_i, and area Area_i.

2. Phase 2 (Direction-Agnostic Beta-Skeleton Graph & Isotropic Edge Features):
   - Isotropic 3D Edge Features e_ij = [log(d_ij + 1), log(Area_i / Area_j), IoU_axial(i, j)]:
       * Normalized Radial Euclidean Distance: d_ij = ||c_i - c_j||_2 / sqrt(h_i * h_j)
       * Relative Area Ratio: log(Area_i / Area_j)
       * Axial Y-Containment (Overlap-over-Minimum): overlap_y(i, j) / min(h_i, h_j)
   - Beta-Skeleton Adjacency (A_ij):
       * Connects nodes (i, j) iff no third distinct node k lies inside their exclusion
         region (beta=1.0 for Gabriel Graph, beta>1 for Lune/Relative Neighborhood Graph).
       * Broadcasts composite node topology back to subword tokens with internal cliques
         (A_ij = 1 within the same composite span) and global [CLS] connectivity.
"""

from __future__ import annotations

import torch


def merge_collinear_boxes(
    bbox: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    y_iou_threshold: float = 0.65,
    x_gap_factor: float = 1.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge horizontally adjacent collinear token boxes into composite span boxes.

    Args:
        bbox: Float tensor of shape (B, L, 4) with coordinates [x0, y0, x1, y1] in [0, 1].
        attention_mask: Optional tensor of shape (B, L) where 1 indicates active tokens.
        y_iou_threshold: Minimum vertical overlap-over-minimum to consider two tokens on the same line.
        x_gap_factor: Maximum horizontal gap (as a multiple of token height) to merge adjacent tokens.

    Returns:
        composite_bbox: Tensor of shape (B, L, 4) where tokens in the same merged visual span
            share the union bounding box of that span.
        span_ids: Long tensor of shape (B, L) assigning each token to its composite span ID.
    """
    B, L, _ = bbox.shape
    device = bbox.device
    composite_bbox = bbox.clone()
    span_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L).clone()

    # Process per batch item on CPU/GPU using fast vectorized adjacency check between t and t-1
    if L <= 1:
        return composite_bbox, span_ids

    x0, y0, x1, y1 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    w = (x1 - x0).clamp(min=0.0)
    h = (y1 - y0).clamp(min=0.0)
    valid = (w > 1e-5) & (h > 1e-5)
    if attention_mask is not None and attention_mask.dim() == 2:
        valid = valid & (attention_mask > 0)

    # Compare token t with token t-1
    prev_x0, prev_y0, prev_x1, prev_y1 = x0[:, :-1], y0[:, :-1], x1[:, :-1], y1[:, :-1]
    curr_x0, curr_y0, curr_x1, curr_y1 = x0[:, 1:], y0[:, 1:], x1[:, 1:], y1[:, 1:]
    prev_h, curr_h = h[:, :-1], h[:, 1:]

    # Same subword or overlapping/adjacent on same line
    inter_y0 = torch.maximum(prev_y0, curr_y0)
    inter_y1 = torch.minimum(prev_y1, curr_y1)
    overlap_y = (inter_y1 - inter_y0).clamp(min=0.0)
    min_h = torch.minimum(prev_h, curr_h).clamp(min=1e-4)
    y_iou = overlap_y / min_h

    x_gap = (curr_x0 - prev_x1).clamp(min=0.0)
    max_h = torch.maximum(prev_h, curr_h).clamp(min=1e-4)

    same_box = (
        (torch.abs(curr_x0 - prev_x0) < 1e-5)
        & (torch.abs(curr_y0 - prev_y0) < 1e-5)
        & (torch.abs(curr_x1 - prev_x1) < 1e-5)
        & (torch.abs(curr_y1 - prev_y1) < 1e-5)
    )
    can_merge = valid[:, :-1] & valid[:, 1:] & (
        same_box | ((y_iou >= y_iou_threshold) & (curr_x0 >= prev_x0 - 1e-3) & (x_gap <= x_gap_factor * max_h))
    )

    # Propagate span groups and compute union bounding boxes
    can_merge_cpu = can_merge.cpu().tolist()
    bbox_cpu = bbox.detach().cpu().tolist()

    out_boxes = []
    out_spans = []
    for b in range(B):
        b_boxes = bbox_cpu[b]
        b_merge = can_merge_cpu[b]
        spans = [0] * L
        cur_span = 0
        span_bounds: dict[int, list[float]] = {0: list(b_boxes[0])}

        for t in range(1, L):
            if b_merge[t - 1]:
                spans[t] = cur_span
                sb = span_bounds[cur_span]
                tb = b_boxes[t]
                sb[0] = min(sb[0], tb[0])
                sb[1] = min(sb[1], tb[1])
                sb[2] = max(sb[2], tb[2])
                sb[3] = max(sb[3], tb[3])
            else:
                cur_span = t
                spans[t] = cur_span
                span_bounds[cur_span] = list(b_boxes[t])

        merged_b = [span_bounds[spans[t]] for t in range(L)]
        out_boxes.append(merged_b)
        out_spans.append(spans)

    composite_bbox = torch.tensor(out_boxes, dtype=bbox.dtype, device=device)
    span_ids = torch.tensor(out_spans, dtype=torch.long, device=device)
    return composite_bbox, span_ids


def build_isotropic_spatial_graph(
    bbox: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    beta: float = 1.0,
    merge_spans: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct Direction-Agnostic Beta-Skeleton Adjacency A_ij and 3D Isotropic Edge Features e_ij.

    Args:
        bbox: Float tensor of shape (B, L, 4) normalized in [0, 1].
        attention_mask: Optional tensor (B, L) where 1 indicates active tokens and 0 padding.
        beta: Beta-skeleton parameter (beta=1.0 is Gabriel Graph; beta>1 prunes more aggressively).
        merge_spans: Whether to merge collinear word fragments into composite nodes prior to graph construction.

    Returns:
        edge_features: Float tensor of shape (B, L, L, 3) containing:
            [log(d_ij + 1), log(Area_i / Area_j), IoU_axial(i, j)]
        adjacency: Float tensor of shape (B, L, L) with values in {0.0, 1.0}.
    """
    if bbox.dim() == 3 and bbox.shape[-1] == 8:
        # Convert 4-corner format [x0, y0, x1, y0, x1, y1, x0, y1] back to [x0, y0, x1, y1]
        bbox = torch.stack([bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 5]], dim=-1)

    B, L, _ = bbox.shape
    device = bbox.device
    dtype = bbox.dtype

    # Optimization: if sequences are padded beyond the max active token in the batch,
    # compute the O(M^2) / O(M^3) spatial graph only on the active prefix M <= L.
    active_L = L
    if attention_mask is not None and attention_mask.dim() == 2:
        active_any = (attention_mask > 0).any(dim=0).nonzero(as_tuple=False)
        if active_any.numel() > 0:
            active_L = min(L, int(active_any.max().item()) + 1)

    if active_L < L:
        sub_feat, sub_adj = build_isotropic_spatial_graph(
            bbox=bbox[:, :active_L],
            attention_mask=attention_mask[:, :active_L] if attention_mask is not None else None,
            beta=beta,
            merge_spans=merge_spans,
        )
        edge_features = torch.zeros((B, L, L, 3), dtype=dtype, device=device)
        adjacency = torch.eye(L, dtype=dtype, device=device).unsqueeze(0).expand(B, L, L).clone()
        edge_features[:, :active_L, :active_L, :] = sub_feat
        adjacency[:, :active_L, :active_L] = sub_adj
        return edge_features, adjacency

    if merge_spans:
        graph_bbox, span_ids = merge_collinear_boxes(bbox, attention_mask=attention_mask)
    else:
        graph_bbox = bbox
        span_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    x0 = graph_bbox[..., 0]
    y0 = graph_bbox[..., 1]
    x1 = graph_bbox[..., 2]
    y1 = graph_bbox[..., 3]

    w = (x1 - x0).clamp(min=0.0)
    h = (y1 - y0).clamp(min=0.0)
    is_valid_box = (w > 1e-5) & (h > 1e-5)
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            is_valid_box = is_valid_box & (attention_mask > 0)

    w_safe = w.clamp(min=1e-3)
    h_safe = h.clamp(min=1e-3)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    area = (w_safe * h_safe).clamp(min=1e-6)

    # -------------------------------------------------------------------------
    # 1. Isotropic Edge Features e_ij: (B, L, L, 3)
    # -------------------------------------------------------------------------
    dx = cx.unsqueeze(2) - cx.unsqueeze(1)  # (B, L, L)
    dy = cy.unsqueeze(2) - cy.unsqueeze(1)  # (B, L, L)
    dist_sq = dx * dx + dy * dy
    euclid_dist = torch.sqrt(dist_sq + 1e-8)

    # Scale-invariant radial distance normalized by geometric mean of token heights
    height_scale = torch.sqrt(h_safe.unsqueeze(2) * h_safe.unsqueeze(1)).clamp(min=1e-3)
    d_ij = euclid_dist / height_scale
    feat_log_dist = torch.log1p(d_ij)

    # Relative area (symmetric log ratio bounded to [-4.6, 4.6])
    area_ratio = (area.unsqueeze(2) / area.unsqueeze(1)).clamp(min=0.01, max=100.0)
    feat_rel_area = torch.log(area_ratio)

    # Axial Y-Containment (Overlap-over-Minimum on Y-axis)
    inter_y0 = torch.maximum(y0.unsqueeze(2), y0.unsqueeze(1))
    inter_y1 = torch.minimum(y1.unsqueeze(2), y1.unsqueeze(1))
    overlap_y = (inter_y1 - inter_y0).clamp(min=0.0)
    min_h = torch.minimum(h_safe.unsqueeze(2), h_safe.unsqueeze(1)).clamp(min=1e-3)
    feat_iou_axial = (overlap_y / min_h).clamp(min=0.0, max=1.0)

    edge_features = torch.stack([feat_log_dist, feat_rel_area, feat_iou_axial], dim=-1)

    valid_pair = is_valid_box.unsqueeze(2) & is_valid_box.unsqueeze(1)
    edge_features.masked_fill_(~valid_pair.unsqueeze(-1), 0.0)

    # -------------------------------------------------------------------------
    # 2. Beta-Skeleton Adjacency Matrix A_ij: (B, L, L)
    #    Using the tropical min-plus formulation:
    #    A third distinct node k lies in the exclusion region of (i, j) iff
    #      beta * (D^2_ik + D^2_kj) < D^2_ij
    # -------------------------------------------------------------------------
    same_span = span_ids.unsqueeze(2) == span_ids.unsqueeze(1)  # (B, L, L)
    coincident = (dist_sq < 1e-7) | same_span

    # Mask out invalid nodes k and coincident nodes (k in same span as i or j) with +inf
    inf_val = 1e6
    d_sq_masked = dist_sq.clone()
    d_sq_masked[coincident] = inf_val
    invalid_k = ~is_valid_box  # (B, L)
    d_sq_masked = d_sq_masked.masked_fill(invalid_k.unsqueeze(1), inf_val)
    d_sq_masked = d_sq_masked.masked_fill(invalid_k.unsqueeze(2), inf_val)

    # Compute min_k (D^2_ik + D^2_kj) in chunks to keep memory O(B * L^2)
    min_sum_sq = torch.full((B, L, L), inf_val, dtype=dtype, device=device)
    chunk_size = 64
    for k_start in range(0, L, chunk_size):
        k_end = min(L, k_start + chunk_size)
        # d_ik: (B, L, 1, C), d_kj: (B, 1, L, C) -> sum: (B, L, L, C)
        d_ik = d_sq_masked[:, :, k_start:k_end].unsqueeze(2)
        d_kj = d_sq_masked[:, k_start:k_end, :].unsqueeze(1).transpose(2, 3)
        chunk_min = (d_ik + d_kj).min(dim=-1).values
        min_sum_sq = torch.minimum(min_sum_sq, chunk_min)

    # Edge (i, j) survives if beta * min_k (D^2_ik + D^2_kj) >= D^2_ij
    beta_val = max(0.5, float(beta))
    is_gabriel_edge = (beta_val * min_sum_sq) >= (dist_sq - 1e-6)

    adjacency = (is_gabriel_edge & valid_pair) | same_span

    # Always include self-loops
    eye = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
    adjacency = adjacency | eye

    # Connect [CLS] (index 0) to all active tokens so document-level context is never isolated
    if attention_mask is not None and attention_mask.dim() == 2:
        active = attention_mask > 0
    else:
        active = torch.ones((B, L), dtype=torch.bool, device=device)
    adjacency[:, 0, :] = active
    adjacency[:, :, 0] = active

    return edge_features, adjacency.to(dtype)
