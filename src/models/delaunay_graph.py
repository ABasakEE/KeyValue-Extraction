"""Delaunay Visibility Graph Builder with Visual Line Pruning and 16-D Structural Edge Features.

Constructs planar Attributed Relational Graphs (ARG) over document tokens:
1. Computes token centroids and generates planar Delaunay triangulation via scipy.spatial.Delaunay.
2. Performs visual line pruning: uses OpenCV morphological line detection to detect horizontal
   and vertical visual rules/borders, pruning any Delaunay edge that crosses a visual line.
3. Computes 16-D structural spatial edge vectors (Geometry, Containment, Line-of-Sight, Typography).
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import os
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    if _var not in os.environ:
        os.environ[_var] = "1"

import numpy as np
import torch


def _segments_intersect(p1: tuple[float, float], p2: tuple[float, float],
                        p3: tuple[float, float], p4: tuple[float, float]) -> bool:
    """Check if segment p1-p2 strictly intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    # Segments intersect iff endpoints of one line lie on opposite sides of the other
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def extract_visual_lines(image: Any, min_line_ratio: float = 0.05) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Detect horizontal and vertical visual lines/rules in document image using morphological filtering."""
    try:
        import cv2
    except ImportError:
        return []

    if image is None:
        return []

    # Handle PIL Image or other array-like types
    if not isinstance(image, np.ndarray):
        try:
            image = np.array(image)
        except Exception:
            return []

    if not hasattr(image, "shape") or len(image.shape) < 2:
        return []

    # Ensure grayscale
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    H, W = gray.shape[:2]
    # Invert binary threshold
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    lines: list[tuple[tuple[int, int], tuple[int, int]]] = []

    # Horizontal lines
    h_len = max(10, int(W * min_line_ratio))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    h_contours, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in h_contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= h_len:
            # Scale to [0, 1000]
            x0 = int(round(1000 * x / W))
            y0 = int(round(1000 * (y + h / 2.0) / H))
            x1 = int(round(1000 * (x + w) / W))
            y1 = y0
            lines.append(((x0, y0), (x1, y1)))

    # Vertical lines
    v_len = max(10, int(H * min_line_ratio))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
    v_contours, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in v_contours:
        x, y, w, h = cv2.boundingRect(c)
        if h >= v_len:
            x0 = int(round(1000 * (x + w / 2.0) / W))
            y0 = int(round(1000 * y / H))
            x1 = x0
            y1 = int(round(1000 * (y + h) / H))
            lines.append(((x0, y0), (x1, y1)))

    return lines


def compute_16d_edge_features(
    boxes: list[list[int]] | np.ndarray,
    edges: list[tuple[int, int]],
    norm_scale: float = 1000.0,
) -> torch.Tensor:
    """Vectorized computation of 16-D structural edge attributes for a list of edges."""
    if not edges:
        return torch.zeros((0, 16), dtype=torch.float32)

    boxes_arr = np.array(boxes, dtype=np.float32)
    src_indices = [u for u, v in edges]
    dst_indices = [v for u, v in edges]

    b_a = boxes_arr[src_indices]  # (E, 4)
    b_b = boxes_arr[dst_indices]  # (E, 4)

    x0_a, y0_a, x1_a, y1_a = b_a[:, 0], b_a[:, 1], b_a[:, 2], b_a[:, 3]
    x0_b, y0_b, x1_b, y1_b = b_b[:, 0], b_b[:, 1], b_b[:, 2], b_b[:, 3]

    w_a = np.maximum(1.0, x1_a - x0_a)
    h_a = np.maximum(1.0, y1_a - y0_a)
    w_b = np.maximum(1.0, x1_b - x0_b)
    h_b = np.maximum(1.0, y1_b - y0_b)

    cx_a = (x0_a + x1_a) / 2.0
    cy_a = (y0_a + y1_a) / 2.0
    cx_b = (x0_b + x1_b) / 2.0
    cy_b = (y0_b + y1_b) / 2.0

    # 1. Continuous Relative Geometry (5 dims)
    dx = (cx_b - cx_a) / norm_scale
    dy = (cy_b - cy_a) / norm_scale
    dist = np.sqrt(dx**2 + dy**2 + 1e-8)
    theta = np.arctan2(dy, dx)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 2. Axial Containment Overlap-over-Minimum (5 dims)
    inter_x = np.maximum(0.0, np.minimum(x1_a, x1_b) - np.maximum(x0_a, x0_b))
    inter_y = np.maximum(0.0, np.minimum(y1_a, y1_b) - np.maximum(y0_a, y0_b))
    min_w = np.minimum(w_a, w_b)
    min_h = np.minimum(h_a, h_b)
    overlap_x = inter_x / min_w
    overlap_y = inter_y / min_h
    is_disjoint_x = (inter_x == 0.0).astype(np.float32)
    is_disjoint_y = (inter_y == 0.0).astype(np.float32)
    is_downward = (cy_b >= cy_a).astype(np.float32)

    # 3. Line-of-Sight Visibility (2 dims)
    los_x = (overlap_y > 0.5).astype(np.float32)
    los_y = (overlap_x > 0.5).astype(np.float32)

    # 4. Typographical Scale Hierarchy (4 dims)
    area_a = w_a * h_a
    area_b = w_b * h_b
    log_h_ratio = np.log(np.clip(h_b / h_a, 1e-3, 1e3))
    log_area_ratio = np.log(np.clip(area_b / area_a, 1e-3, 1e3))

    doc_h = boxes_arr[:, 3] - boxes_arr[:, 1]
    mean_h = np.mean(doc_h)
    std_h = np.std(doc_h) + 1e-6
    z_h_a = (h_a - mean_h) / std_h
    z_h_b = (h_b - mean_h) / std_h

    feats = np.column_stack([
        dx, dy, dist, cos_theta, sin_theta,
        overlap_x, overlap_y, is_disjoint_x, is_disjoint_y, is_downward,
        los_x, los_y,
        log_h_ratio, log_area_ratio, z_h_a, z_h_b,
    ])
    return torch.tensor(feats, dtype=torch.float32)


def build_delaunay_graph(
    boxes: list[list[int]],
    image: np.ndarray | None = None,
    mode: str = "delaunay",
    k_neighbors: int = 5,
    prune_visual_lines: bool = True,
    add_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Attributed Relational Graph over token bounding boxes.

    Args:
        boxes: List of [x0, y0, x1, y1] coordinates (0-1000 scale).
        image: Optional document image for visual line/rule detection.
        mode: Graph topology: 'delaunay' (default), 'knn', or 'fully_connected'.
        k_neighbors: Number of neighbors for 'knn' mode.
        prune_visual_lines: Whether to prune edges crossing visual borders.
        add_self_loops: Whether to include self-loop edges (i, i).

    Returns:
        edge_index: LongTensor of shape (2, E).
        edge_attr: FloatTensor of shape (E, 16) with structural spatial features.
    """
    N = len(boxes)
    if N == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 16), dtype=torch.float32)

    centers = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in boxes], dtype=np.float32)
    raw_edges: set[tuple[int, int]] = set()

    if mode == "fully_connected" or N <= 2:
        for i in range(N):
            for j in range(i + 1, N):
                raw_edges.add((i, j))
    elif mode == "knn":
        # k-NN graph based on Euclidean distance between centroids
        from scipy.spatial import cKDTree
        k = min(N, k_neighbors + 1)
        tree = cKDTree(centers)
        _, indices = tree.query(centers, k=k)
        for i in range(N):
            for j in indices[i][1:]:
                if i != j:
                    raw_edges.add((min(i, j), max(i, j)))
    else:  # 'delaunay'
        try:
            from scipy.spatial import Delaunay
            # Add micro-jitter to prevent Qhull failure on collinear text lines
            jitter = np.random.RandomState(42).normal(0.0, 1e-4, size=centers.shape)
            tri = Delaunay(centers + jitter)
            for simp in tri.simplices:
                for idx in range(3):
                    u, v = int(simp[idx]), int(simp[(idx + 1) % 3])
                    raw_edges.add((min(u, v), max(u, v)))
        except Exception:
            # Fallback to 5-NN if Delaunay fails on degenerate inputs
            from scipy.spatial import cKDTree
            k = min(N, 6)
            tree = cKDTree(centers)
            _, indices = tree.query(centers, k=k)
            for i in range(N):
                for j in indices[i][1:]:
                    if i != j:
                        raw_edges.add((min(i, j), max(i, j)))

    # Visual border/rule pruning via OpenCV line intersection
    visual_lines = extract_visual_lines(image) if (image is not None and prune_visual_lines) else []
    filtered_edges: list[tuple[int, int]] = []

    for u, v in raw_edges:
        if visual_lines:
            pu = (centers[u, 0], centers[u, 1])
            pv = (centers[v, 0], centers[v, 1])
            crosses = any(_segments_intersect(pu, pv, l_start, l_end) for l_start, l_end in visual_lines)
            if crosses:
                continue
        # Make bidirectional
        filtered_edges.append((u, v))
        filtered_edges.append((v, u))

    if add_self_loops:
        for i in range(N):
            filtered_edges.append((i, i))

    if not filtered_edges:
        # Guarantee at least self-loops
        filtered_edges = [(i, i) for i in range(N)]

    edge_attr = compute_16d_edge_features(boxes, filtered_edges)
    edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()

    return edge_index, edge_attr
