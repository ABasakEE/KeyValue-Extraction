"""Spatial bounding box transformations, augmentations, and anchoring strategies."""

import random


def jitter_boxes(
    boxes: list[list[int]],
    shift_range: int = 80,
    scale_range: tuple[float, float] = (0.85, 1.15),
) -> list[list[int]]:
    """Track 1 Coordinate Jitter: random translation and scaling to break spatial memorization."""
    if not boxes:
        return boxes
    dx = random.randint(-shift_range, shift_range)
    dy = random.randint(-shift_range, shift_range)
    sx = random.uniform(scale_range[0], scale_range[1])
    sy = random.uniform(scale_range[0], scale_range[1])

    all_x = [b[0] for b in boxes] + [b[2] for b in boxes]
    all_y = [b[1] for b in boxes] + [b[3] for b in boxes]
    cx = (min(all_x) + max(all_x)) / 2.0
    cy = (min(all_y) + max(all_y)) / 2.0

    new_boxes = []
    for x0, y0, x1, y1 in boxes:
        nx0 = int(round(cx + (x0 - cx) * sx + dx))
        ny0 = int(round(cy + (y0 - cy) * sy + dy))
        nx1 = int(round(cx + (x1 - cx) * sx + dx))
        ny1 = int(round(cy + (y1 - cy) * sy + dy))
        nx0, nx1 = min(nx0, nx1), max(nx0, nx1)
        ny0, ny1 = min(ny0, ny1), max(ny0, ny1)
        nx0 = max(0, min(1000, nx0))
        ny0 = max(0, min(1000, ny0))
        nx1 = max(0, min(1000, nx1))
        ny1 = max(0, min(1000, ny1))
        new_boxes.append([nx0, ny0, nx1, ny1])
    return new_boxes


def _anchor_centroid(boxes: list[list[int]]) -> list[list[int]]:
    """Method 1: Global Midpoint Shift. Centers the document bounding box at (500, 500)."""
    all_x = [b[0] for b in boxes] + [b[2] for b in boxes]
    all_y = [b[1] for b in boxes] + [b[3] for b in boxes]
    cx = (min(all_x) + max(all_x)) / 2.0
    cy = (min(all_y) + max(all_y)) / 2.0

    new_boxes = []
    for x0, y0, x1, y1 in boxes:
        nx0 = max(0, min(1000, int(round(500 + (x0 - cx)))))
        ny0 = max(0, min(1000, int(round(500 + (y0 - cy)))))
        nx1 = max(0, min(1000, int(round(500 + (x1 - cx)))))
        ny1 = max(0, min(1000, int(round(500 + (y1 - cy)))))
        new_boxes.append([min(nx0, nx1), min(ny0, ny1), max(nx0, nx1), max(ny0, ny1)])
    return new_boxes


def _anchor_content_box(
    boxes: list[list[int]],
    active_envelope: tuple[int, int, int, int] | None = None,
) -> list[list[int]]:
    """Method 2: Aspect-Preserving Active Content Envelope Normalization.

    Eliminates scanner margins while preserving true aspect ratios.
    When active_envelope is provided (from the parent document), all sliding windows
    share the exact same coordinate frame.
    """
    if active_envelope is not None and active_envelope[2] > active_envelope[0] and active_envelope[3] > active_envelope[1]:
        min_x, min_y, max_x, max_y = active_envelope
    else:
        min_x = min(b[0] for b in boxes)
        min_y = min(b[1] for b in boxes)
        max_x = max(b[2] for b in boxes)
        max_y = max(b[3] for b in boxes)

    w_act = max(1, max_x - min_x)
    h_act = max(1, max_y - min_y)
    scale = 900.0 / max(w_act, h_act)

    offset_x = 50 + int((900 - w_act * scale) / 2.0)
    offset_y = 50 + int((900 - h_act * scale) / 2.0)

    new_boxes = []
    for x0, y0, x1, y1 in boxes:
        nx0 = max(0, min(1000, offset_x + int(round((x0 - min_x) * scale))))
        ny0 = max(0, min(1000, offset_y + int(round((y0 - min_y) * scale))))
        nx1 = max(0, min(1000, offset_x + int(round((x1 - min_x) * scale))))
        ny1 = max(0, min(1000, offset_y + int(round((y1 - min_y) * scale))))
        new_boxes.append([min(nx0, nx1), min(ny0, ny1), max(nx0, nx1), max(ny0, ny1)])
    return new_boxes


def _anchor_cluster(boxes: list[list[int]], k: int = 4) -> list[list[int]]:
    """Method 3: Multi-Cluster Spatial Quadrant Anchoring.

    Partitions words into 4 layout quadrants and standardizes local cluster positions,
    preserving local reading order while standardizing template layout shifts.
    """
    if len(boxes) < k:
        return _anchor_content_box(boxes)

    centers = [[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in boxes]
    all_x = sorted(c[0] for c in centers)
    all_y = sorted(c[1] for c in centers)
    med_x = all_x[len(all_x) // 2]
    med_y = all_y[len(all_y) // 2]

    # Canonical quadrant anchors
    anchors = {
        (0, 0): (250, 250),  # Top-Left
        (1, 0): (750, 250),  # Top-Right
        (0, 1): (250, 750),  # Bottom-Left
        (1, 1): (750, 750),  # Bottom-Right
    }

    q_assign = [(int(c[0] >= med_x), int(c[1] >= med_y)) for c in centers]
    q_centroids = {}
    for q in anchors:
        pts = [centers[i] for i, qa in enumerate(q_assign) if qa == q]
        if pts:
            q_centroids[q] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        else:
            q_centroids[q] = anchors[q]

    new_boxes = []
    for i, b in enumerate(boxes):
        q = q_assign[i]
        c_orig = q_centroids[q]
        c_target = anchors[q]
        dx = int(round(c_target[0] - c_orig[0]))
        dy = int(round(c_target[1] - c_orig[1]))

        nx0 = max(0, min(1000, b[0] + dx))
        ny0 = max(0, min(1000, b[1] + dy))
        nx1 = max(0, min(1000, b[2] + dx))
        ny1 = max(0, min(1000, b[3] + dy))
        new_boxes.append([min(nx0, nx1), min(ny0, ny1), max(nx0, nx1), max(ny0, ny1)])
    return new_boxes


def anchor_boxes(
    boxes: list[list[int]],
    mode: str = "content_box",
    active_envelope: tuple[int, int, int, int] | None = None,
) -> list[list[int]]:
    """Applies coordinate anchoring based on the selected mode:

    Modes:
      - 'none': returns unmodified boxes
      - 'centroid': global midpoint shift to (500, 500)
      - 'content_box': aspect-preserving active content envelope normalization
      - 'cluster': multi-cluster spatial quadrant anchoring
    """
    if not boxes or mode == "none":
        return boxes
    if mode == "centroid":
        return _anchor_centroid(boxes)
    if mode == "content_box":
        return _anchor_content_box(boxes, active_envelope=active_envelope)
    if mode == "cluster":
        return _anchor_cluster(boxes)
    raise ValueError(f"Unknown anchor mode: {mode!r}. Choose from ['none', 'centroid', 'content_box', 'cluster']")


def make_relative_boxes(
    boxes: list[list[int]],
    active_envelope: tuple[int, int, int, int] | None = None,
) -> list[list[int]]:
    """Backward-compatible wrapper for make_relative_boxes."""
    return anchor_boxes(boxes, mode="content_box", active_envelope=active_envelope)
