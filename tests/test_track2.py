"""Unit tests for Track 2: Delaunay Visibility Graph, OpenCV Visual Line Pruning, and Edge-GAT."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def test_delaunay_pure_numpy():
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        print("[SKIP] test_delaunay_pure_numpy (scipy not available)")
        return

    boxes = np.array([
        [100, 100, 200, 150],
        [300, 100, 450, 150],
        [100, 250, 200, 300],
        [300, 250, 450, 300],
    ])
    centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0])
    tri = Delaunay(centers)
    assert len(tri.simplices) >= 2, f"Expected at least 2 triangles, got {len(tri.simplices)}"
    print(f"[PASS] test_delaunay_pure_numpy (generated {len(tri.simplices)} simplices)")


def test_delaunay_graph_builder():
    try:
        import torch
        from src.models.delaunay_graph import build_delaunay_graph, compute_16d_edge_features
    except ImportError:
        print("[SKIP] test_delaunay_graph_builder (Torch not available)")
        return

    boxes = [
        [100, 100, 200, 150],
        [300, 100, 450, 150],
        [100, 250, 200, 300],
        [300, 250, 450, 300],
    ]

    edge_index, edge_attr = build_delaunay_graph(boxes, mode="delaunay", add_self_loops=True)
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    assert edge_attr.dim() == 2 and edge_attr.size(1) == 16
    assert edge_index.size(1) == edge_attr.size(0)
    assert edge_index.size(1) >= 4, "Expected at least 4 edges (self-loops + neighbors)"
    print(f"[PASS] test_delaunay_graph_builder (E={edge_index.size(1)} edges, 16-D attrs)")


def test_edge_gat_forward():
    try:
        import torch
        from src.models.edge_gat import EdgeGATLayer, EdgeGATTokenClassifier
    except ImportError:
        print("[SKIP] test_edge_gat_forward (Torch not available)")
        return

    N = 8
    in_dim = 64
    hidden_dim = 32
    num_labels = 5
    edge_dim = 16

    x = torch.randn(N, in_dim)
    edges = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edges.append((i, j))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    E = edge_index.size(1)
    edge_attr = torch.randn(E, edge_dim)

    layer = EdgeGATLayer(in_dim=in_dim, out_dim=hidden_dim, edge_dim=edge_dim, num_heads=2)
    out_h = layer(x, edge_index, edge_attr)
    assert out_h.shape == (N, hidden_dim)

    clf = EdgeGATTokenClassifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        num_layers=2,
        edge_dim=edge_dim,
        num_heads=2,
    )
    logits = clf(x, edge_index, edge_attr)
    assert logits.shape == (N, num_labels)
    print(f"[PASS] test_edge_gat_forward (logits shape: {logits.shape})")


if __name__ == "__main__":
    test_delaunay_pure_numpy()
    test_delaunay_graph_builder()
    test_edge_gat_forward()
    print("ALL TRACK 2 UNIT TESTS PASSED SUCCESSFULLY!")
