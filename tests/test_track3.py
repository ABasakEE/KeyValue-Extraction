"""Unit tests for Track 3: EdgeGATJointExtractor with Dual Node & Edge Heads."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from src.models.edge_gat_joint import EdgeGATJointExtractor, focal_binary_loss


def test_joint_extractor_forward():
    model = EdgeGATJointExtractor(
        in_dim=776,
        hidden_dim=64,
        num_node_labels=4,
        num_layers=2,
        edge_dim=16,
        num_heads=2,
    )

    N = 12
    E = 24
    x = torch.randn(N, 776)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 16)

    node_logits, edge_logits = model(x, edge_index, edge_attr)

    assert node_logits.shape == (N, 4), f"Expected ({N}, 4), got {node_logits.shape}"
    assert edge_logits.shape == (E,), f"Expected ({E},), got {edge_logits.shape}"

    # Verify loss calculation
    node_targets = torch.randint(0, 4, (N,))
    edge_targets = torch.randint(0, 2, (E,)).float()

    loss_node = torch.nn.functional.cross_entropy(node_logits, node_targets)
    loss_edge = focal_binary_loss(edge_logits, edge_targets)
    total_loss = loss_node + loss_edge

    total_loss.backward()
    print(f"[PASS] test_joint_extractor_forward (N={N}, E={E}, total_loss={total_loss.item():.4f})")


if __name__ == "__main__":
    test_joint_extractor_forward()
