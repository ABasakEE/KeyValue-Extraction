"""Unit tests for the Graph-BROS architecture (spatial graph + LoRA + GraphBrosSelfAttention)."""

from __future__ import annotations

from pathlib import Path
import sys
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from models.spatial_graph import build_isotropic_spatial_graph, merge_collinear_boxes
from models.graph_bros import (
    GraphBrosForTokenClassification,
    GraphBrosSelfAttention,
    LoRALinear,
    focal_cross_entropy_loss,
)


def test_phase1_merge_collinear_boxes():
    """Verify horizontally adjacent OCR fragments merge into a composite visual span."""
    # Token 0: [CLS] at [0,0,0,0]
    # Tokens 1, 2, 3: collinear phrase "Oct 25, 2026" on the same line (y=100..120)
    # Token 4: distant value on another column (x=500..580, y=100..120)
    # Token 5: token on next line (x=100..160, y=200..220)
    bbox = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [100.0, 100.0, 140.0, 120.0],
                [148.0, 100.0, 185.0, 120.0],
                [192.0, 100.0, 240.0, 120.0],
                [500.0, 100.0, 580.0, 120.0],
                [100.0, 200.0, 160.0, 220.0],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.ones((1, 6), dtype=torch.long)
    merged_bbox, span_ids = merge_collinear_boxes(bbox, mask)

    # Tokens 1, 2, 3 should share the same span_id and composite bounding box [100, 100, 240, 120]
    assert span_ids[0, 1].item() == span_ids[0, 2].item() == span_ids[0, 3].item()
    assert span_ids[0, 4].item() != span_ids[0, 3].item()
    assert span_ids[0, 5].item() != span_ids[0, 3].item()

    expected_union = torch.tensor([100.0, 100.0, 240.0, 120.0])
    for idx in (1, 2, 3):
        assert torch.allclose(merged_bbox[0, idx], expected_union), f"Expected {expected_union}, got {merged_bbox[0, idx]}"
    print("[PASS] Phase 1: merge_collinear_boxes verified.")


def test_phase2_beta_skeleton_and_isotropy():
    """Verify isotropic edge features and Gabriel graph (beta=1) exclusion-disk pruning."""
    # Create 3 distinct non-merged nodes along a line in [0, 1] normalized coordinates:
    # Node 1 at x=0.10..0.14, Node 2 at x=0.30..0.34, Node 3 at x=0.50..0.54 (all y=0.10..0.12)
    # Because gap=0.16 >> 1.25*height(0.02), they are not merged into one span.
    # Node 2 sits directly between Node 1 and Node 3, so Gabriel Graph (beta=1) MUST prune edge (1, 3)!
    bbox = torch.tensor(
        [
            [
                [0.00, 0.00, 0.00, 0.00],  # 0: [CLS]
                [0.10, 0.10, 0.14, 0.12],  # 1: Left node
                [0.30, 0.10, 0.34, 0.12],  # 2: Middle node
                [0.50, 0.10, 0.54, 0.12],  # 3: Right node
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.ones((1, 4), dtype=torch.long)
    edge_features, adj = build_isotropic_spatial_graph(bbox, mask, beta=1.0, merge_spans=True)
    log_adj = torch.log(adj.clamp(min=0.0, max=1.0) + 1e-9)

    assert edge_features.shape == (1, 4, 4, 3)
    assert log_adj.shape == (1, 4, 4)

    # 1. Check symmetry (direction-agnostic distance & axial IoU)
    assert torch.allclose(edge_features[0, :, :, 0], edge_features[0, :, :, 0].T, atol=1e-5)
    assert torch.allclose(edge_features[0, :, :, 2], edge_features[0, :, :, 2].T, atol=1e-5)
    # Relative log-area is antisymmetric: log(A_i/A_j) = -log(A_j/A_i)
    assert torch.allclose(edge_features[0, 1:, 1:, 1], -edge_features[0, 1:, 1:, 1].T, atol=1e-5)

    # 2. Check Gabriel graph adjacency pruning:
    # Edges (1, 2) and (2, 3) must be kept (adj == 1.0, log_adj == 0.0), while (1, 3) is pruned (adj == 0.0, log_adj < -15)
    assert abs(log_adj[0, 1, 2].item()) < 1e-4, f"Edge (1,2) should be active, got {log_adj[0, 1, 2].item()}"
    assert abs(log_adj[0, 2, 3].item()) < 1e-4, f"Edge (2,3) should be active, got {log_adj[0, 2, 3].item()}"
    assert log_adj[0, 1, 3].item() < -15.0, f"Edge (1,3) should be pruned by node 2, got {log_adj[0, 1, 3].item()}"

    # 3. Check [CLS] global connectivity (index 0 connected to all active tokens)
    for j in range(4):
        assert abs(log_adj[0, 0, j].item()) < 1e-4
        assert abs(log_adj[0, j, 0].item()) < 1e-4

    print("[PASS] Phase 2: Isotropic edge features and Beta-skeleton pruning verified.")


def test_phase3_and_4_lora_and_graph_attention():
    """Verify LoRALinear parameter freezing and GraphBrosSelfAttention gradient flow."""
    base_lin = nn.Linear(64, 64)
    lora_lin = LoRALinear(base_lin, r=8, lora_alpha=16.0, lora_dropout=0.0, freeze_base=True)

    assert not lora_lin.base_layer.weight.requires_grad
    assert lora_lin.lora_A.requires_grad and lora_lin.lora_B.requires_grad

    class MockOrigAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_attention_heads = 4
            self.attention_head_size = 16
            self.all_head_size = 64
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)
            self.dropout = nn.Dropout(0.0)
            self.position_embedding_type = "absolute"
            self.is_decoder = False

    orig_attn = MockOrigAttention()
    graph_attn = GraphBrosSelfAttention(orig_attn, lora_r=4, lora_alpha=8.0, freeze_base=True)
    # Give lora_B non-zero values so gradients flow into lora_A on the very first step
    nn.init.normal_(graph_attn.query.lora_B, std=0.02)
    nn.init.normal_(graph_attn.key.lora_B, std=0.02)
    nn.init.normal_(graph_attn.value.lora_B, std=0.02)

    B, S, H = 2, 6, 64
    hidden_states = torch.randn(B, S, H)
    # Native BROS bbox_pos_emb has shape (S, S, B, head_dim=16)
    bbox_pos_emb = torch.randn(S, S, B, 16)
    attention_mask = torch.zeros(B, 1, 1, S)
    graph_attn.set_graph_context(
        edge_features=torch.randn(B, S, S, 3),
        log_adjacency=torch.zeros(B, 1, S, S),
    )

    outputs = graph_attn(
        hidden_states=hidden_states,
        bbox_pos_emb=bbox_pos_emb,
        attention_mask=attention_mask,
    )
    context_layer, attn_probs = outputs[0], outputs[1]
    assert context_layer.shape == (B, S, H)
    assert attn_probs.shape == (B, 4, S, S)

    loss = (context_layer * torch.randn_like(context_layer)).sum()
    loss.backward()

    # Verify gradients flowed to W_g, head_graph_gate, and LoRA weights, but NOT frozen base weights
    assert graph_attn.W_g.weight.grad is not None and graph_attn.W_g.weight.grad.abs().sum() > 0
    assert graph_attn.head_graph_gate.grad is not None
    assert graph_attn.query.lora_A.grad is not None and graph_attn.query.lora_A.grad.abs().sum() > 0
    assert graph_attn.key.lora_A.grad is not None and graph_attn.key.lora_A.grad.abs().sum() > 0
    assert graph_attn.value.lora_A.grad is not None and graph_attn.value.lora_A.grad.abs().sum() > 0
    assert graph_attn.query.base_layer.weight.grad is None
    assert graph_attn.value.base_layer.weight.grad is None
    print("[PASS] Phase 3 & 4: LoRALinear and GraphBrosSelfAttention forward/backward verified.")


def test_phase5_focal_loss():
    """Verify multi-class focal cross-entropy loss handles padding (-100) and computes valid gradients."""
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, -100, -100], [3, 4, 0, 1, -100]], dtype=torch.long)
    loss = focal_cross_entropy_loss(logits, labels, gamma=2.0)
    assert loss.ndim == 0 and loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    # Verify padded positions (-100) receive zero gradient
    assert torch.all(logits.grad[0, 3:] == 0.0)
    assert torch.all(logits.grad[1, 4:] == 0.0)
    print("[PASS] Phase 5: Multi-class Focal Loss verified.")


def test_end_to_end_graph_bros_for_token_classification():
    """Verify full Hugging Face BrosForTokenClassification wrapped in GraphBrosForTokenClassification."""
    from transformers import BrosConfig, BrosForTokenClassification

    cfg = BrosConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        dim_bbox=4,
        dim_bbox_sinusoid_emb_1d=8,
        dim_bbox_sinusoid_emb_2d=32,
        dim_bbox_projection=16,
        num_labels=7,
        id2label={i: f"L_{i}" for i in range(7)},
        label2id={f"L_{i}": i for i in range(7)},
    )
    hf_bros = BrosForTokenClassification(cfg)

    class DummyBaseClassifier:
        def __init__(self, model):
            self.model = model
            self.id2label = cfg.id2label

    wrapper = GraphBrosForTokenClassification(
        base_classifier=DummyBaseClassifier(hf_bros),
        lora_r=4,
        lora_alpha=8.0,
        beta=1.0,
        loss_type="ce",
        merge_spans=True,
        freeze_base=False,
    )
    stats = wrapper.trainable_parameter_summary()
    assert stats["trainable_params"] > 0
    param_groups = wrapper.get_optimizer_param_groups(base_lr=5e-5, graph_lr=2.5e-4, weight_decay=0.01)
    assert len(param_groups) == 2
    assert param_groups[0]["lr"] == 5e-5 and param_groups[1]["lr"] == 2.5e-4

    B, L = 2, 12
    input_ids = torch.randint(0, 100, (B, L))
    bbox = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0]] + [[0.1 * k, 0.1, 0.1 * k + 0.05, 0.14] for k in range(1, L)],
            [[0.0, 0.0, 0.0, 0.0]] + [[0.08 * k, 0.2, 0.08 * k + 0.04, 0.25] for k in range(1, L)],
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.ones((B, L), dtype=torch.long)
    attention_mask[:, -2:] = 0  # test active-prefix optimization on padded tokens
    labels = torch.randint(0, 7, (B, L))
    labels[:, 0] = -100
    labels[:, -2:] = -100

    out = wrapper(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, labels=labels)
    assert out.loss is not None and not torch.isnan(out.loss)
    assert out.logits.shape == (B, L, 7)

    out.loss.backward()
    opt = torch.optim.AdamW(param_groups, weight_decay=0.01)
    opt.step()
    opt.zero_grad()
    print(f"[PASS] End-to-End Hugging Face GraphBrosForTokenClassification verified ({stats['trainable_params']:,} trainable params across {len(param_groups)} differential LR groups).")


if __name__ == "__main__":
    test_phase1_merge_collinear_boxes()
    test_phase2_beta_skeleton_and_isotropy()
    test_phase3_and_4_lora_and_graph_attention()
    test_phase5_focal_loss()
    test_end_to_end_graph_bros_for_token_classification()
    print("\n[+] ALL GRAPH-BROS UNIT TESTS PASSED!")
