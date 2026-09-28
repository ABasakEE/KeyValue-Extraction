"""Unit tests for the Biaffine Key-Value Linking Head."""

import math
import torch

from src.models.linking_head import (
    BiaffineRelationClassifier,
    SpatialEdgeEncoder,
    build_type_compatibility_mask,
    compute_spatial_features,
    pool_entity_representations,
)


def test_spatial_features_shape_and_bounds():
    boxes_a = torch.tensor([
        [100, 100, 200, 150],
        [300, 100, 400, 150],
    ])
    boxes_b = torch.tensor([
        [100, 100, 200, 150],
        [300, 100, 400, 150],
        [500, 100, 600, 150],
    ])
    features = compute_spatial_features(boxes_a, boxes_b)
    assert features.shape == (2, 3, 10)
    # Check that distance is non-negative
    dist = features[..., 4]
    assert torch.all(dist >= 0)
    # Self distance between box 0 and box 0 should be approximately 0
    assert torch.isclose(dist[0, 0], torch.tensor(0.0), atol=1e-3)
    # Cosine and sine should lie in [-1, 1]
    cos_theta = features[..., 5]
    sin_theta = features[..., 6]
    assert torch.all(cos_theta >= -1.0 - 1e-5) and torch.all(cos_theta <= 1.0 + 1e-5)
    assert torch.all(sin_theta >= -1.0 - 1e-5) and torch.all(sin_theta <= 1.0 + 1e-5)


def test_spatial_edge_encoder():
    encoder = SpatialEdgeEncoder(in_features=10, geo_dim=32, dropout=0.0)
    spatial_feats = torch.randn(4, 4, 10)
    out = encoder(spatial_feats)
    assert out.shape == (4, 4, 32)


def test_biaffine_classifier_forward_and_masking():
    M = 5
    hidden_dim = 64
    proj_dim = 32
    geo_dim = 16

    classifier = BiaffineRelationClassifier(
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        geo_dim=geo_dim,
        dropout=0.0,
        use_spatial=True,
    )

    entity_reps = torch.randn(M, hidden_dim)
    entity_boxes = torch.tensor([
        [100, 100, 200, 150],
        [220, 100, 350, 150],
        [100, 200, 200, 250],
        [220, 200, 350, 250],
        [100, 300, 200, 350],
    ])

    types = ["question", "answer", "question", "answer", "header"]
    type_mask = build_type_compatibility_mask(types)
    assert type_mask.shape == (M, M)
    # Question (idx 0) to Answer (idx 1) should be True
    assert type_mask[0, 1].item() is True
    # Answer (idx 1) to Question (idx 0) should be False
    assert type_mask[1, 0].item() is False

    logits = classifier(entity_reps, entity_boxes, type_mask=type_mask)
    assert logits.shape == (M, M)

    # Self-loops must be masked to large negative value
    for i in range(M):
        assert logits[i, i].item() <= -1e8

    # Incompatible pairs must also be masked
    assert logits[1, 0].item() <= -1e8


def test_biaffine_loss_and_gradients():
    M = 4
    hidden_dim = 32
    classifier = BiaffineRelationClassifier(
        hidden_dim=hidden_dim,
        proj_dim=16,
        geo_dim=8,
        dropout=0.0,
        use_spatial=True,
    )

    entity_reps = torch.randn(M, hidden_dim, requires_grad=True)
    entity_boxes = torch.tensor([
        [50, 50, 150, 80],
        [200, 50, 300, 80],
        [50, 150, 150, 180],
        [200, 150, 300, 180],
    ])

    logits = classifier(entity_reps, entity_boxes)
    gold_links = {(0, 1), (2, 3)}

    loss = BiaffineRelationClassifier.compute_loss(logits, gold_links, pos_weight=5.0)
    assert loss.item() > 0.0

    loss.backward()
    assert entity_reps.grad is not None
    assert torch.any(entity_reps.grad != 0.0)
    assert classifier.bilinear.weight.grad is not None


def test_decode_predictions():
    logits = torch.tensor([
        [-1e9, 2.5, -3.0],
        [-1e9, -1e9, -4.0],
        [1.8, -2.0, -1e9],
    ])
    # Sigmoid(2.5) ~ 0.92, Sigmoid(1.8) ~ 0.86, Sigmoid(-2.0) ~ 0.12
    preds = BiaffineRelationClassifier.decode_predictions(logits, threshold=0.5)
    assert (0, 1) in preds
    assert (2, 0) in preds
    assert len(preds) == 2


def test_pool_entity_representations():
    L = 10
    hidden_dim = 16
    token_hidden = torch.randn(L, hidden_dim)
    token_boxes = torch.zeros(L, 4)
    for i in range(L):
        token_boxes[i] = torch.tensor([i * 50, 100, (i + 1) * 50, 150])

    entity_spans = [
        [0, 1, 2],  # entity 0: tokens 0, 1, 2
        [3, 4],     # entity 1: tokens 3, 4
        [5, 6, 7],  # entity 2: tokens 5, 6, 7
    ]

    entity_reps, entity_boxes = pool_entity_representations(token_hidden, token_boxes, entity_spans)
    assert entity_reps.shape == (3, hidden_dim)
    assert entity_boxes.shape == (3, 4)

    # Check that entity 0 representation matches the mean of tokens 0, 1, 2
    expected_rep_0 = token_hidden[0:3].mean(dim=0)
    assert torch.allclose(entity_reps[0], expected_rep_0, atol=1e-5)

    # Check bounding box union for entity 0: x0 should be 0, x1 should be 150
    assert entity_boxes[0, 0].item() == 0
    assert entity_boxes[0, 2].item() == 150
