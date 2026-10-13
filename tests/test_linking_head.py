"""Unit tests for the Biaffine Key-Value Linking Head."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import unittest
import math
import torch

from src.models.linking_head import (
    BiaffineRelationClassifier,
    SpatialEdgeEncoder,
    build_type_compatibility_mask,
    compute_spatial_features,
    pool_entity_representations,
    build_soft_type_priors_from_labels,
    build_knn_spatial_candidate_mask,
)


class TestLinkingHead(unittest.TestCase):

    def test_spatial_features_shape_and_bounds(self):
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
        self.assertEqual(features.shape, (2, 3, 10))
        # Check that distance is non-negative
        dist = features[..., 4]
        self.assertTrue(torch.all(dist >= 0))
        # Self distance between box 0 and box 0 should be approximately 0
        self.assertTrue(torch.isclose(dist[0, 0], torch.tensor(0.0), atol=1e-3))
        # Cosine and sine should lie in [-1, 1]
        cos_theta = features[..., 5]
        sin_theta = features[..., 6]
        self.assertTrue(torch.all(cos_theta >= -1.0 - 1e-5) and torch.all(cos_theta <= 1.0 + 1e-5))
        self.assertTrue(torch.all(sin_theta >= -1.0 - 1e-5) and torch.all(sin_theta <= 1.0 + 1e-5))

    def test_spatial_edge_encoder(self):
        encoder = SpatialEdgeEncoder(in_features=10, geo_dim=32, dropout=0.0)
        spatial_feats = torch.randn(4, 4, 10)
        out = encoder(spatial_feats)
        self.assertEqual(out.shape, (4, 4, 32))

    def test_biaffine_classifier_forward_and_masking(self):
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
        self.assertEqual(type_mask.shape, (M, M))
        # Question (idx 0) to Answer (idx 1) should be True
        self.assertTrue(type_mask[0, 1].item())
        # Answer (idx 1) to Question (idx 0) should be False
        self.assertFalse(type_mask[1, 0].item())

        logits = classifier(entity_reps, entity_boxes, type_mask=type_mask)
        self.assertEqual(logits.shape, (M, M))

        # Self-loops must be masked to large negative value
        for i in range(M):
            self.assertLessEqual(logits[i, i].item(), -1e8)

        # Incompatible pairs must also be masked
        self.assertLessEqual(logits[1, 0].item(), -1e8)

    def test_biaffine_loss_and_gradients(self):
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
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(entity_reps.grad)
        self.assertTrue(torch.any(entity_reps.grad != 0.0))
        self.assertIsNotNone(classifier.bilinear.weight.grad)

    def test_decode_predictions(self):
        logits = torch.tensor([
            [-1e9, 2.5, -3.0],
            [-1e9, -1e9, -4.0],
            [1.8, -2.0, -1e9],
        ])
        # Sigmoid(2.5) ~ 0.92, Sigmoid(1.8) ~ 0.86, Sigmoid(-2.0) ~ 0.12
        preds = BiaffineRelationClassifier.decode_predictions(logits, threshold=0.5)
        self.assertIn((0, 1), preds)
        self.assertIn((2, 0), preds)
        self.assertEqual(len(preds), 2)

    def test_pool_entity_representations(self):
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
        self.assertEqual(entity_reps.shape, (3, hidden_dim))
        self.assertEqual(entity_boxes.shape, (3, 4))

        # Check that entity 0 representation matches the mean of tokens 0, 1, 2
        expected_rep_0 = token_hidden[0:3].mean(dim=0)
        self.assertTrue(torch.allclose(entity_reps[0], expected_rep_0, atol=1e-5))

        # Check bounding box union for entity 0: x0 should be 0, x1 should be 150
        self.assertEqual(entity_boxes[0, 0].item(), 0)
        self.assertEqual(entity_boxes[0, 2].item(), 150)

    def test_extended_spatial_features_12d(self):
        boxes_a = torch.tensor([[100, 100, 200, 150]])
        boxes_b = torch.tensor([[250, 100, 350, 150]])
        features = compute_spatial_features(boxes_a, boxes_b, feature_dim=12)
        self.assertEqual(features.shape, (1, 1, 12))
        # IoU_y for horizontally aligned boxes should be 1.0
        iou_y = features[0, 0, 10].item()
        self.assertLess(abs(iou_y - 1.0), 1e-3)

    def test_soft_type_priors(self):
        types = ["question", "answer", "header"]
        priors = build_soft_type_priors_from_labels(types, penalty_mismatch=-5.0)
        self.assertEqual(priors.shape, (3, 3))
        # (question, answer) is allowed -> prior = 0.0
        self.assertEqual(priors[0, 1].item(), 0.0)
        # (answer, question) is disallowed -> prior = -5.0
        self.assertEqual(priors[1, 0].item(), -5.0)

    def test_knn_spatial_candidate_mask(self):
        boxes = torch.tensor([
            [100, 100, 200, 140],
            [220, 100, 320, 140],
            [100, 200, 200, 240],
            [220, 200, 320, 240],
        ])
        gold_links = {(0, 1), (2, 3)}
        mask = build_knn_spatial_candidate_mask(boxes, k_neighbors=2, gold_links=gold_links)
        self.assertEqual(mask.shape, (4, 4))
        # Gold links must always be in candidate mask
        self.assertTrue(mask[0, 1].item())
        self.assertTrue(mask[2, 3].item())

    def test_focal_loss_vs_bce(self):
        logits = torch.randn(4, 4)
        gold_links = {(0, 1)}
        loss_focal = BiaffineRelationClassifier.compute_loss(logits, gold_links, loss_type="focal")
        loss_bce = BiaffineRelationClassifier.compute_loss(logits, gold_links, loss_type="bce")
        self.assertGreater(loss_focal.item(), 0.0)
        self.assertGreater(loss_bce.item(), 0.0)

    def test_fallback_top_k_decoding(self):
        # Logits all below threshold 0.5 (sigmoid(-2.0) ~ 0.12, sigmoid(-3.0) ~ 0.05)
        logits = torch.tensor([
            [-1e9, -2.0, -5.0],
            [-1e9, -1e9, -5.0],
            [-1e9, -5.0, -1e9],
        ])
        # Standard threshold returns empty set
        preds_standard = BiaffineRelationClassifier.decode_predictions(logits, threshold=0.5, fallback_top_k=0)
        self.assertEqual(len(preds_standard), 0)

        # Fallback decoding returns top-1 candidate (0, 1)
        preds_fallback = BiaffineRelationClassifier.decode_predictions(logits, threshold=0.5, fallback_top_k=1)
        self.assertIn((0, 1), preds_fallback)


if __name__ == "__main__":
    unittest.main()
