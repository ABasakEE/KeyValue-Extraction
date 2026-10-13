"""Integration tests for JointFormExtractor."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import unittest
from dataclasses import dataclass
import torch
from transformers import BertConfig, BertForTokenClassification

from src.models.joint_extractor import JointFormExtractor, JointModelOutput, prepare_entity_batch
from src.models.linking_head import BiaffineRelationClassifier


class TestJointExtractor(unittest.TestCase):

    def test_joint_extractor_forward_and_backward(self):
        # Use lightweight synthetic parameters
        num_labels = 7
        id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        hidden_dim = 64
        proj_dim = 32
        geo_dim = 16

        # Create model directly using a dummy backbone structure
        model = JointFormExtractor.__new__(JointFormExtractor)
        torch.nn.Module.__init__(model)
        model.backbone = "mock"
        model.needs_image = False
        model.linking_loss_weight = 1.0
        model.pos_weight = 5.0

        # Mock encoder: simple BertForTokenClassification with hidden_dim=64
        config = BertConfig(
            vocab_size=100,
            hidden_size=hidden_dim,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            num_labels=num_labels,
        )
        model.encoder = BertForTokenClassification(config)

        # Relation head
        model.relation_head = BiaffineRelationClassifier(
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            geo_dim=geo_dim,
            dropout=0.0,
            use_spatial=True,
        )

        B, L = 2, 8
        inputs = {
            "input_ids": torch.randint(0, 100, (B, L)),
            "attention_mask": torch.ones((B, L), dtype=torch.long),
            "bbox": torch.randint(0, 1000, (B, L, 4)),
            "labels": torch.randint(0, num_labels, (B, L)),
        }

        # Document 0 has 3 entities, Document 1 has 2 entities
        entity_spans = [
            [[0, 1], [2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5]],
        ]
        gold_links = [
            [(0, 1), (0, 2)],
            [(0, 1)],
        ]
        entity_types = [
            ["question", "answer", "answer"],
            ["question", "answer"],
        ]

        output = model(
            inputs=inputs,
            entity_spans=entity_spans,
            gold_links=gold_links,
            entity_types=entity_types,
        )

        self.assertIsInstance(output, JointModelOutput)
        self.assertEqual(output.token_logits.shape, (B, L, num_labels))
        self.assertEqual(len(output.relation_logits), B)
        self.assertEqual(output.relation_logits[0].shape, (3, 3))
        self.assertEqual(output.relation_logits[1].shape, (2, 2))
        self.assertIsNotNone(output.tagging_loss)
        self.assertIsNotNone(output.linking_loss)
        self.assertIsNotNone(output.total_loss)

        # Verify backward pass propagates gradients
        output.total_loss.backward()

        # Check encoder gradients
        self.assertIsNotNone(model.encoder.bert.embeddings.word_embeddings.weight.grad)
        # Check relation head gradients
        self.assertIsNotNone(model.relation_head.bilinear.weight.grad)
        self.assertIsNotNone(model.relation_head.head_mlp[0].weight.grad)

    def test_prepare_entity_batch(self):
        @dataclass
        class MockExample:
            words: list[str]
            entities: dict[int, dict]
            links: list[tuple[int, int]]

        # Document with 4 words: "Total", ":", "$", "50"
        # Entity 10: words 0, 1 ("Total :") -> Question
        # Entity 20: words 2, 3 ("$ 50") -> Answer
        # Link: (10, 20)
        ex = MockExample(
            words=["Total", ":", "$", "50"],
            entities={
                10: {"label": "question", "token_indices": [0, 1]},
                20: {"label": "answer", "token_indices": [2, 3]},
            },
            links=[(10, 20)],
        )

        # Subwords: [<s>, Tot, al, :, $, 50, </s>]
        # word_ids: [None, 0, 0, 1, 2, 3, None]
        word_ids_batch = [[None, 0, 0, 1, 2, 3, None]]

        spans, links, types = prepare_entity_batch([ex], word_ids_batch)

        self.assertEqual(len(spans), 1)
        self.assertEqual(len(spans[0]), 2)
        # Entity 10 (mapped to idx 0) has first-subword of word 0 (idx 1) and word 1 (idx 3)
        self.assertEqual(spans[0][0], [1, 3])
        # Entity 20 (mapped to idx 1) has first-subword of word 2 (idx 4) and word 3 (idx 5)
        self.assertEqual(spans[0][1], [4, 5])

        # Link (10, 20) mapped to (0, 1)
        self.assertEqual(links, [[(0, 1)]])
        self.assertEqual(types, [["question", "answer"]])


if __name__ == "__main__":
    unittest.main()
