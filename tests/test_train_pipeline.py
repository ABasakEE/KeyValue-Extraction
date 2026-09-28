"""Integration tests for the training pipeline in src/train.py.

Verifies that make_collator, train_one, predict, and predict_joint work end-to-end
for both pure token-classification ('tagging') and joint extraction ('joint').
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import tempfile
import torch
from dataclasses import dataclass
from transformers import BertConfig, BertForTokenClassification

from src.models.joint_extractor import JointFormExtractor
from src.models.linking_head import BiaffineRelationClassifier
from src.models.token_clf import TokenClassifier
from src.train import (
    FormDataset,
    TrainConfig,
    build_model,
    make_collator,
    predict,
    predict_joint,
    train_one,
)


@dataclass
class DummyExample:
    words: list[str]
    boxes: list[list[int]]
    ner_tags: list[int]
    entities: dict[int, dict]
    links: list[tuple[int, int]]


def _create_mock_examples():
    """Create 4 mock documents with words, boxes, tags, entities, and links."""
    examples = []
    for doc_idx in range(4):
        ex = DummyExample(
            words=["Total", ":", "$", "50", "Tax", ":", "$", "5"],
            boxes=[
                [100, 100, 200, 150],
                [210, 100, 230, 150],
                [250, 100, 270, 150],
                [280, 100, 350, 150],
                [100, 200, 200, 250],
                [210, 200, 230, 250],
                [250, 200, 270, 250],
                [280, 200, 350, 250],
            ],
            ner_tags=[1, 2, 3, 4, 1, 2, 3, 4],  # B-question, I-question, B-answer, I-answer...
            entities={
                0: {"label": "question", "token_indices": [0, 1]},
                1: {"label": "answer", "token_indices": [2, 3]},
                2: {"label": "question", "token_indices": [4, 5]},
                3: {"label": "answer", "token_indices": [6, 7]},
            },
            links=[(0, 1), (2, 3)],
        )
        examples.append(ex)
    return examples


def _create_mock_joint_extractor(num_labels=7):
    """Instantiate a lightweight JointFormExtractor with mock Bert backbone."""
    id2label = {
        0: "O",
        1: "B-question",
        2: "I-question",
        3: "B-answer",
        4: "I-answer",
        5: "B-header",
        6: "I-header",
    }
    hidden_dim = 64

    model = JointFormExtractor.__new__(JointFormExtractor)
    torch.nn.Module.__init__(model)
    model.backbone = "mock"
    model.needs_image = False
    model.num_labels = num_labels
    model.id2label = id2label
    model.max_length = 32
    model.linking_loss_weight = 1.0
    model.pos_weight = 5.0

    bert_config = BertConfig(
        vocab_size=100,
        hidden_size=hidden_dim,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=num_labels,
    )
    model.encoder = BertForTokenClassification(bert_config)
    model.relation_head = BiaffineRelationClassifier(
        hidden_dim=hidden_dim,
        proj_dim=32,
        geo_dim=16,
        dropout=0.0,
        use_spatial=True,
    )

    # Mock tokenizer and encode
    class MockTokenizer:
        def __call__(self, words_batch, **kwargs):
            B = len(words_batch)
            L = 16
            return {
                "input_ids": torch.randint(0, 100, (B, L)),
                "attention_mask": torch.ones((B, L), dtype=torch.long),
                "bbox": torch.randint(0, 1000, (B, L, 4)),
            }

    model.tokenizer = MockTokenizer()

    def mock_encode(words_batch, boxes_batch, labels_batch=None):
        from src.models.token_clf import EncodedBatch
        B = len(words_batch)
        L = 16
        inputs = {
            "input_ids": torch.randint(0, 100, (B, L)),
            "attention_mask": torch.ones((B, L), dtype=torch.long),
            "bbox": torch.randint(0, 1000, (B, L, 4)),
            "labels": torch.randint(0, num_labels, (B, L)),
        }
        word_ids = []
        for _ in range(B):
            # [None, 0, 1, 2, 3, 4, 5, 6, 7, None, ...]
            wids = [None] + list(range(8)) + [None] * (L - 9)
            word_ids.append(wids)
        return EncodedBatch(inputs=inputs, word_ids=word_ids)

    model.encode = mock_encode
    model.first_subword_predictions = staticmethod(TokenClassifier.first_subword_predictions)
    return model


def test_collator_joint():
    model = _create_mock_joint_extractor()
    examples = _create_mock_examples()
    collate = make_collator(model, is_joint=True)
    batch = collate(examples[:2])

    assert len(batch) == 5
    encoded, batch_ex, spans, links, types = batch
    assert "input_ids" in encoded.inputs
    assert len(batch_ex) == 2
    assert len(spans) == 2
    assert len(links) == 2
    assert len(types) == 2
    # Verify entity spans and links exist
    assert len(spans[0]) == 4  # 4 entities
    assert links[0] == [(0, 1), (2, 3)]  # 2 links


def test_train_one_joint():
    model = _create_mock_joint_extractor()
    examples = _create_mock_examples()

    config = TrainConfig(
        task="joint",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        linking_loss_weight=1.0,
        pos_weight=5.0,
        use_spatial_relations=True,
    )
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir)
        result = train_one(
            model=model,
            train_examples=examples[:2],
            val_examples=examples[2:],
            test_examples=examples[2:],
            config=config,
            device=device,
            checkpoint_dir=ckpt_dir,
        )

        assert "test" in result
        assert "f1" in result["test"]
        assert "test_linking" in result
        assert "f1" in result["test_linking"]
        assert "precision" in result["test_linking"]
        assert "recall" in result["test_linking"]
        assert "best_val_score" in result
        assert len(result["history"]) == 1
        assert "val_entity_f1" in result["history"][0]
        assert "val_linking_f1" in result["history"][0]

        # Verify checkpoints saved
        assert (ckpt_dir / "best_model.pt").exists()
        assert (ckpt_dir / "best_meta.json").exists()
        assert (ckpt_dir / "result.json").exists()
        assert (ckpt_dir / "history.json").exists()


def _create_mock_tagging_classifier(num_labels=7):
    id2label = {
        0: "O",
        1: "B-question",
        2: "I-question",
        3: "B-answer",
        4: "I-answer",
        5: "B-header",
        6: "I-header",
    }
    hidden_dim = 64
    classifier = TokenClassifier.__new__(TokenClassifier)
    classifier.backbone = "mock"
    classifier.needs_image = False
    classifier.max_length = 32

    bert_config = BertConfig(
        vocab_size=100,
        hidden_size=hidden_dim,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=num_labels,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    classifier.model = BertForTokenClassification(bert_config)

    def mock_encode(words_batch, boxes_batch, labels_batch=None):
        from src.models.token_clf import EncodedBatch
        B = len(words_batch)
        L = 16
        inputs = {
            "input_ids": torch.randint(0, 100, (B, L)),
            "attention_mask": torch.ones((B, L), dtype=torch.long),
            "bbox": torch.randint(0, 1000, (B, L, 4)),
            "labels": torch.randint(0, num_labels, (B, L)),
        }
        word_ids = []
        for _ in range(B):
            wids = [None] + list(range(8)) + [None] * (L - 9)
            word_ids.append(wids)
        return EncodedBatch(inputs=inputs, word_ids=word_ids)

    classifier.encode = mock_encode
    return classifier


def test_train_one_tagging():
    classifier = _create_mock_tagging_classifier()
    examples = _create_mock_examples()

    config = TrainConfig(
        task="tagging",
        epochs=1,
        batch_size=2,
        lr=1e-3,
    )
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir)
        result = train_one(
            model=classifier,
            train_examples=examples[:2],
            val_examples=examples[2:],
            test_examples=examples[2:],
            config=config,
            device=device,
            checkpoint_dir=ckpt_dir,
        )

        assert "test" in result
        assert "f1" in result["test"]
        assert "best_val_f1" in result
        assert len(result["history"]) == 1
        assert "val_f1" in result["history"][0]

        # Verify checkpoints saved
        assert (ckpt_dir / "best_model.pt").exists()
        assert (ckpt_dir / "best_meta.json").exists()
        assert (ckpt_dir / "result.json").exists()
        assert (ckpt_dir / "history.json").exists()


if __name__ == "__main__":
    test_collator_joint()
    print("test_collator_joint passed")
    test_train_one_joint()
    print("test_train_one_joint passed")
    test_train_one_tagging()
    print("test_train_one_tagging passed")
    print("ALL PIPELINE INTEGRATION TESTS PASSED")
