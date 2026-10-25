"""Unit tests for BROS Backbone and 2D Relative Positional Layout Integration."""

import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.models.token_clf import TokenClassifier, EncodedBatch, BACKBONES
from src.data.vrdu_loader import VrduExample, chunk_vrdu_example
from src.train import make_collator, FormDataset


def test_bros_backbone_registered():
    assert "bros" in BACKBONES
    checkpoint, tok_checkpoint, needs_img = BACKBONES["bros"]
    assert checkpoint == "jinho8345/bros-base-uncased"
    assert tok_checkpoint == "jinho8345/bros-base-uncased"
    assert not needs_img


def test_bros_encode_bbox_construction():
    """Verify that TokenClassifier.encode builds float bboxes in [0, 1.0] for BROS."""
    num_labels = 4
    id2label = {0: "O", 1: "B-KEY", 2: "B-VAL", 3: "I-VAL"}

    # Mock classifier to test encode logic without instantiating heavy model weights
    class MockBrosClassifier:
        def __init__(self):
            from transformers import BertTokenizerFast
            self.backbone = "bros"
            self.max_length = 32
            # Use basic bert tokenizer (same vocabulary and behavior as bros tokenizer)
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        def encode(self, words_batch, boxes_batch, labels_batch=None):
            return TokenClassifier.encode(self, words_batch, boxes_batch, labels_batch)

    classifier = MockBrosClassifier()
    words = [["Total", "Amount", "$120.00"]]
    boxes = [[[100, 200, 150, 220], [160, 200, 220, 220], [230, 200, 300, 220]]]
    labels = [[1, 1, 2]]

    batch = classifier.encode(words, boxes, labels)

    assert "input_ids" in batch.inputs
    assert "bbox" in batch.inputs
    assert "labels" in batch.inputs
    assert "token_type_ids" in batch.inputs  # BROS expects token_type_ids

    bbox = batch.inputs["bbox"]
    assert bbox.dim() == 3
    assert bbox.shape[0] == 1
    assert bbox.shape[1] == 32
    assert bbox.shape[2] == 4
    assert bbox.dtype == torch.float32

    # Check that special token [CLS] at pos 0 has zero bbox
    assert torch.equal(bbox[0, 0], torch.zeros(4))

    # Check that word tokens are normalized to [0, 1.0]
    word_ids = batch.word_ids[0]
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid < len(boxes[0]):
            expected = torch.tensor([
                boxes[0][wid][0] / 1000.0,
                boxes[0][wid][1] / 1000.0,
                boxes[0][wid][2] / 1000.0,
                boxes[0][wid][3] / 1000.0,
            ], dtype=torch.float32)
            assert torch.allclose(bbox[0, pos], expected)

    print("[PASS] test_bros_encode_bbox_construction")


def test_bros_sliding_window_collation():
    """Verify sliding window chunks are properly encoded with relative coordinates for BROS."""
    num_labels = 3
    id2label = {0: "O", 1: "B-KEY", 2: "B-VAL"}

    class MockBrosClassifier:
        def __init__(self):
            from transformers import BertTokenizerFast
            self.backbone = "bros"
            self.max_length = 64
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        def encode(self, words_batch, boxes_batch, labels_batch=None):
            return TokenClassifier.encode(self, words_batch, boxes_batch, labels_batch)

    classifier = MockBrosClassifier()
    long_ex = VrduExample(
        guid="long_test.pdf",
        words=[f"w_{i}" for i in range(120)],
        boxes=[[100, 100, 200, 150]] * 120,
        ner_tags=[1] * 120,
        template_id="T_TEST",
    )

    chunks = chunk_vrdu_example(long_ex, window_size=50, overlap=10)
    assert len(chunks) >= 3

    collate_fn = make_collator(classifier, is_joint=False, relative_coords=True)
    batch_encoded, batch_chunks = collate_fn(chunks[:2])

    assert "bbox" in batch_encoded.inputs
    assert batch_encoded.inputs["bbox"].dtype == torch.float32
    assert batch_encoded.inputs["bbox"].shape == (2, 64, 4)
    print("[PASS] test_bros_sliding_window_collation")


if __name__ == "__main__":
    test_bros_backbone_registered()
    test_bros_encode_bbox_construction()
    test_bros_sliding_window_collation()
    print("ALL BROS INTEGRATION TESTS PASSED SUCCESSFULLY!")
