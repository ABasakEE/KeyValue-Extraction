"""Unit tests for Track 1: Multi-Page Sliding Window, Coordinate Jitter, and Anchoring Strategies."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.data.vrdu_loader import VrduExample, chunk_vrdu_example, chunk_vrdu_examples
from src.data.transforms import anchor_boxes, jitter_boxes, make_relative_boxes


def test_chunking():
    # 1. Short document (no chunking)
    short_ex = VrduExample(
        guid="short.pdf",
        words=["word"] * 100,
        boxes=[[10, 10, 50, 50]] * 100,
        ner_tags=[0] * 100,
        template_id="T1",
    )
    chunks = chunk_vrdu_example(short_ex, window_size=384, overlap=128)
    assert len(chunks) == 1
    assert chunks[0].doc_guid == "short.pdf"
    assert chunks[0].word_offset == 0
    assert chunks[0].total_doc_words == 100
    assert chunks[0].doc_envelope == (10, 10, 50, 50)

    # 2. Long document (1000 words -> 4 overlapping chunks)
    long_ex = VrduExample(
        guid="long.pdf",
        words=[f"w_{i}" for i in range(1000)],
        boxes=[[100, 100, 200, 200]] * 1000,
        ner_tags=[i % 5 for i in range(1000)],
        template_id="T2",
    )
    chunks = chunk_vrdu_example(long_ex, window_size=384, overlap=128)
    assert len(chunks) == 4
    # Check coverage of all 1000 words & shared doc_envelope
    covered = set()
    for c in chunks:
        assert c.doc_guid == "long.pdf"
        assert c.total_doc_words == 1000
        assert c.doc_envelope == (100, 100, 200, 200)
        for w_idx in range(len(c.words)):
            covered.add(c.word_offset + w_idx)
    assert covered == set(range(1000)), "Sliding window failed to cover all words!"
    print("[PASS] test_chunking")


def test_anchoring_strategies():
    boxes = [
        [50, 60, 150, 100],
        [600, 70, 750, 120],
        [80, 800, 200, 850],
        [650, 820, 800, 900],
    ]

    # Jitter
    jittered = jitter_boxes(boxes, shift_range=50, scale_range=(0.9, 1.1))
    assert len(jittered) == len(boxes)
    for b in jittered:
        assert 0 <= b[0] <= 1000 and 0 <= b[2] <= 1000
        assert 0 <= b[1] <= 1000 and 0 <= b[3] <= 1000
        assert b[2] >= b[0] and b[3] >= b[1]

    # Mode 0: None
    b_none = anchor_boxes(boxes, mode="none")
    assert b_none == boxes

    # Mode 1: Centroid
    b_centroid = anchor_boxes(boxes, mode="centroid")
    assert len(b_centroid) == len(boxes)
    for b in b_centroid:
        assert 0 <= b[0] <= 1000 and 0 <= b[2] <= 1000
        assert 0 <= b[1] <= 1000 and 0 <= b[3] <= 1000
        assert b[2] >= b[0] and b[3] >= b[1]

    # Mode 2: Content Box (Aspect-preserving active envelope)
    b_content = anchor_boxes(boxes, mode="content_box", active_envelope=(50, 60, 800, 900))
    assert len(b_content) == len(boxes)
    for b in b_content:
        assert 0 <= b[0] <= 1000 and 0 <= b[2] <= 1000
        assert 0 <= b[1] <= 1000 and 0 <= b[3] <= 1000
        assert b[2] >= b[0] and b[3] >= b[1]

    # Mode 3: Cluster (Quadrant anchoring)
    b_cluster = anchor_boxes(boxes, mode="cluster")
    assert len(b_cluster) == len(boxes)
    for b in b_cluster:
        assert 0 <= b[0] <= 1000 and 0 <= b[2] <= 1000
        assert 0 <= b[1] <= 1000 and 0 <= b[3] <= 1000
        assert b[2] >= b[0] and b[3] >= b[1]

    print("[PASS] test_anchoring_strategies (tested 'none', 'centroid', 'content_box', 'cluster')")


def test_soft_voting_predict():
    """Verify that multiple chunks for the same document are aggregated into one prediction list."""
    try:
        import torch
        from torch.utils.data import DataLoader
        from src.train import FormDataset, make_collator, predict
        from src.models.token_clf import EncodedBatch
    except ImportError:
        print("[SKIP] test_soft_voting_predict (PyTorch not installed in local environment)")
        return

    long_ex = VrduExample(
        guid="doc_multi.pdf",
        words=[f"w_{i}" for i in range(500)],
        boxes=[[100, 100, 200, 200]] * 500,
        ner_tags=[1] * 500,
        template_id="T1",
    )
    chunks = chunk_vrdu_example(long_ex, window_size=384, overlap=128)
    assert len(chunks) == 2

    class MockModel(torch.nn.Module):
        def forward(self, input_ids=None, bbox=None, attention_mask=None, **kwargs):
            B, L = input_ids.shape
            num_classes = 3
            logits = torch.zeros((B, L, num_classes))
            logits[:, :, 1] = 5.0
            return type("Output", (), {"logits": logits})()

    class MockClassifier:
        def __init__(self):
            self.model = MockModel()

        def encode(self, words_batch, boxes_batch, labels_batch=None):
            B = len(words_batch)
            L = 384
            input_ids = torch.randint(0, 100, (B, L))
            word_ids = []
            for b in range(B):
                n_w = len(words_batch[b])
                wids = list(range(n_w)) + [None] * (L - n_w)
                word_ids.append(wids)
            return EncodedBatch(
                inputs={
                    "input_ids": input_ids,
                    "attention_mask": torch.ones((B, L)),
                    "bbox": torch.randint(0, 1000, (B, L, 4)),
                },
                word_ids=word_ids,
            )

    classifier = MockClassifier()
    collate = make_collator(classifier, is_joint=False, anchor_mode="content_box")
    loader = DataLoader(FormDataset(chunks), batch_size=2, collate_fn=collate)

    preds, refs = predict(classifier, loader, device=torch.device("cpu"))
    assert len(preds) == 1, f"Expected 1 document prediction, got {len(preds)}"
    assert len(refs) == 1, f"Expected 1 document reference, got {len(refs)}"
    assert len(preds[0]) == 500, f"Expected 500 predictions, got {len(preds[0])}"
    assert len(refs[0]) == 500, f"Expected 500 references, got {len(refs[0])}"
    assert all(p == 1 for p in preds[0]), "Soft-voting predicted incorrect class!"
    print("[PASS] test_soft_voting_predict")


if __name__ == "__main__":
    test_chunking()
    test_anchoring_strategies()
    test_soft_voting_predict()
    print("ALL TRACK 1 UNIT TESTS PASSED SUCCESSFULLY!")
