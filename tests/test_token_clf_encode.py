"""Regression test: TokenClassifier.encode must not hand `boxes` to a tokenizer that does not take it.

Only LayoutLMv3's tokenizer (used for LiLT) consumes `boxes`. BROS uses a plain fast BERT tokenizer; transformers 4.51.3
rejects the stray keyword ("_batch_encode_plus() got an unexpected keyword argument 'boxes'"), older and newer versions
ignore it, so the bug only showed up in notebooks that pin 4.51.3 for Qwen2.5-VL. The stubs reproduce both signatures,
so this test needs neither network nor model weights.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.token_clf import IGNORE_INDEX, TokenClassifier  # noqa: E402


class Enc(dict):
    def __init__(self, words_batch, max_length, with_bbox=None, token_type=True):
        ids = torch.zeros((len(words_batch), max_length), dtype=torch.long)
        self._word_ids = []
        for b, words in enumerate(words_batch):
            wid = [None] + list(range(len(words))) + [None]
            ids[b, :len(wid)] = 1
            self._word_ids.append(wid + [None] * (max_length - len(wid)))
        super().__init__(input_ids=ids, attention_mask=(ids > 0).long())
        if token_type:
            self["token_type_ids"] = torch.zeros_like(ids)
        if with_bbox is not None:
            self["bbox"] = torch.tensor(with_bbox)

    def word_ids(self, batch_index=0):
        return self._word_ids[batch_index]


class StrictBertTokenizer:
    """Plain fast tokenizer with explicit keywords: an unknown `boxes` raises TypeError, like transformers 4.51.3."""

    def __call__(self, words_batch, is_split_into_words=False, padding=None, truncation=None, max_length=None,
                 return_tensors=None):
        assert is_split_into_words
        return Enc(words_batch, max_length)


class LayoutLMv3LikeTokenizer:
    """Takes `boxes`, has no is_split_into_words keyword (a TypeError there triggers encode's fallback call)."""

    def __call__(self, words_batch, boxes=None, padding=None, truncation=None, max_length=None, return_tensors=None):
        assert boxes is not None and len(boxes) == len(words_batch)
        bbox = [[[0, 0, 0, 0]] + b + [[0, 0, 0, 0]] + [[0, 0, 0, 0]] * (max_length - len(b) - 2) for b in boxes]
        return Enc(words_batch, max_length, with_bbox=bbox)


def make(backbone, tokenizer):
    tc = object.__new__(TokenClassifier)   # skip the model download: only encode() is exercised
    tc.backbone, tc.max_length, tc.tokenizer = backbone, 16, tokenizer
    return tc


def test_bros_does_not_pass_boxes():
    words = [["Date:", "03-Jan-1994", "TOTAL"]]
    boxes = [[[100, 200, 300, 400], [100, 200, 300, 400], [500, 600, 700, 800]]]
    enc = make("bros", StrictBertTokenizer()).encode(words, boxes, [[1, 2, 0]])
    bbox = enc.inputs["bbox"]
    assert bbox.shape == (1, 16, 4) and bbox.dtype == torch.float32
    assert torch.allclose(bbox[0, 1], torch.tensor([0.1, 0.2, 0.3, 0.4]))          # word 0 box, scaled to 0-1
    assert torch.allclose(bbox[0, 3], torch.tensor([0.5, 0.6, 0.7, 0.8]))
    assert enc.inputs["labels"][0, :5].tolist() == [IGNORE_INDEX, 1, 2, 0, IGNORE_INDEX]


def test_lilt_still_gets_boxes():
    words = [["Date:", "03-Jan-1994"]]
    boxes = [[[100, 200, 300, 400], [500, 600, 700, 800]]]
    enc = make("lilt", LayoutLMv3LikeTokenizer()).encode(words, boxes, [[1, 2]])
    assert enc.inputs["bbox"][0, 1].tolist() == [100, 200, 300, 400] and "token_type_ids" not in enc.inputs


if __name__ == "__main__":
    test_bros_does_not_pass_boxes()
    test_lilt_still_gets_boxes()
    print("[+] token_clf encode regression tests passed")
