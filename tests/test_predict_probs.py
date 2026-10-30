"""predict(return_probs=True) soft-votes window probabilities; dump_test_probs writes them as merged-field percents.

Stub model and loader: no weights, no network. Two overlapping windows over a 3-word document; the probabilities are
chosen so the window average is known exactly.
"""

import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from train import dump_test_probs, predict  # noqa: E402

ID2LABEL = {0: "O", 1: "B-x", 2: "I-x"}


class StubModel:
    def __init__(self, logits_by_call):
        self.calls = iter(logits_by_call)

    def eval(self):
        pass

    def __call__(self, **inputs):
        return SimpleNamespace(logits=torch.log(next(self.calls)))   # softmax(log p) == p


def window(word_ids, offset, n):
    ex = SimpleNamespace(words=["w"] * n, ner_tags=[0] * n, doc_guid="d1", word_offset=offset, total_doc_words=3)
    enc = SimpleNamespace(inputs={"input_ids": torch.zeros((1, len(word_ids)), dtype=torch.long)}, word_ids=[word_ids])
    return enc, [ex]


def test_soft_vote_and_dump():
    a = torch.tensor([[[1, 1, 1], [.2, .7, .1], [.1, .6, .3], [1, 1, 1]]], dtype=torch.float32)   # specials get junk
    b = torch.tensor([[[1, 1, 1], [.1, .2, .7], [.8, .1, .1], [1, 1, 1]]], dtype=torch.float32)
    a[0, 0] = a[0, 3] = b[0, 0] = b[0, 3] = torch.tensor([1 / 3] * 3)
    loader = [window([None, 0, 1, None], 0, 2), window([None, 0, 1, None], 1, 2)]   # words 0-1, then words 1-2
    clf = SimpleNamespace(model=StubModel([a, b]))
    pred, ref, probs = predict(clf, loader, torch.device("cpu"), return_probs=True)
    assert pred == [[1, 2, 0]], pred                                    # word 1: mean of (.1,.6,.3) and (.1,.2,.7)
    guid, p = probs[0]
    assert guid == "d1" and torch.allclose(p[1], torch.tensor([.1, .4, .5]), atol=1e-6), p
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probs.jsonl"
        dump_test_probs(path, probs, ID2LABEL)
        row = json.loads(path.read_text())
        assert row["fields"] == ["x"] and row["p"] == [[80], [90], [20]], row   # B+I merged, integer percent
        dump_test_probs(path, probs, ID2LABEL, keep={"other"})
        assert path.read_text() == ""                                            # filtered out by document id


def test_default_return_unchanged():
    loader = [window([None, 0, 1, None], 0, 2)]
    t = torch.full((1, 4, 3), 1 / 3)
    out = predict(SimpleNamespace(model=StubModel([t])), loader, torch.device("cpu"))
    assert len(out) == 2                                                          # existing callers still get a pair


if __name__ == "__main__":
    test_soft_vote_and_dump()
    test_default_return_unchanged()
    print("[+] predict(return_probs) tests passed")
