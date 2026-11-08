"""Tests for per-group link recall and the table-links verdict rule."""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    sys.path.insert(0, p)

from e2e_link import add_counts, gold_structure, prf  # noqa: E402
from table_links_summary import verdict  # noqa: E402
from xfund_e2e_eval import link_groups  # noqa: E402


def make_ex():
    boxes = [(0, 0, 10, 10), (20, 0, 30, 10), (0, 30, 10, 40), (20, 30, 30, 40)]
    ents = {
        0: {"label": "question", "token_indices": [0]},
        1: {"label": "answer", "token_indices": [1]},
        2: {"label": "question", "token_indices": [2]},
        3: {"label": "answer", "token_indices": [3]},
        4: {"label": "other", "token_indices": []},
    }
    links = [(0, 1), (2, 3), (0, 2), (3, 1), (0, 4)]
    return SimpleNamespace(entities=ents, links=links, boxes=boxes, words=list("abcd"))


def test_groups_partition_links():
    ex = make_ex()
    g = link_groups(ex)
    _, G = gold_structure(ex)
    assert len(G) == 4
    assert g["below"] | g["same_line"] | g["above"] == G
    assert len(g["below"]) + len(g["same_line"]) + len(g["above"]) == 4
    assert len(g["below"]) == 1 and len(g["same_line"]) == 2 and len(g["above"]) == 1
    ranks = [g["rank_0"], g["rank_1"], g["rank_2_4"], g["rank_5plus"]]
    assert set().union(*ranks) == G and sum(len(r) for r in ranks) == 4
    assert len(g["q_to_q"]) == 1


def test_group_recall_counts():
    ex = make_ex()
    g = link_groups(ex)
    pred = set(g["same_line"])
    _, G = gold_structure(ex)
    out = {}
    for name in ("same_line", "below"):
        acc = [0, 0, 0]
        add_counts(acc, pred & G, g[name])
        out[name] = prf(*acc)["recall"]
    assert out["same_line"] == 1.0 and out["below"] == 0.0


def test_verdict_rules():
    assert verdict([2.5, 2.1, 3.0]) == "helps"
    assert verdict([0.5, -0.3, 0.4]) == "no_better"
    assert verdict([-1.5, -2.0, -1.2]) == "worse"
    assert verdict([1.5, 1.2, 1.8]) == "inconclusive"
    assert verdict([2.5, -0.1, 4.0]) == "no_better"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "passed")
    print("ALL TABLE EVAL TESTS PASSED")
