"""Study 40: the two column rules on a made-up page (no data needed beyond what docile_bros imports)."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "scripts")]
from docile_column_rules import column_vote, drop_outside  # noqa: E402


def doc(boxes):
    return {"fboxes": boxes, "pages": [0] * len(boxes)}


def test_column_vote_relabels_the_minority_and_leaves_ties():
    # one column x 0.8-0.9, four rows: three 'gross', one 'net'; a second column at x 0.1-0.3
    boxes = [(0.8, y, 0.9, y + 0.02) for y in (0.1, 0.2, 0.3, 0.4)] + [(0.1, 0.1, 0.3, 0.12)]
    sp = [["g", 0, 1, ""], ["g", 1, 2, ""], ["n", 2, 3, ""], ["g", 3, 4, ""], ["d", 4, 5, ""]]
    out = column_vote(sp, doc(boxes))
    assert [x[0] for x in out] == ["g", "g", "g", "g", "d"]
    tie = [["g", 0, 1, ""], ["n", 1, 2, ""], ["g", 2, 3, ""], ["n", 3, 4, ""], ["d", 4, 5, ""]]
    assert [x[0] for x in column_vote(tie, doc(boxes))] == ["g", "n", "g", "n", "d"]


def test_drop_outside_keeps_the_table_and_drops_far_spans():
    ys = [0.5, 0.55, 0.6, 0.65]   # four anchor rows, pitch 0.05
    boxes = [(0.1, y, 0.2, y + 0.01) for y in ys] + [(0.1, 0.05, 0.2, 0.06), (0.1, 0.9, 0.2, 0.91), (0.1, 0.675, 0.2, 0.685)]
    sp = [["a", i, i + 1, ""] for i in range(4)] + [["x", 4, 5, ""], ["x", 5, 6, ""], ["x", 6, 7, ""]]
    kept = {s[1] for s in drop_outside(sp, doc(boxes))}
    assert kept == {0, 1, 2, 3, 6}   # the span 0.5 pitch below the last row stays; the ones at 0.05 and 0.9 go
