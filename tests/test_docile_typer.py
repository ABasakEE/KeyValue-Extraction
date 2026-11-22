"""Study 43: answer parsing, candidate numbering and retyping of the detector-and-typer (no model, no data beyond what docile_bros imports)."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "scripts")]
from docile_typer import MONEY, candidates, parse_answer, retype  # noqa: E402

A, N, DESC = "line_item_amount_gross", "line_item_amount_net", "line_item_description"


def test_candidates_and_retype():
    spans = [[DESC, 0, 1, ""], [A, 1, 2, ""], [A, 2, 3, ""], [N, 3, 4, ""]]
    pages = [0, 0, 1, 1]
    assert candidates(spans, pages, 0) == [1] and candidates(spans, pages, 1) == [2, 3]
    out = retype(spans, pages, {1: {1: N}}, "vlm")           # page 1: candidate c1 -> net, c2 missing -> kept
    assert [s[0] for s in out] == [DESC, A, N, N] and [s[0] for s in spans] == [DESC, A, A, N]
    gold = {(1, 2): N, (2, 3): A}
    assert [s[0] for s in retype(spans, pages, {}, "oracle", gold)] == [DESC, N, A, N]


def test_parse_answer():
    assert parse_answer('{"c1": "%s", "c2": "%s"}' % (A, N), 2) == {1: A, 2: N}
    assert parse_answer('```json\n{"c1": "%s", "c9": "%s", "c2": "not_a_field"}\n```' % (A, N), 2) == {1: A}   # unknown candidate and unknown field dropped
    assert parse_answer('{"c1": "%s", "c2": "line_item_am' % A, 2) == {}                                       # cut-off: no closing brace -> nothing trusted
    assert parse_answer("no json", 2) == {} and MONEY[0] in MONEY


if __name__ == "__main__":
    test_candidates_and_retype()
    test_parse_answer()
    print("[+] typer unit tests passed")
