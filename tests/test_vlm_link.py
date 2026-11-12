"""Study 27: VLM prompts and parsers score perfect answers perfectly and survive bad ones."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.xfund_loader import load_xfund_split  # noqa: E402
from e2e_link import gold_structure  # noqa: E402
from vlm_link import form_order, ocr_lines, parse, prompt_v2, prompt_v3, render_gold  # noqa: E402

DATA = ROOT / "data" / "xfund"
_VAL = []


def _val():
    if not _VAL:
        _VAL.extend(load_xfund_split(DATA, "val", ["es", "fr", "de", "it", "pt"]))
    return _VAL


def representable(ex):
    """Gold links whose two ends are tagged (H/Q/A) entities: the most an end-to-end arm can find."""
    ents, links = gold_structure(ex)
    rngs = {(a, b) for _, a, b in ents}
    return ents, {l for l in links if l[0] in rngs and l[1] in rngs}


def test_form_order_balanced_and_complete():
    val = _val()
    order = form_order(val)
    assert len(order) == len(val) == 250 and {e.guid for e in order} == {e.guid for e in val}
    first10 = [e.guid.split("_")[0] for e in order[:10]]
    assert sorted(first10) == sorted(["de", "es", "fr", "it", "pt"] * 2)
    assert [e.guid for e in form_order(val)] == [e.guid for e in order]


def test_ocr_lines_cover_every_word_once():
    val = _val()
    for ex in val[:40]:
        lines = ocr_lines(ex)
        covered = [w for a, b in lines for w in range(a, b)]
        assert covered == list(range(len(ex.words)))


def test_v2_perfect_answer_scores_perfectly():
    val = _val()
    for ex in val[:40]:
        ents, links, stats = parse("V2", render_gold("V2", ex), ex)
        g_ents, g_links = representable(ex)
        assert ents == g_ents and links == g_links, ex.guid
        assert stats["parse_fail"] == 0 and stats["bad_ranges"] == 0


def test_v3_perfect_answer_scores_perfectly():
    val = _val()
    for ex in val[:40]:
        _, links, stats = parse("V3", render_gold("V3", ex), ex)
        assert links == gold_structure(ex)[1], ex.guid


def test_v1_grounding_recovers_most_entities():
    val = _val()
    hit = total = 0
    for ex in val:
        ents, _, _ = parse("V1", render_gold("V1", ex), ex)
        g = representable(ex)[0]
        hit, total = hit + len(ents & g), total + len(g)
    # ceiling of the image-only arm: entities made only of symbols cannot be retyped; measured 0.93 on all 250 forms
    assert hit / total > 0.9, hit / total


def test_backward_links_are_flipped():
    val = _val()
    ex = val[0]
    _, links, _ = parse("V2", "A 1-1 > Q 0-0", ex)
    assert links == {((0, 1), (1, 2))}


BAD = ["", "I cannot read this form.", "E1 Q 0-", "E1 Q 5-2\nE1>E2", "E1 Q 0-99999\nE1>E9",
       "```json\n{\"a\": 1}\n```", "E1 H | \nE2 Q | zzzz-not-in-form\nE1>E2"]


def test_bad_outputs_never_raise():
    ex = _val()[0]
    for text in BAD:
        for arm in ("V1", "V2", "V3"):
            _, links, stats = parse(arm, text, ex)
            assert isinstance(links, set) and isinstance(stats, dict)
        _, links, _ = parse("V2", text, ex)
        assert not links, text
    assert parse("V2", "", ex)[2]["parse_fail"] == 1


def test_v2_truncated_answer_keeps_complete_lines():
    ex = _val()[0]
    full = render_gold("V2", ex)
    cut = full[: len(full) // 2]
    _, links, _ = parse("V2", cut, ex)
    complete = full[: full.rfind("\n", 0, len(cut))]
    assert links == parse("V2", complete, ex)[1] and links


def test_prompts_mention_every_word_or_entity():
    val = _val()
    ex = val[0]
    p2 = prompt_v2(ex)
    assert f"[0-" in p2 and f"-{len(ex.words) - 1}]" in p2
    assert f"e{len([i for i in ex.entities if ex.entities[i].get('token_indices')])} " in prompt_v3(ex)


if __name__ == "__main__":
    if not (DATA / "es.val.json").exists():
        sys.exit("needs data/xfund")
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} passed")
    print("ALL VLM LINK TESTS PASSED")
