"""Study 29 data-path tests (CPU, no model): python tests/test_vlm_sft.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.vrdu_loader import load_vrdu  # noqa: E402
from matched_eval import f1_of, span_counts  # noqa: E402
from vlm_sft import decode_pages, fields_and_desc, gold_spans, page_target, page_units, parse_page  # noqa: E402


def test_parse():
    f = ["file_date", "registrant_name"]
    got, st = parse_page("file_date | 3-5, 9-9\nregistrant_name | NONE\njunk\nfile_date | 99-100", 20, f)
    assert got == {"file_date": [(3, 5), (9, 9)]} and st["bad_rows"] == 1 and st["bad_runs"] == 1, (got, st)
    assert parse_page("", 5, f)[0] == {} and parse_page("file_date | x", 5, f)[1]["bad_runs"] == 1


def test_gold_roundtrip():
    examples, _, id2label = load_vrdu(ROOT / "data" / "vrdu")
    by = {e.guid: e for e in examples}
    fields, desc = fields_and_desc()
    assert set(desc) >= set(fields), (fields, set(desc))
    tot, tp_oracle, multi = [0, 0, 0], 0, 0
    n = 0
    for p in ("results/matched_eval/spans.jsonl", "results/confirm_v2/matched/spans.jsonl"):
        for line in open(ROOT / p, encoding="utf-8"):
            r = json.loads(line)
            ex = by[r["guid"]]
            g = gold_spans(ex, id2label)
            assert sorted(s[:3] for s in g) == sorted(s[:3] for s in r["gold"]), r["guid"]  # same spans as the BROS dumps
            units = page_units(ex)
            answers = {u.page: page_target(g, u, fields) for u in units}
            pred, _ = decode_pages(units, answers, fields, ex.words)
            c = span_counts(pred, [s for s in g if any(u.base <= s[1] and s[2] <= u.base + len(u.words) for u in units)])["_micro"]
            allc = span_counts(pred, g)["_micro"]
            tot = [tot[i] + allc[i] for i in range(3)]
            n += 1
    print(f"oracle (gold answers through target -> parse -> decode) over {n} docs: F1 {100 * f1_of(tot):.2f}  tp/pred/gold {tot}")
    assert f1_of(tot) > 0.9


# ---------------------------------------------------------------- study 30: formats S (segments) and T (copied text + line id)
def _refs():
    import json as _j
    return [_j.loads(l) for p in ("results/matched_eval/spans.jsonl", "results/confirm_v2/matched/spans.jsonl") for l in open(ROOT / p, encoding="utf-8")]


def test_segments_and_s_roundtrip():
    from vlm_sft import s_to_words, seg_target
    examples, _, id2label = load_vrdu(ROOT / "data" / "vrdu")
    by = {e.guid: e for e in examples}
    fields, _ = fields_and_desc()
    tot, exact, ngold, nseg = [0, 0, 0], 0, 0, []
    for r in _refs():
        ex = by[r["guid"]]
        g = gold_spans(ex, id2label)
        units = page_units(ex, gap=1.0)
        ans = {}
        for u in units:  # the segments partition the page
            assert u.segs[0][0] == 0 and u.segs[-1][1] == len(u.words) and all(u.segs[i][1] == u.segs[i + 1][0] for i in range(len(u.segs) - 1))
            nseg.append(len(u.segs))
            txt, st = s_to_words(seg_target(g, u, fields), u, fields)
            assert st["bad_rows"] == 0 and st["bad_runs"] == 0, st
            ans[u.page] = txt
        pred, _ = decode_pages(units, ans, fields, ex.words)
        c = span_counts(pred, r["gold"])["_micro"]
        tot = [tot[i] + c[i] for i in range(3)]
        starts = {u.base + a for u in units for a, _ in u.segs}
        ends = {u.base + b for u in units for _, b in u.segs}
        for f, a, b, _ in r["gold"]:
            ngold += 1
            exact += a in starts and b in ends
    print(f"S oracle F1 {100 * f1_of(tot):.2f} {tot}; gold spans that are exact runs of segments {exact}/{ngold} = {100 * exact / ngold:.1f}%; segments per page {sum(nseg) / len(nseg):.1f}")
    assert tot[0] <= exact and f1_of(tot) > 0.75 and exact / ngold > 0.75


def test_s_parser():
    from types import SimpleNamespace
    from vlm_sft import s_to_words
    u = SimpleNamespace(segs=[(0, 2), (2, 5), (5, 6)])
    f = ["file_date", "registrant_name"]
    txt, st = s_to_words("file_date | S0\nregistrant_name | S1-S2, S9\njunk | S1\nregistrant_name | x", u, f)
    assert txt.splitlines()[0] == "file_date | 0-1" and txt.splitlines()[1] == "registrant_name | 2-5", txt
    assert st["bad_rows"] == 1 and st["bad_runs"] == 2, st  # 'junk' is not a field; S9 and x are bad runs
    assert s_to_words("", u, f)[0] == "" and s_to_words("registrant_name | NONE", u, f)[0] == "registrant_name | NONE"


def test_text_format_oracle_and_roundtrip():
    sys.path.insert(0, str(ROOT / "scripts"))
    import schema_guided_kv as sg
    from vrdu_vlm_sft import Fmt
    examples, _, id2label = load_vrdu(ROOT / "data" / "vrdu")
    by = {e.guid: e for e in examples}
    fields, desc = fields_and_desc()
    refs = _refs()
    fmt = Fmt("text", fields, desc, id2label, [r["guid"] for r in refs])
    tot, parsed = [0, 0, 0], 0
    for r in refs:
        ex = by[r["guid"]]
        tg = fmt.targets(ex)
        spans, st = fmt.decode(ex, tg)  # gold answers through the study-25 parser, grounding and vote
        parsed += st["unparsed"] == 0
        c = span_counts(spans, r["gold"])["_micro"]
        tot = [tot[i] + c[i] for i in range(3)]
    print(f"T oracle (gold JSON answers through study 25's decode) over {len(refs)} docs: F1 {100 * f1_of(tot):.2f} {tot}; all answers parse in {parsed}/{len(refs)} docs")
    assert parsed == len(refs) and f1_of(tot) > 0.9
    pg = fmt.pages(by[refs[0]["guid"]])
    assert pg and pg[0][1].startswith("<Document>") and pg[0][1].rstrip().endswith("<Extraction>")


if __name__ == "__main__":
    test_parse()
    test_gold_roundtrip()
    test_segments_and_s_roundtrip()
    test_s_parser()
    test_text_format_oracle_and_roundtrip()
    print("study 30 format tests ok")
