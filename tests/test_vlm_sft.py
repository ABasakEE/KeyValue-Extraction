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


if __name__ == "__main__":
    test_parse()
    print("parse ok")
    test_gold_roundtrip()
    print("all ok")
