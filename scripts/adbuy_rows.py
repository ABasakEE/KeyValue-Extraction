"""Study 34, step 1: VRDU ad-buy line items as rows of word-span cells, the oracle check and the prompt-size table.

    python scripts/adbuy_rows.py oracle [--n 0] [--out results/adbuy_rows/oracle.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
import schema_guided_kv as sg  # noqa: E402
from lir_metric import cell_counts, f1, lir_counts  # noqa: E402
from study25_stage0 import text_match  # noqa: E402

ROOT = sg.ROOT
DATA = ROOT / "data" / "vrdu" / "ad-buy-form" / "main" / "dataset.jsonl.gz"
RULES = json.loads((DATA.parent / "meta.json").read_text(encoding="utf-8"))["entity_name_to_match_func"]
LI_FIELDS = ["channel", "program_desc", "program_start_date", "program_end_date", "sub_amount"]
MAX_PAGES = 12   # the longest ad-buy document has 12 pages; rows are single-page, the model is called per page


def value_same(a: tuple[str, str], b: tuple[str, str]) -> bool:
    """Value-level cell test: same field and the text matches under the field's rule in the dataset's meta.json
    (study 25's reading of the VRDU rules, not run with the official tool; PriceMatch added here: numeric value equal)."""
    if a[0] != b[0]:
        return False
    if RULES[a[0]] == "PriceMatch":
        num = lambda t: "".join(c for c in t if c.isdigit() or c in ".-")
        try:
            return float(num(a[1])) == float(num(b[1]))
        except ValueError:
            return False
    return text_match(RULES[a[0]], a[1], b[1])


def load_records(n: int = 0) -> list[dict]:
    with gzip.open(DATA, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    return rows[:n] if n else rows


def split_lines(rec: dict, doc: sg.Doc, gap: float) -> None:
    """Replace each OCR line by pieces cut where two neighbouring words are more than `gap` line heights apart."""
    boxes = [t["bbox"] for pg in rec["ocr"]["pages"] for t in pg.get("tokens", []) if (t.get("text") or "").strip()]
    for p, lines in doc.lines.items():
        seen: Counter = Counter()
        out = []
        for ln in lines:
            h = max(1e-6, max(boxes[w][4] - boxes[w][2] for w in ln.words)) if ln.words else 1
            cuts = [0] + [k for k in range(1, len(ln.words)) if boxes[ln.words[k]][1] - boxes[ln.words[k - 1]][3] > gap * h] + [len(ln.words)]
            for a, b in zip(cuts, cuts[1:]):
                ws = ln.words[a:b]
                cs, ce = min(doc.char_ranges[w][0] for w in ws), max(doc.char_ranges[w][1] for w in ws)
                lid = sg.coord_id([p, min(boxes[w][1] for w in ws), min(boxes[w][2] for w in ws), max(boxes[w][3] for w in ws), max(boxes[w][4] for w in ws)])
                seen[lid] += 1
                out.append(sg.Line(p, lid if seen[lid] == 1 else f"{lid}#{seen[lid]}", doc.ocr_text[cs:ce].strip(), ws, (cs, ce), doc.ocr_text[cs:ce]))
        doc.lines[p] = out


def ground_set(segs: list[tuple[str, str]], by_id: dict, doc: sg.Doc) -> frozenset | None:
    """Words of the cited text on every cited line (all segments, not only the first run); None if any segment is not found."""
    ws: set[int] = set()
    for text, lid in segs:
        ln = by_id.get(lid)
        sp = ln and (sg._line_span(text, ln, doc) or sg._line_span_lenient(text, ln, doc))
        if not sp:
            return None
        ws.update(range(*sp))
    return frozenset(ws)


def gold_rows(rec: dict, doc: sg.Doc) -> list[list[tuple[str, frozenset, list[tuple[str, str, int]], str]]]:
    """Per gold row: cells (field, word set, [(text on line, line id, page)], gold text); the word set comes from the character offsets."""
    flat = [ln for lines in doc.lines.values() for ln in lines]
    out = []
    for ent in rec["annotations"]:
        if not isinstance(ent[0], list):
            continue
        names, rows = ent
        for row in rows:
            cells = []
            for name, cell in zip(names, row):
                spans = cell[2] if len(cell) > 2 else []
                ws = [i for i, (a, b) in enumerate(doc.char_ranges) if a >= 0 and any(a < ce and b > cs for cs, ce in spans)]
                segs = [(doc.ocr_text[max(cs, ln.chars[0]):min(ce, ln.chars[1])].strip(), ln.id, ln.page)
                        for cs, ce in spans for ln in flat if max(cs, ln.chars[0]) < min(ce, ln.chars[1])]
                if ws:
                    cells.append((name, frozenset(ws), [s for s in segs if s[0]], cell[0].strip()))
            out.append(cells)
    return out


def oracle(records: list[dict], gap: float | None = None) -> dict:
    why: Counter = Counter()
    tot_lir, tot_cell, val_lir, val_cell = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
    n_rows = n_cells = no_words = miss_docs = 0
    toks, misses = [], []
    for rec in records:
        doc = sg.Doc(rec, max_pages=MAX_PAGES)
        if gap is not None:
            split_lines(rec, doc, gap)
        gold, pred, vgold, vpred = [], [], [], []
        for row in gold_rows(rec, doc):
            gold.append({(f, ws) for f, ws, _, _ in row})
            vgold.append({(f, t) for f, _, _, t in row})
            cells, vcells = set(), set()
            for f, ws, segs, _ in row:
                by_id = {ln.id: ln for p in {x[2] for x in segs} for ln in doc.lines.get(p, [])}
                got = ground_set([(t, i) for t, i, _ in segs], by_id, doc) if segs else None
                why["grounded" if got else "not_found"] += 1
                why["right_words" if got == ws else "wrong_words"] += 1
                if got:
                    cells.add((f, got))
                    vcells.add((f, " ".join(doc.words[w] for w in sorted(got))))
            pred.append(cells)
            vpred.append(vcells)
        n_rows += len(gold)
        n_cells += sum(map(len, gold))
        if gold:
            a, b = lir_counts(pred, gold), cell_counts(pred, gold)
            tot_lir, tot_cell = [x + y for x, y in zip(tot_lir, a)], [x + y for x, y in zip(tot_cell, b)]
            c, d = lir_counts(vpred, vgold, value_same), cell_counts(vpred, vgold, value_same)
            val_lir, val_cell = [x + y for x, y in zip(val_lir, c)], [x + y for x, y in zip(val_cell, d)]
            if a[0] < a[2]:
                miss_docs += 1
                misses.append(rec["filename"])
        for p, lines in doc.lines.items():   # prompt size per page: lines with ids, estimated at 3 characters per token
            toks.append(sum(len(ln.text) + 8 for ln in lines) / 3)
    toks.sort()
    q = lambda x: round(toks[min(len(toks) - 1, int(x * len(toks)))])
    return {"docs": len(records), "gold_rows": n_rows, "gold_cells": n_cells, "lir_f1": round(f1(tot_lir), 4), "cell_f1": round(f1(tot_cell), 4),
            "value_lir_f1": round(f1(val_lir), 4), "value_cell_f1": round(f1(val_cell), 4), "value_tp_pred_gold": val_lir,
            "tp_pred_gold_lir": tot_lir, "gap": gap, "grounding": dict(why), "docs_with_misses": miss_docs, "example_misses": misses[:5],
            "page_prompt_tokens_estimate": {"pages": len(toks), "median": q(.5), "p90": q(.9), "p99": q(.99), "max": round(toks[-1]),
                                            "pages_over_6000": sum(t > 6000 for t in toks), "pages_over_12000": sum(t > 12000 for t in toks)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["oracle"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--gap", type=float, default=None, help="split OCR lines at gaps wider than this many line heights")
    ap.add_argument("--out", default=str(ROOT / "results" / "adbuy_rows" / "oracle.json"))
    a = ap.parse_args()
    res = oracle(load_records(a.n), a.gap)
    print(json.dumps(res, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
