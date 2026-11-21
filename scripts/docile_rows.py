"""Study 34, DocILE probe (CPU): line items as rows of word-set cells, the oracle check, layout statistics, prompt sizes.

    python scripts/docile_rows.py oracle [--n 0] [--gap 0.5]

Data: data/docile/{annotations,ocr}/<id>.json (annotated-trainval, token download, see scripts/download_docile.sh).
A cell is the set of words that have at least one pseudo-character centre inside the gold box (DocILE's own rule: characters are
spread evenly along the word box). A cited piece is an OCR line cut where neighbouring words are more than `gap` line heights apart.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from lir_metric import cell_counts, f1, lir_counts  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "data" / "docile"


class Piece:
    def __init__(self, page: int, pid: str, words: list[int], box: tuple):
        self.page, self.id, self.words, self.box = page, pid, words, box


class Doc:
    def __init__(self, doc_id: str, gap: float | None = 0.5):
        self.id = doc_id
        ocr = json.loads((DATA / "ocr" / f"{doc_id}.json").read_text(encoding="utf-8"))
        self.words: list[tuple[int, str, tuple]] = []          # (page, text, (x0, y0, x1, y1)), reading order of the OCR
        self.pieces: dict[int, list[Piece]] = {}
        for p, page in enumerate(ocr["pages"]):
            seen: Counter = Counter()
            out = []
            for blk in page["blocks"]:
                for ln in blk["lines"]:
                    ws = []
                    for w in ln["words"]:
                        (x0, y0), (x1, y1) = w.get("snapped_geometry") or w["geometry"]
                        self.words.append((p, w["value"], (x0, y0, x1, y1)))
                        ws.append(len(self.words) - 1)
                    if not ws:
                        continue
                    h = max(1e-6, max(self.words[i][2][3] - self.words[i][2][1] for i in ws))
                    cuts = [0] + [k for k in range(1, len(ws)) if gap is not None and self.words[ws[k]][2][0] - self.words[ws[k - 1]][2][2] > gap * h] + [len(ws)]
                    for a, b in zip(cuts, cuts[1:]):
                        pw = ws[a:b]
                        box = (min(self.words[i][2][0] for i in pw), min(self.words[i][2][1] for i in pw), max(self.words[i][2][2] for i in pw), max(self.words[i][2][3] for i in pw))
                        q = lambda v: min(99, max(0, int(v * 100)))
                        pid = f"{q((box[0] + box[2]) / 2):02d}|{q((box[1] + box[3]) / 2):02d}"
                        seen[pid] += 1
                        out.append(Piece(p, pid if seen[pid] == 1 else f"{pid}#{seen[pid]}", pw, box))
            self.pieces[p] = out

    def cell_words(self, page: int, bbox: list[float]) -> frozenset:
        x0, y0, x1, y1 = bbox
        ws = set()
        for i, (p, t, (a, b, c, d)) in enumerate(self.words):
            if p != page or not (y0 <= (b + d) / 2 <= y1) or not t:
                continue
            n = len(t)
            if any(x0 <= a + (k + 0.5) / n * (c - a) <= x1 for k in range(n)):
                ws.add(i)
        return frozenset(ws)


def gold_rows(doc: Doc, ann: dict) -> list[list[tuple[str, frozenset, int]]]:
    rows: dict[int, list] = defaultdict(list)
    for e in ann["line_item_extractions"]:
        ws = doc.cell_words(e["page"], e["bbox"])
        if ws:
            rows[e["line_item_id"]].append((e["fieldtype"], ws, e["page"]))
    return list(rows.values())


def cite(doc: Doc, ws: frozenset, page: int) -> list[tuple[str, str]]:
    """What a perfect model would write for a cell: one (text, piece id) per contiguous run of the cell's words inside each piece."""
    out = []
    for pc in doc.pieces[page]:
        run: list[int] = []
        for i in pc.words + [None]:
            if i is not None and i in ws:
                run.append(i)
            elif run:
                out.append((" ".join(doc.words[j][1] for j in run), pc.id))
                run = []
    return out


def ground(doc: Doc, page: int, segs: list[tuple[str, str]]) -> frozenset | None:
    by_id = {pc.id: pc for pc in doc.pieces[page]}
    got: set[int] = set()
    for text, pid in segs:
        pc, toks = by_id.get(pid), text.split()
        if pc is None or not toks:
            return None
        ptoks = [doc.words[i][1] for i in pc.words]
        k = next((k for k in range(len(ptoks) - len(toks) + 1) if ptoks[k:k + len(toks)] == toks), None)
        if k is None:
            return None
        got.update(pc.words[k:k + len(toks)])
    return frozenset(got)


def oracle(ids: list[str], gap: float | None) -> dict:
    tl, tc = [0, 0, 0], [0, 0, 0]
    n_rows = n_cells = empty_gold = multi_page_rows = 0
    toks, rows_page, miss_docs = [], [], 0
    for k, i in enumerate(ids):
        ann = json.loads((DATA / "annotations" / f"{i}.json").read_text(encoding="utf-8"))
        doc = Doc(i, gap)
        empty_gold += sum(1 for e in ann["line_item_extractions"] if not doc.cell_words(e["page"], e["bbox"]))
        gold, pred = [], []
        for row in gold_rows(doc, ann):
            multi_page_rows += len({p for _, _, p in row}) > 1
            gold.append({(f, ws) for f, ws, _ in row})
            cells = set()
            for f, ws, p in row:
                got = ground(doc, p, cite(doc, ws, p))
                if got:
                    cells.add((f, got))
            pred.append(cells)
        per_page = Counter(row[0][2] for row in gold_rows(doc, ann))
        rows_page += list(per_page.values())
        n_rows += len(gold)
        n_cells += sum(map(len, gold))
        if gold:
            a, b = lir_counts(pred, gold), cell_counts(pred, gold)
            tl, tc = [x + y for x, y in zip(tl, a)], [x + y for x, y in zip(tc, b)]
            miss_docs += a[0] < a[2]
        for p, pcs in doc.pieces.items():   # prompt size: pieces with ids, 3 characters per token
            toks.append(sum(sum(len(doc.words[w][1]) + 1 for w in pc.words) + 8 for pc in pcs) / 3)
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(ids)} documents", flush=True)
    toks.sort()
    q = lambda x: round(toks[min(len(toks) - 1, int(x * len(toks)))])
    return {"docs": len(ids), "gap": gap, "gold_rows": n_rows, "gold_cells": n_cells, "gold_cells_with_no_word": empty_gold, "rows_crossing_pages": multi_page_rows,
            "lir_f1": round(f1(tl), 4), "cell_f1": round(f1(tc), 4), "tp_pred_gold": tl, "docs_with_misses": miss_docs,
            "rows_per_page_with_rows": {"median": st.median(rows_page), "p90": sorted(rows_page)[int(.9 * len(rows_page))], "max": max(rows_page)},
            "page_prompt_tokens_estimate": {"pages": len(toks), "median": q(.5), "p90": q(.9), "p99": q(.99), "max": round(toks[-1]), "pages_over_6000": sum(t > 6000 for t in toks)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["oracle"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ids = sorted(json.loads((DATA / "trainval.json").read_text(encoding="utf-8")))
    ids = ids[:a.n] if a.n else ids
    res = oracle(ids, None if a.gap < 0 else a.gap)
    print(json.dumps(res, indent=1))
    out = Path(a.out or ROOT / "results" / "docile_rows" / f"oracle_gap{a.gap}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
