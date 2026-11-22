"""Example figures for the report and poster from SAVED predictions (no weights, no GPU): predicted boxes and labels on the page image, next to the gold.

    python scripts/make_example_figures.py [--out figures]

  * tables : DocILE line items, BROS full-pool seed 21 on an unseen-layout test document: boxes coloured by predicted field, numbered rows from the anchor rule; one good page, one failure.
  * forms  : VRDU registration forms, BROS Track-1 on an unseen template: predicted entities (green = exactly right, red = wrong or extra) and the missed gold entities (dashed).
Key-to-value arrows come from the Gradio linking demo (needs a joint model checkpoint), not from here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src"), str(HERE.parent)]
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = HERE.parent
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#800000", "#808000", "#000075", "#f032e6", "#469990", "#bcbd22", "#7f7f7f"]


def render_page(pdf: Path, page: int, dpi: int = 110) -> Image.Image:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        pix = doc[min(page, len(doc) - 1)].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def draw_box(dr, img, box, color, width=2, dashed=False):
    x0, y0, x1, y1 = box[0] * img.width, box[1] * img.height, box[2] * img.width, box[3] * img.height
    if not dashed:
        dr.rectangle([x0, y0, x1, y1], outline=color, width=width)
        return
    for x in range(int(x0), int(x1), 8):
        dr.line([x, y0, min(x + 4, x1), y0], fill=color, width=width)
        dr.line([x, y1, min(x + 4, x1), y1], fill=color, width=width)
    for y in range(int(y0), int(y1), 8):
        dr.line([x0, y, x0, min(y + 4, y1)], fill=color, width=width)
        dr.line([x1, y, x1, min(y + 4, y1)], fill=color, width=width)


def label(dr, img, xy, text, color, font):
    x, y = xy[0] * img.width, xy[1] * img.height - 11
    w = dr.textlength(text, font=font)
    dr.rectangle([x, y, x + w + 4, y + 11], fill="white")
    dr.text((x + 2, y), text, fill=color, font=font)


def legend(img, items, font):
    h = 16 * ((len(items) + 2) // 3) + 8
    out = Image.new("RGB", (img.width, img.height + h), "white")
    out.paste(img, (0, 0))
    dr = ImageDraw.Draw(out)
    for k, (name, color) in enumerate(items):
        x, y = 8 + (k % 3) * (img.width // 3), img.height + 4 + 16 * (k // 3)
        dr.rectangle([x, y + 2, x + 10, y + 12], fill=color)
        dr.text((x + 14, y), name, fill="black", font=font)
    return out


def side_by_side(left: Image.Image, right: Image.Image, titles: tuple, font) -> Image.Image:
    h = max(left.height, right.height) + 22
    out = Image.new("RGB", (left.width + right.width + 12, h), "white")
    out.paste(left, (0, 22))
    out.paste(right, (left.width + 12, 22))
    dr = ImageDraw.Draw(out)
    dr.text((6, 4), titles[0], fill="black", font=font)
    dr.text((left.width + 18, 4), titles[1], fill="black", font=font)
    return out


# ------------------------------------------------------------------ tables (DocILE)
def table_figure(doc_id: str, tag: str, out: Path, title: str) -> dict:
    import docile_bros as B
    import docile_typer as T
    from row_group import group_rows

    d = B.doc_dict(doc_id)
    spans = T.load_pred(tag)[doc_id]
    gold = [[t, s, e, ""] for t, s, e, _ in B.spans_of(d)]
    font = ImageFont.load_default()
    page = 0
    fields = sorted({sp[0] for sp in spans} | {g[0] for g in gold})
    color = {f: PALETTE[k % len(PALETTE)] for k, f in enumerate(fields)}
    panels = []
    for name, sp in (("prediction (BROS, unseen layout)", spans), ("gold", gold)):
        img = render_page(B.G.D.DATA / "pdfs" / f"{doc_id}.pdf", page)
        dr = ImageDraw.Draw(img)
        rows = group_rows(sp, d, 0.5, merge=False)
        row_of = {ws: k for k, cells in enumerate(rows) for _, ws in cells}
        for f, s, e, _ in sp:
            if d["pages"][s] != page:
                continue
            ws = frozenset(range(s, e))
            for i in range(s, e):
                draw_box(dr, img, d["fboxes"][i], color[f], 2)
            label(dr, img, d["fboxes"][s][:2], f.replace("line_item_", ""), color[f], font)
        ys = sorted({(row_of[ws], min(d["fboxes"][i][1] for i in ws)) for ws in row_of if all(d["pages"][i] == page for i in ws)})
        firsts = {}
        for k, y in ys:
            firsts[k] = min(firsts.get(k, 9), y)
        for k, y in firsts.items():
            dr.ellipse([4, y * img.height - 2, 24, y * img.height + 14], outline="black", fill="white")
            dr.text((8, y * img.height), str(k + 1), fill="black", font=font)
        panels.append(legend(img, [(f.replace("line_item_", ""), color[f]) for f in fields], font))
    side_by_side(panels[0], panels[1], (f"{title}: {name_of(doc_id)} | predicted fields, numbers = rows found", "gold fields and rows"), font).save(out)
    return {"doc": doc_id, "tag": tag, "figure": str(out)}


def name_of(doc_id: str) -> str:
    return doc_id[:8]


def pick_docile(tag: str):
    recs = [json.loads(l) for l in open(ROOT / "results" / "docile_taxonomy" / "per_document.jsonl", encoding="utf-8")]
    recs = [r for r in recs if r["tag"] == tag and r["set"] == "utl" and r["features"]["page_count"] == 1 and 8 <= r["features"]["n_rows"] <= 25]
    f1 = lambda r: 2 * r["lir"][0] / max(1, r["lir"][1] + r["lir"][2])
    recs.sort(key=f1)
    good = recs[-1]
    mid = [r for r in recs if 0.25 <= f1(r) <= 0.45]
    return good["doc"], round(f1(good), 3), (mid[len(mid) // 2]["doc"], round(f1(mid[len(mid) // 2]), 3)) if mid else None


# ------------------------------------------------------------------ forms (VRDU registration)
def form_figure(guid: str, fold: int, out: Path, title: str) -> dict:
    from data.vrdu_loader import load_vrdu

    exs, _, _ = load_vrdu(ROOT / "data" / "vrdu", "registration-form")
    ex = next(e for e in exs if e.guid == guid)
    dump = ROOT / "results" / "vrdu_track1_bros_errdump" / f"UTL_fold{fold}_seed13" / "test_entities.jsonl"
    rec = next(r for r in map(json.loads, open(dump, encoding="utf-8")) if r["guid"] == guid)
    pred, gold = rec["pred"], rec["gold"]
    pset = {(p[0], p[1], p[2]) for p in pred}
    gset = {(g[0], g[1], g[2]) for g in gold}
    font = ImageFont.load_default()
    pages = sorted({ex.page_ids[s] for _, s, e, _ in pred + gold})[:2] or [0]
    imgs = []
    for pg in pages:
        img = render_page(ROOT / "data" / "vrdu" / "registration-form" / "main" / "pdfs" / guid, pg)
        dr = ImageDraw.Draw(img)
        for f, s, e, text in pred:
            if ex.page_ids[s] != pg:
                continue
            ok = (f, s, e) in gset
            for i in range(s, e):
                draw_box(dr, img, [v / 1000 for v in ex.boxes[i]], "#1a9850" if ok else "#d73027", 2)
            label(dr, img, [v / 1000 for v in ex.boxes[s][:2]], f, "#1a9850" if ok else "#d73027", font)
        for f, s, e, text in gold:
            if ex.page_ids[s] == pg and (f, s, e) not in pset:
                for i in range(s, e):
                    draw_box(dr, img, [v / 1000 for v in ex.boxes[i]], "#4575b4", 2, dashed=True)
        imgs.append(img)
    canvas = Image.new("RGB", (sum(i.width for i in imgs) + 8 * (len(imgs) - 1), max(i.height for i in imgs) + 22), "white")
    x = 0
    for i in imgs:
        canvas.paste(i, (x, 22))
        x += i.width + 8
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 4), f"{title}: {guid[:40]} | green = exactly right, red = wrong or extra, blue dashed = missed gold", fill="black", font=font)
    canvas.save(out)
    return {"doc": guid, "fold": fold, "exact": len(pset & gset), "pred": len(pset), "gold": len(gset), "figure": str(out)}


def pick_form(fold: int, good: bool):
    dump = ROOT / "results" / "vrdu_track1_bros_errdump" / f"UTL_fold{fold}_seed13" / "test_entities.jsonl"
    best, best_score = None, None
    for r in map(json.loads, open(dump, encoding="utf-8")):
        p = {(x[0], x[1], x[2]) for x in r["pred"]}
        g = {(x[0], x[1], x[2]) for x in r["gold"]}
        if len(g) < 5:
            continue
        score = len(p & g) - 0.5 * len(p - g) if good else len(g - p) + 0.5 * len(p - g)
        if best_score is None or score > best_score:
            best, best_score = r["guid"], score
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    good, f_good, bad = pick_docile("Nfull_T21")
    report["table_good"] = table_figure(good, "Nfull_T21", out / "example_table_good.png", f"DocILE line items, row F1 {f_good}")
    if bad:
        report["table_failure"] = table_figure(bad[0], "Nfull_T21", out / "example_table_failure.png", f"DocILE line items, row F1 {bad[1]}")
    report["form_good"] = form_figure(pick_form(0, True), 0, out / "example_form_good.png", "VRDU registration form, unseen template Amendment")
    report["form_failure"] = form_figure(pick_form(2, False), 2, out / "example_form_failure.png", "VRDU registration form, unseen template Short-Form")
    (out / "figures.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
