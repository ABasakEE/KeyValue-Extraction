"""One Gradio demo for all four datasets: pick a dataset, a trained checkpoint and an example document; every text on the page follows the choice.

    python demo_all.py                      # http://127.0.0.1:7862
    DEMO_ALLOW_UNTRAINED=1 python demo_all.py   # plumbing test: a missing checkpoint becomes an untrained head (predictions are meaningless, and the page says so)

Datasets (the checkpoints are the ones of the Kaggle notebooks `kaggle_demo_weights_{vrdu,fatura,docile}.ipynb`, seed 13; XFUND is the linking model J1):
  * VRDU registration forms : BROS Track 1, six fields. UTL fold 0 (Amendment held out) and MTL fold 0 (every template seen), N = 200.
  * FATURA invoices         : BROS Track 1, nine fields. Fold 0, trained on K = 40 or K = 2 seen templates.
  * DocILE line items       : BROS Track 1, 19 line-item fields, cells grouped into rows by the anchor rule. N = 1,000 documents.
  * XFUND forms             : BROS joint tagger + biaffine linker (`src/demo.py`): entities and key-value arrows.
Dataset examples use the dataset's own words and boxes (what the model was trained on) and are scored against gold; an uploaded file is read with RapidOCR.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gradio as gr
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from data import fatura_loader as FL
from data.splits import matched_size_protocol
from data.vrdu_loader import VrduExample, build_label_maps, load_vrdu
from models.token_clf import TokenClassifier

DB = importlib.import_module("src.demo_bros")   # not `import demo_bros`: that name is the root launcher
ALLOW_UNTRAINED = os.environ.get("DEMO_ALLOW_UNTRAINED") == "1"
W = ROOT / "results" / "demo_weights"
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#800000", "#808000", "#000075",
           "#f032e6", "#469990", "#bcbd22", "#7f7f7f", "#e67e22", "#16a085", "#c0392b", "#2980b9", "#8e44ad", "#27ae60"]
VRDU_DIR = ROOT / "data" / "vrdu" / "registration-form" / "main"
FATURA_DIR = ROOT / "data" / "fatura" / "raw" / "invoices_dataset_final"
DOCILE_DIR = ROOT / "data" / "docile"
XFUND_NAME = "XFUND forms (key-value linking)"
READER_NAME = "Zero-shot reader: saved Qwen3-VL-8B outputs (VRDU)"
VRDU_NAME, FATURA_NAME, DOCILE_NAME = "VRDU registration forms", "FATURA invoices", "DocILE line items"

# --------------------------------------------------------------------------------------------- texts that follow the dataset
INTRO = {
    VRDU_NAME: (
        "## VRDU registration forms: six fields, three templates\n"
        "Foreign-agent registration forms (the course's main benchmark). The tagger marks six fields: **file date, foreign principle name, registrant name, "
        "registration number, signer name, signer title**. Model: **BROS + Track 1** (384-word windows, overlap 128, soft voting, coordinate jitter, content-box anchoring), "
        "trained on **200 documents**.\n\n"
        "**The point of this tab:** a form model scores about 92 F1 on templates it has seen and loses about 21 points on a template it has not. "
        "Pick the *unseen-template* model, then a form of the held-out template, to see what an unseen template looks like to the model.\n\n"
        "**Two views.** The field tagger (boxes with field names) and, as a second picture, the **key-value linker** (red arrows from each question to its answer). "
        "The linker is the XFUND-trained BROS joint model run on the form's first page: VRDU has no link labels, so these arrows are not scored."),
    FATURA_NAME: (
        "## FATURA invoices: nine fields, 50 templates\n"
        "Synthetic invoices. The tagger marks **title, number, date, due date, seller name, seller address, buyer, sub-total, total**. Model: **BROS + Track 1** trained on invoices of "
        "**K seen templates**; the test invoices come from templates the model has never seen.\n\n"
        "**The point of this tab:** the effect of the number of training templates. With **K = 40** seen templates the tagger is near perfect on new templates (99.9 F1, mean of five folds); "
        "with **K = 2** it reaches 69.2. Our zero-shot reader scores 91.7 whatever K, so it wins below about 5 templates and loses from 10.\n\n"
        "**Two views.** The field tagger and, as a second picture, the **key-value linker** (red arrows from each label to its value; the XFUND-trained BROS joint model, "
        "FATURA has no link labels, so the arrows are not scored)."),
    DOCILE_NAME: (
        "## DocILE invoices: line items (tables)\n"
        "Business documents with a table of line items. The tagger marks the cells of **19 line-item fields** (description, quantity, unit price, amount, ...); an **anchor rule** then "
        "attaches every cell to the row it belongs to: **a row is a link**, drawn as a numbered circle and red arrows from each cell to the anchor cell of its row. Model: **BROS + Track 1** trained on **1,000 documents**; the examples are from "
        "**layouts the model has never seen**.\n\n"
        "**The point of this tab:** a table row is a link between cells. On unseen layouts the tagger reaches **59.6 row F1** (level with the published RoBERTa baseline, 59.8); "
        "with gold cells the anchor rule would reach 94.4, so the loss is in finding and typing cells, not in grouping them."),
    READER_NAME: (
        "## The zero-shot reader: saved outputs of an 8B vision-language model\n"
        "A vision-language model (**Qwen3-VL-8B, 4-bit**) reads the page image and the page's OCR lines (each with an id), is given the six field names with one-line descriptions, "
        "and must **copy the value and cite the line it came from**. Every answer is checked against the cited line (grounding) and becomes a span. **No form of any template was used to train it.**\n\n"
        "**These are preserved outputs, not a live run.** The 8B model needs a GPU (it ran on Kaggle T4s) and does not run on a laptop, so this tab replays its saved answers on "
        "300 VRDU forms that nobody tuned on. The trained tagger's predictions (BROS Track 1, seed 13) are saved too, so you can compare them with the reader and with the gold.\n\n"
        "**The point of this tab:** the two make different mistakes. The reader wins where the field's *wording* varies (`foreign_principle_name`, the registrant), BROS where position and span boundaries matter."),
    XFUND_NAME: (
        "## XFUND forms: entities and key-value links\n"
        "Forms in five languages. The model tags **questions, answers and headers** and a trained biaffine head links each **question to its answer** (red arrows). "
        "Model: **BROS joint tagger + linker**. End to end, on 250 validation forms mostly from templates not seen in training: **53.4 link F1** (LiLT backbone: 47.4), entity F1 80.2.\n\n"
        "**The point of this tab:** relative layout in the encoder helps linking; a multilingual text stream did not."),
}

MODELS = {
    VRDU_NAME: [
        dict(key="UTL", label="Unseen-template model: Amendment held out (UTL fold 0)", paths=[W / "vrdu/UTL_fold0_seed13/best_model.pt", ROOT / "results/vrdu_track1_bros_demo/UTL_fold0_seed13/best_model.pt"],
             text="**Trained on 200 documents of Dissemination_Report and Short-Form only. Amendment never appeared in training.** The examples below are Amendment forms, so every one is a new template to this model. "
                  "**This checkpoint (seed 13): 82.8 F1 on the held-out Amendment forms.** BROS Track 1 averages 83.4 on this fold over three seeds, the best of the three folds; the mean over the three folds is **71.3**."),
        dict(key="MTL", label="Seen-template model (MTL fold 0)", paths=[W / "vrdu/MTL_fold0_seed13/best_model.pt", ROOT / "results/vrdu_track1_bros_demo/MTL_fold0_seed13/best_model.pt"],
             text="**Trained on 200 documents drawn from all three templates.** The examples below are forms it did not train on, but of templates it has seen. **This checkpoint (seed 13): 91.9 F1** on its test documents; BROS Track 1 averages about 92.4 in this setting."),
    ],
    FATURA_NAME: [
        dict(key="K40", label="Trained on 40 seen templates (K = 40)", paths=[W / "fatura/fold0/seed_13/best_model.pt"],
             text="**Fold 0, trained on 200 invoices of the 40 seen templates.** Held-out (unseen) templates: 2, 4, 7, 10, 12, 25, 32, 39, 49, 50. **This checkpoint: 99.9 F1** on the unseen-template test invoices of fold 0 (99.9 over five folds)."),
        dict(key="K2", label="Trained on 2 seen templates (K = 2)", paths=[W / "fatura/K2/fold0/seed_13/best_model.pt"],
             text="**Fold 0, trained on 20 invoices of only 2 seen templates** (10 each). **This checkpoint: 60.7 F1** on the unseen-template test invoices of fold 0 (69.2 mean over five folds and two seeds); this is the data-starved case in which the zero-shot reader (91.7) is better."),
    ],
    DOCILE_NAME: [
        dict(key="S0", label="Trained on 1,000 documents (split seed 0, the poster's setting)", paths=[W / "docile/N1000_S0/seed_13/best_model.pt"],
             text="**N = 1,000 training documents, split seed 0.** Test layouts share no layout or vendor with any training document. Row F1 on unseen layouts: **59.6** (mean of three split seeds)."),
        dict(key="S1", label="Trained on 1,000 documents (split seed 1, a second model)", paths=[W / "docile/N1000_S1/seed_13/best_model.pt"],
             text="**N = 1,000 training documents, split seed 1.** A second, independent model with the same recipe; same test layouts."),
    ],
    READER_NAME: [
        dict(key="V2A2", label="Qwen3-VL-8B reader, prompt v2 (saved outputs)", paths=[W / "reader_vrdu.json"],
             text="**Qwen3-VL-8B (4-bit), prompt v2, one call per page and field, greedy.** On the 300 forms: reader 76.5 F1 against 70.0 for BROS (+6.5 [4.0, 9.1]); with the latest-page rule on ties 80.9; "
                  "a vote of reader, reader without the image and BROS reaches 82.4. The per-form scores shown here are per-document, so they average differently from those pooled numbers."),
    ],
    XFUND_NAME: [
        dict(key="J1", label="BROS joint (J1, seed 42)", paths=[ROOT / "results/bros_joint/J1/seed_42/best_model.pt"],
             text="**BROS joint tagger + biaffine linker, seed 42.** End-to-end link F1 on the 250 validation forms: 53.4 (mean of three seeds)."),
    ],
}

NOTES = {
    VRDU_NAME: "Dataset examples use the benchmark's own words and boxes; the page shows exact-span F1 against gold. An uploaded form is read with RapidOCR.",
    FATURA_NAME: "FATURA is synthetic and its text blocks are the annotations (perfect OCR), so both the tagger and the reader are favoured. An uploaded invoice is read with RapidOCR.",
    DOCILE_NAME: "Dataset examples use DocILE's own OCR words; row F1 uses the DocILE line-item rule against the gold rows of the document. An uploaded invoice is read with RapidOCR (no gold).",
    READER_NAME: "Saved outputs (not a live model run). Boxes come from the benchmark's own words; the reader's answers were grounded to those words.",
    XFUND_NAME: "Dataset examples are XFUND validation forms read with the demo's OCR pipeline; the slider sets the link confidence threshold.",
}

_C: dict[str, Any] = {}


def _short(f: str) -> str:
    return f.replace("line_item_", "")


# --------------------------------------------------------------------------------------------- checkpoints
def find_ckpt(dataset: str, key: str) -> Path | None:
    spec = next(m for m in MODELS[dataset] if m["key"] == key)
    return next((p for p in spec["paths"] if p.exists()), None)


def label_maps(dataset: str):
    if dataset in (VRDU_NAME, READER_NAME):
        labels, _, id2label = build_label_maps(VRDU_DIR / "meta.json")
        return labels, id2label
    if dataset == FATURA_NAME:
        return FL.LABELS, FL.ID2LABEL
    import docile_bros as DBR
    return DBR.LABELS, DBR.ID2LABEL


def field_colors(dataset: str) -> dict[str, str]:
    _, id2label = label_maps(dataset)
    fields = sorted({l.split("-", 1)[1] for l in id2label.values() if l != "O"})
    return {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(fields)}


def legend_md(dataset: str) -> str:
    if dataset == XFUND_NAME:
        return ("**Legend:** <span style='color:#2980b9'>■</span> question  <span style='color:#27ae60'>■</span> answer  "
                "<span style='color:#e67e22'>■</span> header  <span style='color:#c0392b'>→</span> predicted link (question to answer)")
    return "**Fields:** " + "  ".join(f"<span style='color:{c}'>■</span> {_short(f)}" for f, c in field_colors(dataset).items())


def get_model(dataset: str, key: str):
    ck = find_ckpt(dataset, key)
    ident = (dataset, key, str(ck))
    if ident in _C:
        return _C[ident]
    labels, id2label = label_maps(dataset)
    clf = TokenClassifier("bros", len(labels), id2label, max_length=512)
    trained = ck is not None
    if trained:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        clf.model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    elif not ALLOW_UNTRAINED:
        spec = next(m for m in MODELS[dataset] if m["key"] == key)
        raise gr.Error(f"No checkpoint for this model yet. Expected: {spec['paths'][0].relative_to(ROOT)} (train it with the Kaggle notebook and download it).")
    clf.model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf.to(dev)
    _C[ident] = (clf, id2label, dev, trained, ck)
    return _C[ident]


def model_choices(dataset: str):
    out = []
    for m in MODELS[dataset]:
        have = find_ckpt(dataset, m["key"]) is not None
        out.append((m["label"] + ("" if have else "   [no checkpoint yet]"), m["key"]))
    return out


# --------------------------------------------------------------------------------------------- examples (what appears in the example list)
def vrdu_protocol():
    if "vrdu" not in _C:
        exs, _, _ = load_vrdu(ROOT / "data" / "vrdu", "registration-form")
        sp = matched_size_protocol(exs, n_train=200, seed=13)
        _C["vrdu"] = ({e.guid: e for e in exs}, sp["UTL"][0], sp["MTL"][0])
    return _C["vrdu"]


def fatura_pick(key: str):
    """(unseen-template examples, seen-template examples) as (label, invoice name) for the fold-0 checkpoint of `key`."""
    held = FL.folds()[0]
    if key == "K40":
        train, val = FL.train_val_sample(held, 13, 0)
        seen_t = sorted({int(n.split("_")[0][8:]) for n in train})
    else:
        train, val = FL.starved_sample(held, 13, 0, 2)
        seen_t = sorted({int(n.split("_")[0][8:]) for n in train})
    used = set(train) | set(val)
    unseen = [(f"Template {t}: unseen template", f"Template{t}_Instance{(37 * t) % 200}") for t in (2, 4, 7, 10, 12, 25)]
    seen = []
    for t in seen_t[:3]:
        name = next(f"Template{t}_Instance{i}" for i in range(200) if f"Template{t}_Instance{i}" not in used)
        seen.append((f"Template {t}: seen template (not a training invoice)", name))
    return unseen, seen


def docile_pick():
    if "docile_pick" not in _C:
        import json
        import docile_bros as DBR
        sp = json.loads((ROOT / "results" / "docile_rows" / "splits.json").read_text(encoding="utf-8"))
        out = []
        for i in sp["test"]:
            d = DBR.doc_dict(i)
            if 3 <= len(d["rows"]) <= 14 and len(d["words"]) <= 420:
                out.append((f"Document {i[:8]}: {len(d['rows'])} gold rows, unseen layout", i))
            if len(out) == 8:
                break
        _C["docile_pick"] = out
    return _C["docile_pick"]


def xfund_pick():
    """The cached XFUND forms, best link F1 first, labelled with J1's own score on each (scripts/demo_export_pdfs.py writes the files)."""
    import json
    cur = W / "examples_xfund.json"
    base = ROOT / "demo_pdfs" / "tagger" / "xfund"
    if not cur.exists() or not base.exists():
        old = ROOT / "demo_pdfs" / "showcase" / "xfund"
        return [(p.stem, str(p)) for p in sorted(old.glob("*.pdf"))]
    out = []
    for c in json.loads(cur.read_text(encoding="utf-8")):
        stem = Path(c["source"]).stem
        stem = stem.rsplit("_", 1)[0] if stem[-2:] in ("_g", "_m", "_b") else stem
        role = "good case" if c["link_f1"] >= 60 else ("moderate case" if c["link_f1"] >= 25 else "failure case")
        f = next(base.glob(f"{stem}_*.pdf"), None)
        if f:
            out.append((f"{stem.split('_')[0].upper()}: {stem}; {role}, J1 link F1 {c['link_f1']:.0f}% on this form", str(f)))
    return out


def curated(dataset_tag: str, key: str):
    """Examples picked by scripts/demo_curate_examples.py (best / typical / worst for this checkpoint), or None."""
    path = W / f"examples_{dataset_tag}.json"
    if not path.exists():
        return None
    import json
    return json.loads(path.read_text(encoding="utf-8")).get(key)


ROLE = {"best": "good case", "typical": "typical case", "worst": "failure case"}


def reader_data():
    if "reader" not in _C:
        import json
        _C["reader"] = json.loads((W / "reader_vrdu.json").read_text(encoding="utf-8"))
    return _C["reader"]


def example_choices(dataset: str, key: str):
    try:
        if dataset == READER_NAME:
            return [(f"{e['template']}: {e['guid'][:40]} ({e['why']}; reader {e['reader_f1']:.0f}%, BROS {e['bros_f1']:.0f}% on this form)", e["guid"]) for e in reader_data()["examples"]]
        cur = curated("vrdu", key) if dataset == VRDU_NAME else None
        if cur:
            kind = "unseen template" if key == "UTL" else "seen template"
            return [(f"{c['template']}: {c['guid'][:40]} ({kind}; {ROLE[c['role']]}, this checkpoint {c['f1']:.0f}% F1)", c["guid"]) for c in cur]
        if dataset == VRDU_NAME:
            by_guid, utl, mtl = vrdu_protocol()
            docs = utl.test if key == "UTL" else mtl.test
            picked = []
            for tmpl in sorted({d.template_id for d in docs}):
                picked += [d for d in docs if d.template_id == tmpl][:6 if key == "UTL" else 2]
            kind = "unseen template" if key == "UTL" else "seen template"
            return [(f"{d.template_id}: {d.guid[:48]} ({kind})", d.guid) for d in picked]
        if dataset == FATURA_NAME:
            unseen, seen = fatura_pick(key)
            return [(l, v) for l, v in unseen + seen]
        if dataset == DOCILE_NAME:
            cur = curated("docile", key)
            if cur:
                seen = curated("docile", key + "_seen") or []
                return ([(f"Document {c['guid'][:8]}: unseen layout; {ROLE[c['role']]}, this checkpoint row F1 {c['f1']:.0f}%", c["guid"]) for c in cur]
                        + [(f"Document {c['guid'][:8]}: SEEN layout; {ROLE[c['role']]}, this checkpoint row F1 {c['f1']:.0f}%", c["guid"]) for c in seen])
            return docile_pick()
        return xfund_pick()
    except Exception as e:  # missing data folder: the dropdown stays usable for uploads
        print(f"[!] examples for {dataset}: {type(e).__name__}: {e}")
        return []


# --------------------------------------------------------------------------------------------- loading one document
def render_pdf(path: Path, max_pages: int = 6) -> list[Image.Image]:
    return DB.render_pages(path)[:max_pages]


def load_example(dataset: str, value: str):
    """(VrduExample with gold tags, page images, doc dict for rows or None)."""
    if dataset == VRDU_NAME:
        by_guid, _, _ = vrdu_protocol()
        ex = by_guid[value]
        return ex, render_pdf(VRDU_DIR / "pdfs" / value), None
    if dataset == FATURA_NAME:
        doc = FL.load_docs(FATURA_DIR, {int(value.split("_")[0][8:])})[value]
        img = Image.open(FATURA_DIR / "images" / f"{value}.jpg").convert("RGB")
        return FL.to_example(doc), [img], None
    import docile_bros as DBR
    d = DBR.doc_dict(value)
    ex = DBR.to_example(value, d)
    return ex, render_pdf(DOCILE_DIR / "pdfs" / f"{value}.pdf"), d


def load_upload(path: Path):
    pages = DB.render_pages(path)
    ex = DB.words_from_ocr(pages)
    d = {"words": ex.words, "fboxes": [[v / 1000 for v in b] for b in ex.boxes], "pages": ex.page_ids, "rows": []}
    return ex, pages, d


# --------------------------------------------------------------------------------------------- drawing
def _font(size=16):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _arrow(dr, p0, p1, color="#c0392b", width=3, head=12):
    import math

    dr.line([p0, p1], fill=color, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    pts = [p1] + [(p1[0] - head * math.cos(ang + d), p1[1] - head * math.sin(ang + d)) for d in (math.pi / 6, -math.pi / 6)]
    dr.polygon(pts, fill=color)


def row_links(spans, d, rows):
    """Cell -> row-anchor arrows as (page, (x0, y0), (x1, y1)) in page fractions; the anchor field of a page is the one with most spans (the anchor rule)."""
    from collections import Counter

    counts: dict[int, Counter] = {}
    for f, a, b in spans:
        counts.setdefault(d["pages"][a], Counter())[f] += 1
    anchor = {pg: max(c, key=lambda f: (c[f], f)) for pg, c in counts.items()}
    centre = lambda ws: (sum((d["fboxes"][i][0] + d["fboxes"][i][2]) / 2 for i in ws) / len(ws), sum((d["fboxes"][i][1] + d["fboxes"][i][3]) / 2 for i in ws) / len(ws))
    out = []
    for cells in rows:
        pg = d["pages"][min(min(ws) for _, ws in cells)]
        anc = [ws for f, ws in cells if f == anchor.get(pg)]
        if not anc:
            continue
        target = centre(min(anc, key=min))
        out += [(pg, centre(ws), target) for f, ws in cells if f != anchor.get(pg)]
    return out


def draw(pages, ex, spans, gold_spans, colors, rows=None, links=None):
    out = [p.copy() for p in pages]
    font = _font(max(14, min(24, pages[0].width // 60)))
    pset = set(spans)
    for typ, a, b in sorted(gold_spans - pset):            # missed gold: dashed blue
        pi = ex.page_ids[a] if ex.page_ids else 0
        if pi >= len(out):
            continue
        w, h = out[pi].size
        dr = ImageDraw.Draw(out[pi])
        for i in range(a, b):
            x0, y0, x1, y1 = (ex.boxes[i][0] * w / 1000, ex.boxes[i][1] * h / 1000, ex.boxes[i][2] * w / 1000, ex.boxes[i][3] * h / 1000)
            for x in range(int(x0), int(x1), 8):
                dr.line([x, y0, min(x + 4, x1), y0], fill="#1f77b4", width=2)
                dr.line([x, y1, min(x + 4, x1), y1], fill="#1f77b4", width=2)
    for typ, a, b in spans:
        color = colors.get(typ, "#555555")
        pi = ex.page_ids[a] if ex.page_ids else 0
        if pi >= len(out):
            continue
        w, h = out[pi].size
        dr = ImageDraw.Draw(out[pi])
        for i in range(a, b):
            dr.rectangle([ex.boxes[i][0] * w / 1000, ex.boxes[i][1] * h / 1000, ex.boxes[i][2] * w / 1000, ex.boxes[i][3] * h / 1000], outline=color, width=3)
        tag = _short(typ) + ("" if not gold_spans else (" ok" if (typ, a, b) in gold_spans else " wrong"))
        x0, y0 = ex.boxes[a][0] * w / 1000, ex.boxes[a][1] * h / 1000
        tw = dr.textlength(tag, font=font)
        dr.rectangle([x0, max(0, y0 - font.size - 4), x0 + tw + 6, max(0, y0 - 2)], fill="white")
        dr.text((x0 + 3, max(0, y0 - font.size - 4)), tag, fill=color, font=font)
    for pg, c0, c1 in links or []:
        if pg < len(out):
            w, h = out[pg].size
            _arrow(ImageDraw.Draw(out[pg]), (c0[0] * w, c0[1] * h), (c1[0] * w, c1[1] * h))
    if rows:
        for k, (pi, y) in enumerate(rows):
            if pi >= len(out):
                continue
            w, h = out[pi].size
            dr = ImageDraw.Draw(out[pi])
            r = font.size
            cy = y * h
            dr.ellipse([4, cy - r, 4 + 2 * r, cy + r], outline="black", fill="white", width=2)
            dr.text((4 + r - dr.textlength(str(k + 1), font=font) / 2, cy - font.size / 2), str(k + 1), fill="black", font=font)
    return out


# --------------------------------------------------------------------------------------------- one run
def f1_text(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), tp


def doc_path(dataset, example, upload):
    if upload is not None:
        return str(upload if isinstance(upload, str) else upload.name)
    if dataset == VRDU_NAME:
        return str(VRDU_DIR / "pdfs" / example)
    if dataset == FATURA_NAME:
        return str(FATURA_DIR / "images" / f"{example}.jpg")
    return str(DOCILE_DIR / "pdfs" / f"{example}.pdf")


def run_links(path, threshold):
    os.environ.setdefault("DEMO_CKPT_DIR", str(ROOT / "results" / "bros_joint" / "J1" / "seed_42"))
    D = importlib.import_module("src.demo")
    return D.process_uploaded_document(path, float(threshold), True, True, True)


def run_tagger(dataset, key, example, upload, link_on=False, threshold=0.5):
    spec = next(m for m in MODELS[dataset] if m["key"] == key)
    clf, id2label, dev, trained, ck = get_model(dataset, key)
    if upload is not None:
        ex, pages, d = load_upload(Path(upload if isinstance(upload, str) else upload.name))
        has_gold, source = False, "RapidOCR on the uploaded file"
    else:
        if not example:
            raise gr.Error("Pick an example document or upload a file.")
        ex, pages, d = load_example(dataset, example)
        has_gold, source = True, "the dataset's own words and boxes (what the model was trained on)"
    spans = DB.spans_of(DB.tag_document(clf, dev, ex), id2label)
    gold = set(DB.spans_of(ex.ner_tags, id2label)) if has_gold else set()
    colors = field_colors(dataset)

    lines = [spec["text"], "",
             f"**Model:** `{ck.relative_to(ROOT) if ck else 'UNTRAINED head'}`" if trained else
             "**Model: UNTRAINED classifier head. This is a plumbing test, the predictions are meaningless.**",
             f"**Input:** {source}; {len(ex.words)} words on {len(pages)} page(s)."]
    row_marks, arrows = None, None
    if dataset == DOCILE_NAME:
        from lir_metric import f1 as lir_f1
        from row_group import group_rows
        import docile_bros as DBR
        span_list = [[t, a, b, ""] for t, a, b in spans]
        rows = [r for r in group_rows(span_list, d, 0.5, merge=False) if r]   # an anchor whose own cell went to a nearer anchor leaves an empty row
        rows = sorted(rows, key=lambda cells: (min(d["pages"][min(ws)] for _, ws in cells), min(sum(d["fboxes"][i][1] for i in ws) / len(ws) for _, ws in cells)))
        marks = []
        recs = []
        for k, cells in enumerate(rows):
            first = min(cells, key=lambda c: (d["pages"][min(c[1])], d["fboxes"][min(c[1])][1]))
            marks.append((d["pages"][min(first[1])], d["fboxes"][min(first[1])][1]))
            rec = {"Row": k + 1}
            for f, ws in sorted(cells, key=lambda c: min(c[1])):
                rec[_short(f)] = (rec.get(_short(f), "") + " " + " ".join(ex.words[i] for i in sorted(ws))).strip()
            recs.append(rec)
        table = pd.DataFrame(recs)
        row_marks = marks
        arrows = row_links(spans, d, rows)
        if has_gold and d["rows"]:
            c = DBR.counts(group_rows(span_list, d, 0.5, merge=False), d)
            lines.append(f"**Against gold (DocILE line-item rule):** row F1 **{100 * lir_f1(c['lir']):.0f}%**, cell F1 {100 * lir_f1(c['cell']):.0f}%; "
                         f"{len(rows)} rows predicted, {len(d['rows'])} gold rows.")
        lines.append("Numbered circles are the rows found by the anchor rule; red arrows link each cell to the anchor cell of its row. Dashed blue boxes are gold cells the tagger missed.")
    else:
        recs = [{"Field": _short(t), "Extracted text": " ".join(ex.words[a:b]), "Page": (ex.page_ids[a] + 1) if ex.page_ids else 1,
                 "Matches gold": ("yes" if (t, a, b) in gold else "no") if has_gold else "-"} for t, a, b in spans]
        table = pd.DataFrame(recs)
        if has_gold:
            f1, tp = f1_text(set(spans), gold)
            missed = sorted({_short(t) for t, _, _ in gold - set(spans)})
            lines.append(f"**Against gold (exact span):** entity F1 **{100 * f1:.0f}%**: {tp} of {len(gold)} gold fields found, {len(set(spans) - gold)} extra"
                         + (f"; missed: {', '.join(missed)}." if missed else "."))
            lines.append("Labels say *ok* when the span equals gold and *wrong* otherwise; dashed blue boxes are gold fields the tagger missed.")
        else:
            lines.append("No gold labels for an uploaded file.")
    lines += ["", NOTES[dataset]]
    shown = [(im, f"Field tagger, page {i + 1}") for i, im in enumerate(draw(pages, ex, spans, gold, colors, row_marks, arrows))]
    link_table = pd.DataFrame()
    if link_on and dataset != DOCILE_NAME:
        path = doc_path(dataset, example, upload)
        try:
            img, link_table, meta = run_links(path, threshold)
            shown.append((img, "Key-value linker (XFUND-trained BROS joint, first page): question to answer"))
            lines.append(f"**Key-value linker:** {meta.get('num_entities', 0)} entities and {meta.get('num_links', 0)} links on the first page at threshold {threshold}. "
                         "Trained on XFUND, applied here without any training on this dataset; no gold links exist, so these are not scored.")
        except Exception as e:  # the linker is a second view: never lose the tagger result because it failed
            lines.append(f"**Key-value linker unavailable:** {type(e).__name__}: {str(e)[:160]}")
    return shown, table, "\n\n".join(lines), legend_md(dataset), link_table


def run_xfund(example, upload, threshold):
    os.environ.setdefault("DEMO_CKPT_DIR", str(ROOT / "results" / "bros_joint" / "J1" / "seed_42"))
    D = importlib.import_module("src.demo")
    path = upload if upload is not None else example
    if not path:
        raise gr.Error("Pick an example form or upload one.")
    img, df, meta = D.process_uploaded_document(str(path if isinstance(path, str) else path.name), float(threshold), True, True, True)
    lines = [MODELS[XFUND_NAME][0]["text"], "", f"**Model:** `{D.CKPT_PATH.relative_to(ROOT) if D.CKPT_PATH.exists() else 'none'}`",
             f"**Found:** {meta.get('num_entities', 0)} entities and {meta.get('num_links', 0)} key-value links at threshold {threshold}.", "", NOTES[XFUND_NAME]]
    ents = pd.DataFrame([{"Label": e.get("label"), "Text": e.get("text")} for e in meta.get("entities", [])])
    return [(img, "Entities and key-value links (BROS joint)")], ents, "\n\n".join(lines), legend_md(XFUND_NAME), df


PROMPT_V2 = """<Document>
{every OCR line of the page, each followed by its id, e.g. "Date of signature 62|15"}
</Document>
<Task>
From the document, extract the value of the field "{field}" ({one-line description}). Every document line ends with its identifier.
Copy the value exactly from ONE line and write it followed by that line's identifier, for example "Jane Q. Public 42|17".
Copy only the value itself: not the label in front of it and not other words on the same line (for a person's name, leave out their title or position).
If the value spans several lines, write one 'text identifier' per line. If the field does not occur on this page, answer NONE.
</Task>
ANSWER: """


def run_reader(example):
    ent = next(e for e in reader_data()["examples"] if e["guid"] == example)
    by_guid, _, _ = vrdu_protocol()
    ex = by_guid[example]
    pages = render_pdf(VRDU_DIR / "pdfs" / example)
    colors = field_colors(VRDU_NAME)
    tup = lambda spans: [(f, a, b) for f, a, b, _ in spans]
    gold, reader, bros = set(tup(ent["gold"])), tup(ent["reader"]), tup(ent["bros"])
    used = sorted({ex.page_ids[a] for _, a, _ in list(gold) + reader + bros if a < len(ex.page_ids)})
    shown = []
    for name, spans in (("Zero-shot reader (Qwen3-VL-8B, saved)", reader), ("BROS Track 1 (saved, seed 13)", bros)):
        for pi, im in zip(range(len(pages)), draw(pages, ex, spans, gold, colors)):
            if pi in used:
                shown.append((im, f"{name}, page {pi + 1}"))
    txt = lambda spans: {f: t for f, _, _, t in spans}
    r_t, b_t, g_t = txt(ent["reader"]), txt(ent["bros"]), txt(ent["gold"])
    recs = []
    for f in sorted(set(r_t) | set(b_t) | set(g_t) | {x["field"] for x in ent["raw_answers"] if x["answer"].strip().upper() != "NONE"}):
        raw = "; ".join(f"p{x['page']}: {x['answer'].strip()[:60]}" for x in ent["raw_answers"] if x["field"] == f and x["answer"].strip().upper() != "NONE")
        recs.append({"Field": f, "Reader's raw answer (page: text, cited line)": raw or "(NONE on every page)", "Reader span": r_t.get(f, ""), "BROS span": b_t.get(f, ""),
                     "Gold": g_t.get(f, ""), "Reader right": "yes" if any(ff == f and (ff, a, b) in gold for ff, a, b, _ in ent["reader"]) else "no",
                     "BROS right": "yes" if any(ff == f and (ff, a, b) in gold for ff, a, b, _ in ent["bros"]) else "no"})
    st = reader_data()["stats"]
    lines = [MODELS[READER_NAME][0]["text"], "", f"**This form:** reader {ent['reader_f1']:.0f}% F1, BROS {ent['bros_f1']:.0f}% ({ent['why']}). "
             f"Over all {st['n_docs']} forms the per-form mean is {st['reader_mean_f1']}% for the reader and {st['bros_seed13_mean_f1']}% for BROS seed 13.",
             "**Source of these outputs:** " + reader_data()["model"] + ".", "", NOTES[READER_NAME], "",
             "<details><summary>The exact prompt the reader was given (one call per page and field)</summary>\n\n```\n" + PROMPT_V2 + "\n```\n</details>"]
    return shown, pd.DataFrame(recs), "\n\n".join(lines), legend_md(READER_NAME), pd.DataFrame()


def run(dataset, key, example, upload, threshold, link_on):
    if dataset == READER_NAME:
        return run_reader(example)
    if dataset == XFUND_NAME:
        return run_xfund(example, upload, threshold)
    return run_tagger(dataset, key, example, upload, link_on, threshold)


# --------------------------------------------------------------------------------------------- interface
def on_dataset(dataset):
    choices = model_choices(dataset)
    key = choices[0][1]
    spec = next(m for m in MODELS[dataset] if m["key"] == key)
    ex = example_choices(dataset, key)
    return (INTRO[dataset], gr.update(choices=choices, value=key), spec["text"],
            gr.update(choices=ex, value=ex[0][1] if ex else None), gr.update(visible=dataset not in (DOCILE_NAME, READER_NAME)), None, legend_md(dataset),
            gr.update(visible=dataset in (VRDU_NAME, FATURA_NAME)))


def on_model(dataset, key):
    spec = next(m for m in MODELS[dataset] if m["key"] == key)
    ex = example_choices(dataset, key)
    return spec["text"], gr.update(choices=ex, value=ex[0][1] if ex else None)


def create_demo(default_dataset: str = VRDU_NAME) -> gr.Blocks:
    with gr.Blocks(title="Cross-template key-value extraction: four datasets") as demo:
        gr.Markdown("# Cross-Template Key-Value Extraction (IE 643)\n"
                    "One model recipe (BROS with relative layout), four datasets. Pick a dataset; the model choices, the examples and every text below follow it.")
        dataset = gr.Dropdown(label="Dataset", choices=[VRDU_NAME, FATURA_NAME, DOCILE_NAME, XFUND_NAME, READER_NAME], value=default_dataset)
        intro = gr.Markdown(INTRO[default_dataset])
        with gr.Row():
            with gr.Column(scale=4):
                mchoices = model_choices(default_dataset)
                model = gr.Dropdown(label="Model (trained checkpoint)", choices=mchoices, value=mchoices[0][1])
                model_text = gr.Markdown(MODELS[default_dataset][0]["text"])
                ex0 = example_choices(default_dataset, mchoices[0][1])
                example = gr.Dropdown(label="Example document", choices=ex0, value=ex0[0][1] if ex0 else None)
                upload = gr.File(label="...or upload your own document (PDF or image; read with RapidOCR)", file_types=[".pdf", ".png", ".jpg", ".jpeg"])
                link_cb = gr.Checkbox(label="Also run the key-value linker (XFUND-trained; second picture, not scored)", value=True, visible=default_dataset in (VRDU_NAME, FATURA_NAME))
                threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Link confidence threshold", visible=default_dataset not in (DOCILE_NAME, READER_NAME))
                btn = gr.Button("Run the tagger", variant="primary", size="lg")
                legend = gr.Markdown(legend_md(default_dataset))
            with gr.Column(scale=6):
                gallery = gr.Gallery(label="Predictions on the document", columns=2, height=640, object_fit="contain")
                summary = gr.Markdown()
                table = gr.Dataframe(label="Extracted fields / entities / rows", interactive=False, wrap=True)
                link_table = gr.Dataframe(label="Key-value pairs found by the linker", interactive=False, wrap=True)
        dataset.change(on_dataset, inputs=[dataset], outputs=[intro, model, model_text, example, threshold, upload, legend, link_cb])
        model.change(on_model, inputs=[dataset, model], outputs=[model_text, example])
        btn.click(run, inputs=[dataset, model, example, upload, threshold, link_cb], outputs=[gallery, table, summary, legend, link_table])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--server_name", default="127.0.0.1")
    ap.add_argument("--dataset", default=VRDU_NAME, choices=[VRDU_NAME, FATURA_NAME, DOCILE_NAME, XFUND_NAME, READER_NAME])
    a = ap.parse_args()
    for ds in (VRDU_NAME, FATURA_NAME, DOCILE_NAME, XFUND_NAME, READER_NAME):
        print(f"[+] {ds}: " + ", ".join(f"{m['key']}={'ok' if find_ckpt(ds, m['key']) else 'MISSING'}" for m in MODELS[ds]))
    create_demo(a.dataset).launch(server_name=a.server_name, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
