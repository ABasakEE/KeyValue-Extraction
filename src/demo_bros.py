"""Gradio demo for the best entity model so far: BROS + Track 1 (384-word windows, soft voting, content-box anchoring).

It extracts the six VRDU Registration fields (file_date, foreign_principle_name, registrant_name, registration_num,
signer_name, signer_title). It does NOT do key-value linking (that is demo.py).

    python demo_bros.py                          # looks for a checkpoint (see find_checkpoint)
    python demo_bros.py --ckpt results/vrdu_track1_bros_demo/MTL_fold0_seed13/best_model.pt

Inputs are read the way the model was trained: all pages, words with boxes on a 0-1000 grid per page, concatenated.
If the uploaded file is a VRDU document (matched by file name) the dataset's own OCR words are used, which reproduces the
benchmark input exactly and lets the app score the prediction against the gold fields. Anything else is OCR'd with RapidOCR.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF
    import fitz

from data.vrdu_loader import VrduExample, build_label_maps, chunk_vrdu_example, load_vrdu
from evaluate import _extract_spans
from models.token_clf import TokenClassifier
from train import make_collator, predict

DATA_ROOT = ROOT / "data" / "vrdu"
FIELD_COLORS = {
    "file_date": "#e67e22", "foreign_principle_name": "#8e44ad", "registrant_name": "#2980b9",
    "registration_num": "#27ae60", "signer_name": "#c0392b", "signer_title": "#16a085",
}
WINDOW, OVERLAP = 384, 128           # Track 1 settings (configs/vrdu_track1_bros.yaml)
MAX_PAGES = 6

_STATE: dict[str, Any] = {}


# ------------------------------------------------------------------ checkpoint and model
def find_checkpoint(explicit: str | None = None) -> Path | None:
    """--ckpt, then $DEMO_BROS_CKPT, then the usual output folders (MTL fold first: it has seen every template)."""
    cands = [explicit, os.environ.get("DEMO_BROS_CKPT"), ROOT / "checkpoints" / "bros_track1.pt"]
    cands += sorted((ROOT / "results" / "vrdu_track1_bros_demo").glob("MTL_fold*_seed*/best_model.pt"))
    cands += sorted((ROOT / "results" / "vrdu_track1_bros_demo").glob("UTL_fold*_seed*/best_model.pt"))
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def get_model(ckpt: Path | None):
    if "model" in _STATE:
        return _STATE["model"]
    labels, _, id2label = build_label_maps(DATA_ROOT / "registration-form" / "main" / "meta.json")
    clf = TokenClassifier("bros", len(labels), id2label, max_length=512)
    trained = False
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        clf.model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        trained = True
    elif os.environ.get("DEMO_BROS_ALLOW_UNTRAINED") != "1":
        raise gr.Error("No BROS checkpoint found. Train one on Kaggle (see demo_bros.md) and pass --ckpt, "
                       "or set DEMO_BROS_CKPT.")
    clf.model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf.to(dev)
    _STATE.update(model=(clf, id2label, dev, trained, ckpt))
    return _STATE["model"]


# ------------------------------------------------------------------ input: pages, words, boxes
def _ocr():
    if "ocr" not in _STATE:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _STATE["ocr"] = RapidOCR()
        except Exception:
            _STATE["ocr"] = None
    return _STATE["ocr"]


def render_pages(path: Path) -> list[Image.Image]:
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        out = []
        for page in list(doc)[:MAX_PAGES]:
            r = page.rect
            zoom = min(1100.0 / max(1, r.width), 1100.0 / max(1, r.height), 1.6)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            out.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        return out
    return [Image.open(path).convert("RGB")]


def words_from_ocr(pages: list[Image.Image]) -> VrduExample:
    """RapidOCR gives line boxes only: cut each line into words and split its box in proportion to word length."""
    ocr = _ocr()
    if ocr is None:
        raise gr.Error("RapidOCR is not installed; only VRDU documents (dataset OCR) can be processed.")
    words, boxes, page_ids = [], [], []
    for pi, img in enumerate(pages):
        w, h = img.size
        result, _ = ocr(np.array(img))
        for pts, text, _score in result or []:
            toks = str(text).split()
            if not toks:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            total = sum(len(t) for t in toks) + len(toks) - 1
            cx = x0
            for t in toks:
                bx1 = cx + (x1 - x0) * len(t) / total
                words.append(t)
                boxes.append([max(0, min(1000, int(1000 * v))) for v in (cx / w, y0 / h, bx1 / w, y1 / h)])
                page_ids.append(pi)
                cx += (x1 - x0) * (len(t) + 1) / total
    if not words:
        raise gr.Error("OCR found no text on the uploaded document.")
    return VrduExample(guid="upload", words=words, boxes=boxes, ner_tags=[0] * len(words), template_id="unknown",
                       page_ids=page_ids, n_pages=len(pages))


def vrdu_index() -> dict[str, VrduExample]:
    if "vrdu" not in _STATE:
        try:
            examples, _, _ = load_vrdu(DATA_ROOT, "registration-form")
            _STATE["vrdu"] = {e.guid: e for e in examples}
        except Exception:
            _STATE["vrdu"] = {}
    return _STATE["vrdu"]


# ------------------------------------------------------------------ inference
def tag_document(clf, dev, ex: VrduExample) -> list[int]:
    """Windows of 384 words (128 overlap), content-box anchoring, summed word probabilities: the Track 1 inference."""
    windows = chunk_vrdu_example(ex, WINDOW, OVERLAP)
    collate = make_collator(clf, is_joint=False, is_train=False, coordinate_jitter=False, relative_coords=True,
                            anchor_mode="content_box")
    loader = DataLoader(windows, batch_size=4, shuffle=False, collate_fn=collate)
    preds, _ = predict(clf, loader, dev)
    return preds[0]


def spans_of(pred_ids: list[int], id2label: dict[int, str]) -> list[tuple[str, int, int]]:
    tags = [id2label.get(i, "O") for i in pred_ids]
    return sorted(_extract_spans(tags, strict=True), key=lambda s: (s[1], s[2]))


def draw_pages(pages: list[Image.Image], ex: VrduExample, spans: list[tuple[str, int, int]]) -> list[Image.Image]:
    out = [p.copy() for p in pages]
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for typ, a, b in spans:
        color = FIELD_COLORS.get(typ, "#555555")
        for i in range(a, b):
            pi = ex.page_ids[i] if ex.page_ids else 0
            if pi >= len(out):
                continue
            w, h = out[pi].size
            x0, y0, x1, y1 = ex.boxes[i]
            d = ImageDraw.Draw(out[pi])
            d.rectangle([x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000], outline=color, width=3)
        pi = ex.page_ids[a] if ex.page_ids else 0
        if pi < len(out):
            w, h = out[pi].size
            x0, y0 = ex.boxes[a][0] * w / 1000, ex.boxes[a][1] * h / 1000
            ImageDraw.Draw(out[pi]).text((x0, max(0, y0 - 18)), typ, fill=color, font=font)
    return out


def run(file_obj, use_dataset_ocr: bool, ckpt_arg: str | None):
    if file_obj is None:
        raise gr.Error("Upload a PDF or an image of a registration form.")
    path = Path(file_obj.name if hasattr(file_obj, "name") else str(file_obj))
    clf, id2label, dev, trained, ckpt = get_model(find_checkpoint(ckpt_arg))
    pages = render_pages(path)
    guid = re.sub(r"_[gmb](\.pdf)$", r"\1", path.name)
    gold = vrdu_index().get(guid) if use_dataset_ocr else None
    source = "dataset OCR (the benchmark's own words and boxes)" if gold is not None else "RapidOCR on the rendered pages"
    ex = gold if gold is not None else words_from_ocr(pages)
    pred_ids = tag_document(clf, dev, ex)
    spans = spans_of(pred_ids, id2label)

    rows = [{"Field": t, "Extracted text": " ".join(ex.words[a:b]), "Page": (ex.page_ids[a] + 1) if ex.page_ids else 1,
             "Words": b - a} for t, a, b in spans]
    summary = [f"**Model:** BROS + Track 1 — `{ckpt}`" if trained else
               "**Model: UNTRAINED classifier head. This is a plumbing test, the predictions are meaningless.**",
               f"**Input:** {source}; {len(ex.words)} words over {len(pages)} page(s), "
               f"{len(chunk_vrdu_example(ex, WINDOW, OVERLAP))} windows"]
    if gold is not None:
        g = {(t, a, b) for t, a, b in spans_of(gold.ner_tags, id2label)}
        p = set(spans)
        tp = len(g & p)
        prec = tp / len(p) if p else 0.0
        rec = tp / len(g) if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        missed = sorted({t for t, _, _ in g - p})
        summary.append(f"**Against gold (exact span):** entity F1 {100*f1:.0f}% — {tp} of {len(g)} gold fields found, "
                       f"{len(p - g)} extra" + (f"; missed: {', '.join(missed)}" if missed else ""))
        for r in rows:
            r["Matches gold"] = "yes" if any(r["Field"] == t and " ".join(ex.words[a:b]) == r["Extracted text"]
                                              for t, a, b in g) else "no"
    else:
        summary.append("No gold labels (not a VRDU document, or dataset OCR switched off).")
    return draw_pages(pages, ex, spans), pd.DataFrame(rows), "\n\n".join(summary)


def create_demo(ckpt_arg: str | None = None) -> gr.Blocks:
    ck = find_checkpoint(ckpt_arg)
    banner = (f"Checkpoint: `{ck}`" if ck else "**No BROS checkpoint found yet.** Train one on Kaggle, then relaunch with --ckpt.")
    with gr.Blocks(title="BROS + Track 1: VRDU field extraction") as demo:
        gr.Markdown("# BROS + Track 1: the six VRDU Registration fields\n"
                    "Our best trained entity model (92.4 F1 on seen templates, 71.3 on an unseen one). Entities only, no linking.\n\n" + banner)
        with gr.Row():
            f = gr.File(label="Registration form (PDF or image)", file_types=[".pdf", ".png", ".jpg", ".jpeg"])
            use_ds = gr.Checkbox(value=True, label="If this is a VRDU file, use the dataset's OCR and score against gold")
        btn = gr.Button("Extract fields", variant="primary")
        gal = gr.Gallery(label="Predicted fields on each page", columns=2, height=600)
        table = gr.Dataframe(label="Extracted fields")
        md = gr.Markdown()
        ck_state = gr.State(ckpt_arg)
        btn.click(run, inputs=[f, use_ds, ck_state], outputs=[gal, table, md])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="path to a BROS Track-1 best_model.pt")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--server_name", default="127.0.0.1")
    a = ap.parse_args()
    ck = find_checkpoint(a.ckpt)
    print(f"[+] BROS checkpoint: {ck if ck else 'NONE FOUND (set --ckpt or DEMO_BROS_CKPT)'}")
    create_demo(a.ckpt).launch(server_name=a.server_name, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
