"""Why does the demo (OCR + text match) score below the benchmark (gold words + exact spans) on XFUND forms?

For the 8 cached demo forms, per form:
  * sizes: the original XFUND page (JSON), the cached jpg, and the pixels the demo's OCR actually sees;
  * word recall: share of gold words (normalised) that the OCR output contains, for RapidOCR at the demo's resolution, RapidOCR at full resolution, docTR;
  * entity recoverability: share of gold entities (question/answer/header) whose text is a run of consecutive OCR words, and how many gold entities share an OCR line with another entity
    or span several OCR lines (the segmentation mismatch);
  * the scorer: J1 on GOLD words, scored once with the benchmark rule (exact word ranges) and once with the demo's text-match rule.
"""
import difflib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymupdf
import torch
from PIL import Image

ROOT = Path(r'D:\KeyValue-Extraction')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT)]
os.environ['DEMO_CKPT_DIR'] = str(ROOT / 'results' / 'bros_joint' / 'J1' / 'seed_42')
from data.xfund_loader import ID2LABEL, LABEL_LIST, load_xfund_split  # noqa: E402
from data.xfund_windows import word_windows  # noqa: E402
from e2e_link import add_counts, gold_structure, predict_document, prf  # noqa: E402
from train import build_model  # noqa: E402
import importlib  # noqa: E402

D = importlib.import_module('src.demo')

import unicodedata  # noqa: E402

norm = lambda t: re.sub(r'[\W_]+', '', str(t).lower())
def norm2(t):   # accents and case ignored: what is left when only the recognised LETTERS matter
    t = unicodedata.normalize('NFKD', str(t))
    return re.sub(r'[^0-9a-z]+', '', ''.join(c for c in t if not unicodedata.combining(c)).lower())
cache = ROOT / 'results' / 'demo_weights' / 'examples_xfund.json'
forms = [Path(r['source']).stem.rsplit('_', 1)[0] for r in json.loads(cache.read_text(encoding='utf-8'))]
val = load_xfund_split(ROOT / 'data' / 'xfund', 'val', ('es', 'fr', 'de', 'it', 'pt'))
raw = {l: json.loads((ROOT / 'data' / 'xfund' / f'{l}.val.json').read_text(encoding='utf-8'))['documents'] for l in ('es', 'fr', 'de', 'it', 'pt')}

cfg = SimpleNamespace(**json.loads((ROOT / 'results' / 'bros_joint' / 'J1' / 'metrics.json').read_text(encoding='utf-8'))['config'])
model = build_model(cfg, len(LABEL_LIST), ID2LABEL)
sd = torch.load(ROOT / 'results' / 'bros_joint' / 'J1' / 'seed_42' / 'best_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)
model.eval()

rapid = D.get_ocr()
os.environ['DEMO_OCR'] = 'doctr'
D._RAPID_OCR = None
doctr = D.get_ocr()


def render(pdf: Path, zoom: float | None):
    page = pymupdf.open(pdf)[0]
    r = page.rect
    z = zoom if zoom else min(1200.0 / r.width, 1200.0 / r.height, 1.5)   # the demo's rule
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z))
    return np.array(Image.frombytes('RGB', [pix.width, pix.height], pix.samples))


def ocr_words(ocr, img):
    items, _ = ocr(img)
    out = []   # (word, line index, y centre)
    for li, (pts, text, _s) in enumerate(items):
        ys = [p[1] for p in pts]
        for t in str(text).split():
            out.append((t, li, (min(ys) + max(ys)) / 2))
    return out


def word_recall(gold_words, ocr_ws):
    bag = {}
    for w, _, _ in ocr_ws:
        bag[norm(w)] = bag.get(norm(w), 0) + 1
    hit = 0
    for g in gold_words:
        k = norm(g)
        if k and bag.get(k, 0) > 0:
            bag[k] -= 1
            hit += 1
    return hit / max(1, sum(1 for g in gold_words if norm(g)))


def classify(entity_texts, ocr_ws, nf):
    """for each gold entity: found (consecutive OCR words) | scattered (all its words are in the OCR output but not consecutively) | partial (>= half of its words) | missing."""
    stream = [nf(w) for w, _, _ in ocr_ws]
    bag = {}
    for w in stream:
        bag[w] = bag.get(w, 0) + 1
    out = {'found': 0, 'scattered': 0, 'partial': 0, 'missing': 0}
    for words in entity_texts:
        ws = [nf(w) for w in words if nf(w)]
        if not ws:
            continue
        if any(stream[i:i + len(ws)] == ws for i in range(len(stream) - len(ws) + 1)):
            out['found'] += 1
            continue
        present = sum(1 for w in ws if bag.get(w, 0) > 0) / len(ws)
        out['scattered' if present == 1 else 'partial' if present >= 0.5 else 'missing'] += 1
    return out


def recoverable(entity_texts, ocr_ws):
    """exact consecutive-word match of an entity's words inside the OCR word stream; plus how many OCR lines the match touches."""
    stream = [norm(w) for w, _, _ in ocr_ws]
    n_ok = n_merge = n_split = 0
    for words in entity_texts:
        ws = [norm(w) for w in words if norm(w)]
        if not ws:
            continue
        found = None
        for i in range(len(stream) - len(ws) + 1):
            if stream[i:i + len(ws)] == ws:
                found = (i, i + len(ws))
                break
        if found:
            n_ok += 1
            lines = {ocr_ws[j][1] for j in range(*found)}
            n_split += len(lines) > 1
            # entity sits inside a longer OCR line (other words share the line)
            line = ocr_ws[found[0]][1]
            total = sum(1 for _, l, _ in ocr_ws if l == line)
            n_merge += total > len(ws) and len(lines) == 1
    return n_ok, n_merge, n_split, sum(1 for w in entity_texts if any(norm(x) for x in w))


def same(a, b):
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and difflib.SequenceMatcher(None, a, b).ratio() >= 0.75


def text_match_f1(pred_pairs, gold_pairs):
    used, tp = set(), 0
    for gk, gv in gold_pairs:
        for i, (pk, pv) in enumerate(pred_pairs):
            if i not in used and same(pk, gk) and same(pv, gv):
                used.add(i)
                tp += 1
                break
    return tp, len(pred_pairs), len(gold_pairs)


agg = {k: {'found': 0, 'scattered': 0, 'partial': 0, 'missing': 0} for k in ('rapid', 'rapid_noaccent', 'doctr', 'doctr_noaccent')}
rec = {k: [0, 0] for k in agg}
print(f"{'form':10s} {'orig px':>11s} {'OCR px':>9s} | word recall: rapid(demo-res) rapid(full) doctr | entities exact-recoverable: rapid doctr | merged-in-line rapid | benchmark F1 | text-match F1 (gold words)")
tot = {'bench': [0, 0, 0], 'text': [0, 0, 0]}
for name in forms:
    lang, _, idx = name.split('_')
    ex = [v for v in val if v.guid.startswith(lang + '_')][int(idx)]
    doc = raw[lang][int(idx)]
    meta = doc['img']
    pdf = ROOT / 'demo_pdfs' / 'tagger' / 'xfund' / next(p.name for p in (ROOT / 'demo_pdfs' / 'tagger' / 'xfund').glob(f'{name}_*.pdf'))
    img_demo = render(pdf, None)
    img_full = render(pdf, 3.0)
    gold_words = [w['text'] for e in doc['document'] for w in e['words']]
    ents = [[w['text'] for w in e['words']] for e in doc['document'] if e['words']]
    r_demo, r_full, d_demo = ocr_words(rapid, img_demo), ocr_words(rapid, img_full), ocr_words(doctr, img_demo)
    for key, ws_, nf in (('rapid', r_demo, norm), ('rapid_noaccent', r_demo, norm2), ('doctr', d_demo, norm), ('doctr_noaccent', d_demo, norm2)):
        for k, v in classify(ents, ws_, nf).items():
            agg[key][k] += v
        gw = [g for g in gold_words if nf(g)]
        bag = {}
        for w, _, _ in ws_:
            bag[nf(w)] = bag.get(nf(w), 0) + 1
        for g in gw:
            rec[key][1] += 1
            if bag.get(nf(g), 0) > 0:
                bag[nf(g)] -= 1
                rec[key][0] += 1
    ok_r, merge_r, split_r, n = recoverable(ents, r_demo)
    ok_d, _, _, _ = recoverable(ents, d_demo)
    # the scorer: J1 on gold words
    g_ents, g_links = gold_structure(ex)
    wins = word_windows(model.tokenizer, ex.words, 510, 128)
    pe, pl = predict_document(model, ex, wins)
    c = [0, 0, 0]
    add_counts(c, pl, g_links)
    add_counts(tot['bench'], pl, g_links)
    txt = lambda span: ' '.join(ex.words[span[0]:span[1]])
    pred_pairs = [(txt(s), txt(d)) for s, d in pl]
    gold_pairs = [(txt(s), txt(d)) for s, d in g_links]
    tp, npred, ngold = text_match_f1(pred_pairs, gold_pairs)
    tot['text'] = [a + b for a, b in zip(tot['text'], [tp, npred - tp, ngold - tp])]
    tf = 2 * tp / max(1, npred + ngold)
    print(f"{name:10s} {meta['width']}x{meta['height']:<5d} {img_demo.shape[1]}x{img_demo.shape[0]:<5d} | {100 * word_recall(gold_words, r_demo):5.0f}% {100 * word_recall(gold_words, r_full):5.0f}% {100 * word_recall(gold_words, d_demo):5.0f}% "
          f"| {ok_r}/{n} {ok_d}/{n} | {merge_r}/{n} | {100 * prf(*c)['f1']:5.1f} | {100 * tf:5.1f}", flush=True)
print('micro benchmark F1 (gold words, exact):', round(100 * prf(*tot['bench'])['f1'], 1), '| micro text-match F1 (gold words):', round(100 * 2 * tot['text'][0] / max(1, 2 * tot['text'][0] + tot['text'][1] + tot['text'][2]), 1))

print()
print('gold ENTITIES (all labels) by what the OCR output contains; accent-insensitive rows ignore diacritics and case')
for k, v in agg.items():
    tot_e = sum(v.values())
    print(f"  {k:15s} found {100 * v['found'] / tot_e:4.0f}%  scattered {100 * v['scattered'] / tot_e:4.0f}%  partial {100 * v['partial'] / tot_e:4.0f}%  missing {100 * v['missing'] / tot_e:4.0f}%   (n={tot_e}); gold-word recall {100 * rec[k][0] / rec[k][1]:.0f}%")
