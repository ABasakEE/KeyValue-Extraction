"""VRDU tagger (BROS Track 1): dataset OCR words vs RapidOCR words on the same forms, field-level text match to the gold."""
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(r'D:\KeyValue-Extraction')
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]
import src.demo_all as A  # noqa: E402

DB = A.DB
norm = lambda t: re.sub(r'[\W_]+', '', str(t).lower())


def same(a, b):
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and difflib.SequenceMatcher(None, a, b).ratio() >= 0.75


def f1_text(pred, gold):
    """pred, gold: lists of (field, text); a pred matches a gold of the same field with similar text."""
    used, tp = set(), 0
    for gf, gt in gold:
        for i, (pf, pt) in enumerate(pred):
            if i not in used and pf == gf and same(pt, gt):
                used.add(i)
                tp += 1
                break
    p = tp / len(pred) if pred else 0
    r = tp / len(gold) if gold else 0
    return (2 * p * r / (p + r) if p + r else 0), tp, len(pred), len(gold)


by_guid, _, _ = A.vrdu_protocol()
tot = {'dataset': [0, 0, 0], 'rapid': [0, 0, 0]}
print(f"{'model':4s} {'form':44s} {'dataset OCR':>12s} {'RapidOCR':>10s}")
for key in ('UTL', 'MTL'):
    clf, id2label, dev, trained, ck = A.get_model(A.VRDU_NAME, key)
    cur = A.curated('vrdu', key)
    for c in cur[:5]:
        ex = by_guid[c['guid']]
        pages = A.render_pdf(A.VRDU_DIR / 'pdfs' / c['guid'])
        gold = [(t, ' '.join(ex.words[a:b])) for t, a, b in DB.spans_of(ex.ner_tags, id2label)]
        res = {}
        for name, e in (('dataset', ex), ('rapid', DB.words_from_ocr(pages))):
            pred = [(t, ' '.join(e.words[a:b])) for t, a, b in DB.spans_of(DB.tag_document(clf, dev, e), id2label)]
            f, tp, npred, ngold = f1_text(pred, gold)
            res[name] = f
            tot[name] = [x + y for x, y in zip(tot[name], [tp, npred - tp, ngold - tp])]
        print(f"{key:4s} {c['guid'][:44]:44s} {100 * res['dataset']:11.0f}% {100 * res['rapid']:9.0f}%", flush=True)
for name, (tp, fp, fn) in tot.items():
    print(name, 'micro text-match F1', round(100 * 2 * tp / max(1, 2 * tp + fp + fn), 1))
