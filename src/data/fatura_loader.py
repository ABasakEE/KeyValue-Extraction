"""FATURA invoices (Zenodo 10371464, CC-BY-4.0) as word-level examples with block-level gold, plus fixed folds.

Built from ``Annotations/Original_Format`` because the Hugging Face-format files lump 21 field classes under one
tag id (see results/fatura_audit/summary.json). Every labelled text block becomes a run of words that all carry the
block's box (FATURA has no word-level boxes); block text includes its own label, e.g. "TOTAL : 734.33 EUR".

Fields: the 9 core classes present in at least 29 of the 50 templates. Other text blocks are kept as "O" words.
Boxes: original boxes are PDF-style (y up, page 595 x 841); we flip y to image coordinates and scale to 0-1000.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

PAGE_W, PAGE_H = 595, 841
CORE = {"BUYER": "buyer", "DATE": "date", "DUE_DATE": "due_date", "NUMBER": "number",
        "SELLER_ADDRESS": "seller_address", "SELLER_NAME": "seller_name", "SUB_TOTAL": "sub_total",
        "TITLE": "title", "TOTAL": "total"}
FIELDS = sorted(CORE.values())
LABELS = ["O"] + [f"{p}-{f}" for f in FIELDS for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NO_TEXT = {"OTHER", "TABLE", "LOGO", "INVOICE_INFO"}   # no per-block text in the annotations
OFFICIAL_TEST_TEMPLATES = [50, 7, 32, 39, 2, 12, 4, 49, 10, 25]   # Strat2_Split.txt dev_inds (= its test set)
N_FOLDS = 5
FOLD_SEED = 0


def default_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data" / "fatura" / "raw" / "invoices_dataset_final"


def _img_box(bbox) -> tuple[float, float, float, float]:
    (xa, ya), (xb, yb) = bbox
    return min(xa, xb), PAGE_H - max(ya, yb), max(xa, xb), PAGE_H - min(ya, yb)


def parse_doc(guid: str, data: dict) -> dict:
    """{'guid','template','words','boxes'(0-1000),'blocks':[(field|None, start, end)],'gold':[[field,s,e,text]]}."""
    blocks = []
    for cls, v in data.items():
        if cls in NO_TEXT or not (isinstance(v, dict) and v.get("text") and v.get("bbox")):
            continue
        x0, y0, x1, y1 = _img_box(v["bbox"])
        blocks.append((y0, x0, CORE.get(cls), v["text"].split(), (x0, y0, x1, y1)))
    blocks.sort(key=lambda b: (round(b[0]), b[1]))          # reading order: top to bottom, then left to right
    words, boxes, spans, gold = [], [], [], []
    for _, _, field, ws, (x0, y0, x1, y1) in blocks:
        if not ws:
            continue
        box = [max(0, min(1000, round(v))) for v in (x0 * 1000 / PAGE_W, y0 * 1000 / PAGE_H, x1 * 1000 / PAGE_W, y1 * 1000 / PAGE_H)]
        start = len(words)
        words += ws
        boxes += [box] * len(ws)
        spans.append((field, start, len(words)))
        if field:
            gold.append([field, start, len(words), " ".join(ws)])
    return {"guid": guid, "template": int(re.match(r"Template(\d+)_", guid).group(1)),
            "words": words, "boxes": boxes, "blocks": spans, "gold": gold}


def load_docs(root: Path | None = None, templates: set[int] | None = None) -> dict[str, dict]:
    root = root or default_root()
    docs = {}
    for p in sorted((root / "Annotations" / "Original_Format").glob("*.json")):
        t = int(re.match(r"Template(\d+)_", p.stem).group(1))
        if templates is None or t in templates:
            docs[p.stem] = parse_doc(p.stem, json.loads(p.read_text(encoding="utf-8")))
    return docs


def tags_of(doc: dict) -> list[int]:
    tags = [LABEL2ID["O"]] * len(doc["words"])
    for field, s, e, _ in doc["gold"]:
        tags[s] = LABEL2ID[f"B-{field}"]
        for i in range(s + 1, e):
            tags[i] = LABEL2ID[f"I-{field}"]
    return tags


def to_example(doc: dict):
    """A VrduExample (the training code's document type) for a FATURA document."""
    from data.vrdu_loader import VrduExample

    n = len(doc["words"])
    return VrduExample(guid=doc["guid"], words=doc["words"], boxes=doc["boxes"], ner_tags=tags_of(doc),
                       template_id=f"Template{doc['template']}", page_ids=[0] * n, n_pages=1)


def folds() -> list[list[int]]:
    """5 folds of 10 templates: fold 0 is the official unseen-template test set; the other 40 are split by seed 0."""
    rest = sorted(set(range(1, 51)) - set(OFFICIAL_TEST_TEMPLATES))
    random.Random(FOLD_SEED).shuffle(rest)
    return [sorted(OFFICIAL_TEST_TEMPLATES)] + [sorted(rest[i * 10:(i + 1) * 10]) for i in range(4)]


def eval_sample(per_template: int = 20, seed: int = 1, n_instances: int = 200) -> dict[int, dict[str, list[str]]]:
    """template -> {'dev': [...], 'confirm': [...]} instance names; the first half of a seeded shuffle is dev."""
    out = {}
    for t in range(1, 51):
        rng = random.Random(seed * 100 + t)
        picked = rng.sample(range(n_instances), per_template)
        names = [f"Template{t}_Instance{i}" for i in picked]
        out[t] = {"dev": sorted(names[: per_template // 2]), "confirm": sorted(names[per_template // 2:])}
    return out


def train_val_sample(held_out: list[int], seed: int, fold: int, n_train: int = 200, n_val: int = 50,
                     n_instances: int = 200) -> tuple[list[str], list[str]]:
    """N training documents spread evenly over the non-held-out templates, plus a disjoint validation set drawn
    from the same (seen) templates, as in the strict protocol: no held-out-template document is ever used."""
    rng = random.Random(seed * 1000 + fold)
    seen = [t for t in range(1, 51) if t not in held_out]
    per = n_train // len(seen)
    train = [f"Template{t}_Instance{i}" for t in seen for i in rng.sample(range(n_instances), per)]
    left = [f"Template{t}_Instance{i}" for t in seen for i in range(n_instances) if f"Template{t}_Instance{i}" not in set(train)]
    return train, rng.sample(left, n_val)


def _selftest() -> None:
    f = folds()
    flat = [t for fold in f for t in fold]
    assert len(f) == 5 and all(len(x) == 10 for x in f) and sorted(flat) == list(range(1, 51)), f
    assert f[0] == sorted(OFFICIAL_TEST_TEMPLATES)
    s = eval_sample()
    assert all(len(v["dev"]) == len(v["confirm"]) == 10 and not set(v["dev"]) & set(v["confirm"]) for v in s.values())
    for held in f:
        tr, va = train_val_sample(held, 13, 0)
        assert len(tr) == 200 and len(va) == 50 and not set(tr) & set(va)
        assert not {int(re.match(r"Template(\d+)_", g).group(1)) for g in tr + va} & set(held)
    toy = {"TOTAL": {"bbox": [[300, 100], [400, 120]], "text": "TOTAL : 5.00 EUR"},
           "DATE": {"bbox": [[10, 700], [110, 712]], "text": "Date: 03-Jan-1994"},
           "NOTE": {"bbox": [[10, 400], [90, 410]], "text": "Thank you"}, "TABLE": [{"bbox": [[0, 0], [1, 1]]}]}
    d = parse_doc("Template3_Instance7", toy)
    assert d["words"] == ["Date:", "03-Jan-1994", "Thank", "you", "TOTAL", ":", "5.00", "EUR"], d["words"]   # top to bottom
    assert d["gold"] == [["date", 0, 2, "Date: 03-Jan-1994"], ["total", 4, 8, "TOTAL : 5.00 EUR"]], d["gold"]
    assert tags_of(d) == [LABEL2ID["B-date"], LABEL2ID["I-date"], 0, 0, LABEL2ID["B-total"]] + [LABEL2ID["I-total"]] * 3
    assert d["boxes"][0] == [round(10 * 1000 / 595), round((841 - 712) * 1000 / 841), round(110 * 1000 / 595), round((841 - 700) * 1000 / 841)]
    root = default_root()
    if root.exists():   # real data: counts and the audit's findings
        docs = load_docs(root, templates={10})
        assert len(docs) == 200, len(docs)
        d = docs["Template10_Instance0"]
        assert {g[0] for g in d["gold"]} >= {"buyer", "date", "due_date", "number", "total"}, d["gold"]
        assert all(len(set(map(tuple, d["boxes"][s:e]))) == 1 for _, s, e, _ in d["gold"])   # one box per block
    print("[+] fatura_loader selftest passed")


if __name__ == "__main__":
    _selftest()
