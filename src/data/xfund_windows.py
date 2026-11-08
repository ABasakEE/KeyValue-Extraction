"""Word windows over long XFUND forms (62% of val documents exceed 512 subword tokens).

A window is a run of words whose subword span fits in ``size`` content tokens (510 + 2 special = 512).  Training windows keep
only the entities that lie wholly inside the window, plus the links between them, so every window is a valid joint example.
"""
from __future__ import annotations

import dataclasses
import random

try:
    from data.funsd_loader import FormExample
except ImportError:  # imported as src.data.xfund_windows
    from src.data.funsd_loader import FormExample


def word_windows(tokenizer, words: list[str], size: int = 510, stride: int = 255) -> list[tuple[int, int]]:
    """[(first_word, end_word_exclusive)], each window <= ``size`` subwords, successive starts ~``stride`` subwords apart."""
    enc = tokenizer(words, boxes=[[0, 0, 0, 0]] * len(words), truncation=False)
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for p, w in enumerate(enc.word_ids()):
        if w is not None:
            first.setdefault(w, p)
            last[w] = p
    for w in range(len(words)):  # a word that tokenises to nothing borrows its neighbour's position
        first.setdefault(w, first.get(w - 1, 0))
        last.setdefault(w, last.get(w - 1, first[w]))
    n = len(words)
    out, wa = [], 0
    while wa < n:
        base = first[wa]
        wb = wa
        while wb < n and last[wb] - base < size:
            wb += 1
        wb = max(wb, wa + 1)  # a single word longer than the window still advances
        out.append((wa, wb))
        if wb >= n:
            break
        nxt = wa + 1
        while nxt < wb and first[nxt] - base < stride:
            nxt += 1
        wa = nxt
    return out


def sub_example(ex: FormExample, wa: int, wb: int, with_entities: bool = True) -> FormExample:
    """Words [wa, wb) of ``ex``.  Without entities the example is an inference input (no gold structure)."""
    ents = {}
    if with_entities:
        for i, e in ex.entities.items():
            idx = e.get("token_indices", [])
            if idx and idx[0] >= wa and idx[-1] < wb:
                ents[i] = {**e, "token_indices": [w - wa for w in idx]}
    return dataclasses.replace(
        ex,
        guid=f"{ex.guid}#w{wa}",
        words=ex.words[wa:wb], boxes=ex.boxes[wa:wb], ner_tags=ex.ner_tags[wa:wb], entity_ids=ex.entity_ids[wa:wb],
        entities=ents,
        links=[(s, d) for s, d in ex.links if s in ents and d in ents],
    )


def window_examples(examples: list[FormExample], tokenizer, size: int = 510, stride: int = 255) -> list[FormExample]:
    """Training/validation windows; windows with no wholly-contained entity carry no signal and are dropped."""
    out = []
    for ex in examples:
        for wa, wb in word_windows(tokenizer, ex.words, size, stride):
            w = sub_example(ex, wa, wb)
            if w.entities:
                out.append(w)
    return out


def link_components(ex: FormExample, merge_dist: float = 50.0) -> list[tuple[int, int, int, int]]:
    """One bounding region per group of linked entities (a key and its value, or a table header and its cells).

    Entities are grouped by union-find over ``ex.links``; unlinked entities are ignored.  Group boxes that sit within
    ``merge_dist`` of each other on both axes are then merged repeatedly, so adjacent pairs form one table-sized region.
    """
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, d in ex.links:
        if ex.entities.get(s, {}).get("token_indices") and ex.entities.get(d, {}).get("token_indices"):
            parent.setdefault(s, s)
            parent.setdefault(d, d)
            parent[find(s)] = find(d)
    boxes: dict[int, list[int]] = {}
    for i in parent:
        bs = [ex.boxes[w] for w in ex.entities[i]["token_indices"]]
        b = [min(x[0] for x in bs), min(x[1] for x in bs), max(x[2] for x in bs), max(x[3] for x in bs)]
        r = find(i)
        if r in boxes:
            o = boxes[r]
            boxes[r] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
        else:
            boxes[r] = b
    regions = [boxes[r] for r in sorted(boxes)]

    def gap(a0: int, a1: int, b0: int, b1: int) -> int:
        return max(0, max(a0, b0) - min(a1, b1))

    merged = True
    while merged:
        merged = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                a, b = regions[i], regions[j]
                if gap(a[0], a[2], b[0], b[2]) < merge_dist and gap(a[1], a[3], b[1], b[3]) < merge_dist:
                    regions[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del regions[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(r) for r in regions]


def crop_example(ex: FormExample, region: tuple[int, int, int, int], tag: str) -> FormExample | None:
    """Words whose box centre lies in ``region``, re-normalised to 0-1000 over the region.

    An entity survives only if all its words do, a link only if both ends do.  Returns None when no link survives.
    """
    rx0, ry0, rx1, ry1 = region
    keep = [
        w for w, b in enumerate(ex.boxes)
        if rx0 <= (b[0] + b[2]) / 2 <= rx1 and ry0 <= (b[1] + b[3]) / 2 <= ry1
    ]
    new = {w: n for n, w in enumerate(keep)}
    ents = {}
    for i, e in ex.entities.items():
        idx = e.get("token_indices", [])
        if idx and all(w in new for w in idx):
            ents[i] = {**e, "token_indices": [new[w] for w in idx]}
    links = [(s, d) for s, d in ex.links if s in ents and d in ents]
    if not links:
        return None
    sx, sy = 1000 / max(rx1 - rx0, 1), 1000 / max(ry1 - ry0, 1)

    def clamp(v: float) -> int:
        return min(1000, max(0, round(v)))

    boxes = [
        [clamp((b[0] - rx0) * sx), clamp((b[1] - ry0) * sy), clamp((b[2] - rx0) * sx), clamp((b[3] - ry0) * sy)]
        for b in (ex.boxes[w] for w in keep)
    ]
    return dataclasses.replace(
        ex,
        guid=f"{ex.guid}#crop{tag}",
        words=[ex.words[w] for w in keep], boxes=boxes,
        ner_tags=[ex.ner_tags[w] for w in keep], entity_ids=[ex.entity_ids[w] for w in keep],
        entities=ents, links=links,
    )


def crop_examples(
    examples: list[FormExample], rng: random.Random, max_per_form: int = 4, merge_dist: float = 50.0,
    pad: tuple[int, int] = (10, 60),
) -> list[FormExample]:
    """Link-preserving crop augmentation (after arXiv 2608.20868): training crops around linked-entity groups.

    Each form yields up to ``max_per_form`` crops, one per sampled link component, padded by a random margin per side.
    Forms without links yield nothing.
    """
    out = []
    for ex in examples:
        regions = link_components(ex, merge_dist)
        for k, r in enumerate(rng.sample(regions, min(max_per_form, len(regions)))):
            x0, y0, x1, y1 = r
            region = (
                max(0, x0 - rng.randint(*pad)), max(0, y0 - rng.randint(*pad)),
                min(1000, x1 + rng.randint(*pad)), min(1000, y1 + rng.randint(*pad)),
            )
            c = crop_example(ex, region, str(k))
            if c is not None:
                out.append(c)
    return out
