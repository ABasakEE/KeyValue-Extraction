"""Word windows over long XFUND forms (62% of val documents exceed 512 subword tokens).

A window is a run of words whose subword span fits in ``size`` content tokens (510 + 2 special = 512).  Training windows keep
only the entities that lie wholly inside the window, plus the links between them, so every window is a valid joint example.
"""
from __future__ import annotations

import dataclasses

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
