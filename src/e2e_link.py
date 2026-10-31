"""End-to-end key-value linking with a joint model: entities come from the model's OWN tags, not from the annotation.

``predict_window`` tags one window, turns the BIO tags into entity spans, pools those spans and runs the relation head,
mirroring ``JointFormExtractor.forward`` (checked against it by ``tests/test_e2e_link.py``).  ``predict_document`` slides
entity-safe windows over a long form and unions the results.  A predicted link is correct when both of its endpoints
match the word ranges of a linked gold entity pair (types are not required to match for links; entity F1 is typed).
"""
from __future__ import annotations

import torch

try:
    from data.xfund_windows import sub_example, word_windows
    from evaluate import _extract_spans
    from models.joint_extractor import BiaffineRelationClassifier
    from models.joint_extractor import build_soft_type_priors_from_labels, build_type_compatibility_mask
    from models.joint_extractor import pool_entity_representations
    from data.funsd_loader import ID2LABEL
except ImportError:
    from src.data.xfund_windows import sub_example, word_windows
    from src.evaluate import _extract_spans
    from src.models.joint_extractor import BiaffineRelationClassifier
    from src.models.joint_extractor import build_soft_type_priors_from_labels, build_type_compatibility_mask
    from src.models.joint_extractor import pool_entity_representations
    from src.data.funsd_loader import ID2LABEL

Entity = tuple[str, int, int]                     # (type, first_word, end_word_exclusive)
Link = tuple[tuple[int, int], tuple[int, int]]    # ((a, b), (a, b)) word ranges of source and destination


def link_entities(model, hidden: torch.Tensor, boxes: torch.Tensor, spans: list[list[int]], types: list[str]):
    """The relation half of JointFormExtractor.forward for one document: (relation logits, decoded (i, j) pairs)."""
    reps, ent_boxes = pool_entity_representations(hidden, boxes, spans)
    type_mask = priors = None
    masking = getattr(model, "masking_mode", "soft")
    if masking == "hard":
        type_mask = build_type_compatibility_mask(types).to(hidden.device)
    elif masking == "soft":
        priors = build_soft_type_priors_from_labels(types).to(hidden.device)
    logits = model.relation_head(reps, ent_boxes, type_mask=type_mask, soft_priors=priors)
    pairs = BiaffineRelationClassifier.decode_predictions(
        logits, threshold=getattr(model, "link_threshold", 0.5),
        fallback_top_k=getattr(model, "fallback_top_k", 0), one_to_one=getattr(model, "decode_one_to_one", False))
    return logits, pairs


@torch.no_grad()
def predict_window(model, ex) -> tuple[list[Entity], set[tuple[int, int]]]:
    """Entities (local word coordinates) and index pairs into that list, from the model's own tags."""
    model.eval()
    device = next(model.parameters()).device
    enc = model.encode([ex.words], [ex.boxes], None)
    inputs = {k: v.to(device) for k, v in enc.inputs.items() if k != "labels"}
    out = model.encoder(**inputs, output_hidden_states=True)
    word_ids = enc.word_ids[0]
    tags = [ID2LABEL[i] if i >= 0 else "O" for i in model.first_subword_predictions(out.logits[0].cpu(), word_ids, len(ex.words))]
    ents = sorted((t.lower(), a, b) for t, a, b in _extract_spans(tags, strict=False))
    first_sub: dict[int, int] = {}
    for pos, w in enumerate(word_ids):
        if w is not None:
            first_sub.setdefault(w, pos)
    spans, kept = [], []
    for e in ents:
        sub = [first_sub[w] for w in range(e[1], e[2]) if w in first_sub]
        if sub:
            spans.append(sub)
            kept.append(e)
    if len(kept) < 2:
        return kept, set()
    _, pairs = link_entities(model, out.hidden_states[-1][0], inputs["bbox"][0], spans, [e[0] for e in kept])
    return kept, pairs


def predict_windows(model, ex, windows: list[tuple[int, int]] | None = None) -> list[dict]:
    """Raw per-window output, JSON-serialisable: [{wa, wb, ents: [(type, a, b)] (window-local), pairs: [(i, j)]}].

    ``windows=None`` is one window over the whole form (cut at 512 tokens by the encoder)."""
    n = len(ex.words)
    out = []
    for wa, wb in (windows or [(0, n)]):
        ents, pairs = predict_window(model, sub_example(ex, wa, wb, with_entities=False))
        out.append({"wa": wa, "wb": wb, "ents": [list(e) for e in ents], "pairs": [list(p) for p in pairs]})
    return out


def merge_windows(raw: list[dict], n: int, strategy: str = "edge_drop") -> tuple[set[Entity], set[Link]]:
    """Whole-document entities and links from per-window output.

    ``edge_drop``: an entity touching a window edge that is not the document edge may be cut in half, so it is dropped
    (the overlap re-predicts it); everything else is kept, so one entity predicted slightly differently by two windows
    appears twice.  ``center``: nothing is dropped; same-type predictions that share a word form one cluster, and the one
    furthest from its window edge represents it (ties: earliest window); links follow their entities.
    ``edge_drop+center``: both.
    """
    assert strategy in ("edge_drop", "center", "edge_drop+center"), strategy
    cands = []  # (window, local index, type, a, b, distance to the nearest cut edge)
    for w, win in enumerate(raw):
        wa, wb = win["wa"], win["wb"]
        for i, (t, a, b) in enumerate(win["ents"]):
            left = float("inf") if wa == 0 else a
            right = float("inf") if wb == n else (wb - wa) - b
            cands.append((w, i, t, a + wa, b + wa, min(left, right)))
    if strategy != "center":
        cands = [c for c in cands if c[5] > 0]
    rep = {(w, i): (t, a, b) for w, i, t, a, b, _ in cands}                  # candidate -> its (possibly merged) entity
    if strategy != "edge_drop":
        cands.sort(key=lambda c: (c[2], c[3]))
        cluster: list = []
        def flush():
            if cluster:
                win = max(cluster, key=lambda c: (c[5], -c[0]))
                for c in cluster:
                    rep[(c[0], c[1])] = (win[2], win[3], win[4])
        end = -1
        for c in cands:
            if cluster and (c[2] != cluster[0][2] or c[3] >= end):
                flush(); cluster = []
            end = max(end, c[4]) if cluster else c[4]
            cluster.append(c)
        flush()
    links: set[Link] = set()
    for w, win in enumerate(raw):
        for i, j in win["pairs"]:
            if (w, i) in rep and (w, j) in rep and rep[(w, i)] != rep[(w, j)]:
                links.add(((rep[(w, i)][1], rep[(w, i)][2]), (rep[(w, j)][1], rep[(w, j)][2])))
    return set(rep.values()), links


def predict_document(model, ex, windows: list[tuple[int, int]] | None = None, strategy: str = "edge_drop") -> tuple[set[Entity], set[Link]]:
    """Predicted entities and links in whole-document word coordinates (see merge_windows for the strategies)."""
    return merge_windows(predict_windows(model, ex, windows), len(ex.words), strategy)


def gold_structure(ex) -> tuple[set[Entity], set[Link]]:
    """Typed gold entities (tagged ones only) and the links between ANY two annotated entities, as word ranges."""
    rng = {i: (min(e["token_indices"]), max(e["token_indices"]) + 1) for i, e in ex.entities.items() if e.get("token_indices")}
    ents = {(e["label"].lower(), *rng[i]) for i, e in ex.entities.items() if i in rng and e["label"] != "other"}
    links = {(rng[s], rng[d]) for s, d in ex.links if s in rng and d in rng and s != d}
    return ents, links


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "gold": tp + fn, "pred": tp + fp}


def add_counts(acc: list[int], pred: set, gold: set) -> None:
    acc[0] += len(pred & gold)
    acc[1] += len(pred - gold)
    acc[2] += len(gold - pred)
