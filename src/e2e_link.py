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


def predict_document(model, ex, windows: list[tuple[int, int]] | None = None) -> tuple[set[Entity], set[Link]]:
    """Predicted entities and links in whole-document word coordinates.

    ``windows=None`` is the single 512-token reading the earlier checkpoints were evaluated with.  With windows, an entity
    touching a window edge that is not the document edge may be cut in half, so it is dropped (the overlap re-predicts it).
    """
    n = len(ex.words)
    entities: set[Entity] = set()
    links: set[Link] = set()
    for wa, wb in (windows or [(0, n)]):
        win = sub_example(ex, wa, wb, with_entities=False)
        ents, pairs = predict_window(model, win)
        keep = [not (windows and ((a == 0 and wa > 0) or (b == wb - wa and wb < n))) for _, a, b in ents]
        glob = [(t, a + wa, b + wa) for t, a, b in ents]
        entities |= {e for e, k in zip(glob, keep) if k}
        links |= {((glob[i][1], glob[i][2]), (glob[j][1], glob[j][2])) for i, j in pairs if keep[i] and keep[j]}
    return entities, links


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
