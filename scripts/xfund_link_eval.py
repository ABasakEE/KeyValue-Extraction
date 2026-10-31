"""Link F1 of the saved XFUND joint-LiLT checkpoints on held-out forms (CPU).

Modes (all with the annotation's entity spans and types given to the linker, exactly as src/train.py predict_joint does):
  single    one 512-token window, as trained/evaluated before
  windowed  entity-aligned windows (default 510 tokens, stride 128); predicted links are unioned across windows
Gold sets:
  visible   links whose two entities each have >=1 word in the first 512 tokens (what metrics.json scored)
  full      every annotated link (links cut off by truncation count as misses)
Subsets: all 250 val documents, and the ones with no train sibling (question-text Jaccard < 0.5, same language).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.xfund_loader import ID2LABEL, LABEL_LIST, load_xfund_split  # noqa: E402
from train import build_model  # noqa: E402

LANGS = ["es", "fr", "de", "it", "pt"]


def question_set(ex) -> set[str]:
    out = set()
    for e in ex.entities.values():
        if str(e["label"]).lower().endswith("question"):
            t = re.sub(r"[^a-zà-ÿ]+", " ", e["text"].lower()).strip()
            if t:
                out.add(t)
    return out


def sibling_free(val, train) -> set[str]:
    by_lang: dict[str, list[set]] = {}
    for t in train:
        by_lang.setdefault(t.guid.split("_")[0], []).append(question_set(t))
    keep = set()
    for v in val:
        q = question_set(v)
        sib = any(len(q & s) / max(1, len(q | s)) >= 0.5 for s in by_lang.get(v.guid.split("_")[0], []))
        if not sib:
            keep.add(v.guid)
    return keep


def entity_order(ex, word_ids) -> list[int]:
    """Entity ids in the order prepare_entity_batch numbers them (those with >=1 tokenised word)."""
    seen = {w for w in word_ids if w is not None}
    return [i for i in sorted(ex.entities) if any(w in seen for w in ex.entities[i].get("token_indices", []))]


@torch.no_grad()
def predict(model, ex) -> tuple[set, set]:
    """(predicted links, gold links) in the example's own entity-id space."""
    enc = model.encode([ex.words], [ex.boxes], [ex.ner_tags])
    try:
        from models.joint_extractor import prepare_entity_batch
    except ImportError:
        from src.models.joint_extractor import prepare_entity_batch
    spans, gold, types = prepare_entity_batch([ex], enc.word_ids)
    order = entity_order(ex, enc.word_ids[0])
    assert len(order) == len(spans[0]), (ex.guid, len(order), len(spans[0]))
    inputs = {k: v for k, v in enc.inputs.items() if k != "labels"}
    out = model(inputs=inputs, entity_spans=spans, gold_links=None, entity_types=types)
    pred = {(order[a], order[b]) for a, b in out.predicted_links[0]}
    return pred, {(order[a], order[b]) for a, b in gold[0]}


def windows(model, ex, size: int, stride: int):
    """Word ranges [wa, wb) whose subword spans fit in `size` content tokens, advancing ~stride tokens."""
    enc = model.tokenizer(ex.words, boxes=[[0, 0, 0, 0]] * len(ex.words), truncation=False)
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for p, w in enumerate(enc.word_ids()):
        if w is not None:
            first.setdefault(w, p)
            last[w] = p
    n = len(ex.words)
    out, wa = [], 0
    while wa < n:
        base = first[wa]
        wb = wa
        while wb < n and last[wb] - base < size:
            wb += 1
        if wb == wa:  # a single word longer than the window
            wb = wa + 1
        out.append((wa, wb))
        if wb >= n:
            break
        nxt = wa + 1
        while nxt < wb and first[nxt] - base < stride:
            nxt += 1
        wa = nxt
    return out


def sub_example(ex, wa: int, wb: int):
    ents = {}
    for i, e in ex.entities.items():
        idx = e.get("token_indices", [])
        if idx and idx[0] >= wa and idx[-1] < wb:
            ents[i] = {**e, "token_indices": [w - wa for w in idx]}
    return dataclasses.replace(
        ex,
        words=ex.words[wa:wb], boxes=ex.boxes[wa:wb], ner_tags=ex.ner_tags[wa:wb],
        entity_ids=ex.entity_ids[wa:wb],
        entities=ents,
        links=[(s, d) for s, d in ex.links if s in ents and d in ents],
    )


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "gold": tp + fn, "pred": tp + fp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="results/xfund_joint_lilt")
    ap.add_argument("--data", default="data/xfund")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42])
    ap.add_argument("--window", type=int, default=510)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="first N val docs (smoke test)")
    ap.add_argument("--out", default="results/xfund_link_eval/metrics.json")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    ckpt = ROOT / a.ckpt_dir
    cfg = SimpleNamespace(**json.load(open(ckpt / "metrics.json"))["config"])
    val = load_xfund_split(a.data, "val", LANGS)
    train = load_xfund_split(a.data, "train", LANGS)
    keep = sibling_free(val, train)
    if a.limit:
        val = val[: a.limit]
    print(f"val docs {len(val)}, sibling-free {sum(v.guid in keep for v in val)}", flush=True)

    results = {"window": a.window, "stride": a.stride, "n_val": len(val), "n_sibling_free": sum(v.guid in keep for v in val), "seeds": {}}
    for seed in a.seeds:
        model = build_model(cfg, len(LABEL_LIST), ID2LABEL)
        sd = torch.load(ckpt / f"seed_{seed}" / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
        model.eval()
        # counts[(subset, mode, gold)] = [tp, fp, fn]
        counts: dict[tuple, list[int]] = {}

        def add(key, pred, gold):
            c = counts.setdefault(key, [0, 0, 0])
            c[0] += len(pred & gold); c[1] += len(pred - gold); c[2] += len(gold - pred)

        for i, ex in enumerate(val):
            full = {(s, d) for s, d in ex.links if s != d}
            pred1, vis = predict(model, ex)
            predw: set = set()
            for wa, wb in windows(model, ex, a.window, a.stride):
                p, _ = predict(model, sub_example(ex, wa, wb))
                predw |= p
            for sub in ("all", "sibling_free"):
                if sub == "sibling_free" and ex.guid not in keep:
                    continue
                add((sub, "single", "visible"), pred1, vis)
                add((sub, "single", "full"), pred1, full)
                add((sub, "windowed", "visible"), predw & vis, vis)
                add((sub, "windowed", "full"), predw, full)
            if (i + 1) % 25 == 0:
                print(f"  seed {seed}: {i + 1}/{len(val)}", flush=True)
        results["seeds"][seed] = {"|".join(k): prf(*v) for k, v in counts.items()}
        print(seed, {k: v["f1"] for k, v in results["seeds"][seed].items()}, flush=True)
        part = ROOT / a.out
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps(results, indent=2), encoding="utf-8")  # survives a crash on a later seed

    keys = next(iter(results["seeds"].values())).keys()
    results["mean_f1"] = {k: round(sum(results["seeds"][s][k]["f1"] for s in results["seeds"]) / len(results["seeds"]), 4) for k in keys}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
