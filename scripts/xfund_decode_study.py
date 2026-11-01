"""Study 23: decode-time levers for gold-entity linking (CPU).  Registered in STUDIES.md study 23 before this code existed.

    python scripts/xfund_decode_study.py selftest
    python scripts/xfund_decode_study.py cache   --ckpt-dir results/xfund_joint_lilt --cache-dir results/xfund_decode/cache
    python scripts/xfund_decode_study.py analyse --cache-dir results/xfund_decode/cache --out results/xfund_decode/legacy.json

``cache`` runs the model once per window (gold entities, windows of 510 tokens, stride 128) and stores the relation logits
(the soft type prior is already inside them) for the 111 forms carved from the training file (selection split) and the 250
forms of the val file (report split).  ``analyse`` re-decodes every variant of the registered grid from that cache.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    sys.path.insert(0, p)

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
O2O = (True, False)
PRIORS = ("trained", "none", "qq_free", "hard")
KS = (0, 5, 10)
DEFAULT = (0.5, True, "trained", 0)
ALLOWED = {("question", "answer"), ("header", "question"), ("header", "answer")}   # src/models/linking_head.py
PENALTY = -5.0
NEG = -1e9


# ---------------------------------------------------------------- decoding (numpy)
def prior_matrix(types: list[str]) -> np.ndarray:
    n = len(types)
    pm = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(types):
        for j, b in enumerate(types):
            if i != j and (a.lower(), b.lower()) not in ALLOWED:
                pm[i, j] = PENALTY
    return pm


def qq_mask(types: list[str]) -> np.ndarray:
    q = np.array([t.lower() == "question" for t in types])
    m = q[:, None] & q[None, :]
    np.fill_diagonal(m, False)
    return m


def adjust(logits: np.ndarray, types: list[str], prior: str) -> np.ndarray:
    pm = prior_matrix(types)
    if prior == "trained":
        return logits
    if prior == "none":
        return logits - pm
    if prior == "qq_free":
        return logits - pm * qq_mask(types)
    if prior == "hard":
        return np.where(pm < 0, NEG, logits)
    raise ValueError(prior)


def decode_np(logits: np.ndarray, threshold: float, one_to_one: bool) -> set[tuple[int, int]]:
    """Same rule as BiaffineRelationClassifier.decode_predictions (diagonal included in the column argmax)."""
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    n = probs.shape[0]
    if one_to_one:
        best = probs.argmax(axis=0)
        return {(int(best[j]), j) for j in range(n) if probs[best[j], j] >= threshold and best[j] != j}
    ii, jj = np.nonzero(probs >= threshold)
    return {(int(i), int(j)) for i, j in zip(ii, jj) if i != j}


def rank_matrix(boxes: dict, types: dict) -> dict[tuple[int, int], int]:
    """rank[(s, d)]: position of d among the entities of its type ordered by centre distance from s (0 = nearest)."""
    from xfund_link_errors import centre
    out = {}
    for s in boxes:
        cs = centre(boxes[s])
        by_type = defaultdict(list)
        for j in boxes:
            if j != s:
                by_type[types[j]].append((math.dist(cs, centre(boxes[j])), j))
        for lst in by_type.values():
            for r, (_, j) in enumerate(sorted(lst)):
                out[(s, j)] = r
    return out


def decode_doc(windows_cache: list[dict], variant: tuple, ranks: dict) -> set[tuple[int, int]]:
    thr, o2o, prior, k = variant
    pred = set()
    for w in windows_cache:
        order, types = w["order"], w["types"]
        if len(order) < 2:
            continue
        lg = adjust(w["logits"], types, prior)
        if k:
            keep = np.array([[ranks.get((s, d), 10**9) < k for d in order] for s in order])
            lg = np.where(keep, lg, NEG)
        pred |= {(order[i], order[j]) for i, j in decode_np(lg, thr, o2o)}
    return pred


# ---------------------------------------------------------------- data and scoring
def load_splits(data: str):
    from data.xfund_loader import load_xfund_dataset
    from xfund_link_eval import LANGS
    _, carved, test = load_xfund_dataset(data, languages=LANGS)
    return {"val": carved, "test": test}


def doc_context(ex):
    from xfund_link_errors import entity_boxes
    boxes = entity_boxes(ex)
    types = {i: e["label"].lower() for i, e in ex.entities.items()}
    gold = {(s, d) for s, d in ex.links if s != d and s in boxes and d in boxes}
    return boxes, types, gold


def score_variants(cache: dict, docs: list, variants: list[tuple]) -> dict[tuple, list[int]]:
    counts = {v: [0, 0, 0] for v in variants}
    need_ranks = any(v[3] for v in variants)
    for ex in docs:
        boxes, types, gold = doc_context(ex)
        ranks = rank_matrix(boxes, types) if need_ranks else {}
        wc = cache[ex.guid]
        for v in variants:
            pred = decode_doc(wc, v, ranks)
            c = counts[v]
            c[0] += len(pred & gold); c[1] += len(pred - gold); c[2] += len(gold - pred)
    return counts


def f1(c: list[int]) -> float:
    tp, fp, fn = c
    p, r = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    return 2 * p * r / (p + r) if p + r else 0.0


def n_nondefault(v: tuple) -> int:
    return sum(a != b for a, b in zip(v, DEFAULT))


def group_recall(cache: dict, docs: list, variant: tuple) -> dict:
    from xfund_link_errors import direction
    found, total = defaultdict(int), defaultdict(int)
    for ex in docs:
        boxes, types, gold = doc_context(ex)
        pred = decode_doc(cache[ex.guid], variant, rank_matrix(boxes, types) if variant[3] else {})
        for s, d in gold:
            for g in (f"type: {types[s]} -> {types[d]}", f"geometry: {direction(boxes[s], boxes[d])}"):
                total[g] += 1
                found[g] += (s, d) in pred
    return {g: {"found": found[g], "total": total[g], "recall": round(found[g] / total[g], 4)} for g in sorted(total) if total[g] >= 20}


# ---------------------------------------------------------------- subcommands
def cmd_cache(a) -> None:
    import torch
    from data.xfund_loader import ID2LABEL, LABEL_LIST
    from models.joint_extractor import prepare_entity_batch
    from train import build_model
    from xfund_link_eval import entity_order, sub_example, windows
    if a.threads:
        torch.set_num_threads(a.threads)
    ckpt = Path(a.ckpt_dir)
    cfg = SimpleNamespace(**json.loads((ckpt / "metrics.json").read_text(encoding="utf-8"))["config"])
    splits = load_splits(a.data)
    if a.limit:
        splits = {k: v[: a.limit] for k, v in splits.items()}
    Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
    for seed in (a.seeds or list(cfg.seeds)):
        out = Path(a.cache_dir) / f"legacy_seed{seed}.pkl"
        if out.exists():
            print("cached", out, flush=True)
            continue
        model = build_model(cfg, len(LABEL_LIST), ID2LABEL)
        sd = torch.load(ckpt / f"seed_{seed}" / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
        model.eval()
        cache = {}
        for split, docs in splits.items():
            for i, ex in enumerate(docs):
                ws = []
                for wa, wb in windows(model, ex, 510, 128):
                    w = sub_example(ex, wa, wb)
                    enc = model.encode([w.words], [w.boxes], [w.ner_tags])
                    spans, _, types = prepare_entity_batch([w], enc.word_ids)
                    order = entity_order(w, enc.word_ids[0])
                    assert len(order) == len(spans[0]) == len(types[0]), (w.guid, len(order), len(spans[0]))
                    if len(order) < 2:
                        ws.append({"order": order, "types": types[0], "logits": np.zeros((len(order),) * 2, dtype=np.float32)})
                        continue
                    inputs = {k: v for k, v in enc.inputs.items() if k != "labels"}
                    with torch.no_grad():
                        o = model(inputs=inputs, entity_spans=spans, gold_links=None, entity_types=types)
                    ws.append({"order": order, "types": types[0], "logits": o.relation_logits[0].numpy().astype(np.float32)})
                cache[ex.guid] = ws
                if (i + 1) % 25 == 0:
                    print(f"  seed {seed} {split}: {i + 1}/{len(docs)}", flush=True)
        out.write_bytes(pickle.dumps(cache))
        print("wrote", out, flush=True)
        del model


def cmd_analyse(a) -> None:
    splits = load_splits(a.data)
    if a.limit:
        splits = {k: v[: a.limit] for k, v in splits.items()}
    grid = list(itertools.product(THRESHOLDS, O2O, PRIORS, KS))
    seeds = sorted(int(p.stem.split("seed")[1]) for p in Path(a.cache_dir).glob("legacy_seed*.pkl"))
    caches = {s: pickle.loads((Path(a.cache_dir) / f"legacy_seed{s}.pkl").read_bytes()) for s in seeds}
    val_counts = {s: score_variants(caches[s], splits["val"], grid) for s in seeds}
    pooled = {v: [sum(val_counts[s][v][i] for s in seeds) for i in range(3)] for v in grid}
    best = max(grid, key=lambda v: (f1(pooled[v]), -n_nondefault(v), -abs(v[0] - 0.5)))
    sweeps = {"threshold": [(t, *DEFAULT[1:]) for t in THRESHOLDS], "one_to_one": [(DEFAULT[0], o, *DEFAULT[2:]) for o in O2O],
              "prior": [(*DEFAULT[:2], pr, DEFAULT[3]) for pr in PRIORS], "nearness_K": [(*DEFAULT[:3], k) for k in KS]}
    sweep_vars = sorted({v for lst in sweeps.values() for v in lst} | {best, DEFAULT})
    test = {s: score_variants(caches[s], splits["test"], sweep_vars) for s in seeds}
    res = {"seeds": seeds, "n_val_selection": len(splits["val"]), "n_test": len(splits["test"]), "selected": list(best),
           "selection_pooled_f1": {"default": round(f1(pooled[DEFAULT]), 4), "selected": round(f1(pooled[best]), 4)},
           "test": {"default": {s: round(f1(test[s][DEFAULT]), 4) for s in seeds}, "selected": {s: round(f1(test[s][best]), 4) for s in seeds}},
           "sweeps": {k: {str(list(v)): {s: round(f1(test[s][v]), 4) for s in seeds} for v in lst} for k, lst in sweeps.items()}}
    d = [test[s][best] for s in seeds]
    res["delta_per_seed"] = [round(f1(test[s][best]) - f1(test[s][DEFAULT]), 4) for s in seeds]
    res["delta_mean"] = round(sum(res["delta_per_seed"]) / len(seeds), 4)
    res["verdict"] = {"decode_levers_matter": res["delta_mean"] >= 0.02 and min(res["delta_per_seed"]) >= 0,
                      "representation_limited": res["delta_mean"] < 0.01}
    pr = {s: [test[s][v][i] for i in range(3)] for s in seeds for v in (best,)}
    tp, fp, fn = (sum(test[s][best][i] for s in seeds) for i in range(3))
    res["selected_pooled_test_pr"] = {"precision": round(tp / (tp + fp), 4), "recall": round(tp / (tp + fn), 4)}
    tp, fp, fn = (sum(test[s][DEFAULT][i] for s in seeds) for i in range(3))
    res["default_pooled_test_pr"] = {"precision": round(tp / (tp + fp), 4), "recall": round(tp / (tp + fn), 4)}
    s0 = seeds[0]
    res["group_recall_seed%d" % s0] = {"default": group_recall(caches[s0], splits["test"], DEFAULT), "selected": group_recall(caches[s0], splits["test"], best)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("seeds", "selected", "selection_pooled_f1", "test", "delta_per_seed", "delta_mean", "verdict")}, indent=1))


def _selftest() -> None:
    import torch
    from models.linking_head import BiaffineRelationClassifier, build_soft_type_priors_from_labels
    from xfund_link_errors import rank_of
    rng = np.random.default_rng(0)
    types = ["question", "answer", "header", "question", "answer", "other", "answer"]
    pm = prior_matrix(types)
    assert np.allclose(pm, build_soft_type_priors_from_labels(types).numpy()), "prior algebra diverges from the model's"
    for _ in range(40):
        lg = rng.normal(0, 3, (7, 7)).astype(np.float32)
        for thr in THRESHOLDS:
            for o2o in O2O:
                ref = BiaffineRelationClassifier.decode_predictions(torch.tensor(lg), threshold=thr, one_to_one=o2o)
                assert decode_np(lg, thr, o2o) == ref, (thr, o2o)
    lg = rng.normal(0, 3, (7, 7)).astype(np.float32)
    assert np.allclose(adjust(lg, types, "none"), lg - pm)
    qq = adjust(lg, types, "qq_free"); assert np.isclose(qq[0, 3], lg[0, 3] + 5) and np.isclose(qq[0, 1], lg[0, 1]) and np.isclose(qq[1, 0], lg[1, 0])
    h = adjust(lg, types, "hard"); assert h[0, 3] == NEG and h[0, 1] == lg[0, 1] and h[2, 0] == lg[2, 0]
    boxes = {i: (float(rng.integers(0, 900)), float(rng.integers(0, 900)), 0.0, 0.0) for i in range(7)}
    boxes = {i: (b[0], b[1], b[0] + 20, b[1] + 10) for i, b in boxes.items()}
    ty = dict(enumerate(types))
    rm = rank_matrix(boxes, ty)
    assert all(rm[(s, d)] == rank_of(s, d, boxes, ty) for s in boxes for d in boxes if s != d)
    # K limit and window union on a toy cache
    cache = [{"order": [10, 11], "types": ["question", "answer"], "logits": np.array([[0, 4.0], [-4.0, 0]], dtype=np.float32)}]
    assert decode_doc(cache, DEFAULT, {}) == {(10, 11)}
    assert decode_doc(cache, (0.5, True, "trained", 5), {(10, 11): 7, (11, 10): 7}) == set()
    assert decode_doc(cache, (0.5, True, "hard", 0), {}) == {(10, 11)}
    print("selftest passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "cache", "analyse"])
    ap.add_argument("--ckpt-dir", default="results/xfund_joint_lilt")
    ap.add_argument("--data", default="data/xfund")
    ap.add_argument("--cache-dir", default="results/xfund_decode/cache")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/xfund_decode/legacy.json")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    {"selftest": lambda _: _selftest(), "cache": cmd_cache, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
