"""Step 0 for the encoder question: where do gold-entity link errors come from?  (CPU, descriptive.)

Runs the joint-LiLT checkpoints with the annotation's entity spans and types (windowed, 510 tokens, stride 128, exactly as
xfund_link_eval.py) and classifies every gold link (found / missed) and every spurious predicted link:
  gold links   by endpoint types; by geometry of the true partner relative to the source (same line right/left, below, above);
               by rank of the true partner among same-type entities ordered by distance from the source (0 = nearest);
               and whether any 510-token window contains both endpoints (unreachable links cannot be found).
  spurious     reversed direction; source has a different true partner (with the rank of the predicted partner);
               source has no true partner at all.

    python scripts/xfund_link_errors.py --ckpt-dir results/xfund_joint_lilt --out results/xfund_link_errors/legacy.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    sys.path.insert(0, p)

from data.xfund_loader import ID2LABEL, LABEL_LIST, load_xfund_split  # noqa: E402
from train import build_model  # noqa: E402
from xfund_link_eval import LANGS, predict, sub_example, windows  # noqa: E402


def entity_boxes(ex) -> dict[int, tuple[float, float, float, float]]:
    out = {}
    for i, e in ex.entities.items():
        idx = e.get("token_indices", [])
        if idx:
            bs = [ex.boxes[w] for w in idx]
            out[i] = (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
    return out


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def direction(bs, bd) -> str:
    (sx, sy), (dx, dy) = centre(bs), centre(bd)
    if abs(dy - sy) < 0.5 * max(bs[3] - bs[1], bd[3] - bd[1], 1):
        return "same line, partner right" if dx >= sx else "same line, partner left"
    return "partner below" if dy > sy else "partner above"


def rank_of(src: int, dst: int, boxes, types) -> int:
    """Rank of dst among entities of its type, ordered by centre distance from src (0 = nearest)."""
    cs = centre(boxes[src])
    cand = sorted((math.dist(cs, centre(boxes[j])), j) for j in boxes if j != src and types[j] == types[dst])
    return [j for _, j in cand].index(dst)


def bucket(r: int) -> str:
    return "0 (nearest)" if r == 0 else "1" if r == 1 else "2-4" if r <= 4 else "5+"


def analyse(model, docs, size: int, stride: int) -> dict:
    gold_c: dict[str, list[int]] = defaultdict(lambda: [0, 0])      # category -> [found, total]
    fp = Counter()
    fp_rank = Counter()
    totals = Counter()
    for ex in docs:
        boxes = entity_boxes(ex)
        types = {i: e["label"].lower() for i, e in ex.entities.items()}
        gold = {(s, d) for s, d in ex.links if s != d and s in boxes and d in boxes}
        pred, reach = set(), set()
        for wa, wb in windows(model, ex, size, stride):
            w = sub_example(ex, wa, wb)
            p, g = predict(model, w)
            pred |= p
            reach |= g
        by_src = defaultdict(set)
        for s, d in gold:
            by_src[s].add(d)
        for s, d in gold:
            found = (s, d) in pred
            r = rank_of(s, d, boxes, types)
            for cat in (f"type: {types[s]} -> {types[d]}", f"geometry: {direction(boxes[s], boxes[d])}",
                        f"true-partner rank: {bucket(r)}", f"reachable: {(s, d) in reach}"):
                gold_c[cat][0] += found
                gold_c[cat][1] += 1
            totals["gold"] += 1
            totals["found"] += found
        for s, d in pred - gold:
            if s not in boxes or d not in boxes:
                continue
            totals["spurious"] += 1
            if (d, s) in gold:
                fp["reversed direction"] += 1
            elif s in by_src:
                fp["source has a different true partner"] += 1
                fp_rank[bucket(rank_of(s, d, boxes, types)) if types[d] in {types[x] for x in by_src[s]} else "predicted partner of another type"] += 1
            else:
                fp["source has no true partner"] += 1
    return {"totals": dict(totals), "gold_links": {k: {"found": v[0], "total": v[1], "recall": round(v[0] / v[1], 4)} for k, v in sorted(gold_c.items())},
            "spurious": dict(fp), "spurious_wrong_partner_rank": dict(fp_rank)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data", default="data/xfund")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--size", type=int, default=510)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    ckpt = Path(a.ckpt_dir)
    cfg = SimpleNamespace(**json.loads((ckpt / "metrics.json").read_text(encoding="utf-8"))["config"])
    val = load_xfund_split(a.data, "val", LANGS)
    if a.limit:
        val = val[: a.limit]
    res = {"n_val": len(val), "seeds": {}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for seed in (a.seeds or list(cfg.seeds)):
        model = build_model(cfg, len(LABEL_LIST), ID2LABEL)
        sd = torch.load(ckpt / f"seed_{seed}" / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
        model.eval()
        res["seeds"][seed] = analyse(model, val, a.size, a.stride)
        t = res["seeds"][seed]["totals"]
        print(seed, "recall", round(t["found"] / t["gold"], 4), "spurious", t.get("spurious"), flush=True)
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")  # a crash on a later seed keeps the earlier ones
        del model


if __name__ == "__main__":
    main()
