"""Study 20: end-to-end XFUND link/entity F1 under different ways of merging overlapping windows (CPU).

    python scripts/xfund_dedupe_eval.py --ckpt-dir results/xfund_joint_lilt --tag legacy --out results/xfund_dedupe/legacy.json

The model runs once per seed and form; its per-window output is cached in --cache-dir, so a later run (or another strategy)
re-merges without the model.  Strategies are defined in src/e2e_link.py::merge_windows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    sys.path.insert(0, p)

from data.xfund_loader import ID2LABEL, LABEL_LIST, load_xfund_split  # noqa: E402
from data.xfund_windows import word_windows  # noqa: E402
from e2e_link import add_counts, gold_structure, merge_windows, predict_windows, prf  # noqa: E402
from train import build_model  # noqa: E402
from xfund_link_eval import LANGS, sibling_free  # noqa: E402

STRATEGIES = ("edge_drop", "center", "edge_drop+center")


def raw_for_seed(ckpt: Path, cfg, seed: int, docs, stride: int, cache: Path) -> dict[str, list[dict]]:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    model = build_model(cfg, len(LABEL_LIST), ID2LABEL)
    sd = torch.load(ckpt / f"seed_{seed}" / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    model.eval()
    raw = {}
    for i, ex in enumerate(docs):
        raw[ex.guid] = predict_windows(model, ex, word_windows(model.tokenizer, ex.words, 510, stride))
        if (i + 1) % 25 == 0:
            print(f"  seed {seed}: {i + 1}/{len(docs)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(raw), encoding="utf-8")
    return raw


def score(raw: dict, docs, keep: set[str], strategy: str) -> dict:
    counts: dict[str, list[int]] = {}
    for ex in docs:
        g_ents, g_links = gold_structure(ex)
        ents, links = merge_windows(raw[ex.guid], len(ex.words), strategy)
        for sub in ("all", "sibling_free"):
            if sub == "sibling_free" and ex.guid not in keep:
                continue
            add_counts(counts.setdefault(f"{sub}|entity", [0, 0, 0]), ents, g_ents)
            add_counts(counts.setdefault(f"{sub}|link", [0, 0, 0]), links, g_links)
    return {k: prf(*v) for k, v in counts.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="data/xfund")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default="results/xfund_dedupe/cache")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    ckpt = Path(a.ckpt_dir)
    cfg = SimpleNamespace(**json.loads((ckpt / "metrics.json").read_text(encoding="utf-8"))["config"])
    seeds = a.seeds or list(cfg.seeds)
    val, train = load_xfund_split(a.data, "val", LANGS), load_xfund_split(a.data, "train", LANGS)
    keep = sibling_free(val, train)
    if a.limit:
        val = val[: a.limit]
    res = {"tag": a.tag, "stride": a.stride, "n_val": len(val), "seeds": {}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        raw = raw_for_seed(ckpt, cfg, seed, val, a.stride, Path(a.cache_dir) / f"{a.tag}_seed{seed}.json")
        res["seeds"][seed] = {s: score(raw, val, keep, s) for s in STRATEGIES}
        print(seed, {s: (v["all|link"]["f1"], v["all|entity"]["f1"]) for s, v in res["seeds"][seed].items()}, flush=True)
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")  # a crash on a later seed keeps the earlier ones


if __name__ == "__main__":
    main()
