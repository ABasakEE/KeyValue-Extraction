"""Per-document scores used to pick good / bad demo documents (see demo_pdfs/README.md).

    python scripts/demo_select_examples.py vrdu   --out scratch/demo_scores_vrdu.json
    python scripts/demo_select_examples.py xfund  --out scratch/demo_scores_xfund.json
    python scripts/demo_select_examples.py funsd  --out scratch/demo_scores_funsd.json

vrdu  : per-document exact-span entity F1 of Track-1 BROS on unseen templates, from the saved error dumps
        (CPU, no model run; mean over seeds 13 and 42).
xfund : per-form end-to-end entity F1 and link F1 of the legacy joint-LiLT checkpoints, re-merged from the cached
        per-window output (CPU, no model run; mean over 3 seeds).
funsd : per-form entity F1 and gold-entity link F1 of results/funsd_joint_lilt (the checkpoint the Gradio demo loads),
        run on CPU on the 50 FUNSD test forms.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics as st
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    sys.path.insert(0, p)


def prf(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def vrdu() -> dict:
    per = collections.defaultdict(lambda: {"f1": [], "template": None, "gold": 0, "fields_ok": None})
    for path in sorted(glob.glob(str(ROOT / "results/vrdu_track1_bros_errdump/UTL_fold*_seed*/test_entities.jsonl"))):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            gold = {(e[0], e[1], e[2]) for e in r["gold"]}
            pred = {(e[0], e[1], e[2]) for e in r["pred"]}
            tp = len(gold & pred)
            d = per[r["guid"]]
            d["f1"].append(prf(tp, len(pred - gold), len(gold - pred)))
            d["template"] = r["template"]
            d["gold"] = len(gold)
            d["gold_fields"] = sorted({g[0] for g in gold})
            d["missed_fields"] = sorted({g[0] for g in gold - pred})
    return {g: {"entity_f1": round(st.mean(d["f1"]), 3), "template": d["template"], "n_gold": d["gold"],
                "gold_fields": d["gold_fields"], "missed_fields": d["missed_fields"]} for g, d in per.items()}


def xfund() -> dict:
    from data.xfund_loader import load_xfund_split
    from e2e_link import gold_structure, merge_windows
    from xfund_link_eval import LANGS

    val = load_xfund_split(str(ROOT / "data/xfund"), "val", LANGS)
    out: dict[str, dict] = {}
    for ex in val:
        g_ents, g_links = gold_structure(ex)
        ef, lf = [], []
        for seed in (13, 21, 42):
            raw = _raw(seed)
            ents, links = merge_windows(raw[ex.guid], len(ex.words), "edge_drop")
            ef.append(prf(len(ents & g_ents), len(ents - g_ents), len(g_ents - ents)))
            lf.append(prf(len(links & g_links), len(links - g_links), len(g_links - links)))
        out[ex.guid] = {"entity_f1": round(st.mean(ef), 3), "link_f1": round(st.mean(lf), 3),
                        "n_entities": len(g_ents), "n_links": len(g_links), "n_words": len(ex.words),
                        "lang": ex.guid.split("_")[0]}
    return out


_RAW: dict[int, dict] = {}


def _raw(seed: int) -> dict:
    if seed not in _RAW:
        _RAW[seed] = json.loads((ROOT / f"results/xfund_dedupe/cache/legacy_seed{seed}.json").read_text(encoding="utf-8"))
    return _RAW[seed]


def funsd() -> dict:
    import torch
    from torch.utils.data import DataLoader
    from data.funsd_loader import ID2LABEL, LABEL_LIST, load_funsd_split
    from evaluate import entity_f1, linking_f1
    from train import TrainConfig, build_model, make_collator, predict_joint

    ck = ROOT / "results/funsd_joint_lilt"
    cfg = TrainConfig.from_yaml(ROOT / "configs/funsd_joint_lilt.yaml")
    cfg.spatial_in_features = json.loads((ck / "config.json").read_text(encoding="utf-8"))["spatial_in_features"]  # 10-D checkpoint
    test = load_funsd_split(str(ROOT / "data/funsd/dataset"), "test")
    model = build_model(cfg, len(LABEL_LIST), ID2LABEL)
    sd = torch.load(ck / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    model.eval()
    collate = make_collator(model, is_joint=True)
    out = {}
    for ex in test:
        loader = DataLoader([ex], batch_size=1, collate_fn=collate)
        pred, ref, plinks, glinks = predict_joint(model, loader, torch.device("cpu"))
        e = entity_f1(pred, ref, model.id2label)
        pl, gl = plinks[0], glinks[0]
        out[ex.guid] = {"entity_f1": round(e.f1, 3), "link_f1": round(prf(len(pl & gl), len(pl - gl), len(gl - pl)), 3),
                        "n_entities": len(ex.entities), "n_links": len(gl), "n_words": len(ex.words)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["vrdu", "xfund", "funsd"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"vrdu": vrdu, "xfund": xfund, "funsd": funsd}[a.which]()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(a.which, len(res), "documents ->", a.out)


if __name__ == "__main__":
    main()
