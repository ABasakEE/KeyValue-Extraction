"""Rank the 250 XFUND validation forms by how well the finished BROS joint model (study 37, J1) links them, for report and demo-video examples.

    python scripts/rank_xfund_examples.py --ckpt-dir results/bros_joint/J1/seed_42 [--out results/xfund_examples/ranking.json]

Per form, end to end (the model's own entities, windowed 510/128 inference, as the study-37 scorer): entity F1, link F1, gold links. 'sibling_free' = no training form with the same
questions (an approximately unseen template). Gold words are used (no OCR), so this ranks the model, not the OCR; the demo's OCR path is then run on the top forms.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(ROOT)]
import torch  # noqa: E402
from data.xfund_loader import load_xfund_split  # noqa: E402
from data.xfund_windows import word_windows  # noqa: E402
from e2e_link import gold_structure, predict_document, prf  # noqa: E402
from xfund_link_eval import LANGS, sibling_free  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(ROOT / "results" / "bros_joint" / "J1" / "seed_42"))
    ap.add_argument("--out", default=str(ROOT / "results" / "xfund_examples" / "ranking.json"))
    a = ap.parse_args()
    from models.joint_extractor import JointFormExtractor

    d = Path(a.ckpt_dir)
    cfg = json.loads((d.parent / "metrics.json").read_text(encoding="utf-8"))["config"]
    id2label = {int(k): v for k, v in json.loads((d / "config.json").read_text())["id2label"].items()} if (d / "config.json").exists() else None
    val, train = load_xfund_split(ROOT / "data" / "xfund", "val", LANGS), load_xfund_split(ROOT / "data" / "xfund", "train", LANGS)
    keep = sibling_free(val, train)
    sys.path.insert(0, str(ROOT / "src"))
    import importlib
    demo = importlib.import_module("src.demo")
    model = JointFormExtractor(cfg["backbone"], len(demo.ID2LABEL), demo.ID2LABEL, spatial_in_features=int(cfg["spatial_in_features"]))
    st = torch.load(d / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(st["model_state_dict"] if "model_state_dict" in st else st)
    model.eval()
    rows = []
    for k, ex in enumerate(val, 1):
        g_ents, g_links = gold_structure(ex)
        ents, links = predict_document(model, ex, word_windows(model.tokenizer, ex.words, 510, 128))
        e_tp, l_tp = len(ents & g_ents), len(links & g_links)
        rows.append({"guid": ex.guid, "sibling_free": ex.guid in keep, "gold_entities": len(g_ents), "gold_links": len(g_links), "pred_links": len(links),
                     "entity_f1": round(100 * prf(e_tp, len(ents), len(g_ents))["f1"] if isinstance(prf(e_tp, len(ents), len(g_ents)), dict) else 0, 1),
                     "link_f1": round(100 * prf(l_tp, len(links), len(g_links))["f1"] if isinstance(prf(l_tp, len(links), len(g_links)), dict) else 0, 1)})
        if k % 25 == 0:
            print(k, "forms", flush=True)
    rows.sort(key=lambda r: (-r["link_f1"], -r["gold_links"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("done", a.out)


if __name__ == "__main__":
    main()
