"""Study 27, arm V4: write the entities a trained joint-LiLT tagger predicts for every XFUND val form (the VLM then only links them).

    python scripts/xfund_dump_entities.py --ckpt-dir results/xfund_vlm/lilt_A0 --seed 13 --out results/xfund_vlm/lilt_entities.jsonl
    python scripts/xfund_dump_entities.py --stub --limit 3 --out /tmp/ents.jsonl     # CPU plumbing check: the gold entities, not predictions

One row per form: guid, entities [[label, first word, end word], ...] (the entities of ``e2e_link.predict_document`` with entity-safe 510-token windows, stride 128,
the 'windowed' reading of ``xfund_e2e_eval.py``), links [[[a, b], [c, d]], ...] (the tagger's own links, for reference only; V4 does not use them).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    sys.path.insert(0, p)

from data.xfund_loader import ID2LABEL, LABEL_LIST, load_xfund_split  # noqa: E402
from e2e_link import gold_structure  # noqa: E402
from vlm_link import form_order  # noqa: E402
from xfund_link_eval import LANGS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--data", default="data/xfund")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stub", action="store_true", help="write the gold entities instead of predictions (pipeline check only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    forms = form_order(load_xfund_split(a.data, "val", LANGS))
    if a.limit:
        forms = forms[: a.limit]
    model = predict = None
    if not a.stub:
        import torch
        from data.xfund_windows import word_windows
        from e2e_link import predict_document
        from train import build_model

        ckpt = Path(a.ckpt_dir)
        cfg = SimpleNamespace(**json.loads((ckpt / "metrics.json").read_text(encoding="utf-8"))["config"])
        model = build_model(cfg, len(LABEL_LIST), ID2LABEL)
        sd = torch.load(ckpt / f"seed_{a.seed}" / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
        predict = lambda ex: predict_document(model, ex, word_windows(model.tokenizer, ex.words, 510, a.stride))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for ex in forms:
            ents, links = gold_structure(ex) if a.stub else predict(ex)
            fh.write(json.dumps({"guid": ex.guid, "entities": sorted([lab, int(s), int(e)] for lab, s, e in ents),
                                 "links": sorted([list(map(int, src)), list(map(int, dst))] for src, dst in links)}, ensure_ascii=False) + "\n")
    print(f"wrote {len(forms)} forms to {out} ({'gold entities, stub' if a.stub else f'predicted by {a.ckpt_dir} seed {a.seed}'})")


if __name__ == "__main__":
    main()
