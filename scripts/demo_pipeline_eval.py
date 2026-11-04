"""How well does the Gradio demo pipeline (src/demo.py: RapidOCR + colon splitting + funsd joint LiLT + link rules)
reproduce the gold entities and links of one FUNSD / XFUND form?  Text-matched, so approximate.

    python scripts/demo_pipeline_eval.py funsd  --out scratch/demo_pipe_funsd.json [--limit N]
    python scripts/demo_pipeline_eval.py xfund  --out scratch/demo_pipe_xfund.json --ids es_val_38 fr_val_7 ...

A demo entity matches a gold entity if their texts agree (spaces/punctuation/case ignored, similarity >= 0.75) and the
labels agree. A demo pair (key text, value text) matches a gold link if both ends match. Each demo item and each gold
item is used at most once.
"""
from __future__ import annotations
import argparse, difflib, io, json, re, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import demo as D  # noqa: E402  (src/demo.py)
from PIL import Image  # noqa: E402


def norm(t: str) -> str:
    return re.sub(r"[\W_]+", "", t.lower())


def same(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and difflib.SequenceMatcher(None, a, b).ratio() >= 0.75


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return round(2 * p * r / (p + r), 3) if p + r else 0.0


def score(js: dict, gold_ents: list[dict], gold_links: list[tuple[str, str]]) -> dict:
    ents = [(e["text"], e["label"]) for e in js["entities"]]
    used, tp_e = set(), 0
    for gt, gl in [(g["text"], g["label"]) for g in gold_ents]:
        for i, (t, l) in enumerate(ents):
            if i not in used and l == gl and same(t, gt):
                used.add(i); tp_e += 1; break
    pairs = [(p["Key (Question)"], p["Value (Answer)"]) for p in js["extracted_pairs"]]
    used, tp_l = set(), 0
    for gk, gv in gold_links:
        for i, (k, v) in enumerate(pairs):
            if i not in used and same(k, gk) and same(v, gv):
                used.add(i); tp_l += 1; break
    return {"entity_f1": prf(tp_e, len(ents), len(gold_ents)), "link_f1": prf(tp_l, len(pairs), len(gold_links)),
            "n_demo_entities": len(ents), "n_gold_entities": len(gold_ents),
            "n_demo_pairs": len(pairs), "n_gold_links": len(gold_links), "link_tp": tp_l}


def gold_funsd(stem: str):
    d = json.loads((ROOT / f"data/funsd/dataset/testing_data/annotations/{stem}.json").read_text(encoding="utf-8"))["form"]
    by = {e["id"]: e for e in d}
    ents = [{"text": e["text"], "label": e["label"]} for e in d if e["text"].strip()]
    links = {(a, b) for e in d for a, b in e.get("linking", [])}
    return ents, [(by[a]["text"], by[b]["text"]) for a, b in links if a in by and b in by]


def gold_xfund(stem: str):
    lang = stem.split("_")[0]
    doc = next(x for x in json.loads((ROOT / f"data/xfund/{lang}.val.json").read_text(encoding="utf-8"))["documents"]
               if x["id"] == stem)
    by = {e["id"]: e for e in doc["document"]}
    ents = [{"text": e["text"], "label": e["label"]} for e in doc["document"] if e["text"].strip()]
    links = {tuple(l) for e in doc["document"] for l in e.get("linking", [])}
    return ents, [(by[a]["text"], by[b]["text"]) for a, b in links if a in by and b in by]


def pdf_for(which: str, stem: str) -> Path:
    img = ROOT / (f"data/funsd/dataset/testing_data/images/{stem}.png" if which == "funsd"
                  else f"scratch/dl/xfund_{stem.split('_')[0]}/{stem}.jpg")
    out = Path(tempfile.gettempdir()) / f"demo_eval_{which}_{stem}.pdf"
    Image.open(img).convert("RGB").save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["funsd", "xfund"])
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.ids is None:
        ids = sorted(p.stem for p in (ROOT / "data/funsd/dataset/testing_data/annotations").glob("*.json"))
    else:
        ids = a.ids
    if a.limit:
        ids = ids[: a.limit]
    res = {}
    for i, stem in enumerate(ids):
        ents, links = (gold_funsd if a.which == "funsd" else gold_xfund)(stem)
        _, _, js = D.process_uploaded_document(str(pdf_for(a.which, stem)), a.threshold, False, False, False)
        res[stem] = score(js, ents, links)
        print("DONE", i + 1, len(ids), stem, res[stem]["entity_f1"], res[stem]["link_f1"], flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
