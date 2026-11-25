"""Ad-Buy line-item examples for demo_pdfs/tagger/adbuy: good (_g, row F1 >= 90), medium (_m, 50 to 90) and bad (_b, below 50) documents, from SAVED predictions.

    python scripts/demo_export_adbuy.py

No Ad-Buy weights were kept (configs/adbuy_rows_bros.yaml has save_weights: false), so the demo cannot tag these live. The files here come from the saved test predictions
(results/adbuy_bros/run/N200_SD0/seed_13: Track-1 BROS trained on 200 documents of the split's seen templates, scored on documents of unseen templates) and the anchor rule:
the PDF, a preview PNG with the predicted cells and numbered rows drawn on it, and the gold rows in the README.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(HERE)]
import adbuy_bros as AB  # noqa: E402
import src.demo_all as A  # noqa: E402
from lir_metric import f1 as lir_f1  # noqa: E402
from row_group import group_rows  # noqa: E402

RUN = "N200_SD0"
OUT = ROOT / "demo_pdfs" / "tagger" / "adbuy"
PALETTE = {"channel": "#3cb44b", "program_desc": "#4363d8", "program_end_date": "#f58231", "program_start_date": "#911eb4", "sub_amount": "#e6194b"}


def main() -> None:
    primary = json.loads((ROOT / "results" / "adbuy_sg" / "primary.json").read_text(encoding="utf-8"))
    docs = AB.load_docs(set(primary))
    preds = {r["guid"]: r for r in map(json.loads, (ROOT / "results" / "adbuy_bros" / "run" / RUN / "seed_13" / "test_entities.jsonl").read_text(encoding="utf-8").splitlines())}
    scored = []
    for g in primary:
        d = docs[g]
        rows = [r for r in group_rows(preds[g]["pred"], d, 0.5, merge=True) if r]
        c = AB.run_counts(rows, d)
        scored.append({"guid": g, "f1": 100 * lir_f1(c["lir"]), "gold_rows": len(d["rows"]), "pred_rows": len(rows), "pages": max(d["pages"], default=0) + 1, "words": len(d["words"])})
    ok = [s for s in scored if 5 <= s["gold_rows"] <= 20 and s["pages"] <= 3 and s["words"] <= 1500]
    good = sorted([s for s in ok if s["f1"] >= 90], key=lambda s: -s["gold_rows"])
    medium = sorted([s for s in ok if 50 <= s["f1"] < 90], key=lambda s: s["f1"])
    bad = sorted([s for s in scored if s["f1"] < 50 and 3 <= s["gold_rows"] <= 20 and s["pages"] <= 3 and s["words"] <= 1500], key=lambda s: s["f1"])
    pick = lambda xs, n: [xs[i] for i in sorted({round(k * (len(xs) - 1) / max(1, n - 1)) for k in range(n)})]
    chosen = [(s, "g") for s in pick(good, 3)] + [(s, "m") for s in pick(medium, 3)] + [(s, "b") for s in pick(bad, 2)]
    (OUT / "previews").mkdir(parents=True, exist_ok=True)
    readme = ["# demo_pdfs/tagger/adbuy: VRDU Ad-Buy line items (saved predictions, not a live model)", "",
              "Ad-Buy political-ad invoices (641 documents, dozens of templates, five line-item fields: channel, program description, start date, end date, amount). "
              f"The files are documents of templates the model did not train on: Track-1 BROS trained on 200 documents (`{RUN}`, training seed 13), cells grouped into rows by the anchor rule. "
              "`_g` = row F1 on this document >= 90, `_m` = 50 to 90, `_b` = below 50 (failure cases). Each PDF has a preview PNG in `previews/` with the predicted cells (coloured by field) and numbered rows; dashed blue boxes are gold cells that were missed.", "",
              "**Not live:** no Ad-Buy checkpoint was kept (`save_weights: false`), so `demo_all.py` cannot run these; train one with `notebooks/kaggle_adbuy_bros.ipynb` after setting `save_weights: true` if you need that. "
              "Over the 143 documents scored in every split this run averages "
              f"{sum(s['f1'] for s in scored) / len(scored):.0f}% per-document row F1 (the study's pooled figure for N = 200 is 85.7 over three splits).", "",
              "| File | Row F1 (this document) | Gold rows / predicted rows | Gold rows (first two) |", "|---|---|---|---|"]
    truth = {}
    for s, role in chosen:
        g = s["guid"]
        d, ex = docs[g], AB.to_example(g, docs[g])
        stem = g[:-4] if g.endswith(".pdf") else g
        fname = f"{stem}_{role}.pdf"
        shutil.copy(A.ROOT / "data" / "vrdu" / "ad-buy-form" / "main" / "pdfs" / g, OUT / fname)
        pages = A.render_pdf(A.ROOT / "data" / "vrdu" / "ad-buy-form" / "main" / "pdfs" / g, 3)
        spans = [(f, a, b) for f, a, b, _ in preds[g]["pred"]]
        gold = {(f, a, b) for f, a, b, _ in preds[g]["gold"]}
        rows = sorted([r for r in group_rows(preds[g]["pred"], d, 0.5, merge=True) if r], key=lambda cells: (min(d["pages"][min(ws)] for _, ws in cells), min(d["fboxes"][min(ws)][1] for _, ws in cells)))
        marks = [(d["pages"][min(ws)], d["fboxes"][min(ws)][1]) for cells in rows for _, ws in [min(cells, key=lambda c: (d["pages"][min(c[1])], d["fboxes"][min(c[1])][1]))]]
        imgs = A.draw(pages, ex, spans, gold, PALETTE, marks, label_once=True)
        used = sorted({d["pages"][a] for _, a, _ in spans} | {d["pages"][a] for _, a, _ in gold})
        for pi in used:
            if pi < len(imgs):
                imgs[pi].save(OUT / "previews" / f"{stem}_{role}_p{pi + 1}.png")
        gold_rows = [{f: " ".join(d["words"][i] for i in sorted(ws)) for f, ws, _ in row} for row in d["rows"]]
        truth[fname] = {"row_f1": round(s["f1"], 1), "gold_rows": gold_rows, "run": RUN}
        readme.append(f"| `{fname}` | {s['f1']:.0f}% | {s['gold_rows']} / {s['pred_rows']} | " + "; ".join(", ".join(f"{k}: {v[:22]}" for k, v in r.items()) for r in gold_rows[:2]) + " |")
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (OUT / "ground_truth.json").write_text(json.dumps(truth, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", len(chosen), "documents to", OUT)
    for s, role in chosen:
        print(role, round(s["f1"]), s["gold_rows"], s["pred_rows"], s["guid"][:12])


if __name__ == "__main__":
    main()
