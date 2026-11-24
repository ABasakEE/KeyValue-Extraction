"""Write the curated demo examples as files: demo_pdfs/tagger/{vrdu,fatura,docile}/<name>_<g|m|b>.pdf, a README with the gold, and ground_truth.json.

    python scripts/demo_export_pdfs.py

Suffix = the checkpoint's own score on the document, the same bands everywhere: g >= 90, m 50 to 90, b < 50 (the convention of demo_pdfs/showcase).
  * VRDU   : the curated lists of scripts/demo_curate_examples.py; unseen_template/ (Amendment held out) and seen_template/.
  * DocILE : the curated list for split seed 0 (the poster's setting). The PDFs are DocILE documents: they stay out of git (see .gitignore) until the dataset terms are confirmed.
  * FATURA : the six unseen-template invoices of fold 0, scored by BOTH checkpoints; the suffix follows the K = 2 model (g >= 90, m 50 to 90, b < 50), because that is the
             model whose behaviour changes from invoice to invoice; the K = 40 score is in the README.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(HERE)]
import src.demo_all as A  # noqa: E402

DB = A.DB
OUT = ROOT / "demo_pdfs" / "tagger"
def sfx(f1: float) -> str:
    """The same bands for every dataset: g good (F1 >= 90), m moderate (50 to 90), b bad (< 50)."""
    return "g" if f1 >= 90 else ("m" if f1 >= 50 else "b")


def stem_pdf(name: str) -> str:
    return name[:-4] if name.lower().endswith(".pdf") else name


def vrdu(readme: list[str], truth: dict) -> None:
    cur = json.loads((ROOT / "results" / "demo_weights" / "examples_vrdu.json").read_text(encoding="utf-8"))
    by_guid, _, _ = A.vrdu_protocol()
    _, id2label = A.label_maps(A.VRDU_NAME)
    readme += ["", "## VRDU registration forms (BROS Track 1, six fields)", "",
               "| File | Template | Checkpoint | Entity F1 on this form | Gold fields (exact text) |", "|---|---|---|---|---|"]
    for key, folder, model in (("UTL", "unseen_template", "unseen-template model (Amendment held out)"), ("MTL", "seen_template", "seen-template model")):
        d = OUT / "vrdu" / folder
        d.mkdir(parents=True, exist_ok=True)
        for c in cur[key]:
            ex = by_guid[c["guid"]]
            name = f"{stem_pdf(c['guid'])}_{sfx(c['f1'])}.pdf"
            shutil.copy(A.VRDU_DIR / "pdfs" / c["guid"], d / name)
            gold = {f"{t}": " ".join(ex.words[a:b]) for t, a, b in DB.spans_of(ex.ner_tags, id2label)}
            truth[f"vrdu/{folder}/{name}"] = {"checkpoint": key, "entity_f1": c["f1"], "gold": gold}
            readme.append(f"| `vrdu/{folder}/{name}` | {c['template']} | {model} | {c['f1']:.0f}% | " + "; ".join(f"{k}: {v}" for k, v in gold.items()) + " |")


def fatura(readme: list[str], truth: dict) -> None:
    held = A.FL.folds()[0]
    unseen, _ = A.fatura_pick("K2")
    d = OUT / "fatura"
    d.mkdir(parents=True, exist_ok=True)
    readme += ["", "## FATURA invoices (BROS Track 1, nine fields; unseen-template invoices of fold 0)", "",
               "The suffix follows the K = 2 model; both scores are listed. These are image-derived PDFs (the invoices are JPG in the dataset).", "",
               "| File | Template | K = 40 entity F1 | K = 2 entity F1 | Gold fields (exact text) |", "|---|---|---|---|---|"]
    models = {k: A.get_model(A.FATURA_NAME, k) for k in ("K40", "K2")}
    for label, name in unseen:
        ex, pages, _ = A.load_example(A.FATURA_NAME, name)
        sc = {}
        for k, (clf, id2label, dev, trained, ck) in models.items():
            assert trained, f"no checkpoint for FATURA {k}"
            spans = set(DB.spans_of(DB.tag_document(clf, dev, ex), id2label))
            gold = set(DB.spans_of(ex.ner_tags, id2label))
            sc[k] = round(100 * A.f1_text(spans, gold)[0])
        role = sfx(sc["K2"])
        fname = f"{name}_{role}.pdf"
        pages[0].convert("RGB").save(d / fname, "PDF", resolution=72.0)
        g = {t: " ".join(ex.words[a:b]) for t, a, b in DB.spans_of(ex.ner_tags, models["K40"][1])}
        truth[f"fatura/{fname}"] = {"entity_f1": sc, "gold": g, "held_out_templates": held}
        readme.append(f"| `fatura/{fname}` | {label.split(':')[0]} | {sc['K40']}% | {sc['K2']}% | " + "; ".join(f"{k}: {v[:40]}" for k, v in g.items()) + " |")


def docile(readme: list[str], truth: dict) -> None:
    import docile_bros as DBR

    cur = json.loads((ROOT / "results" / "demo_weights" / "examples_docile.json").read_text(encoding="utf-8"))["S0"]
    d = OUT / "docile"
    d.mkdir(parents=True, exist_ok=True)
    readme += ["", "## DocILE line items (BROS Track 1, 19 line-item fields; N = 1,000 split seed 0; unseen layouts)", "",
               "Local only: these are DocILE documents and are not committed (see .gitignore).", "",
               "| File | Row F1 on this document (this checkpoint) | Gold rows |", "|---|---|---|"]
    for c in cur:
        doc = DBR.doc_dict(c["guid"])
        fname = f"{c['guid']}_{sfx(c['f1'])}.pdf"
        shutil.copy(A.DOCILE_DIR / "pdfs" / f"{c['guid']}.pdf", d / fname)
        rows = [{A._short(f): " ".join(doc["words"][i] for i in sorted(ws)) for f, ws, _ in row} for row in doc["rows"]]
        truth[f"docile/{fname}"] = {"row_f1": c["f1"], "gold_rows": rows}
        readme.append(f"| `docile/{fname}` | {c['f1']:.0f}% | {len(rows)} rows; first: " + "; ".join(f"{k}: {v[:30]}" for k, v in (rows[0].items() if rows else [])) + " |")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    readme = ["# demo_pdfs/tagger: examples for `python demo_all.py`", "",
              "Documents of the three tagger datasets, chosen by the demo's own checkpoints (not by file order): a mix of good, moderate and failing cases, "
              "so a viewer sees what the model gets right and wrong. All are **held out** from the checkpoint they are paired with.", "",
              "**Using them:** the dropdown in `demo_all.py` lists the same documents and scores them with the dataset's own words and boxes (what the model was trained on). "
              "Uploading a file from this folder instead reads it with RapidOCR, so the scores can differ from the dropdown labels and no gold comparison is shown.", "",
              "File suffix: `_g` good (F1 >= 90), `_m` moderate (50 to 90), `_b` bad (< 50); the score behind it is the one printed below.",
              "Entity F1 is exact-span on the dataset's words; DocILE rows use the line-item row F1."]
    truth: dict = {}
    vrdu(readme, truth)
    fatura(readme, truth)
    docile(readme, truth)
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (OUT / "ground_truth.json").write_text(json.dumps(truth, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", sum(1 for _ in OUT.rglob("*.pdf")), "pdfs;", OUT)


if __name__ == "__main__":
    main()
