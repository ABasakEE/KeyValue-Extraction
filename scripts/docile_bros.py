"""Study 34, DocILE: ceilings of a BROS line-item tagger plus row grouping (CPU oracle; the training code comes after the ad-buy results).

    python scripts/docile_bros.py oracle

Gold cells (word sets) are tagged as BIO runs over the 19 line-item fields, decoded to spans, then (1) grouped by their true row, and
(2) grouped by `adbuy_bros.group_rows` (anchor field per page, tolerance 0.5). A row may hold several cells of one field in DocILE.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
import adbuy_bros as B  # noqa: E402
import docile_rows as D  # noqa: E402
import docile_sg as G  # noqa: E402
from lir_metric import cell_counts, f1, lir_counts  # noqa: E402

FIELDS = sorted(G.FIELDS)


def doc_dict(i: str) -> dict:
    doc = D.Doc(i, None)
    ann = G.ann_of(i)
    rows = [[(f, ws, "") for f, ws, _ in row] for row in D.gold_rows(doc, ann)]
    return {"words": [w[1] for w in doc.words], "fboxes": [w[2] for w in doc.words], "pages": [w[0] for w in doc.words], "rows": rows}


def spans_of(d: dict) -> list[list]:
    tags = ["O"] * len(d["words"])
    for row in d["rows"]:
        for f, ws, _ in row:
            for s, e in B.runs(ws):
                if all(tags[i] == "O" for i in range(s, e)):
                    tags[s] = f"B-{f}"
                    for i in range(s + 1, e):
                        tags[i] = f"I-{f}"
    return sorted([t, s, e, ""] for t, s, e in B.extract_spans(tags))


def counts(pred: list[set], d: dict) -> dict:
    g = [{(f, ws) for f, ws, _ in row} for row in d["rows"]]
    return {"lir": lir_counts(pred, g), "cell": cell_counts(pred, g)}


def cmd_oracle() -> None:
    sp = json.loads((D.ROOT / "results" / "docile_rows" / "splits.json").read_text(encoding="utf-8"))
    sets = {"test (200 documents)": sp["test"], "training documents, seed 0, N = 200": sp["train"]["0"][:200]}
    for name, ids in sets.items():
        tot_true, tot_clu = {}, {}
        n_cells = n_split = n_dup = 0
        for i in ids:
            d = doc_dict(i)
            if not d["rows"]:
                continue
            n_cells += sum(len(r) for r in d["rows"])
            n_split += sum(len(B.runs(ws)) > 1 for r in d["rows"] for _, ws, _ in r)
            n_dup += sum(len(r) - len({f for f, _, _ in r}) for r in d["rows"])
            spans = spans_of(d)
            true_row = {(f, i2): k for k, r in enumerate(d["rows"]) for f, ws, _ in r for i2 in ws}
            byrow = defaultdict(set)
            for f, s, e, _ in spans:
                byrow[true_row.get((f, s), -1)].add((f, frozenset(range(s, e))))
            B.add(tot_true, counts(list(byrow.values()), d))
            B.add(tot_clu, counts(B.group_rows(spans, d, 0.5, merge=False), d))
        show = lambda t: {"row_f1": round(f1(t["lir"]) * 100, 2), "cell_f1": round(f1(t["cell"]) * 100, 2)}
        print(f"{name}: {n_cells} gold cells, {100 * n_split / n_cells:.1f}% not consecutive words, {100 * n_dup / n_cells:.1f}% repeat a field already in their row\n"
              f"   tags -> spans with true rows : {show(tot_true)}\n   tags -> spans -> clustered rows: {show(tot_clu)}")


if __name__ == "__main__":
    {"oracle": cmd_oracle}[sys.argv[1]]()
