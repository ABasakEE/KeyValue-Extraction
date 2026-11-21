"""Study 34, DocILE: how much of the column link (header -> cell) does geometry alone solve? (CPU; annotations only, no model, no OCR)

    python scripts/docile_headers.py

For every gold line-item cell, a rule picks a header box of the same page and the cell gets that header's field type; the score is how often the type
equals the cell's gold type. The header types are the gold ones, so this measures the linking geometry only (a model would still have to read the header text).
Rules:  R1 = the nearest header above the cell whose horizontal extent overlaps the cell's by at least 30% of the narrower box (no link if none);
        R2 = R1, and if none qualifies the header with the largest horizontal overlap (below or beside), else the nearest by centre distance.
Cells whose gold type has no header box on their page cannot be recovered by any rule and are reported apart.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "docile"


def xoverlap(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(1e-9, min(a[2] - a[0], b[2] - b[0]))


def pick(cell: list[float], headers: list[dict], rule: str) -> dict | None:
    cy = (cell[1] + cell[3]) / 2
    above = [h for h in headers if h["bbox"][3] <= cy and xoverlap(cell, h["bbox"]) >= 0.3]
    if above:
        return min(above, key=lambda h: cy - h["bbox"][3])
    if rule == "R1" or not headers:
        return None
    best = max(headers, key=lambda h: xoverlap(cell, h["bbox"]))
    if xoverlap(cell, best["bbox"]) > 0:
        return best
    cx = (cell[0] + cell[2]) / 2
    return min(headers, key=lambda h: (cx - (h["bbox"][0] + h["bbox"][2]) / 2) ** 2 + (cy - (h["bbox"][1] + h["bbox"][3]) / 2) ** 2)


def run(ids: list[str]) -> dict:
    tot: dict = {r: Counter() for r in ("R1", "R2")}
    per_field: dict = {r: defaultdict(Counter) for r in ("R1", "R2")}
    by_kind: dict = {r: defaultdict(Counter) for r in ("R1", "R2")}
    n_cells = unrecoverable = 0
    for i in ids:
        a = json.loads((DATA / "annotations" / f"{i}.json").read_text(encoding="utf-8"))
        heads = defaultdict(list)
        for h in a["line_item_headers"]:
            heads[h["page"]].append(h)
        for e in a["line_item_extractions"]:
            n_cells += 1
            hs = heads.get(e["page"], [])
            has = any(h["fieldtype"] == e["fieldtype"] for h in hs)
            unrecoverable += not has
            kind = "no header on the page" if not hs else "one header per field" if len({h["fieldtype"] for h in hs}) == len(hs) else "a field has several headers (several tables or repeats)"
            for r in tot:
                h = pick(e["bbox"], hs, r)
                for c in (tot[r], per_field[r][e["fieldtype"]], by_kind[r][kind]):
                    c["cells"] += 1
                    c["linked"] += h is not None
                    c["right"] += h is not None and h["fieldtype"] == e["fieldtype"]
                    c["recoverable"] += has
                    c["right_among_recoverable"] += has and h is not None and h["fieldtype"] == e["fieldtype"]
    return {"documents": len(ids), "cells": n_cells, "cells_with_no_same_type_header_on_page": unrecoverable, "tot": tot, "per_field": per_field, "by_kind": by_kind}


def show(name: str, res: dict) -> None:
    n = res["cells"]
    print(f"\n== {name}: {res['documents']} documents, {n} line-item cells; {100 * res['cells_with_no_same_type_header_on_page'] / n:.1f}% have no header of their type on the page (no rule can recover them)")
    for r in ("R1", "R2"):
        c = res["tot"][r]
        print(f"  {r}: linked {100 * c['linked'] / n:5.1f}% | type right on {100 * c['right'] / n:5.1f}% of all cells, {100 * c['right_among_recoverable'] / c['recoverable']:5.1f}% of the recoverable ones")
    print("  by page kind (R2, right among recoverable):", {k: f"{100 * v['right_among_recoverable'] / max(1, v['recoverable']):.1f}% of {v['cells']}" for k, v in res["by_kind"]["R2"].items()})
    worst = sorted(res["per_field"]["R2"].items(), key=lambda kv: kv[1]["right_among_recoverable"] / max(1, kv[1]["recoverable"]))[:5]
    print("  weakest fields (R2):", {k: f"{100 * v['right_among_recoverable'] / max(1, v['recoverable']):.0f}% of {v['cells']}" for k, v in worst if v["cells"] >= 100})


if __name__ == "__main__":
    ids = sorted(json.loads((DATA / "trainval.json").read_text(encoding="utf-8")))
    test = json.loads((ROOT / "results" / "docile_rows" / "splits.json").read_text(encoding="utf-8"))["test"]
    out = {}
    for name, sel in (("all labelled documents", ids), ("test documents (200)", test)):
        res = run(sel)
        show(name, res)
        out[name] = {"documents": res["documents"], "cells": res["cells"], "no_same_type_header": res["cells_with_no_same_type_header_on_page"],
                     "R1": dict(res["tot"]["R1"]), "R2": dict(res["tot"]["R2"])}
    p = ROOT / "results" / "docile_rows" / "header_rule.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
