"""Study 44 (CPU): a failure taxonomy of the BROS line-item tagger on DocILE unseen layouts (and seen layouts for the full-pool runs).

    python scripts/docile_failure_taxonomy.py [--workers 7] [--out results/docile_taxonomy/taxonomy.json]

Runs (saved predictions, no model): unseen-layout test documents for N1000_S0/S1/S2 and Nfull_T13/T21; seen-layout test documents for Nfull_T13/T21 (one process per job).
Per gold span exactly one outcome: right | wrong field (same extent; monetary family or other) | right field, wrong extent (too short, too long, other overlap) |
wrong field and extent | missed (no predicted span touches it: field absent in the document, split gold cell, other). Per predicted span: false (touches no gold word), inside or outside the table.
Per document: structure from the annotation metadata (table border type, table structure, columns, pages, document type, source, language) and from the gold (rows, cells, amount columns, multi-line cells).
Outputs: loss budget (share of gold spans by outcome), per-field outcomes, row F1 by document property (documents pooled over the runs, bootstrap over documents), Spearman correlations.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src"), str(HERE.parent)]

JOBS = [("N1000_S0", "utl"), ("N1000_S1", "utl"), ("N1000_S2", "utl"), ("Nfull_T13", "utl"), ("Nfull_T21", "utl"), ("Nfull_T13", "mtl"), ("Nfull_T21", "mtl")]
MONEY = {"line_item_unit_price_gross", "line_item_unit_price_net", "line_item_amount_gross", "line_item_amount_net"}


def doc_features(d: dict, ann: dict) -> dict:
    meta = ann["metadata"]
    grids = list(meta.get("page_to_table_grid", {}).values())
    first = grids[0] if grids else {}
    rows, fields = d["rows"], Counter(f for r in rows for f, _, _ in r)
    amount = "both" if fields["line_item_amount_net"] and fields["line_item_amount_gross"] else "net" if fields["line_item_amount_net"] else "gross" if fields["line_item_amount_gross"] else "neither"
    heights = sorted(b[3] - b[1] for b in d["fboxes"]) or [0.01]
    med_h = heights[len(heights) // 2] or 0.01
    cells = ann["line_item_extractions"]
    multi = sum((e["bbox"][3] - e["bbox"][1]) > 1.6 * med_h for e in cells) / max(1, len(cells))
    pages_with_rows = len({p for r in rows for _, _, p in r}) or 1
    return {"document_type": meta.get("document_type"), "source": ann.get("source", meta.get("source")), "language": meta.get("language"), "currency": meta.get("currency"),
            "page_count": meta.get("page_count"), "table_border_type": first.get("table_border_type"), "table_structure": first.get("table_structure"),
            "n_table_columns": max((len(g.get("columns", [])) for g in grids), default=0), "missing_columns": any(g.get("missing_columns") for g in grids),
            "n_words": len(d["words"]), "n_rows": len(rows), "rows_per_page": round(len(rows) / pages_with_rows, 2), "n_fields": len(fields), "amount_columns": amount,
            "share_multiline_cells": round(multi, 3), "share_rows_repeating_a_field": round(sum(len(r) != len({f for f, _, _ in r}) for r in rows) / max(1, len(rows)), 3)}


def outcomes(d: dict, spans: list) -> dict:
    import docile_bros as B
    from row_group import runs

    gold = [(t, s, e) for t, s, e, _ in B.spans_of(d)]
    pr = [(t, s, e) for t, s, e, *_ in spans]
    pset = set(pr)
    split = {(f, a) for row in d["rows"] for f, ws, _ in row if len(runs(ws)) > 1 for a, _ in runs(ws)}
    fields_pred = {t for t, _, _ in pr}
    out: Counter = Counter()
    by_field: dict[str, Counter] = defaultdict(Counter)
    for t, s, e in gold:
        ov = [p for p in pr if p[1] < e and s < p[2]]
        if (t, s, e) in pset:
            c = "right"
        elif any((p[1], p[2]) == (s, e) for p in ov):
            same = [p for p in ov if (p[1], p[2]) == (s, e)][0]
            c = "wrong_field_money" if t in MONEY and same[0] in MONEY else "wrong_field_other"
        elif any(p[0] == t for p in ov):
            q = [p for p in ov if p[0] == t][0]
            c = "extent_too_short" if s <= q[1] and q[2] <= e else "extent_too_long" if q[1] <= s and e <= q[2] else "extent_other_overlap"
        elif ov:
            c = "wrong_field_and_extent"
        else:
            c = "missed_split_gold_cell" if (t, s) in split else "missed_field_absent_in_doc" if t not in fields_pred else "missed_other"
        out[c] += 1
        by_field[t][c] += 1
    gold_words = {i for _, s, e in gold for i in range(s, e)}
    ext = {}
    for i in gold_words:
        pg, (y0, y1) = d["pages"][i], (d["fboxes"][i][1], d["fboxes"][i][3])
        lo, hi = ext.get(pg, (9, -9))
        ext[pg] = (min(lo, y0), max(hi, y1))
    for t, s, e in pr:
        if any(i in gold_words for i in range(s, e)):
            continue
        yc = (d["fboxes"][s][1] + d["fboxes"][s][3]) / 2
        pg = d["pages"][s]
        out["false_span_inside_table" if pg in ext and ext[pg][0] - 0.01 <= yc <= ext[pg][1] + 0.01 else "false_span_outside_table"] += 1
    out["gold_spans"], out["pred_spans"] = len(gold), len(pr)
    return {"outcomes": dict(out), "by_field": {k: dict(v) for k, v in by_field.items()}}


def job(args: tuple) -> list[dict]:
    tag, which = args
    import docile_bros as B
    import docile_typer as T   # pred loading only
    import docile_sg as G

    test = json.loads((B.MTL if which == "mtl" else B.SPLITS).read_text(encoding="utf-8"))["test"]
    pred = T.load_pred(tag)
    recs = []
    for g in test:
        d = B.doc_dict(g)
        if not d["rows"] or g not in pred:
            continue
        ann = G.ann_of(g)
        from row_group import group_rows
        c = B.counts(group_rows(pred[g], d, 0.5, merge=False), d)
        recs.append({"tag": tag, "set": which, "doc": g, "lir": c["lir"], "cell": c["cell"], **outcomes(d, pred[g]), "features": doc_features(d, ann)})
    return recs


def f1c(c) -> float:
    return 100 * 2 * c[0] / max(1, c[1] + c[2])


def group_table(recs: list[dict], key, nboot: int = 500) -> list[dict]:
    """Row F1 by document property: counts pooled over the runs, 95% bootstrap over documents inside the group."""
    docs: dict[str, dict] = {}
    for r in recs:
        o = docs.setdefault(r["doc"], {"lir": [0, 0, 0], "feat": r["features"], "n": 0})
        o["lir"] = [a + b for a, b in zip(o["lir"], r["lir"])]
        o["n"] += 1
    groups: dict = defaultdict(list)
    for o in docs.values():
        groups[key(o["feat"])].append(o["lir"])
    rng, out = random.Random(44), []
    for k, cs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        tot = [sum(c[j] for c in cs) for j in range(3)]
        bs = []
        if len(cs) >= 5:
            for _ in range(nboot):
                samp = [cs[rng.randrange(len(cs))] for _ in cs]
                bs.append(f1c([sum(c[j] for c in samp) for j in range(3)]))
            bs.sort()
        out.append({"group": k, "documents": len(cs), "gold_cells": tot[2] // max(1, len(recs) // max(1, len(docs))), "row_f1": round(f1c(tot), 1),
                    "ci95": [round(bs[int(.025 * nboot)], 1), round(bs[int(.975 * nboot) - 1], 1)] if bs else None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default=str(HERE.parent / "results" / "docile_taxonomy" / "taxonomy.json"))
    a = ap.parse_args()
    with Pool(min(a.workers, len(JOBS))) as pool:
        results = pool.map(job, JOBS)
    recs = [r for rs in results for r in rs]
    report: dict = {"jobs": [f"{t}_{w}" for t, w in JOBS]}
    cats = ["right", "wrong_field_money", "wrong_field_other", "extent_too_short", "extent_too_long", "extent_other_overlap", "wrong_field_and_extent",
            "missed_field_absent_in_doc", "missed_split_gold_cell", "missed_other"]
    false_cats = ["false_span_inside_table", "false_span_outside_table"]
    for which in ("utl", "mtl"):
        sub = [r for r in recs if r["set"] == which]
        if not sub:
            continue
        runs_ = sorted({r["tag"] for r in sub})
        budget = {}
        for tag in runs_:
            tot = Counter()
            for r in sub:
                if r["tag"] == tag:
                    tot.update(r["outcomes"])
            budget[tag] = {c: round(100 * tot[c] / tot["gold_spans"], 2) for c in cats} | {c: round(100 * tot[c] / tot["pred_spans"], 2) for c in false_cats} | {"gold_spans": tot["gold_spans"], "pred_spans": tot["pred_spans"]}
        budget["mean"] = {c: round(st.mean(budget[t][c] for t in runs_), 2) for c in cats + false_cats}
        report[f"loss_budget_{which}"] = budget
        byf: dict = defaultdict(Counter)
        for r in sub:
            for f, cc in r["by_field"].items():
                byf[f].update(cc)
        report[f"by_field_{which}"] = {f: {"gold_spans": sum(c.values()) // len(runs_), **{k: round(100 * c[k] / sum(c.values()), 1) for k in cats}} for f, c in sorted(byf.items(), key=lambda kv: -sum(kv[1].values())) if sum(c.values()) // len(runs_) >= 20}
    utl = [r for r in recs if r["set"] == "utl"]
    props = {"table_border_type": lambda f: f["table_border_type"], "table_structure": lambda f: f["table_structure"], "document_type": lambda f: f["document_type"],
             "source": lambda f: f["source"], "language": lambda f: f["language"], "page_count": lambda f: min(f["page_count"] or 1, 3),
             "n_rows": lambda f: "1-2" if f["n_rows"] <= 2 else "3-6" if f["n_rows"] <= 6 else "7-15" if f["n_rows"] <= 15 else "16+",
             "n_table_columns": lambda f: "0-3" if f["n_table_columns"] <= 3 else "4-5" if f["n_table_columns"] <= 5 else "6+", "amount_columns": lambda f: f["amount_columns"],
             "n_fields": lambda f: "1-3" if f["n_fields"] <= 3 else "4-5" if f["n_fields"] <= 5 else "6+",
             "n_words": lambda f: "<80" if f["n_words"] < 80 else "80-200" if f["n_words"] < 200 else "200-400" if f["n_words"] < 400 else "400+",
             "multiline_cells": lambda f: "none" if f["share_multiline_cells"] == 0 else "<30%" if f["share_multiline_cells"] < 0.3 else "30%+",
             "repeated_field_in_row": lambda f: "no" if f["share_rows_repeating_a_field"] == 0 else "yes", "missing_columns": lambda f: f["missing_columns"]}
    report["row_f1_by_property_utl"] = {k: group_table(utl, fn) for k, fn in props.items()}
    try:
        from scipy.stats import spearmanr
        docs: dict = {}
        for r in utl:
            o = docs.setdefault(r["doc"], {"f": [], "feat": r["features"]})
            o["f"].append(f1c(r["lir"]))
        ys = [st.mean(o["f"]) for o in docs.values()]
        rho = {}
        for k in ("n_words", "n_rows", "rows_per_page", "n_fields", "n_table_columns", "page_count", "share_multiline_cells", "share_rows_repeating_a_field"):
            x = [o["feat"][k] or 0 for o in docs.values()]
            r_, p_ = spearmanr(x, ys)
            rho[k] = {"rho": round(float(r_), 3), "p": round(float(p_), 4)}
        report["spearman_doc_row_f1_vs_feature_utl"] = rho
        report["documents_utl"] = len(docs)
    except Exception as e:   # scipy missing: the rest is still written
        report["spearman_error"] = str(e)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    Path(a.out).with_name("per_document.jsonl").write_text("\n".join(json.dumps({k: r[k] for k in ("tag", "set", "doc", "lir", "cell", "features")}) for r in recs), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k.startswith("loss_budget") for k in [k]}, indent=1)[:3000])


if __name__ == "__main__":
    main()
