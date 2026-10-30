"""Study 14: BROS + zero-shot VLM hybrid on FATURA with held-out template folds (CPU scoring).

Subcommands
  manifest  VLM run manifest: 20 invoices per template (10 dev + 10 confirm), fold and split recorded
  fields    {"fatura": [9 field names]} for `zero_shot_kv_v2.py describe-schema --fields-file`
  score     block-level scoring of BROS and the VLM, router / fill-in / oracle hybrids, validity checks, verdict
  selftest  logic checks without models

Pre-registered rules (STUDIES.md study 14), all inherited from studies 8-12 except where noted:
* VLM: frozen v2 pipeline; one answer per field; `vlm_both` (key + named candidates pooled, best similarity wins).
* Grounding (new, block-level): value -> first exact word match (else fuzzy >= 0.5 token F1) -> the whole text block
  containing it; a block can hold one field (higher similarity wins). Gold = blocks of the 9 core fields.
* Metric: exact block-span micro F1 (a predicted span counts only if it equals a gold block).
* Router: per field, BROS or `vlm_both`, whichever has the higher pooled F1 on the other folds' documents (dev and
  confirm); applied to the held-out fold's confirm documents only. Fill-in: BROS + VLM blocks for fields BROS left empty.
* Primary interval: cluster bootstrap over templates (resample templates, keep all their documents), 1000 draws.
  Document-level intervals are reported as well. Dev documents are for debugging only; claims use confirm documents.
* Pass: pooled router - BROS (mean of the two BROS seeds) has a 95% interval above 0, and no fold's interval lies
  entirely below 0.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import fatura_loader as fl  # noqa: E402
from hybrid_eval import fill_in, fit_router, oracle_counts, route  # noqa: E402
from matched_eval import DocText, f1_of, span_counts  # noqa: E402
from zero_shot_kv import sbert_embedder  # noqa: E402
from zero_shot_kv_v2 import assign_both, assign_fields, read_rows, run_stats, schema_texts  # noqa: E402

SEEDS = (13, 42)
N_BOOT = 1000
VLM_MODES = ("vlm_key", "vlm_named", "vlm_both")


# ---------------------------------------------------------------- manifest
def build_manifest(root: Path, per_template: int = 20, seed: int = 1) -> list[dict]:
    sample = fl.eval_sample(per_template, seed)
    fold_of = {t: f for f, ts in enumerate(fl.folds()) for t in ts}
    docs = []
    for t, parts in sample.items():
        for split, names in parts.items():
            for g in names:
                docs.append({"doc_id": f"fatura/{g}", "dataset": "fatura", "template": f"Template{t}", "fold": fold_of[t],
                             "split": split, "source": str(root / "images" / f"{g}.jpg"), "n_pages": 1,
                             "pages": [{"page": None, "lines": []}], "gold": {}})
    docs.sort(key=lambda d: d["doc_id"])
    random.Random(seed).shuffle(docs)        # a deadline-stopped run still covers every template
    return docs


# ---------------------------------------------------------------- grounding to blocks
def block_spans(doc: dict, assigned: dict) -> list[list]:
    """[field, block_start, block_end, text] for each assigned field whose value grounds into a text block."""
    text = DocText(doc["words"], [0] * len(doc["words"]))
    block_of = {i: (s, e) for _, s, e in doc["blocks"] for i in range(s, e)}
    used, out = set(), []
    for f, (sim, value, page) in sorted(((f, c[0]) for f, c in assigned.items()), key=lambda x: -x[1][0]):
        sp = text.ground(value, page)
        if not sp or sp[0] not in block_of:
            continue
        s, e = block_of[sp[0]]
        if (s, e) not in used:
            used.add((s, e))
            out.append([f, s, e, " ".join(doc["words"][s:e])])
    return out


# ---------------------------------------------------------------- bootstrap
def boot_f1(systems: dict[str, list[dict]], clusters: list, n_boot: int = N_BOOT, seed: int = 0,
            by_cluster: bool = True) -> dict[str, np.ndarray]:
    """F1 of every system over paired resamples: whole clusters (templates) or single documents."""
    names = list(systems)
    arr = np.array([[d.get("_micro", (0, 0, 0)) for d in systems[n]] for n in names], dtype=float)  # (systems, docs, 3)
    rng = np.random.default_rng(seed)
    groups: dict = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = list(groups)
    out = {n: np.empty(n_boot) for n in names}
    for b in range(n_boot):
        if by_cluster:
            idx = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        else:
            idx = rng.integers(0, arr.shape[1], arr.shape[1])
        tot = arr[:, idx].sum(axis=1)
        for j, n in enumerate(names):
            out[n][b] = f1_of(tot[j])
    return out


def ci(x: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])]


def micro_f1(counts: list[dict]) -> float:
    return f1_of([sum(c["_micro"][i] for c in counts) for i in range(3)])


def summarise(systems: dict[str, list[dict]], clusters: list) -> dict:
    """BROS (mean of seeds) vs router / fill-in / oracle / VLM-only: F1 and difference with both interval types."""
    out = {}
    boots = {m: boot_f1(systems, clusters, by_cluster=m) for m in (True, False)}
    for name in ("router", "fill_in", "oracle", *VLM_MODES):
        seeded = name in ("router", "fill_in", "oracle")
        parts = [f"{name}_seed{s}" for s in SEEDS] if seeded else [name]
        entry = {"f1": round(float(np.mean([micro_f1(systems[p]) for p in parts])), 4),
                 "bros_f1": round(float(np.mean([micro_f1(systems[f"bros_seed{s}"]) for s in SEEDS])), 4)}
        for label, by_c in (("template_ci", True), ("document_ci", False)):
            b = boots[by_c]
            delta = np.mean([b[p] for p in parts], axis=0) - np.mean([b[f"bros_seed{s}"] for s in SEEDS], axis=0)
            entry["minus_bros"] = round(float(np.mean(delta)), 4)
            entry[f"minus_bros_{label}95"] = ci(delta)
        entry["minus_bros"] = round(entry["f1"] - entry["bros_f1"], 4)
        out[name] = entry
    return out


# ---------------------------------------------------------------- scoring
def vlm_records(docs: dict, manifest: list[dict], rows: dict, desc: dict, embed) -> tuple[list[dict], dict]:
    fields = fl.FIELDS
    field_emb = {m: embed(schema_texts(m, fields, desc)) for m in ("key", "named")}
    recs, miss = [], 0
    for m in manifest:
        g = m["doc_id"].split("/", 1)[1]
        rs = rows.get(m["doc_id"], [])
        if len(rs) != len(m["pages"]):
            miss += 1
            continue
        rec = {"fold": m["fold"], "template": m["template"], "guid": g, "split": m["split"], "gold": docs[g]["gold"]}
        for mode in ("key", "named", "both"):
            assigned = (assign_both(rs, fields, field_emb, embed, True) if mode == "both"
                        else assign_fields(rs, mode, fields, field_emb[mode], embed, True))
            rec[f"vlm_{mode}"] = block_spans(docs[g], assigned)
        recs.append(rec)
    return recs, {"documents_incomplete": miss}


def same_spans(a: list, b: list) -> bool:
    """The BROS dump lists spans sorted by field, the loader in reading order: compare as sets of spans."""
    return sorted(map(tuple, a)) == sorted(map(tuple, b))


def attach_bros(recs: list[dict], docs: dict, bros_dir: Path) -> dict:
    """Adds bros_seed<s> spans to each record; returns alignment and full-test-set reference numbers."""
    info = {"gold_alignment_mismatches": 0, "bros_missing": 0, "full_test_f1": {}}
    dumps = {}
    for f in sorted({r["fold"] for r in recs}):
        for s in SEEDS:
            d = bros_dir / f"fold{f}" / f"seed_{s}"
            if not (d / "test_entities.jsonl").exists():
                continue
            dumps[(f, s)] = {r["guid"]: r for r in map(json.loads, (d / "test_entities.jsonl").read_text(encoding="utf-8").splitlines())}
            info["full_test_f1"][f"fold{f}_seed{s}"] = json.loads((d / "result.json").read_text())["test"]["f1"]
    for r in recs:
        for s in SEEDS:
            row = dumps.get((r["fold"], s), {}).get(r["guid"])
            if row is None:
                info["bros_missing"] += 1
                continue
            info["gold_alignment_mismatches"] += not same_spans(row["gold"], docs[r["guid"]]["gold"])
            r[f"bros_seed{s}"] = row["pred"]
    return info


def systems_for(test: list[dict], fit: list[dict], with_tables: dict | None = None) -> dict[str, list[dict]]:
    systems: dict[str, list[dict]] = defaultdict(list)
    for s in SEEDS:
        bros = f"bros_seed{s}"
        table = fit_router(fit, bros, ("vlm_both",)) if fit else {}
        if with_tables is not None:
            with_tables[bros] = table
        for r in test:
            src = {bros: r[bros], "vlm_both": r["vlm_both"]}
            systems[bros].append(span_counts(r[bros], r["gold"]))
            systems[f"router_seed{s}"].append(span_counts(route(table, src, bros), r["gold"]))
            systems[f"fill_in_seed{s}"].append(span_counts(fill_in(r[bros], r["vlm_both"]), r["gold"]))
            systems[f"oracle_seed{s}"].append(oracle_counts(r["gold"], src, [bros, "vlm_both"]))
    for m in VLM_MODES:
        systems[m] = [span_counts(r[m], r["gold"]) for r in test]
    return dict(systems)


def verdict(by_fold: dict, pooled: dict) -> dict:
    pooled_ok = pooled["router"]["minus_bros_template_ci95"][0] > 0
    no_loss = all(v["router"]["minus_bros_template_ci95"][1] >= 0 for v in by_fold.values())
    return {"pooled_gain_significant": pooled_ok, "no_fold_significantly_worse": no_loss, "passes": pooled_ok and no_loss}


def analyse(recs: list[dict]) -> dict:
    confirm = [r for r in recs if r["split"] == "confirm"]
    folds = sorted({r["fold"] for r in recs})
    tables, by_fold, parts = {}, {}, []
    for f in folds:
        test = [r for r in confirm if r["fold"] == f]
        tab: dict = {}
        sys_f = systems_for(test, [r for r in recs if r["fold"] != f], tab)
        tables[f"fold{f}"] = tab
        by_fold[f"fold{f}"] = {"n_docs": len(test), **summarise(sys_f, [r["template"] for r in test])}
        parts.append((test, sys_f))
    pooled_sys = {k: [c for _, s in parts for c in s[k]] for k in parts[0][1]}
    pooled = summarise(pooled_sys, [r["template"] for test, _ in parts for r in test])
    # per-field pooled F1 (confirm documents): BROS (mean of seeds), router, vlm_both
    per_field = {}
    for fld in fl.FIELDS:
        def f1_of_field(counts):
            return f1_of([sum(c.get(fld, (0, 0, 0))[i] for c in counts) for i in range(3)])
        per_field[fld] = {"bros": round(float(np.mean([f1_of_field(pooled_sys[f"bros_seed{s}"]) for s in SEEDS])), 4),
                          "router": round(float(np.mean([f1_of_field(pooled_sys[f"router_seed{s}"]) for s in SEEDS])), 4),
                          "vlm_both": round(f1_of_field(pooled_sys["vlm_both"]), 4)}
    return {"by_fold": by_fold, "pooled": pooled, "per_field_f1": per_field, "router_tables": tables,
            "verdict": verdict(by_fold, pooled), "n_confirm_docs": len(confirm)}


def validity(recs: list[dict], info: dict) -> dict:
    out = {"gold_alignment_mismatches": info["gold_alignment_mismatches"], "bros_missing": info["bros_missing"], "per_fold": {}}
    for f in sorted({r["fold"] for r in recs}):
        conf = [r for r in recs if r["fold"] == f and r["split"] == "confirm"]
        dev = [r for r in recs if r["fold"] == f and r["split"] == "dev"]
        cl = [r["template"] for r in conf]
        entry = {}
        for s in SEEDS:
            counts = [span_counts(r[f"bros_seed{s}"], r["gold"]) for r in conf]
            b = boot_f1({"x": counts}, cl)["x"]
            ref = info["full_test_f1"].get(f"fold{f}_seed{s}")
            entry[f"bros_seed{s}"] = {"confirm_f1": round(micro_f1(counts), 4), "confirm_ci95": ci(b), "full_test_f1": ref,
                                      "full_test_inside_ci": ref is not None and ci(b)[0] <= ref <= ci(b)[1]}
        for name, group in (("dev", dev), ("confirm", conf)):
            entry[f"vlm_both_{name}_f1"] = round(micro_f1([span_counts(r["vlm_both"], r["gold"]) for r in group]), 4)
        out["per_fold"][f"fold{f}"] = entry
    return out


def run_score(manifest_path: str, preds: list[str], bros_dir: str, schema_desc: str, out_dir: str, embed) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    desc = json.loads(Path(schema_desc).read_text(encoding="utf-8")).get("fatura", {}) if schema_desc else {}
    docs = fl.load_docs(ROOT / "data" / "fatura" / "raw" / "invoices_dataset_final")
    rows = read_rows(preds)
    recs, miss = vlm_records(docs, manifest, rows, desc, embed)
    info = attach_bros(recs, docs, Path(bros_dir))
    recs = [r for r in recs if all(f"bros_seed{s}" in r for s in SEEDS)]
    report = {"rules": __doc__.split("Pre-registered rules")[1].strip(), "documents": {"manifest": len(manifest), "scored": len(recs), **miss},
              "run_stats": run_stats(manifest, rows), "validity": validity(recs, info)}
    report.update(analyse(recs))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "spans.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    return report


# ---------------------------------------------------------------- CLI and self-test
def _selftest() -> None:
    doc = {"words": ["TAX", "INVOICE", "Date:", "03-Jan-1994", "TOTAL", ":", "5.00"],
           "blocks": [(None, 0, 2), ("date", 2, 4), ("total", 4, 7)]}
    got = block_spans(doc, {"date": [(0.9, "03-Jan-1994", None)], "total": [(0.8, "5.00", None)], "title": [(0.7, "TAX INVOICE", None)]})
    assert got == [["date", 2, 4, "Date: 03-Jan-1994"], ["total", 4, 7, "TOTAL : 5.00"],
                   ["title", 0, 2, "TAX INVOICE"]], got       # a field assigned to a non-core block is kept (a false positive)
    got2 = block_spans(doc, {"date": [(0.9, "03-Jan-1994", None)], "total": [(0.8, "03-Jan-1994", None)]})
    assert got2 == [["date", 2, 4, "Date: 03-Jan-1994"]], got2                                  # one field per block
    assert block_spans(doc, {"date": [(0.9, "no such text", None)]}) == []
    # cluster bootstrap: 4 templates, system a is right on 2 of them, b on all
    ones, zeros = {"_micro": (1, 1, 1)}, {"_micro": (0, 1, 1)}
    sysd = {"a": [ones] * 4 + [zeros] * 4, "b": [ones] * 8}
    cl = ["t1", "t1", "t2", "t2", "t3", "t3", "t4", "t4"]
    b = boot_f1(sysd, cl, 400)
    assert ci(b["b"]) == [1.0, 1.0] and ci(b["a"])[0] < 0.5 < ci(b["a"])[1]
    assert ci(boot_f1(sysd, cl, 400)["a"])[1] - ci(boot_f1(sysd, cl, 400)["a"])[0] > \
        ci(boot_f1(sysd, cl, 400, by_cluster=False)["a"])[1] - ci(boot_f1(sysd, cl, 400, by_cluster=False)["a"])[0] - 1e-9  # clusters widen it
    v = verdict({"fold0": {"router": {"minus_bros_template_ci95": [-0.01, 0.04]}}}, {"router": {"minus_bros_template_ci95": [0.01, 0.05]}})
    assert v["passes"]
    assert not verdict({"fold0": {"router": {"minus_bros_template_ci95": [-0.05, -0.01]}}}, {"router": {"minus_bros_template_ci95": [0.01, 0.05]}})["passes"]
    assert same_spans([["a", 0, 2, "x"], ["b", 3, 4, "y"]], [["b", 3, 4, "y"], ["a", 0, 2, "x"]])
    assert not same_spans([["a", 0, 2, "x"]], [["a", 0, 3, "x"]])
    m = build_manifest(Path("/x"), 20, 1)
    assert len(m) == 1000 and Counter(d["split"] for d in m) == {"dev": 500, "confirm": 500}
    assert Counter(d["fold"] for d in m) == {f: 200 for f in range(5)} and len({d["doc_id"] for d in m}) == 1000
    print("[+] fatura_eval selftest passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--per-template", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    p = sub.add_parser("fields")
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("--manifest", required=True)
    p.add_argument("--preds", nargs="+", required=True)
    p.add_argument("--bros-dir", required=True)
    p.add_argument("--schema-desc", default=None)
    p.add_argument("--out-dir", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        _selftest()
    elif a.cmd == "fields":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"fatura": fl.FIELDS}), encoding="utf-8")
    elif a.cmd == "manifest":
        m = build_manifest(fl.default_root(), a.per_template, a.seed)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(m), encoding="utf-8")
        print(f"[+] {len(m)} invoices -> {a.out}: {dict(Counter(d['split'] for d in m))}, folds {dict(Counter(d['fold'] for d in m))}")
    else:
        rep = run_score(a.manifest, a.preds, a.bros_dir, a.schema_desc, a.out_dir, sbert_embedder())
        print(json.dumps({k: rep[k] for k in ("documents", "validity", "pooled", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
