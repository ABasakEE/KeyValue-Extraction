"""Study 25, stage 0 (CPU only, post hoc): re-analysis of the saved outputs.

  python scripts/study25_stage0.py --split confirm

Reads `results/study25_stage0/<model>_<arm>_<split>/spans_<arm>.jsonl` (made by `schema_guided_kv.py score --fallback`) and reports,
for every arm: exact-span F1 (as registered), an *official-style text-match* F1 (VRDU `match_utils` rules, see below), the same
after snapping the arm's spans to BROS's overlapping span, the paired A2 - A1 difference, and an error dump.

Official-style text match (read from google-research/vrdu `match_utils.py` and `benchmark_utils.py` through a summary, NOT run with the
official tool): a predicted entity is correct if its text matches ANY annotated occurrence of the same field, with the per-field rule
of the dataset's `meta.json` (NameMatch: whitespace removed; NumericalStringMatch: digits only; GeneralStringMatch: alphanumerics only;
DateMatch: parsed date, else alphanumerics).  One prediction per field (the first), gold = one entity per annotated field.
Assumption, unverified: the official tool counts one gold entity per unrepeated field.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import schema_guided_kv as sg  # noqa: E402
from matched_eval import ci, f1_of, paired_bootstrap, span_counts  # noqa: E402

ROOT = sg.ROOT
OUT = ROOT / "results" / "study25_stage0"
MANIFESTS = {"confirm": "results/confirm_v2/manifest.json", "dev": "results/zero_shot_kv_v2/manifest.json"}
META = json.loads((ROOT / "data/vrdu/registration-form/main/meta.json").read_text(encoding="utf-8"))["entity_name_to_match_func"]
DATE_FORMATS = ["%m/%d/%y", "%m/%d/%Y", "%m/%d", "%b%d/%y", "%m-%d-%Y", "%m-%d-%y", "%B%d,%Y", "%Y/%m/%d"]


def _ws(s: str) -> str:
    return " ".join(s.strip().split())


def _date(s: str):
    s = re.sub(r"[^0-9a-zA-Z/\-,]", "", _ws(s))
    for f in DATE_FORMATS:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    return None


def text_match(rule: str, a: str, b: str) -> bool:
    a, b = _ws(a), _ws(b)
    if rule == "NameMatch":
        return re.sub(r"\s", "", a) == re.sub(r"\s", "", b)
    if rule == "NumericalStringMatch":
        return re.sub(r"[^0-9]", "", a) == re.sub(r"[^0-9]", "", b)
    if rule == "DateMatch":
        da, db = _date(a), _date(b)
        if da and db:
            return (da.year, da.month, da.day) == (db.year, db.month, db.day) or (da.year == db.year and (da.month, da.day) == (db.day, db.month))
    alnum = lambda s: re.sub(r"[^0-9a-zA-Z]", "", s)
    return alnum(a) == alnum(b)


def text_counts(pred: list[list], gold_texts: dict[str, list[str]]) -> dict:
    out: dict = defaultdict(lambda: [0, 0, 0])
    first = {}
    for f, _, _, t in pred:
        first.setdefault(f, t)
    for f, t in first.items():
        out[f][1] += 1
        out[f][0] += any(text_match(META.get(f, "DefaultMatch"), t, g) for g in gold_texts.get(f, []))
    for f in gold_texts:
        out[f][2] += 1
    out["_micro"] = [sum(v[i] for k, v in out.items() if k != "_micro") for i in range(3)]
    return {k: tuple(v) for k, v in out.items()}


def overlap(a, b) -> bool:
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


def snap(vspans: list[list], bspans: list[list], fill: bool) -> list[list]:
    out = [next((b for b in bspans if overlap(v, b)), v) for v in vspans]
    if fill:
        have = {s[0] for s in out}
        out += [b for b in bspans if b[0] not in have]
    return out


def load_gold_texts(guids: set[str]) -> dict[str, dict[str, list[str]]]:
    recs = sg.load_records(ROOT / "data" / "vrdu", guids)
    return {g: {n: [o[0] for o in occs] for n, occs in r["annotations"] if isinstance(n, str)} for g, r in recs.items()}


def table(rows: list[dict], systems: list[str], counter) -> dict:
    per = {s: [counter(r, s) for r in rows] for s in systems}
    boot = paired_bootstrap(per, "_micro")
    return {"f1": {s: round(f1_of([sum(x["_micro"][i] for x in per[s]) for i in range(3)]) * 100, 1) for s in systems},
            "ci95": {s: [round(v * 100, 1) for v in ci(boot[s])] for s in systems}, "boot": boot,
            "per_field": {f: {s: round(f1_of([sum(x.get(f, (0, 0, 0))[i] for x in per[s]) for i in range(3)]) * 100, 1) for s in systems}
                          for f in META}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="confirm", choices=list(MANIFESTS))
    ap.add_argument("--dump", type=int, default=12, help="error examples per field")
    a = ap.parse_args()
    arms = {}
    for d in sorted(OUT.glob(f"*_{a.split}")):
        model, arm = d.name.split("_")[:2]
        f = d / f"spans_{arm}.jsonl"
        if f.exists() and arm != "A3":  # A3 is incomplete (258/287 docs for 3B, 2 for 8B): it would shrink the common set
            arms[f"{model}-{arm}"] = {r["guid"]: r for r in map(json.loads, f.read_text(encoding="utf-8").splitlines())}
    guids = set.intersection(*[set(v) for v in arms.values()])
    gold_t = load_gold_texts(guids)
    base = next(iter(arms.values()))
    rows = [{"guid": g, "gold": base[g]["gold"], "bros13": base[g]["bros_seed13"], "bros42": base[g]["bros_seed42"],
             "vlm_both": base[g]["vlm_both"], "template": base[g]["template"],
             **{k: v[g][f"vlm_{k.split('-')[1]}"] for k, v in arms.items()}} for g in sorted(guids)]
    print(f"split {a.split}: {len(rows)} documents common to all arms {list(arms)}")
    for k in arms:  # snapped variants, one per BROS seed
        for sd in ("13", "42"):
            for fill in (False, True):
                for r in rows:
                    r[f"{k}|snap{sd}{'+fill' if fill else ''}"] = snap(r[k], r[f"bros{sd}"], fill)
    systems = ["bros13", "bros42", "vlm_both"] + list(arms)
    snaps = [k for k in rows[0] if "|snap" in k]
    exact = lambda r, s: span_counts(r[s], r["gold"])
    textc = lambda r, s: text_counts(r[s], gold_t[r["guid"]])
    report = {"split": a.split, "n_docs": len(rows)}
    for name, counter in (("exact_span", exact), ("official_style_text", textc)):
        t = table(rows, systems + snaps, counter)
        print(f"\n== {name} F1 (95% CI)")
        for s in systems + snaps:
            print(f"  {s:28s} {t['f1'][s]:5.1f} {t['ci95'][s]}")
        pairs = {}
        for m in ("3B", "8B"):
            if f"{m}-A2" in arms and f"{m}-A1" in arms:
                d = t["boot"][f"{m}-A2"] - t["boot"][f"{m}-A1"]
                pairs[f"{m} A2-A1"] = {"mean": round(float(d.mean()) * 100, 1), "ci95": [round(v * 100, 1) for v in ci(d)]}
        if "8B-A2" in arms and "3B-A2" in arms:
            d = t["boot"]["8B-A2"] - t["boot"]["3B-A2"]
            pairs["A2 8B-3B"] = {"mean": round(float(d.mean()) * 100, 1), "ci95": [round(v * 100, 1) for v in ci(d)]}
        print("  paired:", pairs)
        print("  per field:")
        for f, v in t["per_field"].items():
            print(f"    {f:24s}", {s: v[s] for s in ["bros13", "vlm_both", "8B-A2", "8B-A2|snap13", "3B-A2"] if s in v})
        report[name] = {"f1": t["f1"], "ci95": t["ci95"], "paired": pairs, "per_field": t["per_field"]}
    # error dump for 8B-A2: where is the prediction relative to the gold occurrence, and does the text match officially?
    kinds = Counter()
    dump = defaultdict(list)
    for r in rows:
        g = {x[0]: x for x in r["gold"]}
        gt = gold_t[r["guid"]]
        for f, s, e, t in r.get("8B-A2", []):
            if f not in g:
                kinds[(f, "no gold field")] += 1
                continue
            gs, ge = g[f][1], g[f][2]
            exact_ok = (s, e) == (gs, ge)
            txt_ok = any(text_match(META[f], t, x) for x in gt.get(f, []))
            kinds[(f, "exact" if exact_ok else "text-match, different span" if txt_ok else "text differs")] += 1
            if not exact_ok and len(dump[f]) < a.dump:
                dump[f].append({"guid": r["guid"][:40], "pred": t[:90], "gold": [x[:60] for x in gt.get(f, [])][:2], "text_ok": txt_ok})
    print("\n== 8B-A2 predictions: span vs official-style text agreement")
    for f in META:
        print(f"  {f:24s}", {k[1]: v for k, v in kinds.items() if k[0] == f})
    report["error_kinds"] = {f"{k[0]}|{k[1]}": v for k, v in kinds.items()}
    report["error_dump"] = dump
    (OUT / f"analysis_{a.split}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT / f'analysis_{a.split}.json'}")


if __name__ == "__main__":
    main()
