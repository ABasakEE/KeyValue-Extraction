"""Study 29 post hoc diagnostics (CPU only, existing outputs): where does the tuned VLM lose, and does it add anything?

    python scripts/vlm_sft_diagnostics.py [--out results/vlm_sft/diagnostics.json]

All tests use results/vlm_sft/spans_E1.jsonl and spans_E1r.jsonl (600 UTL documents: 300 dev + 300 confirm, with the BROS and
zero-shot-VLM reference spans).  Written after the Study 29 results were seen, so everything here is post hoc.
  D1  boundary anatomy: how far off are near-miss spans, and the F1 if boundaries are allowed to be off by k words
  D2  deployable snapping rules (no gold): snap the VLM span to BROS spans or to visual-line boundaries
  D3  complementarity: per-field cross-template router and oracle with and without the tuned VLM; error correlation
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from matched_eval import f1_of, span_counts  # noqa: E402

R = ROOT / "results" / "vlm_sft"
FIELDS = ["file_date", "foreign_principle_name", "registrant_name", "registration_num", "signer_name", "signer_title"]


def load(arm: str) -> list[dict]:
    return [json.loads(l) for l in (R / f"spans_{arm}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def micro(per_doc: list[dict], fields=None) -> float:
    fs = fields or [None]
    tot = [0, 0, 0]
    for c in per_doc:
        for f in fs:
            x = c["_micro"] if f is None else c.get(f, (0, 0, 0))
            tot = [tot[i] + x[i] for i in range(3)]
    return 100 * f1_of(tot)


def f1_spans(rows: list[dict], key: str, fields=None) -> float:
    return micro([span_counts(r[key], r["gold"]) for r in rows], fields)


# ---------------------------------------------------------------- D1: boundary anatomy
def tolerant_f1(rows, key, k: int | None) -> float:
    """Micro F1 when a prediction matches a gold span of the same field if both boundaries are within k words (None = any overlap)."""
    tp = npred = ngold = 0
    for r in rows:
        for f in FIELDS:
            g = [(s[1], s[2]) for s in r["gold"] if s[0] == f]
            p = [(s[1], s[2]) for s in r[key] if s[0] == f]
            ngold += len(g)
            npred += len(p)
            used = set()
            for a, b in g:
                best, bi = None, None
                for i, (s, e) in enumerate(p):
                    if i in used:
                        continue
                    ok = (max(a, s) < min(b, e)) if k is None else (abs(a - s) <= k and abs(b - e) <= k)
                    cost = abs(a - s) + abs(b - e)
                    if ok and (best is None or cost < best):
                        best, bi = cost, i
                if bi is not None:
                    used.add(bi)
                    tp += 1
    return 100 * f1_of([tp, npred, ngold])


def anatomy(rows, key) -> dict:
    kinds, ds, de, ex = Counter(), [], [], defaultdict(list)
    for r in rows:
        for f, a, b, t in r["gold"]:
            for g, s, e, pt in [x for x in r[key] if x[0] == f]:
                if max(a, s) < min(b, e) and (s, e) != (a, b):
                    kinds["pred_inside_gold (too short)" if (s >= a and e <= b) else "pred_contains_gold (too long)" if (s <= a and e >= b) else "shifted"] += 1
                    ds.append(s - a)
                    de.append(e - b)
                    if len(ex[f]) < 6:
                        ex[f].append((t[:70], pt[:70]))
                    break
    n = max(1, len(ds))
    tot = [abs(x) + abs(y) for x, y in zip(ds, de)]
    return {"n_near_miss": len(ds), "kinds": dict(kinds),
            "total_abs_offset_le1": round(sum(t <= 1 for t in tot) / n, 3), "le2": round(sum(t <= 2 for t in tot) / n, 3),
            "le3": round(sum(t <= 3 for t in tot) / n, 3), "le5": round(sum(t <= 5 for t in tot) / n, 3),
            "mean_start_offset": round(float(np.mean(ds)), 2) if ds else 0, "mean_end_offset": round(float(np.mean(de)), 2) if de else 0,
            "examples": {f: v[:4] for f, v in ex.items()}}


# ---------------------------------------------------------------- D2: deployable snapping
def snap_to(rows, key, src_key, any_field: bool) -> list[dict]:
    out = []
    for r in rows:
        new = []
        for f, a, b, t in r[key]:
            cand = [(max(0, min(b, e) - max(a, s)), s, e) for g, s, e, _ in r[src_key] if (any_field or g == f) and max(a, s) < min(b, e)]
            if cand:
                _, s, e = max(cand)
                a, b = s, e
            new.append([f, a, b, ""])
        seen, dedup = set(), []
        for x in new:
            if (x[0], x[1], x[2]) not in seen:
                seen.add((x[0], x[1], x[2]))
                dedup.append(x)
        out.append({**r, "_snapped": dedup})
    return out


def line_snap(rows, key, lines_by_doc) -> tuple[list[dict], float]:
    out, whole = [], [0, 0]
    for r in rows:
        ln = lines_by_doc[r["guid"]]
        for f, a, b, _ in r["gold"]:
            whole[1] += 1
            whole[0] += any(s == a for s, _ in ln) and any(e == b for _, e in ln)
        new = []
        for f, a, b, t in r[key]:
            hit = [(s, e) for s, e in ln if max(a, s) < min(b, e)]
            if hit:
                a, b = min(s for s, _ in hit), max(e for _, e in hit)
            new.append([f, a, b, ""])
        out.append({**r, "_snapped": new})
    return out, 100 * whole[0] / max(1, whole[1])


# ---------------------------------------------------------------- D3: complementarity
def router_choice(rows_fit, systems) -> dict[str, str]:
    choice = {}
    for f in FIELDS:
        sc = {s: f1_of([sum(span_counts(r[s], r["gold"]).get(f, (0, 0, 0))[i] for r in rows_fit) for i in range(3)]) for s in systems}
        choice[f] = max(systems, key=lambda s: sc[s])
    return choice


def apply_router(r, choice) -> list[list]:
    return [x for f in FIELDS for x in r[choice[f]] if x[0] == f]


def cross_template_router(dev, conf, systems):
    """Per fold k: fit the per-field system choice on the dev documents of the other folds, apply to the confirm documents of fold k."""
    out, tables = [], {}
    for k in sorted({r["fold"] for r in conf}):
        choice = router_choice([r for r in dev if r["fold"] != k], systems)
        tables[k] = choice
        out += [{**r, "_router": apply_router(r, choice)} for r in conf if r["fold"] == k]
    return out, tables


def oracle(rows, systems) -> float:
    tot = [0, 0, 0]
    for r in rows:
        for f in FIELDS:
            best = max((span_counts(r[s], r["gold"]).get(f, (0, 0, 0)) for s in systems), key=lambda c: (c[0], -c[1]))
            tot = [tot[i] + best[i] for i in range(3)]
    return 100 * f1_of(tot)


def boot_diff(rows, a_key, b_key, n=1000, seed=0) -> tuple[float, float, float]:
    ca = np.array([span_counts(r[a_key], r["gold"])["_micro"] for r in rows], dtype=float)
    cb = np.array([span_counts(r[b_key], r["gold"])["_micro"] for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(rows), len(rows))
        d.append(100 * (f1_of(ca[i].sum(0)) - f1_of(cb[i].sum(0))))
    return round(float(np.mean(d)), 2), round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "diagnostics.json"))
    a = ap.parse_args()
    e1, e1r = load("E1"), {r["guid"]: r for r in load("E1r")}
    rows = [{**r, "vlm_E1r": e1r[r["guid"]]["vlm_E1r"]} for r in e1]
    dev_ids = {json.loads(l)["guid"] for l in (ROOT / "results" / "matched_eval" / "spans.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    dev, conf = [r for r in rows if r["guid"] in dev_ids], [r for r in rows if r["guid"] not in dev_ids]
    res: dict = {"n_dev": len(dev), "n_confirm": len(conf)}
    print(f"documents: dev {len(dev)}, confirm {len(conf)}")

    # D1
    print("\n== D1 boundary anatomy (E1 and E1r, all 600 documents)")
    res["D1"] = {}
    for key in ("vlm_E1", "vlm_E1r"):
        an = anatomy(rows, key)
        tol = {("any_overlap" if k is None else f"within_{k}_words"): round(tolerant_f1(rows, key, k), 2) for k in (0, 1, 2, 3, 5, None)}
        res["D1"][key] = {"anatomy": an, "f1_with_boundary_tolerance": tol}
        print(f"{key}: near-misses {an['n_near_miss']}  kinds {an['kinds']}")
        print(f"   total |start offset|+|end offset| <=1: {an['total_abs_offset_le1']:.0%}  <=2: {an['le2']:.0%}  <=3: {an['le3']:.0%}  <=5: {an['le5']:.0%}")
        print(f"   F1 if boundaries may be off by k words: {tol}")
        for f, v in an["examples"].items():
            if f in ("registrant_name", "signer_name"):
                for g, p in v[:3]:
                    print(f"   [{f}] gold: {g!r}\n   {'':{len(f) + 3}}pred: {p!r}")

    # D2
    print("\n== D2 deployable snapping (no gold used)")
    res["D2"] = {}
    from data.vrdu_loader import load_vrdu
    from vlm_sft import lines_of

    by = {e.guid: e for e in load_vrdu(ROOT / "data" / "vrdu")[0]}
    lines_by_doc = {}
    for r in rows:
        ex = by[r["guid"]]
        ln = []
        for p in range(ex.n_pages):
            idx = [i for i, q in enumerate(ex.page_ids) if q == p]
            if idx:
                ln += [(idx[0] + s, idx[0] + e) for s, e in lines_of([ex.boxes[i] for i in idx])]
        lines_by_doc[r["guid"]] = ln
    for key in ("vlm_E1", "vlm_E1r"):
        base = f1_spans(rows, key)
        out = {"unsnapped": round(base, 2)}
        for name, src, anyf in (("snap_to_BROS_seed13_same_field", "bros_seed13", False), ("snap_to_BROS_seed13_any_field", "bros_seed13", True),
                                ("snap_to_zero_shot_VLM_same_field", "vlm_both", False)):
            sn = snap_to(rows, key, src, anyf)
            out[name] = round(micro([span_counts(r["_snapped"], r["gold"]) for r in sn]), 2)
        ls, whole = line_snap(rows, key, lines_by_doc)
        out["snap_to_visual_lines"] = round(micro([span_counts(r["_snapped"], r["gold"]) for r in ls]), 2)
        out["gold_spans_that_are_whole_visual_lines_pct"] = round(whole, 1)
        res["D2"][key] = out
        print(key, out)

    # D3
    print("\n== D3 complementarity (cross-template router fitted on dev of the other folds, scored on confirm; oracle = best system per field per document)")
    res["D3"] = {}
    for tuned in ("vlm_E1", "vlm_E1r"):
        for bros in ("bros_seed13", "bros_seed42"):
            two, three = [bros, "vlm_both"], [bros, "vlm_both", tuned]
            r2, t2 = cross_template_router(dev, conf, two)
            r3, t3 = cross_template_router(dev, conf, three)
            both = [{**x, "_r2": x["_router"], "_r3": y["_router"]} for x, y in zip(r2, r3)]
            single = {s: round(f1_spans(conf, s), 2) for s in three}
            d = boot_diff(both, "_r3", "_r2")
            wrong = [(r, f) for r in conf for f in FIELDS if not any(s[0] == f and s in r["gold"] for s in r[bros])]
            corr = {}
            for s in ("vlm_both", tuned):
                hit = sum(any(x[0] == f and x[:3] in [g[:3] for g in r["gold"]] for x in r[s]) for r, f in wrong)
                corr[s] = round(hit / max(1, len(wrong)), 3)
            out = {"singles_on_confirm": single, "router_BROS+zeroshot": round(micro([span_counts(r["_r2"], r["gold"]) for r in both]), 2),
                   "router_BROS+zeroshot+tuned": round(micro([span_counts(r["_r3"], r["gold"]) for r in both]), 2),
                   "router3_minus_router2_boot": {"mean": d[0], "ci95": [d[1], d[2]]},
                   "oracle_BROS+zeroshot": round(oracle(conf, two), 2), "oracle_BROS+zeroshot+tuned": round(oracle(conf, three), 2),
                   "p_correct_given_BROS_wrong(gold field occurrences)": corr, "tables3": t3}
            res["D3"][f"{tuned}|{bros}"] = out
            print(f"\n{tuned} with {bros}: singles {single}\n  router BROS+zero-shot {out['router_BROS+zeroshot']}  router +tuned {out['router_BROS+zeroshot+tuned']}  diff {d}\n"
                  f"  oracle {out['oracle_BROS+zeroshot']} -> {out['oracle_BROS+zeroshot+tuned']}   P(right | BROS wrong) {corr}\n  fitted tables (+tuned): {t3}")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
