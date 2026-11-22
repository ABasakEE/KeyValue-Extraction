"""Study 40 (CPU): two parameter-free column rules applied to the saved BROS DocILE predictions, scored with the same row F1.

    python scripts/docile_column_rules.py [--runs results/docile_bros/run] [--sizes 200,1000] [--out results/docile_column_rules/results.json]

R1 column vote: on each page, predicted spans whose horizontal extents overlap by at least OVERLAP of the narrower span are chained into columns (union-find);
   a column of at least MIN_COL spans gets the field that most of its spans have (a tie changes nothing).
R2 table extent: on each page with at least MIN_ANCHORS spans of the anchor field (the field `group_rows` uses), a span whose vertical centre is more than one median
   anchor pitch above the first anchor or below the last anchor is dropped.
R1+R2 applies R1, then R2. Constants were fixed before any run (the 0.5 mirrors the row tolerance of `group_rows`); no other value was run.
Rows are then built by `group_rows` (tau 0.5) and scored as in `docile_bros.py score`. Paired bootstrap over the 200 test documents (runs of the same N averaged).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src"), str(HERE.parent)]
import docile_bros as B  # noqa: E402
from lir_metric import f1  # noqa: E402
from row_group import group_rows  # noqa: E402

OVERLAP, MIN_COL, MIN_ANCHORS = 0.5, 3, 3
ARMS = ["base", "R1", "R2", "R1+R2"]


def box(d: dict, s: int, e: int) -> tuple:
    fb = d["fboxes"]
    return (min(fb[i][0] for i in range(s, e)), min(fb[i][1] for i in range(s, e)), max(fb[i][2] for i in range(s, e)), max(fb[i][3] for i in range(s, e)))


def by_page(spans: list, d: dict) -> dict:
    out = defaultdict(list)
    for k, sp in enumerate(spans):
        out[d["pages"][sp[1]]].append(k)
    return out


def column_vote(spans: list, d: dict) -> list:
    out = [list(sp) for sp in spans]
    for ks in by_page(spans, d).values():
        bx = {k: box(d, spans[k][1], spans[k][2]) for k in ks}
        parent = {k: k for k in ks}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                ov = min(bx[a][2], bx[b][2]) - max(bx[a][0], bx[b][0])
                if ov > 0 and ov >= OVERLAP * min(bx[a][2] - bx[a][0], bx[b][2] - bx[b][0]):
                    parent[find(a)] = find(b)
        cols = defaultdict(list)
        for k in ks:
            cols[find(k)].append(k)
        for c in cols.values():
            if len(c) < MIN_COL:
                continue
            cnt = Counter(spans[k][0] for k in c).most_common()
            if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:
                continue
            for k in c:
                out[k][0] = cnt[0][0]
    return out


def drop_outside(spans: list, d: dict) -> list:
    keep = []
    for ks in by_page(spans, d).values():
        counts = Counter(spans[k][0] for k in ks)
        anchor = max(counts, key=lambda f: (counts[f], f))
        yc = {k: sum(box(d, spans[k][1], spans[k][2])[1::2]) / 2 for k in ks}
        a = sorted(yc[k] for k in ks if spans[k][0] == anchor)
        gaps = [y - x for x, y in zip(a, a[1:]) if y > x]
        if len(a) < MIN_ANCHORS or not gaps:
            keep += ks
            continue
        pitch = st.median(gaps)
        keep += [k for k in ks if a[0] - pitch <= yc[k] <= a[-1] + pitch]
    return [spans[k] for k in sorted(keep)]


def apply(arm: str, spans: list, d: dict) -> list:
    if arm in ("R1", "R1+R2"):
        spans = column_vote(spans, d)
    if arm in ("R2", "R1+R2"):
        spans = drop_outside(spans, d)
    return spans


def boot(deltas_by_run: list, n_docs: int, iters: int = 2000) -> list:
    """95% interval of the mean (over runs) of F1(modified) - F1(base) over resampled documents; each element: per-run (base, mod) arrays of per-document [tp, np, ng]."""
    rng, out = random.Random(40), []
    for _ in range(iters):
        idx = [rng.randrange(n_docs) for _ in range(n_docs)]
        ds = []
        for base, mod in deltas_by_run:
            tb, tm = [0, 0, 0], [0, 0, 0]
            for i in idx:
                for j in range(3):
                    tb[j] += base[i][j]
                    tm[j] += mod[i][j]
            ds.append(100 * (f1(tm) - f1(tb)))
        out.append(st.mean(ds))
    out.sort()
    return [round(out[int(0.025 * iters)], 2), round(out[int(0.975 * iters) - 1], 2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(B.ROOT / "results" / "docile_bros" / "run"))
    ap.add_argument("--sizes", default="200,1000")
    ap.add_argument("--out", default=str(B.ROOT / "results" / "docile_column_rules" / "results.json"))
    a = ap.parse_args()
    test = [g for g in json.loads(B.SPLITS.read_text(encoding="utf-8"))["test"]]
    docs = {g: B.doc_dict(g) for g in test}
    test = [g for g in test if docs[g]["rows"]]
    report: dict = {"constants": {"overlap": OVERLAP, "min_col": MIN_COL, "min_anchors": MIN_ANCHORS}, "sizes": {}}
    for n in a.sizes.split(","):
        runs, per = sorted(Path(a.runs).glob(f"N{n}_S*")), {}
        diag = {arm: Counter() for arm in ARMS[1:]}
        for run in runs:
            f = next(run.glob("seed_*/test_entities.jsonl"))
            pred = {r["guid"]: r["pred"] for r in map(json.loads, f.read_text(encoding="utf-8").splitlines())}
            per[run.name] = {arm: [] for arm in ARMS}
            for g in test:
                d, spans = docs[g], pred[g]
                gold = {(t, s, e) for t, s, e, _ in B.spans_of(d)}
                for arm in ARMS:
                    sp = apply(arm, spans, d)
                    per[run.name][arm].append(B.counts(group_rows(sp, d, 0.5, merge=False), d))
                    if arm != "base":
                        key = {(x[0], x[1], x[2]) for x in sp}
                        was = {(x[0], x[1], x[2]) for x in spans}
                        diag[arm]["predicted_before"] += len(spans)
                        diag[arm]["predicted_after"] += len(sp)
                        diag[arm]["exact_right_before"] += len(was & gold)
                        diag[arm]["exact_right_after"] += len(key & gold)
        res = {}
        for arm in ARMS:
            tot = {r: {m: [0, 0, 0] for m in ("lir", "cell")} for r in per}
            for r in per:
                for c in per[r][arm]:
                    for m in tot[r]:
                        tot[r][m] = [x + y for x, y in zip(tot[r][m], c[m])]
            res[arm] = {"row_f1_per_run": {r: round(100 * f1(tot[r]["lir"]), 2) for r in per}, "cell_f1_per_run": {r: round(100 * f1(tot[r]["cell"]), 2) for r in per}}
            res[arm]["row_f1_mean"] = round(st.mean(res[arm]["row_f1_per_run"].values()), 2)
            if arm != "base":
                bf = res["base"]["row_f1_per_run"]
                res[arm]["delta_per_run"] = {r: round(res[arm]["row_f1_per_run"][r] - bf[r], 2) for r in per}
                res[arm]["delta_mean"] = round(st.mean(res[arm]["delta_per_run"].values()), 2)
                res[arm]["delta_ci95_over_documents"] = boot([([c["lir"] for c in per[r]["base"]], [c["lir"] for c in per[r][arm]]) for r in per], len(test))
                halves = {}
                for h in (0, 1):
                    idx = [i for i in range(len(test)) if i % 2 == h]
                    ds = []
                    for r in per:
                        tb, tm = [0, 0, 0], [0, 0, 0]
                        for i in idx:
                            tb = [x + y for x, y in zip(tb, per[r]["base"][i]["lir"])]
                            tm = [x + y for x, y in zip(tm, per[r][arm][i]["lir"])]
                        ds.append(100 * (f1(tm) - f1(tb)))
                    halves[f"half_{h}"] = round(st.mean(ds), 2)
                res[arm]["delta_by_document_half"] = halves
                res[arm]["span_diagnostics_all_runs"] = dict(diag[arm])
        report["sizes"][n] = res
        print(f"N={n}  documents {len(test)}  runs {sorted(per)}")
        for arm in ARMS:
            r = res[arm]
            extra = "" if arm == "base" else f" | delta {r['delta_mean']:+.2f} per run {list(r['delta_per_run'].values())} CI {r['delta_ci95_over_documents']} halves {r['delta_by_document_half']} | spans {dict(diag[arm])}"
            print(f"  {arm:6s} row F1 {r['row_f1_mean']:.2f}{extra}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
