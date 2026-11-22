"""Study 39 post hoc (CPU): where do the BROS DocILE row-F1 points go? Reads results/docile_bros/run/N*_S*/seed_13/test_entities.jsonl.

    python scripts/docile_bros_errors.py [--runs results/docile_bros/run] [--sizes 200,1000] [--out results/docile_bros/errors.json]

For each run, on the 200 test documents, compares predicted spans with the gold spans (tags -> spans of the gold rows):
  * typed exact span F1, untyped exact span F1 (field ignored), and where each gold span ends up (right, wrong field, wrong extent, missed);
  * row F1 with the anchor rule on predicted spans (the reported number), with the TRUE row of each predicted span (grouping oracle),
    and with predicted extents relabelled with the gold field (typing oracle) -- each oracle changes one thing, so the gaps show which stage costs the points.
A predicted span that starts on no gold word goes into one shared row "-1" under the grouping oracle (the same convention as `docile_bros.py oracle`).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src"), str(HERE.parent)]
import docile_bros as B  # noqa: E402
from lir_metric import f1  # noqa: E402
from row_group import group_rows  # noqa: E402


def prf(tp: int, np_: int, ng: int) -> float:
    return round(100 * f1([tp, np_, ng]), 2)


def analyse(run: Path, test: list[str], docs: dict) -> dict:
    f = next(run.glob("seed_*/test_entities.jsonl"))
    pred = {r["guid"]: r["pred"] for r in map(json.loads, f.read_text(encoding="utf-8").splitlines())}
    typed = [0, 0, 0]; untyped = [0, 0, 0]
    where = Counter(); confuse = Counter(); per_field = defaultdict(lambda: [0, 0, 0])
    spurious = Counter()
    tot = {"reported": {}, "group_oracle": {}, "typing_oracle": {}}
    for g in test:
        d = docs[g]
        if not d["rows"] or g not in pred:
            continue
        gold = [(t, s, e) for t, s, e, _ in B.spans_of(d)]
        ps = [(t, s, e) for t, s, e, *_ in pred[g]]
        gset, pset = set(gold), set(ps)
        typed = [typed[0] + len(gset & pset), typed[1] + len(pset), typed[2] + len(gset)]
        gu, pu = {(s, e) for _, s, e in gold}, {(s, e) for _, s, e in ps}
        untyped = [untyped[0] + len(gu & pu), untyped[1] + len(pu), untyped[2] + len(gu)]
        for t in {x[0] for x in gold} | {x[0] for x in ps}:
            tg, tp = {x for x in gset if x[0] == t}, {x for x in pset if x[0] == t}
            for k, v in enumerate((len(tg & tp), len(tp), len(tg))):
                per_field[t][k] += v
        for t, s, e in gold:   # where does each gold span end up?
            ov = [p for p in ps if p[1] < e and s < p[2]]
            if (t, s, e) in pset:
                where["right"] += 1
            elif any((p[1], p[2]) == (s, e) for p in ov):
                where["wrong field, same extent"] += 1
                for p in ov:
                    if (p[1], p[2]) == (s, e):
                        confuse[(t, p[0])] += 1
            elif any(p[0] == t for p in ov):
                where["right field, wrong extent"] += 1
            elif ov:
                where["wrong field and extent"] += 1
            else:
                where["missed (no predicted span touches it)"] += 1
        gold_words = {i for _, s, e in gold for i in range(s, e)}
        for t, s, e in ps:
            if not any(i in gold_words for i in range(s, e)):
                spurious[t] += 1
        # row F1 under the three decodes
        B.add(tot["reported"], B.counts(group_rows(pred[g], d, 0.5, merge=False), d))
        word_row = {i: k for k, r in enumerate(d["rows"]) for _, ws, _ in r for i in ws}
        byrow = defaultdict(set)
        for t, s, e in ps:
            byrow[word_row.get(s, -1)].add((t, frozenset(range(s, e))))
        B.add(tot["group_oracle"], B.counts(list(byrow.values()), d))
        gold_field = {(s, e): t for t, s, e in gold}
        relabel = [[gold_field.get((s, e), t), s, e, ""] for t, s, e in ps]
        B.add(tot["typing_oracle"], B.counts(group_rows(relabel, d, 0.5, merge=False), d))
    return {"typed_span_f1": prf(*typed), "untyped_span_f1": prf(*untyped), "typed_counts": typed, "untyped_counts": untyped, "gold_span_outcome": dict(where),
            "top_field_confusions": [[a, b, n] for (a, b), n in confuse.most_common(8)], "spurious_by_field": dict(spurious.most_common(6)),
            "per_field": {k: {"f1": prf(*v), "tp": v[0], "pred": v[1], "gold": v[2]} for k, v in sorted(per_field.items(), key=lambda kv: -kv[1][2])[:10]},
            **{k: B.show(v) for k, v in tot.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(B.ROOT / "results" / "docile_bros" / "run"))
    ap.add_argument("--sizes", default="200,1000")
    ap.add_argument("--out", default=str(B.ROOT / "results" / "docile_bros" / "errors.json"))
    a = ap.parse_args()
    test = json.loads(B.SPLITS.read_text(encoding="utf-8"))["test"]
    docs = {g: B.doc_dict(g) for g in test}
    out = {}
    for n in a.sizes.split(","):
        for run in sorted(Path(a.runs).glob(f"N{n}_S*")):
            out[run.name] = analyse(run, test, docs)
            r = out[run.name]
            print(run.name, "typed span F1", r["typed_span_f1"], "untyped", r["untyped_span_f1"], "| row F1: reported", r["reported"]["row_f1"],
                  "group oracle", r["group_oracle"]["row_f1"], "typing oracle", r["typing_oracle"]["row_f1"])
    for n in a.sizes.split(","):
        rs = [v for k, v in out.items() if k.startswith(f"N{n}_")]
        if rs:
            m = lambda f: round(st.mean(f(r) for r in rs), 2)
            print(f"N={n} mean: typed {m(lambda r: r['typed_span_f1'])} untyped {m(lambda r: r['untyped_span_f1'])} | row F1 reported {m(lambda r: r['reported']['row_f1'])} "
                  f"group oracle {m(lambda r: r['group_oracle']['row_f1'])} typing oracle {m(lambda r: r['typing_oracle']['row_f1'])}")
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
