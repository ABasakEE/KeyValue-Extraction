"""Hybrid feasibility study (CPU): is combining Track-1 BROS with the zero-shot VLM worth pursuing?

Reads the per-document spans written by scripts/matched_eval.py (same 100 documents per UTL fold, same gold,
exact entity-span F1). Every rule below was fixed before any hybrid result was computed.

Systems (each built separately with BROS seed 13 and seed 42, compared against the same seed):
* ``oracle``   per (document, field), the source with more correct spans (ties: fewer predictions, then BROS).
               An upper bound; it uses the gold labels, so it is never a usable system.
* ``fill_in``  BROS spans, plus VLM spans for fields BROS predicted nothing for in that document (VLM spans
               overlapping a BROS span are dropped). The VLM mode (key / named) is chosen by pooled VLM F1 on the
               other two templates.
* ``router``   per field, the source (BROS, VLM key, VLM named) with the best pooled F1 for that field on the other
               two templates (ties and fields without gold there: BROS); applied to the held-out template.

Pass rule (pre-declared): a usable hybrid (fill_in or router) is worth pursuing if, pooled over the 300 documents,
its F1 minus BROS F1 (mean over seeds, paired bootstrap) has a 95% interval above 0, and on no single template
is that interval entirely below 0.

Usage: python scripts/hybrid_eval.py --spans results/matched_eval/spans.jsonl --out results/hybrid_eval/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matched_eval import SEEDS, ci, f1_of, paired_bootstrap, span_counts  # noqa: E402

VLM_MODES = ("vlm_key", "vlm_named")


def _overlaps(span: list, used: set[int]) -> bool:
    return bool(used.intersection(range(span[1], span[2])))


def fill_in(bros: list[list], vlm: list[list]) -> list[list]:
    have = {s[0] for s in bros}
    used = {i for s in bros for i in range(s[1], s[2])}
    return bros + [s for s in vlm if s[0] not in have and not _overlaps(s, used)]


def route(choice: dict[str, str], sources: dict[str, list[list]], default: str) -> list[list]:
    """Spans of each field from its chosen source; the default source's spans are placed first, overlaps dropped."""
    out, used = [], set()
    for prefer_default in (True, False):
        for name, spans in sources.items():
            for s in spans:
                if (choice.get(s[0], default) == name) and ((name == default) == prefer_default) and not _overlaps(s, used):
                    out.append(s)
                    used.update(range(s[1], s[2]))
    return out


def oracle_counts(gold: list[list], sources: dict[str, list[list]], order: list[str]) -> dict:
    """Per field, the counts of the source with the most true positives (ties: fewer predictions, then order)."""
    per_source = {n: span_counts(sources[n], gold) for n in order}
    fields = {f for c in per_source.values() for f in c if f != "_micro"}
    out = {}
    for f in fields:
        ranked = [(c[0], -c[1], -i, c) for i, c in enumerate(per_source[n].get(f, (0, 0, 0)) for n in order)]
        out[f] = max(ranked)[3]
    out["_micro"] = tuple(sum(c[i] for f, c in out.items()) for i in range(3))
    return out


def pooled_f1(records: list[dict], source: str, field: str | None = None) -> float:
    key = field or "_micro"
    c = [0, 0, 0]
    for r in records:
        x = span_counts(r[source], r["gold"]).get(key, (0, 0, 0))
        c = [a + b for a, b in zip(c, x)]
    return f1_of(c), c[2]


def fit_router(train: list[dict], bros: str) -> dict[str, str]:
    fields = sorted({s[0] for r in train for s in r["gold"]})
    choice = {}
    for f in fields:
        scores = [(pooled_f1(train, src, f)[0], -i, src) for i, src in enumerate([bros, *VLM_MODES])]
        best = max(scores)
        choice[f] = best[2] if best[0] > 0 else bros
    return choice


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="results/matched_eval/spans.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.spans).read_text(encoding="utf-8").splitlines() if l.strip()]
    folds = sorted({r["fold"] for r in records})
    per_doc: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    decisions: dict = {}
    for fold in folds:
        test = [r for r in records if r["fold"] == fold]
        train = [r for r in records if r["fold"] != fold]  # the other two templates
        mode = max(VLM_MODES, key=lambda m: (pooled_f1(train, m)[0], m == "vlm_key"))
        decisions[fold] = {"fill_in_vlm_mode": mode}
        for s in SEEDS:
            bros = f"bros_seed{s}"
            router = fit_router(train, bros)
            decisions[fold][f"router_seed{s}"] = router
            for r in test:
                sources = {bros: r[bros], **{m: r[m] for m in VLM_MODES}}
                per_doc[f"bros_seed{s}"][fold].append(span_counts(r[bros], r["gold"]))
                per_doc[f"fill_in_seed{s}"][fold].append(span_counts(fill_in(r[bros], r[mode]), r["gold"]))
                per_doc[f"router_seed{s}"][fold].append(span_counts(route(router, sources, bros), r["gold"]))
                per_doc[f"oracle_seed{s}"][fold].append(oracle_counts(r["gold"], sources, [bros, *VLM_MODES]))

    report: dict = {"rules": __doc__.split("Usage:")[0].strip(), "decisions": decisions, "by_template": {}, "pooled": {}}

    def summarise(sel: dict[str, list[dict]]) -> dict:
        boot = paired_bootstrap(sel, "_micro")
        out = {}
        for name in ("bros", "fill_in", "router", "oracle"):
            f1s = [f1_of([sum(x["_micro"][i] for x in sel[f"{name}_seed{s}"]) for i in range(3)]) for s in SEEDS]
            entry = {"f1_per_seed": [round(v, 4) for v in f1s], "f1_mean": round(float(np.mean(f1s)), 4)}
            if name != "bros":
                delta = np.mean([boot[f"{name}_seed{s}"] - boot[f"bros_seed{s}"] for s in SEEDS], axis=0)
                entry["minus_bros"] = round(float(np.mean(delta)), 4)
                entry["minus_bros_ci95"] = ci(delta)
            out[name] = entry
        return out

    for fold in folds:
        template = next(r["template"] for r in records if r["fold"] == fold)
        report["by_template"][template] = summarise({k: v[fold] for k, v in per_doc.items()})
    report["pooled"] = summarise({k: [x for f in folds for x in v[f]] for k, v in per_doc.items()})

    verdict = {}
    for name in ("fill_in", "router"):
        pooled_ok = report["pooled"][name]["minus_bros_ci95"][0] > 0
        no_loss = all(report["by_template"][t][name]["minus_bros_ci95"][1] >= 0 for t in report["by_template"])
        verdict[name] = {"pooled_gain_significant": pooled_ok, "no_template_significantly_worse": no_loss,
                         "passes": pooled_ok and no_loss}
    report["verdict"] = verdict
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("pooled", "verdict")}, indent=2))


def _selftest() -> None:
    gold = [["foreign_principle_name", 10, 12, "x"], ["registrant_name", 0, 2, "y"], ["signer_name", 20, 22, "z"]]
    bros = [["registrant_name", 5, 7, "decoy"], ["signer_name", 20, 22, "z"]]
    vlm = [["foreign_principle_name", 10, 12, "x"], ["registrant_name", 0, 2, "y"], ["signer_name", 30, 32, "w"]]
    # fill-in adds only fields BROS left empty
    assert fill_in(bros, vlm) == bros + [["foreign_principle_name", 10, 12, "x"]]
    assert fill_in(bros, [["file_date", 6, 8, "d"]]) == bros  # overlaps a BROS span -> dropped
    # router takes each field from its chosen source
    routed = route({"registrant_name": "vlm"}, {"bros": bros, "vlm": vlm}, "bros")
    assert sorted(routed) == sorted([["signer_name", 20, 22, "z"], ["registrant_name", 0, 2, "y"]]), routed
    # oracle picks the better source per field: fpn + registrant from VLM, signer from BROS -> 3/3 correct
    o = oracle_counts(gold, {"bros": bros, "vlm": vlm}, ["bros", "vlm"])
    assert o["_micro"] == (3, 3, 3), o
    # fit_router chooses VLM for a field only where it is better on the training records
    train = [{"gold": gold, "bros_seed13": bros, "vlm_key": vlm, "vlm_named": []}]
    ch = fit_router(train, "bros_seed13")
    assert ch["registrant_name"] == "vlm_key" and ch["signer_name"] == "bros_seed13", ch
    print("[+] hybrid_eval selftest passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        _selftest()
    else:
        main()
