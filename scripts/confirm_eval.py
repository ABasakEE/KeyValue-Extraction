"""Study 12: confirmation of the study-11 hybrid on fresh unseen-template documents (CPU scoring).

Everything was frozen before the GPU run: extraction/naming prompts, schema descriptions (copied from v2),
one-answer-per-field, the mode-free `vlm_both` mapping, grounding, and the pass rule of studies 10-11.

* primary   : the per-fold router tables learned in study 11 (each from the *other two* templates' old documents),
              applied unchanged to fresh documents of the held-out template.
* secondary : the same routing procedure re-fitted on the fresh documents of the other two templates.
* validity  : BROS on the fresh sample vs its full test-set F1; VLM on the fresh sample vs the old sample.

Pass (unchanged): pooled router - BROS (mean over seeds, paired bootstrap) has a 95% interval above 0, and no
template's interval lies entirely below 0.

Usage: python scripts/confirm_eval.py --spans results/confirm_v2/matched/spans.jsonl --out results/confirm_v2/confirm.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_eval import fit_router, route  # noqa: E402
from matched_eval import SEEDS, ci, f1_of, paired_bootstrap, span_counts  # noqa: E402


def micro(counts: list[dict]) -> float:
    return f1_of([sum(c["_micro"][i] for c in counts) for i in range(3)])


def compare(per_doc: dict[str, list[dict]], name: str) -> dict:
    """Router `name` vs BROS: F1 per seed, mean delta and its paired-bootstrap 95% interval."""
    boot = paired_bootstrap(per_doc, "_micro")
    delta = np.mean([boot[f"{name}_seed{s}"] - boot[f"bros_seed{s}"] for s in SEEDS], axis=0)
    return {"bros_f1": round(float(np.mean([micro(per_doc[f"bros_seed{s}"]) for s in SEEDS])), 4),
            "f1": round(float(np.mean([micro(per_doc[f"{name}_seed{s}"]) for s in SEEDS])), 4),
            "minus_bros": round(float(np.mean(delta)), 4), "minus_bros_ci95": ci(delta)}


def verdict(by_template: dict, pooled: dict) -> dict:
    pooled_ok = pooled["minus_bros_ci95"][0] > 0
    no_loss = all(v["minus_bros_ci95"][1] >= 0 for v in by_template.values())
    return {"pooled_gain_significant": pooled_ok, "no_template_significantly_worse": no_loss,
            "passes": pooled_ok and no_loss}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", required=True, help="spans.jsonl written by matched_eval.py on the fresh documents")
    ap.add_argument("--frozen", default="results/hybrid_eval/metrics.json", help="study-11 router decisions")
    ap.add_argument("--old-matched", default="results/matched_eval/metrics.json", help="study 9/11 VLM results")
    ap.add_argument("--bros-metrics", default="results/vrdu_track1_bros_errdump/metrics.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.spans).read_text(encoding="utf-8").splitlines() if l.strip()]
    frozen = json.loads(Path(args.frozen).read_text())["decisions"]
    old = json.loads(Path(args.old_matched).read_text())["folds"]
    bros_full = {(r["fold"], r["seed"]): r["test"]["f1"] for r in json.loads(Path(args.bros_metrics).read_text())["UTL"]}
    folds = sorted({r["fold"] for r in records})

    per_doc: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    tables: dict = {}
    for fold in folds:
        test = [r for r in records if r["fold"] == fold]
        fresh_train = [r for r in records if r["fold"] != fold]
        for s in SEEDS:
            bros = f"bros_seed{s}"
            primary = frozen[str(fold)][f"router_both_seed{s}"]
            secondary = fit_router(fresh_train, bros, ("vlm_both",))
            tables[f"fold{fold}_seed{s}"] = {"primary_frozen": primary, "secondary_refit": secondary}
            for r in test:
                sources = {bros: r[bros], "vlm_both": r["vlm_both"]}
                per_doc[f"bros_seed{s}"][fold].append(span_counts(r[bros], r["gold"]))
                per_doc[f"primary_seed{s}"][fold].append(span_counts(route(primary, sources, bros), r["gold"]))
                per_doc[f"secondary_seed{s}"][fold].append(span_counts(route(secondary, sources, bros), r["gold"]))
        for m in ("vlm_key", "vlm_named", "vlm_both"):
            per_doc[m][fold] = [span_counts(r[m], r["gold"]) for r in test]

    report: dict = {"rules": __doc__.split("Usage:")[0].strip(), "router_tables": tables, "validity": {},
                    "primary": {}, "secondary": {}, "verdict": {}}
    for fold in folds:
        template = next(r["template"] for r in records if r["fold"] == fold)
        sel = {k: v[fold] for k, v in per_doc.items()}
        boot = paired_bootstrap(sel, "_micro")
        old_fold = old[f"fold{fold}_{template}"]["systems"]
        report["validity"][template] = {
            "n_docs": len(sel["bros_seed13"]),
            "bros": {f"seed{s}": {"fresh_f1": round(micro(sel[f"bros_seed{s}"]), 4),
                                  "fresh_ci95": ci(boot[f"bros_seed{s}"]),
                                  "full_test_f1": bros_full[(fold, s)],
                                  "full_test_inside_ci": ci(boot[f"bros_seed{s}"])[0] <= bros_full[(fold, s)]
                                  <= ci(boot[f"bros_seed{s}"])[1]} for s in SEEDS},
            "vlm": {m: {"fresh_f1": round(micro(sel[m]), 4), "fresh_ci95": ci(boot[m]),
                        "old_f1": old_fold[m]["f1"], "old_ci95": old_fold[m]["f1_ci95"],
                        "intervals_overlap": ci(boot[m])[0] <= old_fold[m]["f1_ci95"][1]
                        and old_fold[m]["f1_ci95"][0] <= ci(boot[m])[1]}
                    for m in ("vlm_key", "vlm_named", "vlm_both")},
        }
        for name in ("primary", "secondary"):
            report[name][template] = compare(sel, name)
    pooled = {k: [x for f in folds for x in v[f]] for k, v in per_doc.items()}
    for name in ("primary", "secondary"):
        report[name]["_pooled"] = compare(pooled, name)
        report["verdict"][name] = verdict({t: v for t, v in report[name].items() if t != "_pooled"},
                                          report[name]["_pooled"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("primary", "secondary", "verdict")}, indent=2))


def _selftest() -> None:
    gold = [["foreign_principle_name", 10, 12, "x"], ["signer_name", 20, 22, "z"]]
    bros = [["signer_name", 20, 22, "z"]]
    vlm = [["foreign_principle_name", 10, 12, "x"], ["signer_name", 30, 32, "w"]]
    table = {"foreign_principle_name": "vlm_both", "signer_name": "bros_seed13"}
    assert span_counts(route(table, {"bros_seed13": bros, "vlm_both": vlm}, "bros_seed13"), gold)["_micro"] == (2, 2, 2)
    ok = {"a": {"minus_bros_ci95": [0.01, 0.05]}, "b": {"minus_bros_ci95": [-0.02, 0.03]}}
    assert verdict(ok, {"minus_bros_ci95": [0.01, 0.04]})["passes"]
    assert not verdict({**ok, "c": {"minus_bros_ci95": [-0.05, -0.01]}}, {"minus_bros_ci95": [0.01, 0.04]})["passes"]
    assert not verdict(ok, {"minus_bros_ci95": [-0.01, 0.04]})["passes"]
    print("[+] confirm_eval selftest passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        _selftest()
    else:
        main()
