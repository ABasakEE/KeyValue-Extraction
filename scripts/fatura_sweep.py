"""Study 15 (CPU scoring): does the BROS + VLM router help once BROS has seen few templates?

Re-uses the VLM spans of study 14 (results/fatura_v1/spans.jsonl: gold, vlm_key/vlm_named/vlm_both for the 1,000
evaluated invoices) and replaces the BROS predictions with those of BROS models trained on only K seen templates
(10 invoices each; K = 1, 2, 5, 10), from scripts/fatura_bros.py --k-list ... --test-set eval. For each K the
unchanged study-14 analysis runs: router fitted on the other four folds' invoices, applied to the held-out fold's
confirm invoices; template-bootstrap intervals; oracle; fill-in; VLM alone.

Registered before the run:
* Reference: K = 40 / 200 invoices is study 14 (BROS 99.9%).
* A setting "has a template gap" if BROS's pooled F1 there is below 90%.
* The hybrid is judged helpful in the data-starved regime if at least one setting has a template gap AND the router
  passes the study-14 rule there (pooled router - BROS 95% template-bootstrap interval above 0, no fold's interval
  entirely below 0). If BROS stays at or above 90% for every K, FATURA cannot test the hybrid.
Nothing else is a pass criterion; per-field and per-K detail is descriptive.

Usage: python scripts/fatura_sweep.py --sweep-dir results/fatura_v2_sweep --out results/fatura_v2_sweep/sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatura_eval import SEEDS, analyse, same_spans  # noqa: E402

GAP_THRESHOLD = 0.90


def with_bros(recs: list[dict], sweep_dir: Path, k: int) -> tuple[list[dict], dict]:
    """Copies of the records whose BROS spans come from the K-template models; drops invoices missing a prediction."""
    dumps, info = {}, {"alignment_mismatches": 0, "missing": 0, "runs_found": 0}
    for f in sorted({r["fold"] for r in recs}):
        for s in SEEDS:
            path = sweep_dir / f"K{k}" / f"fold{f}" / f"seed_{s}" / "test_entities.jsonl"
            if path.exists():
                dumps[(f, s)] = {r["guid"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
                info["runs_found"] += 1
    out = []
    for r in recs:
        new = {key: v for key, v in r.items() if not key.startswith("bros_seed")}
        ok = True
        for s in SEEDS:
            row = dumps.get((r["fold"], s), {}).get(r["guid"])
            if row is None:
                ok = False
                break
            info["alignment_mismatches"] += not same_spans(row["gold"], r["gold"])
            new[f"bros_seed{s}"] = row["pred"]
        info["missing"] += not ok
        if ok:
            out.append(new)
    return out, info


def regime_verdict(per_k: dict[str, dict]) -> dict:
    gap = {k: v["bros_f1"] < GAP_THRESHOLD for k, v in per_k.items()}
    helpful = [k for k, v in per_k.items() if gap[k] and v["router_passes"]]
    return {"settings_with_template_gap": [k for k, g in gap.items() if g], "settings_where_router_passes": [k for k, v in per_k.items() if v["router_passes"]],
            "helpful_in_starved_regime": bool(helpful), "helpful_settings": helpful,
            "fatura_can_test_hybrid": any(gap.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="results/fatura_v1/spans.jsonl")
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--ks", default="1,2,5,10")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.loads(l) for l in Path(a.spans).read_text(encoding="utf-8").splitlines() if l.strip()]
    sweep_dir = Path(a.sweep_dir)
    settings, per_k = {}, {}
    reference = analyse(recs)
    for name, rep, n_train in [("40", reference, 200)] + [(k, None, 10 * int(k)) for k in a.ks.split(",")]:
        info = {}
        if rep is None:
            sub, info = with_bros(recs, sweep_dir, int(name))
            if not sub or {r["fold"] for r in sub} != {r["fold"] for r in recs}:
                settings[name] = {"n_train_invoices": n_train, "error": "no complete BROS predictions", **info}
                continue
            rep = analyse(sub)
            rep["invoices_scored"] = len(sub)
        settings[name] = {"n_train_invoices": n_train, "n_seen_templates": int(name), "alignment": info, **rep}
        p = rep["pooled"]
        per_k[name] = {"bros_f1": p["router"]["bros_f1"], "router_f1": p["router"]["f1"], "router_minus_bros": p["router"]["minus_bros"],
                       "router_ci95": p["router"]["minus_bros_template_ci95"], "router_passes": rep["verdict"]["passes"],
                       "oracle_minus_bros": p["oracle"]["minus_bros"], "fill_in_minus_bros": p["fill_in"]["minus_bros"],
                       "vlm_both_f1": p["vlm_both"]["f1"]}
    out = {"rules": __doc__.split("Registered before the run:")[1].split("Usage:")[0].strip(), "summary": per_k,
           "verdict": regime_verdict(per_k), "settings": settings}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"summary": per_k, "verdict": out["verdict"]}, indent=2))


def _selftest() -> None:
    per_k = {"40": {"bros_f1": 0.999, "router_passes": False}, "5": {"bros_f1": 0.80, "router_passes": True},
             "2": {"bros_f1": 0.70, "router_passes": False}}
    v = regime_verdict(per_k)
    assert v["settings_with_template_gap"] == ["5", "2"] and v["helpful_settings"] == ["5"] and v["helpful_in_starved_regime"]
    assert not regime_verdict({"1": {"bros_f1": 0.95, "router_passes": True}})["fatura_can_test_hybrid"]
    assert not regime_verdict({"1": {"bros_f1": 0.95, "router_passes": True}})["helpful_in_starved_regime"]
    assert not regime_verdict({"1": {"bros_f1": 0.5, "router_passes": False}})["helpful_in_starved_regime"]
    print("[+] fatura_sweep selftest passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        _selftest()
    else:
        main()
