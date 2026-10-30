"""Study 15, exploratory post-hoc analysis (NOT part of the registered rule; written after seeing the sweep results).

For each number of seen templates K: how does the router compare with the VLM alone (the natural baseline when BROS
is weak), how much does BROS's F1 vary with which templates it saw, and how much of the oracle headroom does a static
per-field router capture?

Usage: python scripts/fatura_sweep_posthoc.py --sweep-dir results/fatura_v2_sweep --out results/fatura_v2_sweep/posthoc.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatura_eval import SEEDS, boot_f1, ci, micro_f1, systems_for  # noqa: E402
from fatura_sweep import with_bros  # noqa: E402
from matched_eval import span_counts  # noqa: E402


def analyse_k(recs: list[dict]) -> dict:
    confirm = [r for r in recs if r["split"] == "confirm"]
    parts, per_run = [], {}
    for f in sorted({r["fold"] for r in recs}):
        test = [r for r in confirm if r["fold"] == f]
        sys_f = systems_for(test, [r for r in recs if r["fold"] != f])
        parts.append((test, sys_f))
        for s in SEEDS:
            per_run[f"fold{f}_seed{s}"] = round(micro_f1(sys_f[f"bros_seed{s}"]), 4)
    pooled = {k: [c for _, s in parts for c in s[k]] for k in parts[0][1]}
    clusters = [r["template"] for test, _ in parts for r in test]
    b = boot_f1(pooled, clusters)
    bros = np.mean([b[f"bros_seed{s}"] for s in SEEDS], axis=0)
    router = np.mean([b[f"router_seed{s}"] for s in SEEDS], axis=0)
    oracle = np.mean([b[f"oracle_seed{s}"] for s in SEEDS], axis=0)
    f1 = lambda name: float(np.mean([micro_f1(pooled[f"{name}_seed{s}"]) for s in SEEDS])) if name != "vlm_both" else micro_f1(pooled["vlm_both"])
    best_single = np.maximum(bros.mean(), b["vlm_both"].mean())
    runs = list(per_run.values())
    return {"bros_f1": round(f1("bros"), 4), "vlm_f1": round(f1("vlm_both"), 4), "router_f1": round(f1("router"), 4),
            "oracle_f1": round(f1("oracle"), 4),
            "router_minus_vlm": round(f1("router") - f1("vlm_both"), 4), "router_minus_vlm_ci95": ci(router - b["vlm_both"]),
            "router_minus_bros_ci95": ci(router - bros),
            "oracle_headroom_over_bros": round(f1("oracle") - f1("bros"), 4),
            "router_share_of_headroom": round((f1("router") - f1("bros")) / (f1("oracle") - f1("bros")), 3) if f1("oracle") > f1("bros") else None,
            "better_single_system": "vlm" if b["vlm_both"].mean() > bros.mean() else "bros",
            "router_minus_better_single": round(f1("router") - max(f1("bros"), f1("vlm_both")), 4),
            "bros_f1_per_fold_seed": per_run, "bros_f1_run_min": min(runs), "bros_f1_run_max": max(runs),
            "bros_f1_run_std": round(float(np.std(runs)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="results/fatura_v1/spans.jsonl")
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--ks", default="1,2,5,10")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.loads(l) for l in Path(a.spans).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {"note": "exploratory, written after the registered sweep results were seen", "reference_k40": analyse_k(recs)}
    for k in a.ks.split(","):
        sub, _ = with_bros(recs, Path(a.sweep_dir), int(k))
        out[f"K{k}"] = analyse_k(sub)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, v in out.items():
        if name != "note":
            print(f"{name:13s} BROS {100*v['bros_f1']:5.1f} VLM {100*v['vlm_f1']:5.1f} router {100*v['router_f1']:5.1f} | router-VLM {100*v['router_minus_vlm']:+5.1f} "
                  f"{[round(100*c,1) for c in v['router_minus_vlm_ci95']]} | oracle headroom {100*v['oracle_headroom_over_bros']:+5.1f}, router captures {v['router_share_of_headroom']} "
                  f"| BROS per-run F1 min/max/std {100*v['bros_f1_run_min']:.1f}/{100*v['bros_f1_run_max']:.1f}/{100*v['bros_f1_run_std']:.1f} | better single: {v['better_single_system']}")


if __name__ == "__main__":
    main()
