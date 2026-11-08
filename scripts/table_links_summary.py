"""Study 26 summary: per-arm means and seed-matched deltas vs A0, with a verdict per arm.

    python scripts/table_links_summary.py --arms A0=results/a0.json A1=results/a1.json --out results/summary.json
    python scripts/table_links_summary.py selftest

Inputs are the per-arm JSON files of xfund_e2e_eval.py.  Deltas are in F1 points (x100), matched by seed id.
Verdict precedence (registered in STUDIES.md), first match applies:
  helps = mean >= +2.0 and every seed > 0;  no_better = |mean| < 1.0 or signs differ across seeds;
  worse = mean < 0 and every seed < 0;  else inconclusive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMARY = "all|windowed|link"


def verdict(deltas_pts: list[float]) -> str:
    mean = sum(deltas_pts) / len(deltas_pts)
    if mean >= 2.0 and all(d > 0 for d in deltas_pts):
        return "helps"
    if abs(mean) < 1.0 or min(deltas_pts) < 0 < max(deltas_pts):
        return "no_better"
    if mean < 0 and all(d < 0 for d in deltas_pts):
        return "worse"
    return "inconclusive"


def _metric(k: str) -> str:
    return "recall" if "|group|" in k else "f1"


def summarise(arms: dict[str, dict]) -> dict:
    base = arms["A0"]
    out = {"primary_cell": PRIMARY, "arms": {}}
    for name, res in arms.items():
        seeds = sorted(set(base["seeds"]) & set(res["seeds"]), key=int)
        keys = [k for k, v in base["seeds"][seeds[0]].items()
                if isinstance(v, dict) and (k == PRIMARY or "|group|" in k) and k in res["seeds"][seeds[0]]]
        cells = {}
        for k in keys:
            m = _metric(k)
            a = [res["seeds"][s][k][m] for s in seeds]
            b = [base["seeds"][s][k][m] for s in seeds]
            d = [round((x - y) * 100, 2) for x, y in zip(a, b)]
            cells[k] = {"metric": m, "mean": round(sum(a) / len(a) * 100, 2), "delta_pts_per_seed": d,
                        "delta_pts_mean": round(sum(d) / len(d), 2)}
        arm = {"seeds": seeds, "cells": cells}
        if name != "A0":
            arm["verdict"] = verdict(cells[PRIMARY]["delta_pts_per_seed"])
        out["arms"][name] = arm
    return out


def _selftest() -> None:
    assert verdict([2.5, 2.1, 3.0]) == "helps"
    assert verdict([0.5, -0.3, 0.4]) == "no_better"
    assert verdict([-1.5, -2.0, -1.2]) == "worse"
    assert verdict([1.5, 1.2, 1.8]) == "inconclusive"
    assert verdict([2.5, -0.1, 4.0]) == "no_better"

    def run(f1s, rec):
        return {"seeds": {str(s): {PRIMARY: {"f1": f}, "all|windowed|group|below": {"recall": r, "f1": 0.0}}
                          for s, f, r in zip((13, 21, 42), f1s, rec)}}
    r = summarise({"A0": run([.40, .41, .39], [.5, .5, .5]), "A1": run([.43, .44, .42], [.6, .6, .6])})
    assert r["arms"]["A1"]["verdict"] == "helps" and "verdict" not in r["arms"]["A0"]
    assert r["arms"]["A1"]["cells"][PRIMARY]["delta_pts_per_seed"] == [3.0, 3.0, 3.0]
    assert r["arms"]["A1"]["cells"]["all|windowed|group|below"]["delta_pts_mean"] == 10.0
    print("selftest passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run")
    ap.add_argument("--arms", nargs="+", help="NAME=path.json, one per arm; A0 is the reference")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return _selftest()
    arms = {n: json.loads(Path(p).read_text(encoding="utf-8")) for n, p in (x.split("=", 1) for x in a.arms)}
    res = summarise(arms)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({n: {"verdict": v.get("verdict"), PRIMARY: v["cells"][PRIMARY]} for n, v in res["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
