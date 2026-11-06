"""Study 19 summary: windows-trained vs baseline-trained joint LiLT, end-to-end (predicted entities), seed-matched.

    python scripts/xfund_e2e_summary.py --base results/xfund_e2e/base_e2e.json --windows results/xfund_e2e/windows_e2e.json \
        --out results/xfund_e2e/summary.json
    python scripts/xfund_e2e_summary.py selftest

Primary quantity: link F1, windowed inference, all val documents.  Verdict (registered in STUDIES.md study 19):
helps = mean seed-matched delta >= +0.02 and > 0 for every seed;  no_better = |mean delta| < 0.01 or the sign differs across seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMARY = "all|windowed|link"


def summarise(base: dict, windows: dict) -> dict:
    seeds = sorted(set(base["seeds"]) & set(windows["seeds"]), key=int)
    out = {"seeds": seeds, "cells": {}}
    keys = [k for k, v in base["seeds"][seeds[0]].items() if isinstance(v, dict) and k in windows["seeds"][seeds[0]]]
    for k in keys:
        b = [base["seeds"][s][k]["f1"] for s in seeds]
        w = [windows["seeds"][s][k]["f1"] for s in seeds]
        d = [round(x - y, 4) for x, y in zip(w, b)]
        out["cells"][k] = {"base": round(sum(b) / len(b), 4), "windows": round(sum(w) / len(w), 4),
                           "delta_per_seed": d, "delta_mean": round(sum(d) / len(d), 4)}
    p = out["cells"][PRIMARY]
    out["verdict"] = {
        "helps": p["delta_mean"] >= 0.02 and all(x > 0 for x in p["delta_per_seed"]),
        "no_better": abs(p["delta_mean"]) < 0.01 or min(p["delta_per_seed"]) < 0 < max(p["delta_per_seed"]),
        "primary_cell": PRIMARY,
    }
    return out


def _selftest() -> None:
    def run(f1s, ent=0.5):
        return {"seeds": {str(s): {PRIMARY: {"f1": f}, "all|single|entity": {"f1": ent}} for s, f in zip((13, 21, 42), f1s)}}
    r = summarise(run([.40, .41, .39]), run([.45, .44, .43]))
    assert r["verdict"]["helps"] and not r["verdict"]["no_better"], r["verdict"]
    assert r["cells"][PRIMARY]["delta_per_seed"] == [0.05, 0.03, 0.04]
    r = summarise(run([.40, .41, .39]), run([.45, .40, .43]))
    assert not r["verdict"]["helps"] and r["verdict"]["no_better"], r["verdict"]      # one seed worse
    r = summarise(run([.40, .41, .39]), run([.405, .415, .395]))
    assert not r["verdict"]["helps"] and r["verdict"]["no_better"], r["verdict"]      # +0.5 points
    print("selftest passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run")
    ap.add_argument("--base")
    ap.add_argument("--windows")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return _selftest()
    res = summarise(json.loads(Path(a.base).read_text(encoding="utf-8")), json.loads(Path(a.windows).read_text(encoding="utf-8")))
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": res["verdict"], PRIMARY: res["cells"][PRIMARY]}, indent=2))


if __name__ == "__main__":
    main()
