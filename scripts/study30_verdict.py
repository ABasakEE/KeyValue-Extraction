"""Study 30 verdicts (CPU): S and T against the word-number reference P (study 29's E1 and E1r) and BROS, by the registered rules.

    python scripts/study30_verdict.py [--dir results/vlm_sft30] [--p-dir results/vlm_sft] [--out results/vlm_sft30/verdict.json]

Reads spans_{S,T,Sr,Tr}.jsonl (written by `vrdu_vlm_sft.py score`) and the study-29 files spans_E1.jsonl / spans_E1r.jsonl.
Arms that are missing are skipped.  Primary cell: pooled exact-span micro F1 on the 600 UTL documents, paired bootstrap over documents.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from matched_eval import f1_of, span_counts  # noqa: E402
from vlm_sft_diagnostics import FIELDS, tolerant_f1  # noqa: E402

N_BOOT = 1000


def load(path: Path, key: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["guid"]: {**r, "pred": r[key]} for r in rows}


def counts(rows, pred_key="pred"):
    return np.array([span_counts(r[pred_key], r["gold"])["_micro"] for r in rows], dtype=float)


def boot(arrs: dict[str, np.ndarray], fn, seed=0):
    """fn(f1 dict) over document resamples shared by every system."""
    n = len(next(iter(arrs.values())))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        out.append(fn({k: 100 * f1_of(v[i].sum(0)) for k, v in arrs.items()}))
    out = np.array(out)
    return {"mean": round(float(out.mean()), 2), "ci95": [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]}


def compare(arm_rows: dict, p_rows: dict) -> dict:
    guids = [g for g in arm_rows if g in p_rows]
    a = [arm_rows[g] for g in guids]
    p = [{**p_rows[g]} for g in guids]
    arrs = {"arm": counts(a), "p": counts(p), "b13": counts(a, "bros_seed13"), "b42": counts(a, "bros_seed42")}
    return {"n_docs": len(guids),
            "f1": {k: round(100 * f1_of(v.sum(0)), 2) for k, v in arrs.items()},
            "arm_minus_P": boot(arrs, lambda f: f["arm"] - f["p"]),
            "arm_minus_BROS": boot(arrs, lambda f: f["arm"] - (f["b13"] + f["b42"]) / 2),
            "per_field_f1": {f: round(100 * f1_of([sum(span_counts(r["pred"], r["gold"]).get(f, (0, 0, 0))[i] for r in a) for i in range(3)]), 1) for f in FIELDS},
            "boundary_tolerance_f1": {("any" if k is None else f"+-{k}"): round(tolerant_f1([{**r, "k": r["pred"]} for r in a], "k", k), 1) for k in (0, 1, 2, 3, None)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/vlm_sft30")
    ap.add_argument("--p-dir", default="results/vlm_sft")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d, pd = Path(a.dir), Path(a.p_dir)
    P, Pr = load(pd / "spans_E1.jsonl", "vlm_E1"), load(pd / "spans_E1r.jsonl", "vlm_E1r")
    res = {}
    for arm in ("S", "T"):
        main_rows, rep_rows = load(d / f"spans_{arm}.jsonl", f"vlm_{arm}"), load(d / f"spans_{arm}r.jsonl", f"vlm_{arm}r")
        if not main_rows:
            continue
        r = {"seed0": compare(main_rows, P)}
        if rep_rows:
            r["seed1"] = compare(rep_rows, Pr)
        m = r["seed0"]
        rep = r.get("seed1")
        replicated_P = bool(rep and rep["arm_minus_P"]["mean"] > 0 and rep["arm_minus_P"]["ci95"][0] > 0)
        replicated_B = bool(rep and rep["arm_minus_BROS"]["mean"] > 0 and rep["arm_minus_BROS"]["ci95"][0] > 0)
        dp, db = m["arm_minus_P"], m["arm_minus_BROS"]
        r["verdict"] = {
            "format_matters": bool(dp["mean"] >= 5.0 and dp["ci95"][0] > 0 and replicated_P),
            "beats_BROS": bool(db["mean"] >= 3.0 and db["ci95"][0] > 0 and replicated_B),
            "competitive_with_BROS": bool(abs(db["mean"]) <= 3.0 or db["ci95"][0] <= 0 <= db["ci95"][1]),
            "no_better_than_word_numbers": bool(dp["mean"] < 5.0 or dp["ci95"][0] <= 0),
            "replication_available": rep is not None}
        res[arm] = r
        print(f"\n=== {arm}: seed 0 F1 {m['f1']['arm']} (P {m['f1']['p']}, BROS {m['f1']['b13']} / {m['f1']['b42']}) on {m['n_docs']} docs")
        print(f"    arm - P {dp}   arm - BROS {db}")
        if rep:
            print(f"    seed 1: F1 {rep['f1']['arm']} (E1r {rep['f1']['p']});  arm - E1r {rep['arm_minus_P']}  arm - BROS {rep['arm_minus_BROS']}")
        print(f"    per field {m['per_field_f1']}\n    boundary tolerance {m['boundary_tolerance_f1']}\n    verdict {r['verdict']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    if not res:
        print("no S or T spans found yet")


if __name__ == "__main__":
    main()
