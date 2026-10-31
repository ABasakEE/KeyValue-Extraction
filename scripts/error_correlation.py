"""Are BROS's and the VLM's errors correlated? (CPU, post-hoc on the study-16 dumps.)

Gold-span level (recall side): for every gold span, did BROS predict it exactly, did the VLM?  The lift
P(VLM right | BROS missed) / P(VLM right | BROS right) is ~1 for independent errors and < 1 when the two fail together.
Prediction level (what a gate sees): for every span BROS predicted, was it right, was the VLM's span for that field right,
and what was BROS's confidence (mean field probability over the span)?  Split at TAU.
Intervals: cluster bootstrap (FATURA: templates; VRDU: documents), seeds pooled inside each cluster.

    python scripts/error_correlation.py fatura|vrdu|selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from routing_eval import attach_runs, base_records, detect_seeds, read_jsonl  # noqa: E402

TAU = 0.7
N_BOOT = 500
# gold level: [n_gold, both, bros_only, vlm_only, neither]
# pred level: [n_pred, wrong, wrong&vlm_right, right, right&vlm_right, low, low&wrong, low&wrong&vlm_right]
N_G, N_P = 5, 8


def doc_counts(rec: dict, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    gold = {(s[0], s[1], s[2]) for s in rec["gold"]}
    vlm_keys = {(s[0], s[1], s[2]) for s in rec["vlm"]}
    vlm_right = {f for f, a, b in vlm_keys if (f, a, b) in gold}
    g, p = np.zeros(N_G), np.zeros(N_P)
    for sd in seeds:
        bros = [tuple(s[:3]) for s in rec["bros"][sd]]
        bset = set(bros)
        for k in gold:
            b, v = k in bset, k in vlm_keys
            g[0] += 1
            g[1 if b and v else 2 if b else 3 if v else 4] += 1
        for f, a, e in bros:
            arr = rec["probs"][sd].get(f)
            conf = float(arr[a:e].mean()) if arr is not None and e > a else 0.0
            ok, vr = (f, a, e) in gold, f in vlm_right
            p[0] += 1
            if ok:
                p[3] += 1; p[4] += vr
            else:
                p[1] += 1; p[2] += vr
            if conf < TAU:
                p[5] += 1
                if not ok:
                    p[6] += 1; p[7] += vr
    return g, p


def ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def summarise(g: np.ndarray, p: np.ndarray) -> dict:
    both, b_only, v_only, neither = g[1:5]
    p_v_miss = ratio(v_only, v_only + neither)         # P(VLM right | BROS missed)
    p_v_hit = ratio(both, both + b_only)               # P(VLM right | BROS right)
    return {
        "gold_spans": int(g[0]),
        "bros_recall": round(ratio(both + b_only, g[0]), 4), "vlm_recall": round(ratio(both + v_only, g[0]), 4),
        "both_right": round(ratio(both, g[0]), 4), "bros_only": round(ratio(b_only, g[0]), 4),
        "vlm_only": round(ratio(v_only, g[0]), 4), "both_wrong": round(ratio(neither, g[0]), 4),
        "p_vlm_right_given_bros_missed": round(p_v_miss, 4), "p_vlm_right_given_bros_right": round(p_v_hit, 4),
        "lift": round(ratio(p_v_miss, p_v_hit), 4),
        "bros_preds": int(p[0]), "bros_precision": round(ratio(p[3], p[0]), 4),
        "p_vlm_right_given_bros_pred_wrong": round(ratio(p[2], p[1]), 4),
        "p_vlm_right_given_bros_pred_right": round(ratio(p[4], p[3]), 4),
        f"low_conf_lt_{TAU}": {"share_of_preds": round(ratio(p[5], p[0]), 4), "share_wrong": round(ratio(p[6], p[5]), 4),
                               "p_vlm_right_given_wrong": round(ratio(p[7], p[6]), 4)},
    }


def analyse(recs: list[dict], seeds: list[int], cluster_of) -> dict:
    per = [(cluster_of(r), *doc_counts(r, seeds)) for r in recs]
    clusters = sorted({c for c, _, _ in per})
    by_c = {c: (sum(g for k, g, _ in per if k == c), sum(p for k, _, p in per if k == c)) for c in clusters}
    G, P = sum(v[0] for v in by_c.values()), sum(v[1] for v in by_c.values())
    out = summarise(G, P)
    rng = np.random.default_rng(0)
    keys = ["lift", "p_vlm_right_given_bros_missed", "p_vlm_right_given_bros_right", "p_vlm_right_given_bros_pred_wrong"]
    draws = {k: [] for k in keys}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(clusters), len(clusters))
        s = summarise(sum(by_c[clusters[i]][0] for i in pick), sum(by_c[clusters[i]][1] for i in pick))
        for k in keys:
            draws[k].append(s[k])
    out["ci95"] = {k: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)] for k, v in draws.items()}
    out["clusters"], out["seeds"] = len(clusters), seeds
    return out


def run_fatura(spans: str, routing_dir: str, ks: list[str]) -> dict:
    base = base_records(read_jsonl(Path(spans)))
    res = {}
    for k in ks:
        d = lambda f, s, k=k: Path(routing_dir) / f"K{k}" / f"fold{f}" / f"seed_{s}"
        seeds = detect_seeds(d, sorted({r["fold"] for r in base}))
        recs, info = attach_runs(base, seeds, d)
        res[f"K{k}"] = {**analyse(recs, seeds, lambda r: r["template"]), "alignment": info}
    return res


def run_vrdu(dev: str, confirm: str, bros_dir: str) -> dict:
    base = base_records(read_jsonl(Path(dev)), "dev") + base_records(read_jsonl(Path(confirm)), "confirm")
    d = lambda f, s: Path(bros_dir) / f"UTL_fold{f}_seed{s}"
    seeds = detect_seeds(d, sorted({r["fold"] for r in base}), candidates=(13, 42))
    recs, info = attach_runs(base, seeds, d)
    res = {"all": {**analyse(recs, seeds, lambda r: r["guid"]), "alignment": info}}
    res["confirm"] = analyse([r for r in recs if r["split"] == "confirm"], seeds, lambda r: r["guid"])
    return res


def _selftest() -> None:
    rec = {"gold": [["a", 0, 1, "x"], ["b", 2, 3, "y"], ["c", 4, 5, "z"], ["d", 6, 7, "w"]],
           "vlm": [["a", 0, 1, "x"], ["c", 4, 5, "z"]],
           "bros": {1: [["a", 0, 1, "x"], ["b", 2, 3, "y"], ["d", 6, 8, "wrong"]]},
           "probs": {1: {"a": np.array([.9, 0, 0, 0, 0, 0, 0, 0]), "b": np.array([0, 0, .8, 0, 0, 0, 0, 0]),
                         "d": np.array([0, 0, 0, 0, 0, 0, .2, .2])}}}
    g, p = doc_counts(rec, [1])
    assert list(g) == [4, 1, 1, 1, 1], g                 # a both, b bros-only, c vlm-only, d neither
    assert list(p) == [3, 1, 0, 2, 1, 1, 1, 0], p        # d wrong (vlm silent on d), low conf; a right & vlm right
    s = summarise(g, p)
    assert s["p_vlm_right_given_bros_missed"] == 0.5 and s["p_vlm_right_given_bros_right"] == 0.5 and s["lift"] == 1.0
    print("selftest passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fatura", "vrdu", "selftest"])
    ap.add_argument("--routing-dir", default="results/routing_v3/fatura")
    ap.add_argument("--bros-dir", default="results/routing_v3/vrdu_bros")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return _selftest()
    res = (run_fatura("results/fatura_v1/spans.jsonl", a.routing_dir, ["2", "5", "10"]) if a.cmd == "fatura"
           else run_vrdu("results/matched_eval/spans.jsonl", "results/confirm_v2/matched/spans.jsonl", a.bros_dir))
    out = Path(a.out or f"results/error_corr/{a.cmd}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
