"""Study 34, DocILE: the recipe against the BROS baseline on the 200 test documents, bootstrap over documents (each test document is a different layout).

    python scripts/docile_compare.py --recipe results/docile_sg/run/report.json --bros results/docile_bros/run/report.json [--arm A2] [--metric lir]

Reads only the two reports (per-document counts, no document text). For each training size N the BROS value is the mean over its split seeds of the F1 on the resampled documents;
the difference recipe minus BROS has a 95% interval over 1,000 document resamples (the same resample for every system). Rule (as study 34): the recipe wins at N if the interval is above 0,
ties if it contains 0, BROS wins if it is below 0. Only documents that both sides scored are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_BOOT = 1000


def f1_of(c: np.ndarray) -> np.ndarray:
    """F1 from counts (..., 3) = (tp, n_pred, n_gold)."""
    tp, npred, ngold = c[..., 0], c[..., 1], c[..., 2]
    p = np.divide(tp, npred, out=np.zeros_like(tp, float), where=npred > 0)
    r = np.divide(tp, ngold, out=np.zeros_like(tp, float), where=ngold > 0)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)


def compare(recipe: dict, bros: dict, metric: str = "lir", seed: int = 0) -> dict:
    docs = sorted(set(recipe) & set.intersection(*[set(v["per_doc"]) for v in bros.values()]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(docs), size=(N_BOOT, len(docs)))
    arr = lambda per_doc: np.array([per_doc[g][metric] for g in docs], float)
    boot = lambda a: f1_of(a[idx].sum(axis=1))
    full = lambda a: float(f1_of(a.sum(axis=0)))
    rec = arr(recipe)
    res = {"documents": len(docs), "metric": metric, "recipe_f1": round(full(rec) * 100, 2), "per_size": {}}
    for n in sorted({int(k.split("_")[0][1:]) for k in bros}):
        runs = [arr(v["per_doc"]) for k, v in bros.items() if int(k.split("_")[0][1:]) == n]
        b_boot, b_full = np.mean([boot(a) for a in runs], axis=0), float(np.mean([full(a) for a in runs]))
        d = boot(rec) - b_boot
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        res["per_size"][str(n)] = {"splits": len(runs), "bros_f1": round(b_full * 100, 2), "minus_bros": round((full(rec) - b_full) * 100, 2),
                                   "ci": [round(lo * 100, 2), round(hi * 100, 2)], "verdict": "LLM wins" if lo > 0 else "BROS wins" if hi < 0 else "tie"}
    wins = [int(n) for n, e in res["per_size"].items() if e["verdict"] == "LLM wins"]
    loses = [int(n) for n, e in res["per_size"].items() if e["verdict"] == "BROS wins"]
    res["crossover"] = {"largest_N_llm_wins": max(wins, default=None), "smallest_N_bros_wins": min(loses, default=None)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--bros", required=True)
    ap.add_argument("--arm", default="A2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recipe = json.loads(Path(a.recipe).read_text(encoding="utf-8"))[a.arm]["per_doc"]
    bros = json.loads(Path(a.bros).read_text(encoding="utf-8"))["runs"]
    out = {}
    for m in ("lir", "cell"):
        out[m] = compare(recipe, bros, m)
        print(f"== {m} ({out[m]['documents']} documents); recipe {out[m]['recipe_f1']:.2f}")
        for n, e in out[m]["per_size"].items():
            print(f"N={n:>5s} BROS {e['bros_f1']:6.2f} | recipe minus BROS {e['minus_bros']:+7.2f} {e['ci']} {e['verdict']}")
        print("   crossover:", out[m]["crossover"])
    Path(a.out or Path(a.bros).parent / "comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
