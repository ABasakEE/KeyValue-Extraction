"""Study 34, step 5 (CPU): the recipe against BROS on the 143 primary ad-buy documents, bootstrap over documents.

    python scripts/adbuy_compare.py --rows-dir results/adbuy_sg/run --bros-dir results/adbuy_bros/run [--metric lir]

BROS per-document counts come from `adbuy_bros.py score` (report.json), the recipe's from its saved answers decoded with the frozen repair.
For each training size N the BROS value is the mean over its splits (SD 0, 1, 2) of the F1 on the resampled documents; the difference
recipe minus BROS has a 95% interval over 1,000 document resamples (the same resample for every system). Registered rule: the recipe wins
at N if the interval is above 0, ties if it contains 0, BROS wins if it is below 0. Primary metric: strict row F1 (`lir`).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
import adbuy_bros as B  # noqa: E402
import adbuy_rows as R  # noqa: E402
import adbuy_sg as S  # noqa: E402

METRICS = ("lir", "cell", "vlir", "vcell")
N_BOOT = 1000


def f1_arr(c: np.ndarray) -> np.ndarray:
    """F1 from counts (..., 3) = (tp, n_pred, n_gold), summed over the documents axis by the caller."""
    tp, npred, ngold = c[..., 0], c[..., 1], c[..., 2]
    p, r = np.divide(tp, npred, out=np.zeros_like(tp, float), where=npred > 0), np.divide(tp, ngold, out=np.zeros_like(tp, float), where=ngold > 0)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)


def recipe_counts(rows_dir: Path, docs: dict) -> dict[str, dict[str, dict]]:
    """arm -> doc -> metric -> counts, from rows_A2_*.jsonl / rows_A1_*.jsonl."""
    out: dict[str, dict] = {}
    recs = {r["filename"]: r for r in R.load_records() if r["filename"] in docs}
    for arm in ("A2", "A1"):
        pages = S.read_pages(sorted(glob.glob(str(rows_dir / f"rows_{arm}_*.jsonl"))))
        if not pages:
            continue
        out[arm] = {}
        for g, d in docs.items():
            if not d["rows"]:
                continue
            doc = S.make_doc(recs[g])
            if not S.complete(doc, pages.get(g, {})):
                continue   # a document the deadline or an error left half answered is not scored
            assert len(doc.words) == len(d["words"]), f"word indexing differs for {g}"
            pred = [set(r) for p in sorted(doc.lines) for r in S.decode_page(doc, p, pages.get(g, {}).get(p, '{"rows": []}'), Counter())]
            out[arm][g] = B.run_counts(pred, d)
    return out


def compare(rec: dict, bros: dict, metric: str, seed: int = 0) -> dict:
    """rec: arm -> doc -> metric -> counts; bros: run -> doc -> metric -> counts. Returns per N and arm the difference with its interval."""
    docs = sorted(set.intersection(*[set(v) for v in bros.values()], *[set(v) for v in rec.values()]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(docs), size=(N_BOOT, len(docs)))
    arr = lambda per_doc: np.array([per_doc[g][metric] for g in docs], float)          # (docs, 3)
    boot = lambda a: f1_arr(a[idx].sum(axis=1))                                         # (N_BOOT,)
    full = lambda a: float(f1_arr(a.sum(axis=0)))
    res = {"documents": len(docs), "metric": metric, "per_size": {}}
    sizes = sorted({int(k.split("_")[0][1:]) for k in bros})
    for n in sizes:
        runs = [arr(v) for k, v in bros.items() if int(k.split("_")[0][1:]) == n]
        b_boot, b_full = np.mean([boot(a) for a in runs], axis=0), float(np.mean([full(a) for a in runs]))
        entry = {"splits": len(runs), "bros_f1": round(b_full * 100, 2)}
        for arm, per_doc in rec.items():
            a = arr(per_doc)
            d = boot(a) - b_boot
            lo, hi = np.percentile(d, [2.5, 97.5])
            entry[arm] = {"f1": round(full(a) * 100, 2), "minus_bros": round((full(a) - b_full) * 100, 2), "ci": [round(float(lo) * 100, 2), round(float(hi) * 100, 2)],
                          "verdict": "LLM wins" if lo > 0 else "BROS wins" if hi < 0 else "tie"}
        res["per_size"][str(n)] = entry
    for arm in rec:
        wins = [int(n) for n, e in res["per_size"].items() if e[arm]["verdict"] == "LLM wins"]
        loses = [int(n) for n, e in res["per_size"].items() if e[arm]["verdict"] == "BROS wins"]
        res.setdefault("crossover", {})[arm] = {"largest_N_llm_wins": max(wins, default=None), "smallest_N_bros_wins": min(loses, default=None)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--bros-dir", required=True)
    ap.add_argument("--metric", choices=METRICS, default="lir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    primary = json.loads(S.OUT.joinpath("primary.json").read_text(encoding="utf-8"))
    docs = B.load_docs(set(primary))
    rep = json.loads((Path(a.bros_dir) / "report.json").read_text(encoding="utf-8"))
    bros = {k: v["per_doc"] for k, v in rep["runs"].items()}
    rec = recipe_counts(Path(a.rows_dir), docs)
    out = {m: compare(rec, bros, m) for m in METRICS}
    for m in METRICS:
        print(f"== {m} ({out[m]['documents']} documents)")
        for n, e in out[m]["per_size"].items():
            print(f"N={n:>4s} BROS {e['bros_f1']:6.2f} | " + " | ".join(f"{arm} {e[arm]['f1']:6.2f} ({e[arm]['minus_bros']:+6.2f} {e[arm]['ci']} {e[arm]['verdict']})" for arm in rec))
        print("   crossover:", out[m]["crossover"])
    Path(a.out or Path(a.rows_dir) / "comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
