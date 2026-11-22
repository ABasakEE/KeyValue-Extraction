"""Study 44, follow-up (CPU): where in the table do the unexplained misses ('missed, other') fall? Miss rate of gold spans by row position, row height and column.

    python scripts/docile_miss_position.py [--workers 5] [--out results/docile_taxonomy/miss_position.json]

Same saved predictions as `docile_failure_taxonomy.py` (unseen-layout test documents of the five runs). A gold span is *touched* if any predicted span shares a word with it;
the rate below is untouched / all gold spans in the group, pooled over the runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src"), str(HERE.parent)]
TAGS = ["N1000_S0", "N1000_S1", "N1000_S2", "Nfull_T13", "Nfull_T21"]


def job(tag: str) -> dict:
    import docile_bros as B
    import docile_typer as T
    from row_group import runs

    pred = T.load_pred(tag)
    test = json.loads(B.SPLITS.read_text(encoding="utf-8"))["test"]
    c: dict[str, Counter] = defaultdict(Counter)
    for g in test:
        d = B.doc_dict(g)
        if not d["rows"] or g not in pred:
            continue
        pw = {i for _, s, e, *_ in pred[g] for i in range(s, e)}
        heights = sorted(b[3] - b[1] for b in d["fboxes"])
        med_h = heights[len(heights) // 2] or 0.01
        rows = sorted(d["rows"], key=lambda r: (min(d["pages"][next(iter(ws))] for _, ws, _ in r), min(d["fboxes"][i][1] for _, ws, _ in r for i in ws)))
        n = len(rows)
        for k, row in enumerate(rows):
            ys = [d["fboxes"][i][1] for _, ws, _ in row for i in ws]
            ye = [d["fboxes"][i][3] for _, ws, _ in row for i in ws]
            tall = (max(ye) - min(ys)) > 2.2 * med_h
            pos = "only row" if n == 1 else "first row" if k == 0 else "last row" if k == n - 1 else "middle rows"
            for f, ws, _ in row:
                for a, b in runs(ws):
                    miss = not any(i in pw for i in range(a, b))
                    for key in (("position", pos), ("row_height", "multi-line row" if tall else "single-line row"), ("table_size", "1-2 rows" if n <= 2 else "3-6 rows" if n <= 6 else "7+ rows"),
                                ("cells_in_row", "1-2 cells" if len(row) <= 2 else "3-4 cells" if len(row) <= 4 else "5+ cells"), ("field", f)):
                        c[key[0] + "|" + key[1]]["n"] += 1
                        c[key[0] + "|" + key[1]]["missed"] += miss
    return {k: dict(v) for k, v in c.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=str(HERE.parent / "results" / "docile_taxonomy" / "miss_position.json"))
    a = ap.parse_args()
    with Pool(min(a.workers, len(TAGS))) as pool:
        res = pool.map(job, TAGS)
    tot: dict[str, Counter] = defaultdict(Counter)
    for r in res:
        for k, v in r.items():
            tot[k].update(v)
    table = {k: {"gold_spans_all_runs": v["n"], "untouched_pct": round(100 * v["missed"] / v["n"], 1)} for k, v in sorted(tot.items()) if v["n"] >= 100}
    Path(a.out).write_text(json.dumps(table, indent=1), encoding="utf-8")
    for k, v in table.items():
        print(f"{k:40s} n={v['gold_spans_all_runs']:6d}  untouched {v['untouched_pct']:5.1f}%")


if __name__ == "__main__":
    main()
