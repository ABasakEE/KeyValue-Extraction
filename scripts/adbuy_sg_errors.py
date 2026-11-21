"""Study 34: where do the recipe's line-item errors on ad-buy come from? (CPU; the saved A2 answers of results/adbuy_sg/run)

    python scripts/adbuy_sg_errors.py [--rows-dir results/adbuy_sg/run] [--arm A2]

Per field: strict cell precision and recall. Every predicted cell that is not an exact gold cell is put in one class: wrong field (its words are a gold cell of another field),
boundary (it overlaps a gold cell of the same field without being equal), repeat (it equals a gold cell that another predicted cell already matched), or extra (touches no gold cell).
Also: predicted rows per page against gold rows, pages that hit the output cap, and the grounding failures of the decode.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
import adbuy_rows as R  # noqa: E402
import adbuy_sg as S  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", default=str(S.OUT / "run"))
    ap.add_argument("--arm", default="A2")
    a = ap.parse_args()
    primary = json.loads((S.OUT / "primary.json").read_text(encoding="utf-8"))
    recs = {r["filename"]: r for r in R.load_records() if r["filename"] in set(primary)}
    pages = S.read_pages(sorted(glob.glob(str(Path(a.rows_dir) / f"rows_{a.arm}_*.jsonl"))))
    cap_pages = {(r["doc_id"], r["page"]) for p in glob.glob(str(Path(a.rows_dir) / f"rows_{a.arm}_*.jsonl")) for r in map(json.loads, open(p, encoding="utf-8")) if (r.get("new_tokens") or 0) >= 2400}
    tp, npred, ngold = Counter(), Counter(), Counter()
    cls, st = Counter(), Counter()
    rows_pred = rows_gold = pages_n = 0
    ratio_pages = []
    cap_stats = Counter()
    for fn in primary:
        doc = S.make_doc(recs[fn])
        if not S.complete(doc, pages.get(fn, {})):
            continue
        gold = R.gold_rows(recs[fn], doc)
        gold_cells = {(f, ws) for row in gold for f, ws, _, _ in row}
        by_set = defaultdict(set)
        for f, ws in gold_cells:
            by_set[ws].add(f)
        matched = set()
        for p in sorted(doc.lines):
            rows = S.decode_page(doc, p, pages[fn].get(p, '{"rows": []}'), st)
            g_rows = [r for r in gold if r and r[0][2] and r[0][2][0][2] == p]
            rows_pred += len(rows); rows_gold += len(g_rows); pages_n += 1
            if g_rows or rows:
                ratio_pages.append((len(rows), len(g_rows), (fn, p) in cap_pages))
            for row in rows:
                for f, ws in row:
                    npred[f] += 1
                    if (f, ws) in gold_cells:
                        if (f, ws) in matched:
                            cls["repeat (equals a gold cell already matched)"] += 1
                        else:
                            matched.add((f, ws)); tp[f] += 1
                    elif ws in by_set:
                        cls["wrong field (words are a gold cell of another field)"] += 1
                    elif any(g_f == f and g_ws & ws for g_f, g_ws in gold_cells):
                        cls["boundary (overlaps a gold cell of the same field)"] += 1
                    elif any(g_ws & ws for _, g_ws in gold_cells):
                        cls["overlaps a gold cell of another field only"] += 1
                    else:
                        cls["extra (touches no gold cell)"] += 1
        for f, ws in gold_cells:
            ngold[f] += 1
    print(f"{a.arm}: pages scored {pages_n}; predicted rows {rows_pred} against {rows_gold} gold rows ({rows_pred / max(1, rows_gold):.2f}x)")
    print("per field (strict cell): " + " | ".join(f"{f} P {100 * tp[f] / max(1, npred[f]):.0f} R {100 * tp[f] / max(1, ngold[f]):.0f} ({npred[f]} predicted, {ngold[f]} gold)" for f in R.LI_FIELDS))
    n_wrong = sum(cls.values())
    print(f"{n_wrong} predicted cells are not exact gold cells (of {sum(npred.values())}):")
    for k, v in cls.most_common():
        print(f"   {v:6d} ({100 * v / n_wrong:4.1f}%)  {k}")
    capped = [(p, g) for p, g, c in ratio_pages if c]
    normal = [(p, g) for p, g, c in ratio_pages if not c]
    print(f"pages that hit the 2,400-token cap: {len(capped)} (predicted rows {sum(p for p, _ in capped)} vs gold {sum(g for _, g in capped)}); other pages with rows: predicted {sum(p for p, _ in normal)} vs gold {sum(g for _, g in normal)}")
    print("decode:", dict(st))


if __name__ == "__main__":
    main()
