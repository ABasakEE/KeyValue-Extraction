"""Study 34, CPU check: can a wrong cited id be repaired on tables? (no GPU, no new model output)

  python scripts/adbuy_repair.py slips      # how the real model's id errors on FATURA (study 33 rows) are distributed, and which rule fixes them
  python scripts/adbuy_repair.py simulate   # ad-buy oracle answers with wrong ids put in by hand (four slip types), four repair rules
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
import adbuy_rows as R  # noqa: E402
import adbuy_sg as S  # noqa: E402
import fatura_sg as F  # noqa: E402
import schema_guided_kv as sg  # noqa: E402
from lir_metric import cell_counts, f1, lir_counts  # noqa: E402
from zero_shot_kv import norm  # noqa: E402

N_DOCS = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def cmd_slips() -> None:
    """FATURA, real answers: for every cell whose cited block does not hold its text but another block does, which rule finds the right block?"""
    recs = {r["guid"]: r for r in map(json.loads, (F.V1 / "spans.jsonl").read_text(encoding="utf-8").splitlines()) if r["split"] == "confirm"}
    docs = F.load_fdocs(set(recs))
    for arm in ("A2", "A1"):
        by = sg.read_rows([str(F.OUT / "run" / f"rows_{arm}_{s}.jsonl") for s in (0, 1)])
        c: Counter = Counter()
        for g, doc in docs.items():
            lines = doc.lines[0]
            gold = {f: (s, e) for f, s, e, _ in doc.gold}
            for r in by[g]:
                p = sg.parse_json(r["samples"][0], F.FIELDS)
                for f, segs in (p or {}).items():
                    if not segs or segs[0][1] not in doc.by_id:
                        continue
                    text, lid = segs[0]
                    q = norm(text)
                    cited = doc.by_id[lid]
                    cand = [ln for ln in lines if q and q in norm(ln.text)]
                    if not cand or cited in cand or f not in gold:
                        continue
                    c["slips"] += 1
                    truth = [ln for ln in cand if (ln.words[0], ln.words[-1] + 1) == gold[f]]
                    c["truth_among_candidates"] += bool(truth)
                    c["unique_candidate"] += len(cand) == 1
                    c["unique_and_right"] += len(cand) == 1 and bool(truth)
                    xy = S._xy(lid)
                    if xy and truth:
                        key = lambda ln: sum((a - b) ** 2 for a, b in zip(S._xy(ln.id) or (0, 0), xy))
                        best = min(cand, key=key)
                        c["nearest_right"] += best is truth[0]
                        c["truth_is_nearest_candidate_unique"] += best is truth[0] and sorted(map(key, cand))[:2].count(key(best)) == 1
                    c["cited_block_is_gold_of_another_field"] += any(gold[o] == (cited.words[0], cited.words[-1] + 1) for o in gold if o != f)
        print(arm, dict(c))


RULES = ("none", "unique", "nearest", "row", "row+")
QS = (0.0, 0.1, 0.3)
SLIPS = ("next", "random", "row_neighbor", "same_row")


def corrupted_answer(doc, rec, page, slip: str, q: float, rng: random.Random) -> str:
    ids = [ln.id for ln in doc.lines.get(page, [])]
    rows = [row for row in R.gold_rows(rec, doc) if row and row[0][2] and row[0][2][0][2] == page]
    out = []
    for k, row in enumerate(rows):
        obj = {}
        for j, (f, _, segs, _) in enumerate(row):
            cite = [(t, i) for t, i, _ in segs]
            if rng.random() < q and cite:
                if slip == "next":
                    new = ids[(ids.index(cite[0][1]) + 1) % len(ids)]
                elif slip == "row_neighbor":
                    other = rows[k + 1 if k + 1 < len(rows) else k - 1] if len(rows) > 1 else None
                    same = [c for c in (other or []) if c[0] == f and c[2]]
                    new = same[0][2][0][1] if same else rng.choice(ids)
                elif slip == "same_row":
                    others = [c for i2, c in enumerate(row) if i2 != j and c[2]]
                    new = others[0][2][0][1] if others else rng.choice(ids)
                else:
                    new = rng.choice(ids)
                cite = [(t, new) for t, _ in cite]
            obj[f] = "\n".join(f"{t} {i}" for t, i in cite) or None
        out.append(obj)
    return json.dumps({"rows": out})


def cmd_simulate() -> None:
    recs = R.load_records(N_DOCS)
    res: dict = {}
    for rec in recs:
        doc = S.make_doc(rec)
        gold = [{(f, ws) for f, ws, _, _ in row} for row in R.gold_rows(rec, doc)]
        if not gold:
            continue
        for slip in SLIPS:
            for q in QS:
                rng = random.Random(7)
                raws = {p: corrupted_answer(doc, rec, p, slip, q, rng) for p in doc.lines}
                for rule in RULES:
                    pred = [set(r) for p in doc.lines for r in S.decode_page(doc, p, raws[p], Counter(), rule)]
                    a = res.setdefault((slip, q, rule), {"cell": [0, 0, 0], "lir": [0, 0, 0]})
                    for k, c in (("cell", cell_counts(pred, gold)), ("lir", lir_counts(pred, gold))):
                        a[k] = [x + y for x, y in zip(a[k], c)]
    print(f"{len(recs)} documents; cell F1 / row-grouped F1 (%), wrong ids on a share q of the cells")
    print(f"{'slip':14s}{'q':>5s}" + "".join(f"{r:>14s}" for r in RULES))
    for slip in SLIPS:
        for q in QS:
            print(f"{slip:14s}{q:5.1f}" + "".join(f"{f1(res[(slip, q, r)]['cell']) * 100:8.1f}/{f1(res[(slip, q, r)]['lir']) * 100:5.1f}" for r in RULES))
    Path(HERE.parent / "results" / "adbuy_sg" / "repair_simulation.json").write_text(
        json.dumps({f"{s}|{q}|{r}": {k: round(f1(v) * 100, 2) for k, v in d.items()} for (s, q, r), d in res.items()}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"slips": cmd_slips, "simulate": cmd_simulate}[sys.argv[1]]()
