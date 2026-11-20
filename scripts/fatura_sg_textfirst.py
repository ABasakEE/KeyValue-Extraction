"""Study 33, post hoc: re-score the saved FATURA rows with a text-first decode (not the registered decode).

Rule: when the cited id's block does not contain the quoted text but exactly one block does, use that block.
Otherwise the registered decode is unchanged. Writes results/fatura_sg/run/report_textfirst.json and prints
how many answers changed and whether each change turned a wrong answer right or the reverse.

    python scripts/fatura_sg_textfirst.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
import fatura_sg as F  # noqa: E402
import schema_guided_kv as sg  # noqa: E402
from zero_shot_kv import norm  # noqa: E402

RUN = F.OUT / "run"
registered = F.decode
changes: Counter = Counter()


def text_first(doc, rows, fmt, fallback=True):
    """Same as the registered decode, with each answer's id replaced by the one block that holds its text."""
    fixed = []
    for r in rows:
        p = sg.parse_json(r["samples"][0], F.FIELDS) if fmt == "json" else {r["field"]: sg.parse_field(r["samples"][0])}
        if p is None:
            fixed.append(r)
            continue
        obj = {}
        for f, segs in p.items():
            if not segs:
                continue
            text, lid = segs[0]
            q = norm(text)
            cand = [c.id for c in doc.lines[0] if q and q in norm(c.text)] if lid in doc.by_id else []
            if lid in doc.by_id and cand and lid not in cand and len(cand) == 1:
                changes[f] += 1
                lid = cand[0]
            obj[f] = f"{text} {lid}" if lid else text
        fixed.append({**r, "samples": [json.dumps(obj) if fmt == "json" else "ANSWER: " + next(iter(obj.values()), "NONE")]})
    return registered(doc, fixed, fmt, fallback)


if __name__ == "__main__":
    F.decode = text_first
    with tempfile.TemporaryDirectory() as t:
        for f in RUN.glob("rows_A*.jsonl"):
            shutil.copy(f, t)
        shutil.copy(RUN / "recipe.json", t)
        F.cmd_score(type("A", (), {"rows_dir": t})())
        shutil.copy(Path(t) / "report.json", RUN / "report_textfirst.json")
    print("answers whose id was replaced, by field:", dict(changes))
