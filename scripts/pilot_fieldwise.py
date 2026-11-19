"""Study 31 pilot check: per-field official-style text-match F1 of a set of pilot rows, decoded with the post hoc fallback.

  python scripts/pilot_fieldwise.py --manifest M --rows rows_*.jsonl --format json|field [--out x.json]

Used to compare the study-25 prompt (rows already in the repo) with the study-31 prompt on the same 30 pilot documents.
Only a guard against a broken prompt (30 documents are far too few for a significance test).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import schema_guided_kv as sg  # noqa: E402
import study25_stage0 as st  # noqa: E402
from matched_eval import f1_of  # noqa: E402
from zero_shot_kv_v2 import schema_fields  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--format", choices=["json", "field"], required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fields = schema_fields(sg.SUBCORPUS)
    docs = sg.vrdu_manifest_docs(a.manifest)
    by_doc = sg.read_rows(a.rows)
    keys = {d["doc_id"].split("/", 1)[1] for d in docs}
    recs = sg.load_records(sg.ROOT / "data" / "vrdu", keys)
    gold = st.load_gold_texts(keys)
    tot = {f: [0, 0, 0] for f in list(fields) + ["_micro"]}
    for d in docs:
        rows = by_doc.get(d["doc_id"], [])
        if not rows:
            continue
        g = d["doc_id"].split("/", 1)[1]
        spans, _ = sg.decode_doc(sg.Doc(recs[g]), rows, a.format, fields, True)
        for f, c in st.text_counts(spans, gold[g]).items():
            for i in range(3):
                tot.setdefault(f, [0, 0, 0])[i] += c[i]
    res = {f: round(f1_of(c) * 100, 1) for f, c in tot.items()}
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
