"""Preserve some outputs of the zero-shot reader (Qwen3-VL-8B, 4-bit, prompt v2) for the demo: the 8B model cannot run on a laptop without a GPU, so the demo replays them.

    python scripts/demo_reader_outputs.py        # writes results/demo_weights/reader_vrdu.json

Source: the fresh-forms run (300 VRDU registration forms nobody tuned on): the model's raw answers per page and field, the grounded spans (latest-page tie-break),
BROS Track-1 seed 13's saved spans and the gold. Twelve forms are kept: where the reader beats BROS, where BROS beats the reader, where both are right, where both fail.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(ROOT)]
import schema_guided_kv as sg  # noqa: E402
import study31_cpu as c  # noqa: E402
import study32_fresh as S  # noqa: E402
from matched_eval import f1_of, span_counts  # noqa: E402

FRESH = ROOT / "results" / "fresh_set3"


def f1(pred, gold) -> float:
    return 100 * f1_of(span_counts(pred, gold)["_micro"])


def main() -> None:
    docs = {d["doc_id"].split("/", 1)[1]: d for d in json.loads((FRESH / "manifest.json").read_text(encoding="utf-8"))}
    recs = sg.load_records(ROOT / "data" / "vrdu", set(docs))
    base = {r["guid"]: r for r in map(json.loads, (FRESH / "bros_spans.jsonl").read_text(encoding="utf-8").splitlines())}
    paths = [str(FRESH / "run" / f"rows_V2A2_fresh_{s}.jsonl") for s in (0, 1)]
    late = {f: "latest" for f in c.LATE}
    reader = S.decode_arm(paths, docs, recs, late)
    by_doc = sg.read_rows(paths)
    scored = []
    for g in sorted(docs):
        b = base[g]
        scored.append({"guid": g, "template": b["template"], "reader_f1": f1(reader[g], b["gold"]), "bros_f1": f1(b["bros_seed13"], b["gold"])})
    n = len(scored)
    stats = {"n_docs": n, "reader_mean_f1": round(sum(s["reader_f1"] for s in scored) / n, 1), "bros_seed13_mean_f1": round(sum(s["bros_f1"] for s in scored) / n, 1)}

    def pick(key, k, reverse=True, cond=lambda s: True):
        return [s for s in sorted((s for s in scored if cond(s)), key=key, reverse=reverse)][:k]

    chosen = (
        [dict(s, why="reader beats BROS") for s in pick(lambda s: s["reader_f1"] - s["bros_f1"], 3)]
        + [dict(s, why="BROS beats the reader") for s in pick(lambda s: s["bros_f1"] - s["reader_f1"], 2)]
        + [dict(s, why="both right") for s in pick(lambda s: s["reader_f1"] + s["bros_f1"], 3, cond=lambda s: s["reader_f1"] >= 90 and s["bros_f1"] >= 90)]
        + [dict(s, why="both fail") for s in pick(lambda s: -(s["reader_f1"] + s["bros_f1"]), 2)]
    )
    seen, out = set(), []
    for s in chosen:
        if s["guid"] in seen:
            continue
        seen.add(s["guid"])
        g = s["guid"]
        raw = [{"page": r["page"] + 1, "field": r["field"], "answer": r["samples"][0]} for r in sorted(by_doc[docs[g]["doc_id"]], key=lambda r: (r["page"], r["field"]))]
        out.append({**s, "gold": base[g]["gold"], "bros": base[g]["bros_seed13"], "reader": reader[g], "raw_answers": raw})
    path = ROOT / "results" / "demo_weights" / "reader_vrdu.json"
    path.write_text(json.dumps({"stats": stats, "model": "Qwen3-VL-8B-Instruct, 4-bit nf4, prompt v2, one call per page and field, greedy; fresh-forms run (Kaggle T4 x 2)",
                                "examples": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(stats)
    for s in out:
        print(f"{s['why']:22s} reader {s['reader_f1']:5.1f} BROS {s['bros_f1']:5.1f} {s['guid'][:50]}")


if __name__ == "__main__":
    main()
