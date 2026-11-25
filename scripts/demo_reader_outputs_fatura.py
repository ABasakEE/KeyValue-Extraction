"""Preserve some outputs of the zero-shot reader on FATURA invoices (Qwen3-VL-8B, 4-bit, prompt v2) next to BROS trained on K = 2 and K = 40 templates.

    python scripts/demo_reader_outputs_fatura.py     # writes results/demo_weights/reader_fatura.json

Source: the 500 FATURA confirm invoices of the K sweep (10 per template, 50 templates). For each invoice the reader's raw JSON answer, its grounded spans (registered decode), the saved BROS predictions
(seed 13) of the fold that holds the invoice's template out, K = 40 and K = 2, and the gold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(ROOT)]
import fatura_sg as fs  # noqa: E402
import schema_guided_kv as sg  # noqa: E402
from matched_eval import f1_of, span_counts  # noqa: E402


def f1(pred, gold) -> float:
    return 100 * f1_of(span_counts(pred, gold)["_micro"])


def main() -> None:
    rd = ROOT / "results" / "fatura_sg" / "run"
    recs = {r["guid"]: r for r in map(json.loads, (fs.V1 / "spans.jsonl").read_text(encoding="utf-8").splitlines()) if r["split"] == "confirm"}
    docs = fs.load_fdocs(set(recs))
    fmt = json.loads((rd / "recipe.json").read_text(encoding="utf-8"))["format"]
    by_doc = sg.read_rows([str(rd / f"rows_A2_{s}.jsonl") for s in (0, 1)])
    dumps = {}
    for k in (2, 40):
        for fold in range(5):
            p = fs.bros_paths(k, fold, 13)
            dumps[(k, fold)] = {r["guid"]: r["pred"] for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
    scored = []
    for g in sorted(recs):
        if g not in by_doc:
            continue
        r = recs[g]
        reader = fs.decode(docs[g], by_doc[g], fmt)[0]
        k2, k40 = dumps[(2, r["fold"])].get(g), dumps[(40, r["fold"])].get(g)
        if k2 is None or k40 is None:
            continue
        scored.append({"guid": g, "template": r["template"], "fold": r["fold"], "reader_f1": f1(reader, r["gold"]), "k2_f1": f1(k2, r["gold"]), "k40_f1": f1(k40, r["gold"]),
                       "gold": r["gold"], "reader": reader, "k2": k2, "k40": k40,
                       "raw": [x["samples"][0] for x in by_doc[g]]})
    n = len(scored)
    mean = lambda key: round(sum(s[key] for s in scored) / n, 1)
    stats = {"n_invoices": n, "reader_mean_f1": mean("reader_f1"), "k2_mean_f1": mean("k2_f1"), "k40_mean_f1": mean("k40_f1")}

    def pick(key, k, cond=lambda s: True, reverse=True):
        return sorted((s for s in scored if cond(s)), key=key, reverse=reverse)[:k]

    chosen = (
        [dict(s, why="reader beats the K = 2 tagger (K = 40 right too)") for s in pick(lambda s: s["reader_f1"] - s["k2_f1"], 3, lambda s: s["k40_f1"] >= 95)]
        + [dict(s, why="K = 40 tagger beats the reader") for s in pick(lambda s: s["k40_f1"] - s["reader_f1"], 2)]
        + [dict(s, why="all three right") for s in pick(lambda s: s["reader_f1"] + s["k2_f1"], 2, lambda s: min(s["reader_f1"], s["k2_f1"], s["k40_f1"]) >= 95)]
        + [dict(s, why="reader and K = 2 tagger both weak (K = 40 right)") for s in pick(lambda s: -(s["reader_f1"] + s["k2_f1"]), 1, lambda s: s["k40_f1"] >= 95)]
    )
    out, seen = [], set()
    for s in chosen:
        if s["guid"] not in seen:
            seen.add(s["guid"])
            out.append(s)
    path = ROOT / "results" / "demo_weights" / "reader_fatura.json"
    path.write_text(json.dumps({"stats": stats, "model": "Qwen3-VL-8B-Instruct, 4-bit nf4, prompt v2, one JSON answer per invoice (the format gate kept JSON), greedy; Kaggle T4 x 2",
                                "examples": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(stats)
    for s in out:
        print(f"{s['why'][:46]:46s} reader {s['reader_f1']:5.1f} K40 {s['k40_f1']:5.1f} K2 {s['k2_f1']:5.1f} {s['guid']}")


if __name__ == "__main__":
    main()
