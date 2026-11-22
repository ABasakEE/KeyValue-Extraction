"""Study 43 post hoc (CPU): what does the VLM typer get wrong? Confusion with the gold field and agreement with BROS on the exactly-found monetary candidates.

    python scripts/docile_typer_diagnose.py --rows results/docile_typer/run/rows_*_0.jsonl results/docile_typer/run/rows_*_1.jsonl [--out results/docile_typer/diagnosis.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src"), str(HERE.parent)]
import docile_bros as B  # noqa: E402
import docile_typer as T  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for f in a.rows for l in open(f, encoding="utf-8") if l.strip()]
    answers = {(r["tag"], r["doc_id"], r["page"]): T.parse_answer(r["samples"][0], r["n_cand"]) for r in rows if not r["error"] and r["samples"]}
    docs = {g: B.doc_dict(g) for g in json.loads(B.SPLITS.read_text(encoding="utf-8"))["test"]}
    short = lambda f: f.replace("line_item_", "")
    conf, vlm_dist, bros_dist = Counter(), Counter(), Counter()
    two = Counter()
    by_gold: dict = defaultdict(Counter)
    by_doc_kind: dict = defaultdict(Counter)
    for tag in sorted({r["tag"] for r in rows}):
        pred = T.load_pred(tag)
        for g, d in docs.items():
            spans = pred[g]
            gold = {(s, e): f for f, s, e, _ in B.spans_of(d)}
            used = {f for r_ in d['rows'] for f, _, _ in r_}
            kind = 'net-only documents' if 'line_item_amount_net' in used and 'line_item_amount_gross' not in used else 'gross-only documents' if 'line_item_amount_gross' in used and 'line_item_amount_net' not in used else 'other documents'
            for p in sorted({d["pages"][sp[1]] for sp in spans if sp[0] in T.MONEY}):
                ans = answers.get((tag, g, p), {})
                for n, k in enumerate(T.candidates(spans, d["pages"], p), 1):
                    f0, s, e = spans[k][0], spans[k][1], spans[k][2]
                    v = ans.get(n)
                    vlm_dist[short(v) if v else "(none)"] += 1
                    bros_dist[short(f0)] += 1
                    if (s, e) in gold and gold[(s, e)] in T.MONEY and v:
                        gf = gold[(s, e)]
                        conf[(short(gf), short(v))] += 1
                        two[("bros_right" if f0 == gf else "bros_wrong", "vlm_right" if v == gf else "vlm_wrong")] += 1
                        by_doc_kind[kind]["n"] += 1
                        by_doc_kind[kind]["bros_right"] += f0 == gf
                        by_doc_kind[kind]["vlm_right"] += v == gf
                        by_gold[short(gf)]["n"] += 1
                        by_gold[short(gf)]["bros_right"] += f0 == gf
                        by_gold[short(gf)]["vlm_right"] += v == gf
    n = sum(two.values())
    out = {"candidates_with_a_valid_answer_and_a_monetary_gold_span": n, "bros_vs_vlm": {f"{a_}/{b_}": c for (a_, b_), c in two.items()},
           "vlm_answer_distribution": dict(vlm_dist.most_common()), "bros_field_distribution": dict(bros_dist.most_common()),
           "accuracy_by_gold_field": {k: {"n": v["n"], "bros_pct": round(100 * v["bros_right"] / v["n"], 1), "vlm_pct": round(100 * v["vlm_right"] / v["n"], 1)} for k, v in by_gold.items()},
           "accuracy_by_document_kind": {k: {"n": v["n"], "bros_pct": round(100 * v["bros_right"] / v["n"], 1), "vlm_pct": round(100 * v["vlm_right"] / v["n"], 1)} for k, v in by_doc_kind.items()},
           "confusion_gold_to_vlm": {f"{a_} -> {b_}": c for (a_, b_), c in conf.most_common()}}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
