"""Study 36 groundwork (CPU): how much could a perfect router between the recipe (A2, text-first decode) and BROS(K) gain on the 500 FATURA confirm invoices?

    python scripts/router_headroom_fatura.py

Per invoice and field, a prediction is right if its block span equals the gold block. The oracle router picks the right system when either is right, abstains when the field has no
gold, and otherwise keeps the recipe's answer. Both systems start from the same annotation OCR, so this is an optimistic bound on any router.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
import fatura_sg as F  # noqa: E402
import fatura_sg_textfirst as TF  # noqa: E402
import schema_guided_kv as sg  # noqa: E402
from matched_eval import f1_of  # noqa: E402


def main() -> None:
    recs = {r["guid"]: r for r in map(json.loads, (F.V1 / "spans.jsonl").read_text(encoding="utf-8").splitlines()) if r["split"] == "confirm"}
    docs = F.load_fdocs(set(recs))
    by = sg.read_rows([str(F.OUT / "run" / f"rows_A2_{s}.jsonl") for s in (0, 1)])
    a2 = {g: {sp[0]: (sp[1], sp[2]) for sp in TF.text_first(docs[g], by[g], "json")[0]} for g in recs}
    gold = {g: {sp[0]: (sp[1], sp[2]) for sp in r["gold"]} for g, r in recs.items()}
    print(f"{len(recs)} invoices; recipe A2 alone")
    rows = {}
    for k in F.KS:
        res = {"bros": [], "a2": [], "oracle": [], "either_right": [], "only_a2": [], "only_bros": []}
        for seed in F.SEEDS:
            bros = {}
            for fold in range(5):
                p = F.bros_paths(k, fold, seed)
                for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()):
                    if r["guid"] in recs:   # BROS may output several spans for one field; all of them count, as in the official scorer
                        d = bros.setdefault(r["guid"], {})
                        for sp in r["pred"]:
                            d.setdefault(sp[0], []).append((sp[1], sp[2]))
            c = {n: [0, 0, 0] for n in ("bros", "a2", "oracle")}
            n_either = n_a = n_b = n_gold = 0
            for g in recs:
                for f in set(gold[g]) | set(a2[g]) | set(bros[g]):
                    gf, af, bl = gold[g].get(f), a2[g].get(f), bros[g].get(f, [])
                    ra, rb = gf is not None and af == gf, gf is not None and gf in bl
                    n_gold += gf is not None
                    c["bros"][0] += rb; c["bros"][1] += len(bl); c["bros"][2] += gf is not None
                    c["a2"][0] += ra; c["a2"][1] += af is not None; c["a2"][2] += gf is not None
                    if ra or rb:
                        c["oracle"][0] += 1; c["oracle"][1] += 1; n_either += 1
                    elif gf is not None and af is not None:
                        c["oracle"][1] += 1
                    n_a += ra and not rb
                    n_b += rb and not ra
                    c["oracle"][2] += gf is not None
            for n in c:
                res[n].append(f1_of(c[n]) * 100)
            res["either_right"].append(100 * n_either / n_gold); res["only_a2"].append(100 * n_a / n_gold); res["only_bros"].append(100 * n_b / n_gold)
        m = {n: round(sum(v) / len(v), 2) for n, v in res.items()}
        m["headroom_over_best_single"] = round(m["oracle"] - max(m["bros"], m["a2"]), 2)
        rows[str(k)] = m
        print(f"K={k:2d}: BROS {m['bros']:5.1f} | recipe {m['a2']:5.1f} | oracle router {m['oracle']:5.1f} (+{m['headroom_over_best_single']:4.1f}) | "
              f"gold fields right in both-or-either {m['either_right']:.1f}%, only the recipe {m['only_a2']:.1f}%, only BROS {m['only_bros']:.1f}%")
    out = HERE.parent / "results" / "router" / "headroom_fatura.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
