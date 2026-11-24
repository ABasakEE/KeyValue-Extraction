"""Pick demo examples that show good, typical and failing cases, scored with the actual demo checkpoint (not just the first documents in file order).

    python scripts/demo_curate_examples.py vrdu          # writes results/demo_weights/examples_vrdu.json
    python scripts/demo_curate_examples.py docile        # same for DocILE (needs its checkpoints)

Each entry is {guid, template, f1 (exact-span entity F1 or DocILE row F1, in %), role: best | typical | worst}. The demo labels every example with its role and score,
so a viewer sees that the list mixes successes and failures. Candidates are pre-filtered (documents with at least 4 gold fields) and ranked by the checkpoint itself.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(HERE)]
import src.demo_all as A  # noqa: E402

DB = A.DB


def entity_f1(clf, dev, id2label, ex) -> float:
    spans = set(DB.spans_of(DB.tag_document(clf, dev, ex), id2label))
    gold = set(DB.spans_of(ex.ner_tags, id2label))
    return 100 * A.f1_text(spans, gold)[0]


def pick(scored: list[tuple[float, str, str]], n_best: int, n_typ: int, n_worst: int):
    """best = highest scores; typical = closest to the candidates' mean score; worst = lowest."""
    scored = sorted(scored, key=lambda x: -x[0])
    mean = sum(x[0] for x in scored) / len(scored)
    best = scored[:n_best]
    worst = scored[len(scored) - n_worst:] if n_worst else []
    rest = [x for x in scored if x not in best and x not in worst]
    typical = sorted(rest, key=lambda x: abs(x[0] - mean))[:n_typ]
    out = [(*x, "best") for x in best] + [(*x, "typical") for x in typical] + [(*x, "worst") for x in worst]
    return [{"guid": guid, "template": tmpl, "f1": round(f1, 1), "role": role} for f1, guid, tmpl, role in out]


def vrdu() -> dict:
    by_guid, utl, mtl = A.vrdu_protocol()
    out = {}
    for key, split in (("UTL", utl), ("MTL", mtl)):
        clf, id2label, dev, trained, ck = A.get_model(A.VRDU_NAME, key)
        assert trained, f"no checkpoint for {key}"
        cands = [d for d in split.test if sum(t != 0 for t in d.ner_tags) >= 4]
        rng = random.Random(0)
        rng.shuffle(cands)
        picked = []
        if key == "UTL":
            picked = cands[:40]
        else:
            for tmpl in sorted({d.template_id for d in cands}):
                picked += [d for d in cands if d.template_id == tmpl][:14]
        scored = []
        for d in picked:
            scored.append((entity_f1(clf, dev, id2label, d), d.guid, d.template_id))
            print(key, scored[-1][0].__round__(0), d.guid[:50], flush=True)
        (ROOT / "results" / "demo_weights" / f"scores_vrdu_{key}.json").write_text(json.dumps(scored), encoding="utf-8")   # every candidate's score, for the record
        if key == "UTL":
            out[key] = pick(scored, 3, 2, 1)
        else:
            res = []
            for tmpl in sorted({t for _, _, t in scored}):
                res += pick([x for x in scored if x[2] == tmpl], 1, 1, 0)
            out[key] = res
    return out


def docile() -> dict:
    import docile_bros as DBR
    from lir_metric import f1 as lir_f1
    from row_group import group_rows

    sp = json.loads((ROOT / "results" / "docile_rows" / "splits.json").read_text(encoding="utf-8"))
    out = {}
    for key in ("S0", "S1"):
        if A.find_ckpt(A.DOCILE_NAME, key) is None:
            print(f"[skip] no checkpoint for {key}")
            continue
        clf, id2label, dev, trained, ck = A.get_model(A.DOCILE_NAME, key)
        scored = []
        for i in sp["test"]:
            d = DBR.doc_dict(i)
            if not (3 <= len(d["rows"]) <= 14 and len(d["words"]) <= 420):
                continue
            ex = DBR.to_example(i, d)
            spans = DB.spans_of(DB.tag_document(clf, dev, ex), id2label)
            rows = [r for r in group_rows([[t, a, b, ""] for t, a, b in spans], d, 0.5, merge=False) if r]
            c = DBR.counts(rows, d)
            scored.append((100 * lir_f1(c["lir"]), i, "unseen layout"))
            print(key, round(scored[-1][0]), i[:8], flush=True)
            if len(scored) >= 40:
                break
        out[key] = pick(scored, 3, 2, 1)
    return out


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "vrdu"
    if which not in ("vrdu", "docile"):
        raise SystemExit("usage: demo_curate_examples.py vrdu|docile")
    res = vrdu() if which == "vrdu" else docile()
    path = ROOT / "results" / "demo_weights" / f"examples_{which}.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
