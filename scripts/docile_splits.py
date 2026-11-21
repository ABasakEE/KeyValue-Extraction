"""Study 34: layout-disjoint splits of the DocILE labelled documents (CPU). Writes results/docile_rows/splits.json.

    python scripts/docile_splits.py

Rules (frozen here, before any model sees DocILE):
  * unit = documents with at least one line item; layout = metadata.cluster_id; all 5,680 labelled documents (train + validation of DocILE) form one pool.
  * reserved first, one random document per layout, layouts drawn uniformly at random (seed 34): test 200 layouts, validation 100 layouts (BROS early stopping,
    so its stopping is also judged on unseen layouts), pilot 30 layouts (prompt format gate).
  * stricter than the cluster id: every pool document whose normalised vendor_name equals the vendor of a reserved document is dropped from the pool
    (the same company can sit in two clusters).
  * training sets: for each seed 0, 1, 2 a random order of the pool; the training set of size N is its first N documents (nested in N).
"""
from __future__ import annotations

import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "docile"
SIZES = (10, 50, 100, 200, 500, 1000)
SEEDS = (0, 1, 2)
N_TEST, N_VAL, N_PILOT = 200, 100, 30


def main() -> None:
    ids = sorted(json.loads((DATA / "trainval.json").read_text(encoding="utf-8")))
    info = {}
    for i in ids:
        a = json.loads((DATA / "annotations" / f"{i}.json").read_text(encoding="utf-8"))
        vend = next((e["text"] for e in a["field_extractions"] if e["fieldtype"] == "vendor_name"), "")
        info[i] = {"layout": a["metadata"]["cluster_id"], "rows": len({e["line_item_id"] for e in a["line_item_extractions"]}),
                   "pages": a["metadata"]["page_count"], "vendor": re.sub(r"[^0-9a-z]", "", vend.lower())}
    cand = [i for i in ids if info[i]["rows"] > 0]
    by_layout = defaultdict(list)
    for i in cand:
        by_layout[info[i]["layout"]].append(i)
    rng = random.Random(34)
    layouts = sorted(by_layout)
    rng.shuffle(layouts)
    take = lambda ls: [rng.choice(by_layout[l]) for l in ls]
    test, val, pilot = take(layouts[:N_TEST]), take(layouts[N_TEST:N_TEST + N_VAL]), take(layouts[N_TEST + N_VAL:N_TEST + N_VAL + N_PILOT])
    reserved = set(test + val + pilot)
    res_layouts = {info[i]["layout"] for i in reserved}
    res_vendors = {info[i]["vendor"] for i in reserved if info[i]["vendor"]}
    pool = [i for i in cand if info[i]["layout"] not in res_layouts and info[i]["vendor"] not in res_vendors]
    dropped_vendor = sum(1 for i in cand if info[i]["layout"] not in res_layouts and info[i]["vendor"] in res_vendors)
    train = {str(s): random.Random(s).sample(pool, SIZES[-1]) for s in SEEDS}
    # checks
    assert not (set(test) & set(val) | set(test) & set(pilot) | set(val) & set(pilot))
    assert len({info[i]["layout"] for i in test + val + pilot}) == N_TEST + N_VAL + N_PILOT
    for t in train.values():
        assert not ({info[i]["layout"] for i in t} & res_layouts) and not ({info[i]["vendor"] for i in t} & res_vendors - {""}) and len(set(t)) == len(t)
    stats = {
        "candidates": len(cand), "layouts_with_line_items": len(by_layout), "pool_documents": len(pool), "pool_layouts": len({info[i]["layout"] for i in pool}),
        "pool_documents_dropped_for_vendor_overlap": dropped_vendor,
        "test": {"documents": len(test), "pages": sum(info[i]["pages"] for i in test), "rows": sum(info[i]["rows"] for i in test),
                 "documents_with_rows_per_page_over_15": None},
        "val_documents": len(val), "pilot": {"documents": len(pilot), "pages": sum(info[i]["pages"] for i in pilot)},
        "train_sets": {str(n): {"layouts_per_seed": [len({info[i]["layout"] for i in train[str(s)][:n]}) for s in SEEDS],
                                "rows_per_seed": [sum(info[i]["rows"] for i in train[str(s)][:n]) for s in SEEDS]} for n in SIZES},
        "test_rows_per_document": {"median": st.median(info[i]["rows"] for i in test), "max": max(info[i]["rows"] for i in test)},
        "vendor_known_in_test": sum(1 for i in test if info[i]["vendor"]),
    }
    stats["test"].pop("documents_with_rows_per_page_over_15")
    out = ROOT / "results" / "docile_rows" / "splits.json"
    out.write_text(json.dumps({"rules": __doc__, "stats": stats, "test": test, "val": val, "pilot": pilot, "train": train}, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
