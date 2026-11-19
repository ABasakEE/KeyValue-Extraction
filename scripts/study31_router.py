"""Study 31 stage 0 (CPU, post hoc): snapping to BROS and a per-field router with the prompt-v2 spans, fitted on the OTHER document set.

  python scripts/study31_router.py

For each document set (dev, confirm) and each held-out template T: the per-field source choice is fitted on the other set's documents of the
other two templates (never on the test documents or on the test template), then applied to this set's documents of T.
Sources per field: BROS (seed s), the VLM spans (V2A2), the VLM spans snapped to BROS (each VLM span replaced by the BROS span of the same field it
overlaps), and the snapped spans plus BROS spans for fields the VLM left empty.  Reported: pooled F1 (mean over BROS seeds), difference to BROS with a
paired bootstrap over documents, the per-field F1 of every source, and the choices.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from confirm_eval import compare, micro  # noqa: E402
from hybrid_eval import fit_router, route  # noqa: E402
from matched_eval import SEEDS, f1_of, span_counts  # noqa: E402
from study25_stage0 import snap  # noqa: E402

ROOT = HERE.parent
R = ROOT / "results" / "schema_guided_kv_v2"
SETS = {"dev": R / "dev_vsA0" / "spans_V2A2.jsonl", "confirm": R / "vsA0" / "spans_V2A2.jsonl"}
OUT = R / "stage0_router.json"


def load(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["vlm"] = r["vlm_V2A2"]
        for s in SEEDS:
            r[f"snap{s}"] = snap(r["vlm"], r[f"bros_seed{s}"], False)
            r[f"fill{s}"] = snap(r["vlm"], r[f"bros_seed{s}"], True)
    return rows


def run(test_set: str, train_set: str, data: dict[str, list[dict]]) -> dict:
    per_doc: dict[str, list[dict]] = defaultdict(list)
    choices: dict = {}
    fields = ["file_date", "foreign_principle_name", "registrant_name", "registration_num", "signer_name", "signer_title"]
    per_field: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for fold in sorted({r["fold"] for r in data[test_set]}):
        test = [r for r in data[test_set] if r["fold"] == fold]
        train = [r for r in data[train_set] if r["fold"] != fold]
        for s in SEEDS:
            bros = f"bros_seed{s}"
            variants = {"router_vlm": ("vlm",), "router_snap": ("vlm", f"snap{s}"), "router_fill": ("vlm", f"snap{s}", f"fill{s}")}
            fits = {name: fit_router(train, bros, srcs) for name, srcs in variants.items()}
            choices[f"fold{fold}_seed{s}"] = fits
            for r in test:
                gold = r["gold"]
                per_doc[bros].append(span_counts(r[bros], gold))
                per_doc[f"vlm_seed{s}"].append(span_counts(r["vlm"], gold))
                per_doc[f"snap_seed{s}"].append(span_counts(r[f"snap{s}"], gold))
                per_doc[f"fill_seed{s}"].append(span_counts(r[f"fill{s}"], gold))
                for name, srcs in variants.items():
                    src = {bros: r[bros], **{k: r[k] for k in srcs}}
                    per_doc[f"{name}_seed{s}"].append(span_counts(route(fits[name], src, bros), gold))
                if s == SEEDS[0]:
                    for sys_name, key in (("bros", bros), ("vlm", "vlm"), ("snap", f"snap{s}")):
                        for f, c in span_counts(r[key], gold).items():
                            if f != "_micro":
                                for i in range(3):
                                    per_field[sys_name][f][i] += c[i]
    out = {"n_docs": len(per_doc[f"bros_seed{SEEDS[0]}"])}  # every system has the same document order per seed
    for name in ("vlm", "snap", "fill", "router_vlm", "router_snap", "router_fill"):
        out[name] = compare({f"{name}_seed{s}": per_doc[f"{name}_seed{s}"] for s in SEEDS} | {f"bros_seed{s}": per_doc[f"bros_seed{s}"] for s in SEEDS}, name)
    out["per_field_f1_seed13"] = {sys_name: {f: round(f1_of(per_field[sys_name][f]) * 100, 1) for f in fields} for sys_name in per_field}
    out["choices"] = choices
    return out


def main() -> None:
    data = {k: load(v) for k, v in SETS.items()}
    assert all(len(v) == 300 for v in data.values())
    report = {}
    for test_set, train_set in (("dev", "confirm"), ("confirm", "dev")):
        res = run(test_set, train_set, data)
        report[f"test_{test_set}_fit_{train_set}"] = res
        print(f"\n== test on {test_set} ({res['n_docs']} docs), router fitted on {train_set} documents of the other templates")
        for name in ("vlm", "snap", "fill", "router_vlm", "router_snap", "router_fill"):
            c = res[name]
            print(f"  {name:12s} F1 {c['f1'] * 100:5.1f}  BROS {c['bros_f1'] * 100:5.1f}  diff {c['minus_bros'] * 100:+5.1f}  {[round(x * 100, 1) for x in c['minus_bros_ci95']]}")
        print("  per field (seed 13):", json.dumps(res["per_field_f1_seed13"]))
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
