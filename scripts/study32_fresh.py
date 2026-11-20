"""Study 32: prompt v2 and two post hoc rules on a fresh third document set (300 registration forms outside dev, confirm and the pilot).

  python scripts/study32_fresh.py bros                       BROS spans of the fresh documents from the saved Track-1 dumps (CPU, no model)
  python scripts/study32_fresh.py score --rows-dir DIR       decode the V2A2 / V2A1 rows, apply the rules, score against BROS (CPU)

Registered rules (STUDIES.md, study 32), all on the pooled 300 documents, mean over BROS seeds 13 and 42, paired bootstrap over documents:
  prompt replicates : V2A2 (registered decode: earliest page on ties) minus BROS >= +3.0 official-style text-match F1, interval above 0
  tie-break helps   : V2A2 with the latest page for file_date / signer_name / signer_title minus V2A2 earliest >= +2.0 text-match F1, interval above 0
  vote helps        : vote of (V2A2, V2A1, BROS), abstaining unless two agree, minus V2A2 with the tie-break >= +1.0 text-match F1, interval above 0
Exact-span F1 is reported next to every text-match figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import schema_guided_kv as sg  # noqa: E402
import study31_cpu as c  # noqa: E402
import study25_stage0 as st  # noqa: E402
from matched_eval import SEEDS, f1_of, span_counts  # noqa: E402

ROOT = sg.ROOT
FRESH = ROOT / "results" / "fresh_set3"
FOLD = {"Amendment": 0, "Dissemination_Report": 1, "Short-Form": 2}
DUMP = ROOT / "results" / "vrdu_track1_bros_errdump"


def cmd_bros(_a) -> None:
    docs = {d["doc_id"].split("/", 1)[1]: d for d in json.loads((FRESH / "manifest.json").read_text(encoding="utf-8"))}
    bros: dict = {s: {} for s in SEEDS}
    for fold in range(3):
        for s in SEEDS:
            for l in (DUMP / f"UTL_fold{fold}_seed{s}" / "test_entities.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(l)
                if r["guid"] in docs:
                    bros[s][r["guid"]] = r
    missing = [g for g in docs if any(g not in bros[s] for s in SEEDS)]
    assert not missing, f"{len(missing)} fresh documents have no BROS dump"
    out = []
    for g, d in sorted(docs.items()):
        assert bros[SEEDS[0]][g]["gold"] == bros[SEEDS[1]][g]["gold"]
        out.append({"fold": FOLD[d["template"]], "template": d["template"], "guid": g, "gold": bros[SEEDS[0]][g]["gold"],
                    **{f"bros_seed{s}": bros[s][g]["pred"] for s in SEEDS}})
    (FRESH / "bros_spans.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    full = {s: f1_of([sum(span_counts(r[f"bros_seed{s}"], r["gold"])["_micro"][i] for r in out) for i in range(3)]) for s in SEEDS}
    print(f"wrote {len(out)} documents; BROS exact-span F1 on them: " + ", ".join(f"seed {s} {v * 100:.1f}" for s, v in full.items()))


def decode_arm(paths: list[str], docs: dict, recs: dict, tie: dict[str, str]) -> dict[str, list[list]]:
    by_doc = sg.read_rows(paths)
    out = {}
    for g, d in docs.items():
        doc = sg.Doc(recs[g])
        cands: dict[str, list] = defaultdict(list)
        for row in by_doc.get(d["doc_id"], []):
            lines = {ln.id: ln for ln in doc.lines.get(row["page"], [])}
            segs = sg.parse_field(row["samples"][0])
            if segs:
                span, _ = sg.ground(segs, lines, doc, True)
                if span:
                    cands[row["field"]].append((row["page"], span))
        out[g] = c.decode_variant(doc, cands, tie, False)
    return out


def cmd_score(a) -> None:
    rd = Path(a.rows_dir)
    docs = {d["doc_id"].split("/", 1)[1]: d for d in json.loads((FRESH / "manifest.json").read_text(encoding="utf-8"))}
    recs = sg.load_records(ROOT / "data" / "vrdu", set(docs))
    base = {r["guid"]: r for r in map(json.loads, (FRESH / "bros_spans.jsonl").read_text(encoding="utf-8").splitlines())}
    late = {f: "latest" for f in c.LATE}
    arms = {}
    for name in ("V2A2", "V2A1"):
        paths = [str(rd / f"rows_{name}_fresh_{s}.jsonl") for s in (0, 1)]
        if not any(Path(p).exists() for p in paths):
            continue
        arms[name] = {"e": decode_arm(paths, docs, recs, {}), "l": decode_arm(paths, docs, recs, late)}
    need = ("V2A2",)
    assert all(n in arms for n in need), "V2A2 rows missing"
    complete = [g for g in sorted(docs) if all(g in arms[n]["e"] for n in arms)]
    rows = []
    for g in complete:
        r = dict(base[g])
        r["v2a2_e"], r["v2a2_l"] = arms["V2A2"]["e"][g], arms["V2A2"]["l"][g]
        if "V2A1" in arms:
            r["v2a1_e"], r["v2a1_l"] = arms["V2A1"]["e"][g], arms["V2A1"]["l"][g]
        rows.append(r)
    have_text = "V2A1" in arms
    print(f"{len(rows)} fresh documents scored; arms {list(arms)}")
    report: dict = {"n_docs": len(rows), "arms": list(arms)}
    verdict: dict = {}
    for metric in ("exact", "text"):
        txt = metric == "text"
        bases = ["bros", "v2a2_e", "v2a2_l"]
        acc: dict = defaultdict(lambda: defaultdict(lambda: {"f1": [], "d": [], "lo": [], "hi": []}))
        for s in SEEDS:
            systems = {"bros": lambda r, s=s: r[f"bros_seed{s}"], "v2a2_e": lambda r: r["v2a2_e"], "v2a2_l": lambda r: r["v2a2_l"],
                       "fill_l": lambda r, s=s: r["v2a2_l"] + [b for b in r[f"bros_seed{s}"] if b[0] not in {x[0] for x in r["v2a2_l"]}]}
            if have_text:
                systems.update({"v2a1_l": lambda r: r["v2a1_l"],
                                "v3_l": lambda r, s=s: c.ensemble(r, ["v2a2_l", "v2a1_l", "bros"], s, 1, False),
                                "v3_min2_l": lambda r, s=s: c.ensemble(r, ["v2a2_l", "v2a1_l", "bros"], s, 2, False),
                                "v3_min2_e": lambda r, s=s: c.ensemble(r, ["v2a2_e", "v2a1_e", "bros"], s, 2, False)})
            ev = c.evaluate(rows, systems, bases, txt)
            for b, tbl in ev.items():
                for n, v in tbl.items():
                    acc[b][n]["f1"].append(v["f1"]); acc[b][n]["d"].append(v["minus_base"]); acc[b][n]["lo"].append(v["ci"][0]); acc[b][n]["hi"].append(v["ci"][1])
        res = {b: {n: {"f1": round(float(np.mean(v["f1"])), 1), "diff": round(float(np.mean(v["d"])), 1),
                       "ci": [round(float(np.mean(v["lo"])), 1), round(float(np.mean(v["hi"])), 1)]} for n, v in tbls.items()} for b, tbls in acc.items()}
        report[metric] = res
        print(f"\n== {metric}: F1 [diff to the base, 95% CI]")
        for base_name, tbl in res.items():
            print(f" base = {base_name}")
            for n, v in tbl.items():
                print(f"   {n:12s} {v['f1']:5.1f}  {v['diff']:+5.1f} {v['ci']}")
    t = report["text"]
    verdict["prompt_replicates"] = bool(t["bros"]["v2a2_e"]["diff"] >= 3.0 and t["bros"]["v2a2_e"]["ci"][0] > 0)
    verdict["tie_break_helps"] = bool(t["v2a2_e"]["v2a2_l"]["diff"] >= 2.0 and t["v2a2_e"]["v2a2_l"]["ci"][0] > 0)
    if have_text:
        verdict["vote_helps"] = bool(t["v2a2_l"]["v3_min2_l"]["diff"] >= 1.0 and t["v2a2_l"]["v3_min2_l"]["ci"][0] > 0)
    report["verdict"] = verdict
    print("\nverdict (registered rules):", verdict)
    # per template and per field (exact span, V2A2 earliest / latest vs BROS seed 13), descriptive
    by_t = {}
    for tpl in FOLD:
        sub = [r for r in rows if r["template"] == tpl]
        by_t[tpl] = {n: round(f1_of([sum(span_counts(r[n], r["gold"])["_micro"][i] for r in sub) for i in range(3)]) * 100, 1)
                     for n in ("bros_seed13", "bros_seed42", "v2a2_e", "v2a2_l")}
    report["by_template_exact"] = by_t
    per_field = {n: {f: round(f1_of([sum(span_counts(r[n], r["gold"]).get(f, (0, 0, 0))[i] for r in rows) for i in range(3)]) * 100, 1) for f in c.FIELDS}
                 for n in ("bros_seed13", "v2a2_e", "v2a2_l")}
    report["per_field_exact"] = per_field
    print("by template (exact):", json.dumps(by_t)); print("per field (exact):", json.dumps(per_field))
    (rd / "fresh_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bros")
    sub.add_parser("score").add_argument("--rows-dir", required=True)
    a = ap.parse_args()
    {"bros": cmd_bros, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
