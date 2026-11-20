"""Study 31, CPU-only post hoc experiments on the saved outputs (dev and confirm, 300 documents each).

  python scripts/study31_cpu.py vote --split dev|confirm        span-level voting across systems
  python scripts/study31_cpu.py decode --split dev|confirm      decode variants on the saved V2A2 rows (page tie-break, overlap resolution)
  python scripts/study31_cpu.py taxonomy                        error taxonomy of the V2A2 spans
  python scripts/study31_cpu.py manifest                        a fresh third document set (outside dev, confirm and pilot)

Everything here is post hoc: the variants were listed before looking at their scores, and are reported all together (dev and confirm), but they were chosen
after seeing the study-31 results, so a gain is a hypothesis for a fresh set, not a result.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import schema_guided_kv as sg  # noqa: E402
import study25_stage0 as st  # noqa: E402
from matched_eval import SEEDS, ci, f1_of, paired_bootstrap, span_counts  # noqa: E402

ROOT = sg.ROOT
V2 = ROOT / "results" / "schema_guided_kv_v2"
OUT = ROOT / "results" / "study31_cpu"
OUT.mkdir(parents=True, exist_ok=True)
PATHS = {  # (V2A2 spans, V2A1 spans, study-25 8B A2 spans) per document set
    "dev": (V2 / "dev_vsA0/spans_V2A2.jsonl", V2 / "dev_vsA0/spans_V2A1.jsonl", ROOT / "results/study25_stage0/8B_A2_dev/spans_A2.jsonl"),
    "confirm": (V2 / "vsA0/spans_V2A2.jsonl", V2 / "confirm_full_vsA0/spans_V2A1.jsonl", ROOT / "results/study25_stage0/8B_A2_confirm/spans_A2.jsonl"),
}
MANIFESTS = {"confirm": "results/confirm_v2/manifest.json", "dev": "results/zero_shot_kv_v2/manifest.json"}
FIELDS = ["file_date", "foreign_principle_name", "registrant_name", "registration_num", "signer_name", "signer_title"]


def _rows(p: Path) -> dict[str, dict]:
    return {r["guid"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}


def load_split(split: str) -> list[dict]:
    a2, a1, s25 = (_rows(p) for p in PATHS[split])
    out = []
    for g, r in sorted(a2.items()):
        out.append({"guid": g, "template": r["template"], "fold": r["fold"], "gold": r["gold"], "bros_seed13": r["bros_seed13"],
                    "bros_seed42": r["bros_seed42"], "a0": r["vlm_both"], "v2a2": r["vlm_V2A2"], "v2a1": a1[g]["vlm_V2A1"], "s25": s25[g]["vlm_A2"]})
    assert len(out) == 300, len(out)
    return out


_GOLD_T: dict = {}


def evaluate(rows: list[dict], systems: dict, base: str, text: bool = False) -> dict:
    """systems: name -> function(row) -> spans.  Per-system F1 and the paired difference to `base`.

    text=False: exact spans (as registered).  text=True: official-style text match (any annotated occurrence of the field, VRDU rules),
    which does not penalise picking another occurrence of the right text."""
    if text:
        missing = {r["guid"] for r in rows} - set(_GOLD_T)
        if missing:
            _GOLD_T.update(st.load_gold_texts(missing))
        count = lambda r, fn: st.text_counts(fn(r), _GOLD_T[r["guid"]])
    else:
        count = lambda r, fn: span_counts(fn(r), r["gold"])
    per = {n: [count(r, fn) for r in rows] for n, fn in systems.items()}
    boot = paired_bootstrap(per, "_micro")
    res = {}
    for n in systems:
        f1 = f1_of([sum(x["_micro"][i] for x in per[n]) for i in range(3)])
        d = boot[n] - boot[base]
        res[n] = {"f1": round(f1 * 100, 1), "minus_base": round(float(d.mean()) * 100, 1), "ci": [round(v * 100, 1) for v in ci(d)]}
    return res


# ------------------------------------------------------------------ vote
def ensemble(r: dict, order: list[str], seed: int, min_votes: int, cluster: bool) -> list[list]:
    names = [f"bros_seed{seed}" if n == "bros" else n for n in order]
    out = []
    for f in FIELDS:
        cands = []  # (rank, span)
        for rank, n in enumerate(names):
            sp = next((s for s in r[n] if s[0] == f), None)
            if sp:
                cands.append((rank, sp))
        if not cands:
            continue
        groups: list[list] = []  # each: [rep span, rep rank, set of ranks]
        for rank, sp in cands:
            for g in groups:
                same = (sp[1], sp[2]) == (g[0][1], g[0][2])
                near = sp[1] < g[0][2] and g[0][1] < sp[2]
                if (near if cluster else same):
                    g[2].add(rank)
                    break
            else:
                groups.append([sp, rank, {rank}])
        best = max(groups, key=lambda g: (len(g[2]), -g[1]))
        if len(best[2]) >= min_votes:
            out.append(best[0])
    return out


def cmd_vote(a) -> None:
    rows = load_split(a.split)
    variants = {  # fixed before scoring
        "v3": (["v2a2", "v2a1", "bros"], 1, False), "v3_cluster": (["v2a2", "v2a1", "bros"], 1, True),
        "v5": (["v2a2", "v2a1", "bros", "a0", "s25"], 1, False), "v5_cluster": (["v2a2", "v2a1", "bros", "a0", "s25"], 1, True),
        "v3_min2": (["v2a2", "v2a1", "bros"], 2, False), "v3_cluster_min2": (["v2a2", "v2a1", "bros"], 2, True),
        "v5_min2": (["v2a2", "v2a1", "bros", "a0", "s25"], 2, False), "v5_cluster_min2": (["v2a2", "v2a1", "bros", "a0", "s25"], 2, True),
    }
    res = {}
    for seed in SEEDS:
        systems = {"bros": lambda r, s=seed: r[f"bros_seed{s}"], "v2a2": lambda r: r["v2a2"], "v2a1": lambda r: r["v2a1"],
                   "fill": lambda r, s=seed: r["v2a2"] + [b for b in r[f"bros_seed{s}"] if b[0] not in {x[0] for x in r["v2a2"]}]}
        for name, (order, mv, cl) in variants.items():
            systems[name] = lambda r, o=order, s=seed, m=mv, c=cl: ensemble(r, o, s, m, c)
        for text in (False, True):
            res[(f"seed{seed}", text)] = evaluate(rows, systems, "v2a2", text)
    allmean = {}
    for text in (False, True):
        print(f"== vote on {a.split}, {'official-style text match' if text else 'exact span'} (F1 [diff to V2A2 alone, 95% CI]; mean over BROS seeds {SEEDS})")
        mean = {}
        for n in res[(f"seed{SEEDS[0]}", text)]:
            g = lambda k: np.mean([res[(f"seed{s}", text)][n][k] for s in SEEDS], axis=0)
            f1, d, (lo, hi) = g("f1"), g("minus_base"), g("ci")
            mean[n] = {"f1": round(float(f1), 1), "diff": round(float(d), 1), "ci": [round(float(lo), 1), round(float(hi), 1)]}
            print(f"  {n:18s} {f1:5.1f}  {d:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]")
        allmean["text" if text else "exact"] = mean
    (OUT / f"vote_{a.split}.json").write_text(json.dumps(allmean, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ decode variants
LATE = {"file_date", "signer_name", "signer_title"}  # signature-block fields, generic form structure


def pick(cands: list[tuple[int, tuple[int, int]]], tie: str, richness: dict[int, int]) -> list[tuple[int, int]]:
    """Spans of one field ranked best first: votes, then the page tie-break."""
    votes: dict[tuple[int, int], list[int]] = defaultdict(list)
    for page, span in cands:
        votes[span].append(page)
    if tie == "earliest":
        key = lambda sp: (len(votes[sp]), -min(votes[sp]), -sp[0])
    elif tie == "latest":
        key = lambda sp: (len(votes[sp]), max(votes[sp]), -sp[0])
    else:  # richest page: the page on which the most fields were found
        key = lambda sp: (len(votes[sp]), max(richness[p] for p in votes[sp]), -min(votes[sp]), -sp[0])
    return sorted(votes, key=key, reverse=True)


def decode_variant(doc, cands: dict[str, list], tie_by_field: dict[str, str], next_best: bool) -> list[list]:
    richness: Counter = Counter()
    for f, cs in cands.items():
        for p in {p for p, _ in cs}:
            richness[p] += 1
    ranked = {f: pick(cs, tie_by_field.get(f, "earliest"), richness) for f, cs in cands.items() if cs}
    votes = {f: Counter(sp for _, sp in cands[f]) for f in ranked}
    used: set[int] = set()
    out = []
    for f in sorted(ranked, key=lambda f: (-votes[f][ranked[f][0]], FIELDS.index(f))):
        for sp in (ranked[f] if next_best else ranked[f][:1]):
            if not used.intersection(range(*sp)):
                used.update(range(*sp))
                out.append([f, sp[0], sp[1], " ".join(doc.words[sp[0]:sp[1]])])
                break
    return out


def cmd_decode(a) -> None:
    docs = {d["doc_id"].split("/", 1)[1]: d for d in sg.vrdu_manifest_docs(MANIFESTS[a.split])}
    recs = sg.load_records(ROOT / "data" / "vrdu", set(docs))
    by_doc = sg.read_rows([str(V2 / f"rows_V2A2_{a.split}_{s}.jsonl") for s in (0, 1)])
    base_rows = load_split(a.split)
    cand_by_guid = {}
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
        cand_by_guid[g] = (doc, cands)
    rows = []
    for r in base_rows:
        doc, cands = cand_by_guid[r["guid"]]
        rows.append({**r, "_doc": doc, "_cands": cands})
    ties = {"earliest": {}, "latest_signature": {f: "latest" for f in LATE}, "richest_page": {f: "richest" for f in FIELDS},
            "latest_all": {f: "latest" for f in FIELDS}}
    systems = {"saved_v2a2": lambda r: r["v2a2"]}
    for tn, tb in ties.items():
        for nb in (False, True):
            systems[f"{tn}{'+next_best' if nb else ''}"] = lambda r, tb=tb, nb=nb: decode_variant(r["_doc"], r["_cands"], tb, nb)
    res = {}
    for text in (False, True):
        res["text" if text else "exact"] = evaluate(rows, systems, "saved_v2a2", text)
        print(f"== decode variants on {a.split}, {'official-style text match' if text else 'exact span'} (F1 [diff to the saved registered-style decode, 95% CI])")
        for n, v in res["text" if text else "exact"].items():
            print(f"  {n:28s} {v['f1']:5.1f}  {v['minus_base']:+5.1f} {v['ci']}")
    same = np.mean([span_counts(decode_variant(r['_doc'], r['_cands'], {}, False), r['gold'])['_micro'] == span_counts(r['v2a2'], r['gold'])['_micro'] for r in rows])
    print(f"  sanity: the 'earliest' variant reproduces the saved counts on {same * 100:.1f}% of documents")
    (OUT / f"decode_{a.split}.json").write_text(json.dumps({"results": res, "reproduces_saved": float(same)}, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ taxonomy
def _n(s: str) -> str:
    return " ".join(s.lower().split())


def cmd_taxonomy(_a) -> None:
    kinds: dict = defaultdict(Counter)
    ex: dict = defaultdict(list)
    for split in ("dev", "confirm"):
        rows = load_split(split)
        gold_t = st.load_gold_texts({r["guid"] for r in rows})
        for r in rows:
            gt = gold_t[r["guid"]]
            gold = {s[0]: s for s in r["gold"]}
            pred = {s[0]: s for s in r["v2a2"]}
            for f in FIELDS:
                key = (f, r["template"])
                if f in pred and f not in gold:
                    cat = "predicted, field not annotated"
                elif f in gold and f not in pred:
                    cat = "missing"
                elif f in pred:
                    p, g = pred[f], gold[f]
                    pt = _n(p[3])
                    others = {_n(t) for o, ts in gt.items() if o != f for t in ts}
                    if (p[1], p[2]) == (g[1], g[2]):
                        cat = "exact"
                    elif any(st.text_match(st.META[f], p[3], t) for t in gt.get(f, [])):
                        cat = "right text, other occurrence"
                    elif pt in others:
                        cat = "text of another field"
                    elif any(pt and _n(t).startswith(pt) for t in gt.get(f, [])):
                        cat = "gold continues after it (too short)"
                    elif any(pt and pt in _n(t) for t in gt.get(f, [])):
                        cat = "inside the gold text"
                    elif any(_n(t) and _n(t) in pt for t in gt.get(f, [])):
                        cat = "contains the gold text (too long)"
                    else:
                        cat = "other text"
                else:
                    continue
                kinds[key][cat] += 1
                if cat != "exact" and len(ex[(f, cat)]) < 4:
                    ex[(f, cat)].append({"split": split, "template": r["template"], "pred": pred.get(f, [0, 0, 0, ""])[3][:70], "gold": [t.strip()[:60] for t in gt.get(f, [])][:2]})
    table: dict = defaultdict(Counter)
    for (f, tpl), c in kinds.items():
        table[f].update(c)
    print("== error taxonomy of the V2A2 spans (dev + confirm, 600 documents)")
    for f in FIELDS:
        print(f"  {f}: {dict(table[f].most_common())}")
    print("\n  by template (non-exact categories only), signer_name / foreign_principle_name / file_date:")
    for f in ("signer_name", "foreign_principle_name", "file_date"):
        for tpl in ("Amendment", "Dissemination_Report", "Short-Form"):
            print(f"    {f:24s} {tpl:21s}", {k: v for k, v in kinds[(f, tpl)].most_common()})
    (OUT / "taxonomy.json").write_text(json.dumps({"by_field": {f: dict(c) for f, c in table.items()},
                                                   "by_field_template": {f"{f}|{t}": dict(c) for (f, t), c in kinds.items()},
                                                   "examples": {f"{f}|{c}": v for (f, c), v in ex.items()}}, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ fresh third set
def cmd_manifest(_a) -> None:
    from zero_shot_kv_v2 import build_manifest

    exclude = set()
    for p in (*MANIFESTS.values(), "results/schema_guided_kv_q3vl8b/pilot_manifest.json"):
        exclude |= {d["doc_id"].split("/", 1)[1] for d in json.loads((ROOT / p).read_text(encoding="utf-8")) if d["dataset"] == sg.SUBCORPUS}
    docs = build_manifest(None, ROOT / "data" / "vrdu", 100, 11, sg.MAX_PAGES, subcorpora=(sg.SUBCORPUS,), exclude=exclude)
    assert not {d["doc_id"].split("/", 1)[1] for d in docs} & exclude
    out = ROOT / "results" / "fresh_set3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    print(f"== fresh set: {len(docs)} documents, per template {dict(Counter(d['template'] for d in docs))}; excluded {len(exclude)} documents")
    ckpt = sorted(str(p.relative_to(ROOT)) for pat in ("*.pt", "*.bin", "*.safetensors", "*.ckpt") for p in ROOT.glob(f"**/{pat}") if "data" not in p.parts[:2])[:10]
    print("   local checkpoints found (BROS inference on the new documents would need one of these):", ckpt or "none")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("vote", "decode"):
        sub.add_parser(n).add_argument("--split", choices=["dev", "confirm"], required=True)
    sub.add_parser("taxonomy")
    sub.add_parser("manifest")
    a = ap.parse_args()
    {"vote": cmd_vote, "decode": cmd_decode, "taxonomy": cmd_taxonomy, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    main()
