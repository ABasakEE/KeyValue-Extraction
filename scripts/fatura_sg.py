"""Study 33: the schema-guided, grounded recipe (prompt v2, Qwen3-VL-8B) on FATURA unseen templates, against BROS trained on K seen templates.

  python scripts/fatura_sg.py manifests                      write the pilot (30 dev invoices) and confirm (500 invoices) lists
  python scripts/fatura_sg.py oracle                         gold answers through the decode path must score ~100 (validates blocks, ids, decode)
  python scripts/fatura_sg.py run --manifest M --arm image|text --format json|field --out rows.jsonl [--shard i --num-shards n --quant4 --stub]
  python scripts/fatura_sg.py gate --manifest M --rows ROWS --format json|field
  python scripts/fatura_sg.py score --rows-dir DIR           score the arms against BROS(K) for K = 1, 2, 5, 10, 40 (CPU)

The OCR 'lines' of a FATURA page are its annotated text blocks (a block holds its own label, e.g. 'Date: 20-Mar-2008'); the same block text is what
BROS reads.  A cited block id becomes the whole block (gold is block level).  Everything else (prompt v2, descriptions written from the field names,
one call per page or per field, fallback by text when an id is missing) is the study-31 recipe; nothing is tuned on FATURA.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import schema_guided_kv as sg  # noqa: E402
from data import fatura_loader as fl  # noqa: E402
from fatura_eval import boot_f1, ci as fci  # noqa: E402
from hybrid_eval import fill_in  # noqa: E402
from matched_eval import f1_of, span_counts  # noqa: E402
from zero_shot_kv import norm  # noqa: E402

ROOT = sg.ROOT
OUT = ROOT / "results" / "fatura_sg"
V1 = ROOT / "results" / "fatura_v1"
FIELDS = fl.FIELDS
EXAMPLE = '{"date": "Date: March 5, 2012 31|12", "number": "Invoice 6123 08|40", "total": null}'   # invented values and ids
SEEDS = (13, 42)
KS = (1, 2, 5, 10, 40)


# ------------------------------------------------------------------ documents
class FDoc:
    """A FATURA invoice with its text blocks as identified lines."""

    def __init__(self, doc: dict):
        self.words, self.gold = doc["words"], doc["gold"]
        seen: Counter = Counter()
        lines = []
        for _field, s, e in doc["blocks"]:
            x0, y0, x1, y1 = (v / 1000 for v in doc["boxes"][s])
            lid = sg.coord_id([0, x0, y0, x1, y1])
            seen[lid] += 1
            if seen[lid] > 1:
                lid = f"{lid}#{seen[lid]}"
            text = " ".join(self.words[s:e])
            lines.append(sg.Line(0, lid, text, list(range(s, e)), (0, 0), text))
        self.lines = {0: lines}
        self.by_id = {ln.id: ln for ln in lines}


def load_fdocs(guids: set[str]) -> dict[str, FDoc]:
    root = fl.default_root() / "Annotations" / "Original_Format"
    return {g: FDoc(fl.parse_doc(g, json.loads((root / f"{g}.json").read_text(encoding="utf-8")))) for g in sorted(guids)}


def desc() -> dict[str, str]:
    return json.loads((V1 / "schema_desc.json").read_text(encoding="utf-8"))["fatura"]


def guids_of(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ decode (block level)
def decode(doc: FDoc, rows: list[dict], fmt: str, fallback: bool = True) -> tuple[list[list], Counter]:
    st: Counter = Counter()
    answers: dict[str, list | None] = {}
    for r in rows:
        raw = r["samples"][0]
        st["outputs"] += 1
        if fmt == "json":
            p = sg.parse_json(raw, FIELDS)
            if p is None:
                st["unparsed"] += 1
                continue
            answers.update(p)
        else:
            answers[r["field"]] = sg.parse_field(raw)
    out, used = [], set()
    for f in FIELDS:
        segs = answers.get(f)
        if not segs:
            st["null"] += 1
            continue
        st["values"] += 1
        text, lid = segs[0]
        ln = doc.by_id.get(lid) if lid else None
        if ln is not None:
            st["id_valid"] += 1
        elif fallback and norm(text).split():
            t = norm(re.sub(r"\s+\d{1,2}$", "", text) if lid is None else text)
            ln = next((c for c in doc.lines[0] if t and t in norm(c.text)), None)
            st["by_text" if ln else "ungrounded"] += ln is not None or 1
        else:
            st["ungrounded"] += 1
        if ln is None:
            continue
        s, e = ln.words[0], ln.words[-1] + 1
        if used.intersection(range(s, e)):
            st["dropped_overlap"] += 1
            continue
        used.update(range(s, e))
        out.append([f, s, e, " ".join(doc.words[s:e])])
    return out, st


# ------------------------------------------------------------------ manifests, oracle
def cmd_manifests(_a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((V1 / "manifest.json").read_text(encoding="utf-8"))
    confirm = sorted(d["doc_id"].split("/", 1)[1] for d in man if d["split"] == "confirm")
    dev = sorted(d["doc_id"].split("/", 1)[1] for d in man if d["split"] == "dev")
    pilot = sorted(random.Random(5).sample(dev, 30))
    (OUT / "confirm.json").write_text(json.dumps(confirm), encoding="utf-8")
    (OUT / "pilot.json").write_text(json.dumps(pilot), encoding="utf-8")
    print(f"confirm {len(confirm)} invoices, pilot {len(pilot)} dev invoices ({len({g.split('_')[0] for g in pilot})} templates)")


def cmd_oracle(_a) -> None:
    guids = guids_of(OUT / "confirm.json")[:100]
    docs = load_fdocs(set(guids))
    for fmt in ("json", "field"):
        counts, stats = [], Counter()
        for g in guids:
            d = docs[g]
            by_start = {ln.words[0]: ln for ln in d.lines[0]}
            ans = {f: f"{by_start[s].text} {by_start[s].id}" for f, s, e, t in d.gold}
            if fmt == "json":
                rows = [{"samples": [json.dumps({f: ans.get(f) for f in FIELDS})]}]
            else:
                rows = [{"field": f, "samples": [f"ANSWER: {ans[f]}" if f in ans else "ANSWER: NONE"]} for f in FIELDS]
            spans, st = decode(d, rows, fmt, False)
            stats.update(st)
            counts.append(span_counts(spans, d.gold))
        f1 = f1_of([sum(c["_micro"][i] for c in counts) for i in range(3)])
        print(f"oracle {fmt}: exact block-span F1 {f1 * 100:.2f} on {len(guids)} invoices; {dict(stats)}")
        assert f1 >= 0.98, f"oracle {fmt} below 0.98"


# ------------------------------------------------------------------ run
def cmd_run(a) -> None:
    from PIL import Image

    guids = guids_of(Path(a.manifest))
    docs = load_fdocs(set(guids))
    d = desc()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size:
        done = {(r["doc_id"], r.get("field")) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    units = [(g, f) for g in guids for f in ([None] if a.format == "json" else FIELDS)]
    units = [u for u in units[a.shard::a.num_shards] if u not in done]
    if a.limit:
        units = units[:a.limit]
    print(f"[shard {a.shard}] {len(units)} units on {a.device} (arm {a.arm}, format {a.format})", flush=True)
    if not units:
        return

    class _Stub:
        def sample(self, prompt, image, k, max_new_tokens, prefill=""):
            return ["{}" if "JSON" in prompt else "ANSWER: NONE"] * k

    gen = _Stub() if a.stub else sg._vlm_class()(a.model, a.device, a.dtype, a.max_pixels, a.quant4)
    deadline = time.time() + a.deadline_hours * 3600 if a.deadline_hours else None
    root = fl.default_root()
    with open(out, "a", encoding="utf-8") as fh:
        for k, (g, f) in enumerate(units, 1):
            if deadline and time.time() > deadline:
                print(f"[shard {a.shard}] deadline reached after {k - 1} units; stopping cleanly (re-run to resume)", flush=True)
                break
            row = {"doc_id": g, "page": 0, "field": f, "arm": a.arm, "format": a.format, "samples": [], "error": None}
            t0 = time.time()
            try:
                lines = docs[g].lines[0]
                image = Image.open(root / "images" / f"{g}.jpg").convert("RGB") if a.arm == "image" else None
                prompt = (sg.build_prompt(lines, FIELDS, d, True, EXAMPLE) if f is None else sg.field_prompt(lines, f, d, True))
                row["samples"] = gen.sample(prompt, image, 1, 400 if f is None else 64)
            except Exception as e:  # one bad page must not kill a multi-hour run
                row["error"] = f"{type(e).__name__}: {e}"
            row["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[shard {a.shard}] {k}/{len(units)} {g} {f or ''} {row['seconds']}s{' ERR ' + row['error'] if row['error'] else ''}", flush=True)


def cmd_gate(a) -> None:
    guids = guids_of(Path(a.manifest))
    docs = load_fdocs(set(guids))
    by_doc = sg.read_rows(a.rows)
    tot: Counter = Counter()
    for g in guids:
        if by_doc.get(g):
            tot.update(decode(docs[g], by_doc[g], a.format, False)[1])
    parse = 1 - tot["unparsed"] / max(1, tot["outputs"])
    ids = tot["id_valid"] / max(1, tot["values"])
    res = {"format": a.format, "outputs": tot["outputs"], "parse_rate": round(parse, 4), "values": tot["values"], "id_valid_rate": round(ids, 4),
           "decision": "keep" if parse >= 0.9 and ids >= 0.9 else "switch_to_field_format"}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


# ------------------------------------------------------------------ score
def bros_paths(k: int, fold: int, seed: int) -> Path:
    base = V1 / "bros" / f"fold{fold}" / f"seed_{seed}" if k == 40 else ROOT / "results" / "fatura_v2_sweep" / f"K{k}" / f"fold{fold}" / f"seed_{seed}"
    return base / "test_entities.jsonl"


def cmd_score(a) -> None:
    rd = Path(a.rows_dir)
    recs = {r["guid"]: r for r in map(json.loads, (V1 / "spans.jsonl").read_text(encoding="utf-8").splitlines()) if r["split"] == "confirm"}
    docs = load_fdocs(set(recs))
    fmt = json.loads((rd / "recipe.json").read_text(encoding="utf-8"))["format"]
    arms = {}
    for name in ("A2", "A1"):
        paths = [str(rd / f"rows_{name}_{s}.jsonl") for s in (0, 1)]
        by_doc = sg.read_rows(paths)
        if by_doc:
            arms[name] = {g: decode(docs[g], by_doc[g], fmt)[0] for g in recs if g in by_doc}
    assert arms, "no rows"
    guids = sorted(set.intersection(*[set(v) for v in arms.values()]))
    print(f"{len(guids)} confirm invoices scored; arms {list(arms)}; format {fmt}")
    report: dict = {"n_invoices": len(guids), "arms": list(arms), "format": fmt, "per_k": {}}
    clusters = [recs[g]["template"] for g in guids]
    for k in KS:
        dumps = {}
        for f in range(5):
            for s in SEEDS:
                p = bros_paths(k, f, s)
                if p.exists():
                    dumps[(f, s)] = {r["guid"]: r["pred"] for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
        if len({s for _, s in dumps}) < len(SEEDS) or any((recs[g]["fold"], s) not in dumps or g not in dumps[(recs[g]["fold"], s)] for g in guids for s in SEEDS):
            print(f"K={k}: BROS dumps incomplete, skipped")
            continue
        systems: dict[str, list[dict]] = defaultdict(list)
        for g in guids:
            gold = recs[g]["gold"]
            for s in SEEDS:
                b = dumps[(recs[g]["fold"], s)][g]
                systems[f"bros_seed{s}"].append(span_counts(b, gold))
                for n, sp in arms.items():
                    systems[f"{n}_fill_seed{s}"].append(span_counts(fill_in(b, sp[g]), gold))      # BROS, the LLM fills fields BROS left empty
                    systems[f"{n}_first_seed{s}"].append(span_counts(fill_in(sp[g], b), gold))     # the LLM, BROS fills fields it left empty
            for n, sp in arms.items():
                systems[n].append(span_counts(sp[g], gold))
            systems["old_vlm"].append(span_counts(recs[g]["vlm_both"], gold))
        boot = boot_f1(dict(systems), clusters)
        bros_boot = np.mean([boot[f"bros_seed{s}"] for s in SEEDS], axis=0)
        mf = lambda n: f1_of([sum(c["_micro"][i] for c in systems[n]) for i in range(3)])
        entry = {"bros_f1": round(float(np.mean([mf(f"bros_seed{s}") for s in SEEDS])) * 100, 1)}
        for n in arms:
            d = boot[n] - bros_boot
            entry[n] = {"f1": round(mf(n) * 100, 1), "minus_bros": round(float(d.mean()) * 100, 1), "ci": [round(v * 100, 1) for v in fci(d)]}
            for tag in ("fill", "first"):
                parts = [f"{n}_{tag}_seed{s}" for s in SEEDS]
                dd = np.mean([boot[p] for p in parts], axis=0) - bros_boot
                entry[f"{n}_{tag}"] = {"f1": round(float(np.mean([mf(p) for p in parts])) * 100, 1), "minus_bros": round(float(dd.mean()) * 100, 1),
                                       "ci": [round(v * 100, 1) for v in fci(dd)]}
        report["per_k"][str(k)] = entry
        print(f"K={k:2d}  BROS {entry['bros_f1']:5.1f} | " + " | ".join(f"{n} {entry[n]['f1']:5.1f} ({entry[n]['minus_bros']:+5.1f} {entry[n]['ci']})" for n in arms))
    old = {"old_vlm": [span_counts(recs[g]["vlm_both"], recs[g]["gold"]) for g in guids]} | {n: [span_counts(sp[g], recs[g]["gold"]) for g in guids] for n, sp in arms.items()}
    ob = boot_f1(old, clusters)
    report["vs_old_vlm"] = {n: {"minus": round(float((ob[n] - ob["old_vlm"]).mean()) * 100, 1), "ci": [round(v * 100, 1) for v in fci(ob[n] - ob["old_vlm"])]} for n in arms}
    print("arm minus the study-14 zero-shot VLM (template-cluster CI):", json.dumps(report["vs_old_vlm"]))
    # per template and per field (arms, old VLM), descriptive
    report["old_vlm_f1"] = round(f1_of([sum(span_counts(recs[g]["vlm_both"], recs[g]["gold"])["_micro"][i] for g in guids) for i in range(3)]) * 100, 1)
    report["per_field"] = {n: {f: round(f1_of([sum(span_counts(sp[g], recs[g]["gold"]).get(f, (0, 0, 0))[i] for g in guids) for i in range(3)]) * 100, 1) for f in FIELDS}
                           for n, sp in arms.items()}
    report["template_f1"] = {n: sorted(round(f1_of([sum(span_counts(sp[g], recs[g]["gold"])["_micro"][i] for g in guids if recs[g]["template"] == t) for i in range(3)]) * 100, 1)
                                       for t in sorted({recs[g]["template"] for g in guids})) for n, sp in arms.items()}
    print("old zero-shot VLM (study 14):", report["old_vlm_f1"]); print("per field:", json.dumps(report["per_field"]))
    (rd / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifests")
    sub.add_parser("oracle")
    p = sub.add_parser("run")
    p.add_argument("--manifest", required=True); p.add_argument("--arm", choices=["image", "text"], required=True)
    p.add_argument("--format", choices=["json", "field"], required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct"); p.add_argument("--device", default="cuda:0"); p.add_argument("--dtype", default="float16")
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28); p.add_argument("--quant4", action="store_true")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--deadline-hours", type=float, default=0.0); p.add_argument("--stub", action="store_true")
    p = sub.add_parser("gate")
    p.add_argument("--manifest", required=True); p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--format", choices=["json", "field"], required=True); p.add_argument("--out", default=None)
    sub.add_parser("score").add_argument("--rows-dir", required=True)
    a = ap.parse_args()
    {"manifests": cmd_manifests, "oracle": cmd_oracle, "run": cmd_run, "gate": cmd_gate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
