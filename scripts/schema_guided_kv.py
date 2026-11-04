"""Study 25: schema-guided, grounded VLM extraction on VRDU registration forms (LMDX-style), with a pixel ablation.

Registered in STUDIES.md (study 25) before this code existed.  Pipeline per page:
  OCR lines, each with an identifier XX|YY (centre coordinates in 100 buckets)  ->  prompt with the schema (keys + one-line
  descriptions)  ->  the VLM answers a JSON object whose values are `text identifier`  ->  every value is verified against the
  cited OCR line (exact normalised containment, else dropped as ungrounded) and becomes a word span  ->  one answer per field
  by vote over pages and samples.  Spans use the same word indexing as matched_eval.py, so scoring and the hybrid tools apply.

    python scripts/schema_guided_kv.py selftest
    python scripts/schema_guided_kv.py oracle  --manifest results/zero_shot_kv_v2/manifest.json --ref results/matched_eval/spans.jsonl
    python scripts/schema_guided_kv.py pilot-manifest --out results/schema_guided_kv/pilot_manifest.json
    python scripts/schema_guided_kv.py run   --manifest M --arm image|text --format json|field --out rows.jsonl [--samples K]
    python scripts/schema_guided_kv.py gate  --rows rows.jsonl
    python scripts/schema_guided_kv.py score --manifest M --ref spans.jsonl --rows rows.jsonl --arm A2 --out metrics.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from matched_eval import ci, f1_of, paired_bootstrap, span_counts  # noqa: E402
from zero_shot_kv import norm  # noqa: E402

BUCKETS = 100
MAX_PAGES = 4
SUBCORPUS = "registration-form"
SEG = re.compile(r"^(.*?)\s*(\d{2}\|\d{2}(?:#\d+)?)\s*$")


# ---------------------------------------------------------------- documents: OCR lines, ids, line -> words
class Line:
    def __init__(self, page: int, lid: str, text: str, words: list[int], chars: tuple[int, int] = (-1, -1), raw: str = ""):
        self.page, self.id, self.text, self.words, self.chars, self.raw = page, lid, text, words, chars, raw or text


def coord_id(bbox: list[float]) -> str:
    """Identifier from the line's centre: `XX|YY`, 100 buckets (LMDX: line-level segments, 2 coordinates, B = 100)."""
    q = lambda v: min(BUCKETS - 1, max(0, int(v * BUCKETS)))
    return f"{q((bbox[1] + bbox[3]) / 2):02d}|{q((bbox[2] + bbox[4]) / 2):02d}"


class Doc:
    """One VRDU record: the loader's word stream (same indexing as the BROS dumps) plus identified OCR lines."""

    def __init__(self, record: dict, max_pages: int = MAX_PAGES):
        from data.vrdu_loader import _token_char_range

        self.filename = record["filename"]
        self.words: list[str] = []
        self.page_ids: list[int] = []
        ranges: list[tuple[int, int]] = []
        for page in record["ocr"]["pages"]:
            for tok in page.get("tokens", []):
                t = (tok.get("text") or "").strip()
                if t:
                    self.words.append(t)
                    self.page_ids.append(int(tok["bbox"][0]))
                    ranges.append(_token_char_range(tok))
        self.n_pages = len(record["ocr"]["pages"])
        self.lines: dict[int, list[Line]] = {}
        self.collisions = 0
        for p, page in enumerate(record["ocr"]["pages"][:max_pages]):
            seen: Counter = Counter()
            out = []
            for ln in page.get("lines", []):
                text = ln["text"].strip()
                segs = ln.get("segments") or []
                if not text or not segs:
                    continue
                cs, ce = min(s for s, _ in segs), max(e for _, e in segs)
                words = [i for i, (a, b) in enumerate(ranges) if self.page_ids[i] == p and a >= 0 and a < ce and b > cs]
                lid = coord_id(ln["bbox"])
                seen[lid] += 1
                if seen[lid] > 1:
                    self.collisions += 1
                    lid = f"{lid}#{seen[lid]}"
                out.append(Line(p, lid, text, words, (cs, ce), ln["text"]))
            self.lines[p] = out
        self.char_ranges = ranges
        self.ocr_text = record["ocr"].get("text", "")


def load_records(root: Path, filenames: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with gzip.open(root / SUBCORPUS / "main" / "dataset.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["filename"] in filenames:
                out[r["filename"]] = r
                if len(out) == len(filenames):
                    break
    return out


# ---------------------------------------------------------------- prompts
def build_prompt(lines: list[Line], fields: list[str], desc: dict[str, str]) -> str:
    doc = "\n".join(f"{ln.text} {ln.id}" for ln in lines)
    descs = "\n".join(f"- {f}: {desc[f]}" for f in fields if desc.get(f))
    schema = "{\n" + ",\n".join(f'  "{f}": ""' for f in fields) + "\n}"
    return (f"<Document>\n{doc}\n</Document>\n<Task>\nFrom the document, extract the text values of the following fields. "
            f"Every document line ends with its identifier (two numbers separated by |). Write each value as the exact text "
            f"copied from the line, followed by that line's identifier. A value that spans several lines is written as one "
            f"'text identifier' per line, separated by a newline. Use null for a field that does not occur on this page.\n"
            f"Field descriptions:\n{descs}\nAnswer with this JSON object:\n{schema}\n</Task>\n<Extraction>\n")


def field_prompt(lines: list[Line], field: str, desc: dict[str, str]) -> str:
    doc = "\n".join(f"{ln.text} {ln.id}" for ln in lines)
    return (f"<Document>\n{doc}\n</Document>\n<Task>\nFrom the document, extract the value of the field \"{field}\"" +
            (f" ({desc[field]})" if desc.get(field) else "") +
            ". Every document line ends with its identifier. Answer with the exact text copied from the line, followed by that "
            "line's identifier. If the field does not occur on this page, answer NONE.\n</Task>\nANSWER: ")


FIELD_PREFILL = "ANSWER: "


# ---------------------------------------------------------------- parsing, grounding, voting
def _segments(value) -> list[tuple[str, str | None]] | None:
    items = value if isinstance(value, list) else [value]
    segs = []
    for item in items:
        for part in str(item).split("\n"):
            part = part.strip()
            if part and part.upper() != "NONE":
                m = SEG.match(part)
                segs.append((m.group(1).strip(), m.group(2)) if m else (part, None))
    return segs or None


def parse_json(raw: str, fields: list[str]) -> dict[str, list | None] | None:
    s = re.sub(r"```(?:json)?", "", raw).strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b < a:
        return None
    try:
        obj = json.loads(s[a:b + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return {f: (None if obj.get(f) in (None, "", []) else _segments(obj[f])) for f in fields}


def parse_field(raw: str) -> list | None:
    text = raw.split("ANSWER:", 1)[-1].strip().splitlines()
    return _segments("\n".join(text)) if text else None


def _line_span(text: str, ln: Line, doc: Doc) -> tuple[int, int] | None:
    """Words of `ln` covered by `text`: exact (whitespace-insensitive) match in the line, mapped by character overlap."""
    toks = text.split()
    if not toks:
        return None
    m = re.search(r"\s+".join(re.escape(t) for t in toks), ln.raw)
    if not m:
        return None
    a, b = ln.chars[0] + m.start(), ln.chars[0] + m.end()
    ws = [w for w in ln.words if doc.char_ranges[w][0] < b and doc.char_ranges[w][1] > a]
    return (min(ws), max(ws) + 1) if ws else None


def _line_span_lenient(text: str, ln: Line, doc: Doc) -> tuple[int, int] | None:
    """Fallback: the same text after the matched_eval normalisation (case, punctuation, spacing)."""
    vt = norm(text).split()
    toks = [(t, w) for w in ln.words for t in norm(doc.words[w]).split()]
    pos = next((i for i in range(len(toks) - len(vt) + 1) if [t for t, _ in toks[i:i + len(vt)]] == vt), None) if vt else None
    return None if pos is None else (toks[pos][1], toks[pos + len(vt) - 1][1] + 1)


def ground(segs: list[tuple[str, str | None]], lines_by_id: dict[str, Line], doc: Doc) -> tuple[tuple[int, int] | None, str]:
    """Verify every segment against its cited line; return (word span, how) or (None, reason).

    A segment is grounded when its text occurs in the cited line (exactly, else after normalisation).  Segments are merged
    while they are contiguous; a later disjoint segment is ignored, because the gold alignment keeps the first contiguous run."""
    lo = hi = None
    how = "exact"
    for text, lid in segs:
        ln = lines_by_id.get(lid) if lid else None
        if ln is None:
            return None, "unknown_id"
        if not norm(text).split():
            return None, "empty"
        span = _line_span(text, ln, doc)
        if span is None:
            span = _line_span_lenient(text, ln, doc)
            if span is None:
                return None, "not_in_line"
            how = "lenient"
        if lo is None:
            lo, hi = span
        elif span[0] <= hi and span[1] >= lo:
            lo, hi = min(lo, span[0]), max(hi, span[1])
        else:
            how = "disjoint_truncated" if how == "exact" else how
    return ((lo, hi), how) if lo is not None else (None, "empty")


def decode_doc(doc: Doc, rows: list[dict], fmt: str, fields: list[str]) -> tuple[list[list], Counter]:
    """Spans [field, start, end, text] for one document from its page rows, plus parse/grounding counters."""
    stats: Counter = Counter()
    votes: dict[str, Counter] = defaultdict(Counter)
    first_page: dict[tuple[str, tuple[int, int]], int] = {}
    for r in rows:
        by_id = {ln.id: ln for ln in doc.lines.get(r["page"], [])}
        for raw in r["samples"]:
            if fmt == "json":
                parsed = parse_json(raw, fields)
                stats["outputs"] += 1
                if parsed is None:
                    stats["unparsed"] += 1
                    continue
                answers = parsed
            else:
                answers = {r["field"]: parse_field(raw)}
                stats["outputs"] += 1
            for f, segs in answers.items():
                if not segs:
                    stats["null"] += 1
                    continue
                stats["values"] += 1
                span, why = ground(segs, by_id, doc)
                if span is None:
                    stats[f"ungrounded_{why}"] += 1
                    continue
                stats["grounded"] += 1
                stats[f"grounded_{why}"] += 1
                votes[f][span] += 1
                first_page.setdefault((f, span), r["page"])
    chosen = []
    for f, c in votes.items():
        span = max(c, key=lambda s: (c[s], -first_page[(f, s)], -s[0]))
        chosen.append((c[span], f, span))
    used: set[int] = set()
    spans = []
    for n, f, (a, b) in sorted(chosen, key=lambda x: (-x[0], fields.index(x[1]))):
        if used.intersection(range(a, b)):
            stats["dropped_overlap"] += 1
            continue
        used.update(range(a, b))
        spans.append([f, a, b, " ".join(doc.words[a:b])])
    return spans, stats


# ---------------------------------------------------------------- manifests, runs
def vrdu_manifest_docs(path: str) -> list[dict]:
    return [d for d in json.loads(Path(path).read_text(encoding="utf-8")) if d["dataset"] == SUBCORPUS]


def cmd_pilot_manifest(args) -> None:
    from zero_shot_kv_v2 import build_manifest, fresh_sample  # noqa: F401

    exclude = {d["doc_id"].split("/", 1)[1] for p in args.exclude for d in json.loads(Path(p).read_text(encoding="utf-8"))
               if d["dataset"] == SUBCORPUS}
    docs = build_manifest(None, ROOT / "data" / "vrdu", args.n_per_template, args.seed, MAX_PAGES,
                          subcorpora=(SUBCORPUS,), exclude=exclude)
    assert not {d["doc_id"].split("/", 1)[1] for d in docs} & exclude, "pilot overlaps dev/confirm"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    print(f"pilot manifest: {len(docs)} documents, templates {dict(Counter(d['template'] for d in docs))}")


def _vlm_class():
    from zero_shot_kv_v2 import VLM

    class SamplingVLM(VLM):
        def sample(self, prompt: str, image, k: int, max_new_tokens: int, prefill: str = "") -> list[str]:
            import torch

            content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
            text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                 add_generation_prompt=True) + prefill
            kw = {"text": [text], "return_tensors": "pt"}
            if image is not None:
                kw["images"] = [image]
            inputs = self.proc(**kw).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=k > 1, temperature=0.5, top_k=40,
                                          num_return_sequences=k)
            return [t for t in self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]

    return SamplingVLM


def cmd_run(args) -> None:
    from zero_shot_kv import load_image
    from zero_shot_kv_v2 import schema_fields

    docs = vrdu_manifest_docs(args.manifest)
    fields = schema_fields(SUBCORPUS)
    desc = json.loads((ROOT / "results" / "zero_shot_kv_v2" / "schema_desc.json").read_text(encoding="utf-8"))[SUBCORPUS]
    records = load_records(ROOT / "data" / "vrdu", {d["doc_id"].split("/", 1)[1] for d in docs})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["doc_id"], r["page"], r.get("field")) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines() if out.stat().st_size else [])}
    units = [(d, p, f) for d in docs for p in range(min(d["n_pages"], MAX_PAGES)) for f in ([None] if args.format == "json" else fields)]
    units = [u for u in units[args.shard::args.num_shards] if (u[0]["doc_id"], u[1], u[2]) not in done]
    if args.limit:
        units = units[:args.limit]
    print(f"[shard {args.shard}] {len(units)} units to do on {args.device} (arm {args.arm}, format {args.format}, K={args.samples})", flush=True)
    if not units:
        return
    class _Stub:  # local test of the run loop without a GPU: always answers "no field on this page"
        def sample(self, prompt, image, k, max_new_tokens, prefill=""):
            return ['{"file_date": null}' if "JSON" in prompt else "ANSWER: NONE"] * k

    gen = _Stub() if args.stub else _vlm_class()(args.model, args.device, args.dtype, args.max_pixels)
    deadline = time.time() + args.deadline_hours * 3600 if args.deadline_hours else None
    cache: dict[str, Doc] = {}
    with open(out, "a", encoding="utf-8") as fh:
        for k, (d, p, f) in enumerate(units, 1):
            if deadline and time.time() > deadline:
                print(f"[shard {args.shard}] deadline reached after {k - 1} units; stopping cleanly (re-run to resume)", flush=True)
                break
            fn = d["doc_id"].split("/", 1)[1]
            doc = cache.setdefault(fn, Doc(records[fn]))
            row = {"doc_id": d["doc_id"], "page": p, "field": f, "arm": args.arm, "format": args.format, "samples": [], "error": None}
            t0 = time.time()
            try:
                lines = doc.lines.get(p, [])
                image = load_image({"image": d["source"], "pdf_page": p}) if args.arm == "image" else None
                prompt = build_prompt(lines, fields, desc) if f is None else field_prompt(lines, f, desc)
                row["samples"] = gen.sample(prompt, image, args.samples, args.max_new_tokens if f is None else 64,
                                            prefill="" if f is None else "")
            except Exception as e:  # one bad page must not kill a multi-hour run
                row["error"] = f"{type(e).__name__}: {e}"
            row["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[shard {args.shard}] {k}/{len(units)} {fn[:48]} p{p} {f or ''} {row['seconds']}s{' ERR ' + row['error'] if row['error'] else ''}", flush=True)


def read_rows(paths: list[str]) -> dict[str, list[dict]]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        if Path(p).exists():
            for line in Path(p).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if not r.get("error"):
                        by_doc[r["doc_id"]].append(r)
    return by_doc


def cmd_gate(args) -> None:
    from zero_shot_kv_v2 import schema_fields

    fields = schema_fields(SUBCORPUS)
    by_doc = read_rows(args.rows)
    docs = vrdu_manifest_docs(args.manifest)
    records = load_records(ROOT / "data" / "vrdu", {d["doc_id"].split("/", 1)[1] for d in docs})
    total: Counter = Counter()
    for d in docs:
        rows = by_doc.get(d["doc_id"], [])
        if rows:
            _, st = decode_doc(Doc(records[d["doc_id"].split("/", 1)[1]]), rows, args.format, fields)
            total.update(st)
    parse = 1 - total["unparsed"] / max(1, total["outputs"])
    ground_rate = total["grounded"] / max(1, total["values"])
    res = {"format": args.format, "outputs": total["outputs"], "parse_rate": round(parse, 4), "values": total["values"],
           "ground_rate": round(ground_rate, 4), "counters": dict(total),
           "decision": "keep" if (parse >= 0.9 and ground_rate >= 0.9) else "switch_to_field_format"}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


# ---------------------------------------------------------------- scoring against BROS and the reference VLM spans
def cmd_score(args) -> None:
    from zero_shot_kv_v2 import schema_fields

    fields = schema_fields(SUBCORPUS)
    ref = [json.loads(l) for l in Path(args.ref).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_doc = read_rows(args.rows)
    manifest = {d["doc_id"].split("/", 1)[1]: d for d in vrdu_manifest_docs(args.manifest)}
    records = load_records(ROOT / "data" / "vrdu", set(manifest))
    name = f"vlm_{args.arm}"
    keep, stats = [], Counter()
    for r in ref:
        d = manifest.get(r["guid"])
        rows = by_doc.get(d["doc_id"], []) if d else []
        need = min(d["n_pages"], MAX_PAGES) * (1 if args.format == "json" else len(fields)) if d else 0
        if not d or len(rows) < need:
            stats["docs_incomplete"] += 1
            continue
        spans, st = decode_doc(Doc(records[r["guid"]]), rows, args.format, fields)
        keep.append({**r, name: spans})
        stats.update(st)
    systems = ["bros_seed13", "bros_seed42", args.reference, name]
    per_fold: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in keep:
        for s in systems:
            per_fold[int(r["fold"])][s].append(span_counts(r[s], r["gold"]))
    pooled = {s: [x for f in per_fold.values() for x in f[s]] for s in systems}
    report: dict = {"arm": args.arm, "format": args.format, "reference": args.reference, "n_docs": len(keep),
                    "docs_incomplete": stats["docs_incomplete"], "decode": dict(stats), "folds": {}, "pooled": {}}
    fields_f1 = lambda v, f: round(f1_of([sum(x.get(f, (0, 0, 0))[i] for x in v) for i in range(3)]), 4)
    for scope, data in [(f"fold{f}_{next(r['template'] for r in keep if int(r['fold']) == f)}", per_fold[f]) for f in sorted(per_fold)] + [("pooled", pooled)]:
        boot = paired_bootstrap(data, "_micro")
        bros = (boot["bros_seed13"] + boot["bros_seed42"]) / 2
        out = {"n_docs": len(data[name]),
               "f1": {s: round(f1_of([sum(x["_micro"][i] for x in data[s]) for i in range(3)]), 4) for s in systems},
               "f1_ci95": {s: ci(boot[s]) for s in systems},
               "new_minus_reference_ci95": ci(boot[name] - boot[args.reference]),
               "new_minus_reference_mean": round(float(np.mean(boot[name] - boot[args.reference])), 4),
               "new_minus_bros_ci95": ci(boot[name] - bros),
               "per_field_f1": {f: {s: fields_f1(data[s], f) for s in (name, args.reference, "bros_seed13")} for f in fields}}
        (report["pooled"] if scope == "pooled" else report["folds"]).update({} if scope != "pooled" else out)
        if scope != "pooled":
            report["folds"][scope] = out
    pm = report["pooled"]
    lower, mean = pm["new_minus_reference_ci95"][0], pm["new_minus_reference_mean"]
    fold_upper = [v["new_minus_reference_ci95"][1] for v in report["folds"].values()]
    no_worse = all(u >= 0 for u in fold_upper)
    report["verdict"] = {"strong_ge_5": bool(mean * 100 >= 5.0 and lower > 0 and no_worse),
                         "weaker_ge_3": bool(mean * 100 >= 3.0 and lower > 0 and no_worse),
                         "not_worse_than_bros_by_more_than_3": bool(pm["f1"][name] * 100 >= np.mean([pm["f1"]["bros_seed13"], pm["f1"]["bros_seed42"]]) * 100 - 3.0),
                         "competitive_with_bros": bool(abs(pm["f1"][name] - np.mean([pm["f1"]["bros_seed13"], pm["f1"]["bros_seed42"]])) * 100 <= 3.0
                                                       or pm["new_minus_bros_ci95"][0] <= 0 <= pm["new_minus_bros_ci95"][1]),
                         "delta_pooled_f1_points": round(mean * 100, 2), "no_template_significantly_worse": no_worse}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.out).with_name(f"spans_{args.arm}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    print(json.dumps({"n_docs": report["n_docs"], "pooled_f1": pm["f1"], "delta": report["verdict"], "decode": report["decode"]}, indent=1))


# ---------------------------------------------------------------- oracle check and self-test
def gold_occurrences(record: dict, doc: Doc) -> list[tuple[str, list[tuple[str, str, int]]]]:
    """(field, [(text on line, line id, page), ...]) for every annotated occurrence, from the character offsets of the gold."""
    out = []
    flat = [ln for lines in doc.lines.values() for ln in lines]
    for name, occs in record["annotations"]:
        if not isinstance(name, str):
            continue
        for occ in occs:
            segs = []
            for cs, ce in (occ[2] if len(occ) > 2 else []):
                for ln in flat:
                    a, b = max(cs, ln.chars[0]), min(ce, ln.chars[1])
                    if a < b and doc.ocr_text[a:b].strip():
                        segs.append((doc.ocr_text[a:b].strip(), ln.id, ln.page))
            if segs:
                out.append((name, segs))
    return out


def cmd_oracle(args) -> None:
    ref = {r["guid"]: r for r in map(json.loads, Path(args.ref).read_text(encoding="utf-8").splitlines()) if r}
    manifest = [d for d in vrdu_manifest_docs(args.manifest) if d["doc_id"].split("/", 1)[1] in ref][: args.n]
    records = load_records(ROOT / "data" / "vrdu", {d["doc_id"].split("/", 1)[1] for d in manifest})
    tot, why, fail = [0, 0, 0], Counter(), []
    for d in manifest:
        fn = d["doc_id"].split("/", 1)[1]
        doc = Doc(records[fn])
        pred = []
        for name, segs in gold_occurrences(records[fn], doc):
            by_id = {ln.id: ln for p in {s[2] for s in segs} for ln in doc.lines[p]}
            span, reason = ground([(t, i) for t, i, _ in segs], by_id, doc)
            why[reason] += 1
            if span:
                pred.append([name, span[0], span[1], ""])
        gold = [g for g in ref[fn]["gold"]]
        c = span_counts(pred, gold)["_micro"]
        tot = [a + b for a, b in zip(tot, c)]
        if c[0] < c[2]:
            fail.append(fn)
    res = {"docs": len(manifest), "oracle_f1": round(f1_of(tot), 4), "tp_fp_fn": tot, "grounding_reasons": dict(why),
           "collisions": None, "docs_with_misses": len(fail), "example_misses": fail[:5]}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    if res["oracle_f1"] < args.min_f1:
        raise SystemExit(f"oracle grounding F1 {res['oracle_f1']} < {args.min_f1}: the line -> word mapping is wrong")


def _toy_record() -> dict:
    """Two pages; page 0 has lines 'FILE DATE' / 'June 5, 2020' and a colliding pair of 'Total' lines."""
    text = "FILE DATE\nJune 5, 2020\nTotal 1\nTotal 2\nSigned by John Q Public\n( Smith )\n"
    def toks(words, page, y, line_text):
        out, pos = [], text.index(line_text)
        for k, w in enumerate(words):
            pos = text.index(w, pos)
            out.append({"text": w, "bbox": [page, 0.1 + 0.1 * k, y, 0.18 + 0.1 * k, y + 0.02], "segments": [[pos, pos + len(w)]]})
            pos += len(w)
        return out
    l0 = ["FILE", "DATE"]; l1 = ["June", "5,", "2020"]; l2 = ["Total", "1"]; l3 = ["Total", "2"]; l4 = ["Signed", "by", "John", "Q", "Public"]
    lines = lambda spans, page, ys: [{"text": t + "\n", "bbox": [page, 0.1, y, 0.5, y + 0.02], "segments": [[text.index(t), text.index(t) + len(t) + 1]]}
                                     for t, y in zip(spans, ys)]
    page0_lines = lines(["FILE DATE", "June 5, 2020", "Total 1"], 0, [0.1, 0.2, 0.4]) + \
        [{"text": "Total 2\n", "bbox": [0, 0.1, 0.4, 0.5, 0.42], "segments": [[text.index("Total 2"), text.index("Total 2") + 8]]}]
    p0 = {"tokens": toks(l0, 0, 0.1, "FILE DATE") + toks(l1, 0, 0.2, "June 5, 2020") + toks(l2, 0, 0.4, "Total 1")
          + toks(l3, 0, 0.4, "Total 2"), "lines": page0_lines}
    p1 = {"tokens": toks(l4, 1, 0.3, "Signed by John Q Public") + toks(["(", "Smith", ")"], 1, 0.5, "( Smith )"),
          "lines": lines(["Signed by John Q Public", "( Smith )"], 1, [0.3, 0.5])}
    return {"filename": "toy.pdf", "ocr": {"text": text, "pages": [p0, p1]}, "annotations": []}


def cmd_selftest(_args) -> None:
    assert coord_id([0, 0.10, 0.20, 0.30, 0.22]) == "20|21" and coord_id([0, 0.99, 0.99, 1.0, 1.0]) == "99|99"
    doc = Doc(_toy_record())
    ids = [ln.id for ln in doc.lines[0]]
    assert len(set(ids)) == len(ids) and doc.collisions == 1 and ids[3].endswith("#2"), ids          # colliding ids get a suffix
    by0 = {ln.id: ln for ln in doc.lines[0]}
    by1 = {ln.id: ln for ln in doc.lines[1]}
    date_id = doc.lines[0][1].id
    assert ground([("June 5, 2020", date_id)], by0, doc)[0] == (2, 5)
    assert ground([("jun 5 2020", date_id)], by0, doc) == (None, "not_in_line")                        # a hallucinated value is rejected
    assert ground([("June 5, 2020", "99|99")], by0, doc) == (None, "unknown_id")
    assert ground([("June", date_id)], by0, doc)[0] == (2, 3)                                         # a sub-span of a line
    assert ground([("Total 1", doc.lines[0][2].id)], by0, doc)[0] == (5, 7) and ground([("Total 2", doc.lines[0][3].id)], by0, doc)[0] == (7, 9)
    assert ground([("FILE DATE", ids[0]), ("June 5, 2020", date_id)], by0, doc)[0] == (0, 5)            # two lines -> one covering span
    smith_id = doc.lines[1][1].id
    assert ground([("( Smith )", smith_id)], by1, doc)[0] == (14, 17)                                    # punctuation-only words are covered
    assert ground([("Smith", smith_id)], by1, doc)[0] == (15, 16)
    assert ground([("FILE DATE", ids[0]), ("Total 1", doc.lines[0][2].id)], by0, doc) == ((0, 2), "disjoint_truncated")  # first run only
    assert ground([("jun 5 2020", date_id)], by0, doc) == (None, "not_in_line")
    f = ["file_date", "signer_name"]
    raw = '```json\n{"file_date": "June 5, 2020 %s", "signer_name": null}\n```' % date_id
    assert parse_json(raw, f) == {"file_date": [("June 5, 2020", date_id)], "signer_name": None}
    assert parse_json("no json here", f) is None and parse_json('{"file_date": "x"', f) is None
    assert parse_json('{"file_date": ["A 10|20", "B 11|22"]}', f)["file_date"] == [("A", "10|20"), ("B", "11|22")]
    assert parse_field("ANSWER: June 5, 2020 %s" % date_id) == [("June 5, 2020", date_id)] and parse_field("ANSWER: NONE") is None
    # voting: page 0 says June 5, 2020 twice (two samples), page 1 says a different value once -> majority wins
    rows = [{"page": 0, "samples": ['{"file_date": "June 5, 2020 %s"}' % date_id] * 2},
            {"page": 1, "samples": ['{"file_date": "Signed by John Q Public %s"}' % doc.lines[1][0].id]}]
    spans, st = decode_doc(doc, rows, "json", f)
    assert spans == [["file_date", 2, 5, "June 5, 2020"]] and st["grounded"] == 3 and st["outputs"] == 3, (spans, st)
    rows = [{"page": 0, "samples": ['{"file_date": "made up 00|00"}', "oops"]}]
    spans, st = decode_doc(doc, rows, "json", f)
    assert spans == [] and st["unparsed"] == 1 and st["ungrounded_unknown_id"] == 1
    # overlap rule: a lower-voted field overlapping a better-voted one is dropped
    rows = [{"page": 0, "samples": ['{"file_date": "June 5, 2020 %s", "signer_name": "June 5, 2020 %s"}' % (date_id, date_id)]}]
    spans, st = decode_doc(doc, rows, "json", f)
    assert len(spans) == 1 and st["dropped_overlap"] == 1
    assert "June 5, 2020" in build_prompt(doc.lines[0], f, {"file_date": "the filing date"}) and date_id in build_prompt(doc.lines[0], f, {})
    print("selftest passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("oracle")
    p.add_argument("--manifest", required=True); p.add_argument("--ref", required=True)
    p.add_argument("--n", type=int, default=60); p.add_argument("--min-f1", type=float, default=0.98); p.add_argument("--out", default=None)
    p = sub.add_parser("pilot-manifest")
    p.add_argument("--exclude", nargs="+", default=["results/zero_shot_kv_v2/manifest.json", "results/confirm_v2/manifest.json"])
    p.add_argument("--n-per-template", type=int, default=10); p.add_argument("--seed", type=int, default=7); p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--arm", choices=["image", "text"], default="image"); p.add_argument("--format", choices=["json", "field"], default="json")
    p.add_argument("--samples", type=int, default=1); p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--device", default="cuda:0"); p.add_argument("--dtype", default="float16")
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28); p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--deadline-hours", type=float, default=0.0)
    p.add_argument("--stub", action="store_true", help="no model: exercise the run loop only (tests)")
    p = sub.add_parser("gate")
    p.add_argument("--manifest", required=True); p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--format", choices=["json", "field"], default="json"); p.add_argument("--out", default=None)
    p = sub.add_parser("score")
    p.add_argument("--manifest", required=True); p.add_argument("--ref", required=True); p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--arm", required=True); p.add_argument("--format", choices=["json", "field"], default="json")
    p.add_argument("--reference", default="vlm_both"); p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"selftest": cmd_selftest, "oracle": cmd_oracle, "pilot-manifest": cmd_pilot_manifest, "run": cmd_run, "gate": cmd_gate,
     "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
