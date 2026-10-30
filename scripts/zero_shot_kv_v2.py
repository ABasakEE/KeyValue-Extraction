"""Zero-shot open-schema KV extraction, v2: whole documents + an LM naming step.

Changes from v1 (scripts/zero_shot_kv.py, kept as is for reproducibility):

* VLM only (it beat the OCR-text arm on every comparison in v1).
* Whole documents: every page up to --max-pages (4 pages cover 100% of registration
  and 97% of ad-buy gold values; page 1 alone covered 60% / 46%). Same documents as v1.
* Longer generation (2048 tokens; v1's 768 truncated 75% of ad-buy pages) and the
  prompt asks the model to skip empty fields.
* Naming step: the same model, text-only, one short prompt per pair (batched per page), answers
  <field>snake_case</field><desc>...</desc><value>...</value> (one line per piece),
  splitting combined values (name + address, date ranges).
* Schema descriptions: the model describes each VRDU field from the list of field
  names alone (no document or template text), so the schema side is template-blind.
* Scoring in two modes, both with stratified bootstrap 95% CIs over documents:
  ``key``   predicted key text  vs field name             (v1 behaviour, whole document)
  ``named`` predicted field+desc vs field name+description (the naming step)

Subcommands: prepare, describe-schema, run, score, selftest.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from zero_shot_kv import (
    FIELD_SIM_THRESHOLD, MATCH_THRESHOLD, PROMPT, ROOT, bootstrap_f1_ci, funsd_pages, greedy_match, load_image,
    norm, page_level_fields, parse_pairs, prf, sbert_embedder, short_form_filer, tok_f1, vrdu_pages,
)

EXTRACT_PROMPT = PROMPT + "\nDo not output keys whose value is empty or blank."
# One pair per prompt, a worked example, and the answer prefilled with NAMING_PREFILL: on Kaggle the 3B model
# ignored a many-pairs tagged format, then answered a placeholder-only per-pair prompt with untagged lines
# ("invoice_number_invoice_number\nInvoice number is ..."). The example is deliberately unrelated to the test data.
NAMING_PROMPT = (
    "A key-value pair was extracted from a document. Give its field a short canonical name in snake_case and a "
    "one-sentence description of the information it holds, in your own words. If the value combines separate pieces "
    "of information, write one line per piece with the part of the value it covers.\n\n"
    "Example\nKey: Ship To\nValue: Jane Smith, 12 Oak Road, Austin TX\nAnswer:\n"
    "<field>recipient_name</field><desc>Name of the person receiving the shipment.</desc><value>Jane Smith</value>\n"
    "<field>recipient_address</field><desc>Postal address of the recipient.</desc><value>12 Oak Road, Austin TX</value>\n\n"
    "Now this pair\nKey: {key}\nValue: {value}\nAnswer:"
)
NAMING_PREFILL = "<field>"
DESC_RE = re.compile(r"<desc>(.*?)(?:</desc>|$)", re.S)


# ---------------------------------------------------------------- prompts & parsing
def naming_prompt(key: str, value: str) -> str:
    return NAMING_PROMPT.format(key=key, value=value)


def _tag(block: str, name: str) -> str:
    m = re.search(rf"<{name}>(.*?)</{name}>", block, re.S)
    return m.group(1).strip() if m else ""


def _snake(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def _clean_field(name: str) -> str:
    """snake_case, minus echoed placeholders ('x_snake_case_name') and doubled names ('a_b_a_b'), max 8 words."""
    parts = [p for p in _snake(re.sub(r"<[^>]+>", " ", name)).split("_") if p]
    while parts[-2:] == ["case", "name"] or parts[-2:] == ["snake", "case"] or parts[-1:] == ["snake"]:
        parts = parts[:-1] if parts[-1:] == ["snake"] else parts[:-2]
    half = len(parts) // 2
    if len(parts) % 2 == 0 and parts[:half] == parts[half:]:
        parts = parts[:half]
    return "_".join(parts[:8])


def _first_line(text: str) -> str:
    return next((" ".join(re.sub(r"<[^>]+>", " ", l).split()) for l in text.splitlines() if l.strip()), "")


def parse_named(text: str, pair_value: str) -> list[list[str]]:
    """[field, desc, value] lines for one pair; value falls back to the pair's value.

    Each line starts at <field>. Tolerated drift (all seen on Kaggle): a missing </field> or no tags at all, in
    which case the first line is the name and the next line the description.
    """
    blocks = text.split("<field>")[1:] if "<field>" in text else [text]
    out = []
    for block in blocks:
        if "</field>" in block:
            field, rest = block.split("</field>", 1)
        elif "<desc>" in block:
            field, rest = block.split("<desc>", 1)
            rest = "<desc>" + rest
        else:
            lines = [l for l in block.splitlines() if l.strip()]
            field, rest = (lines[0] if lines else ""), "\n".join(lines[1:])
        field = _clean_field(field)
        if field:
            desc = _tag(rest, "desc") or ("" if "<desc>" in rest else _first_line(rest))
            out.append([field, desc, _tag(rest, "value") or pair_value])
    return out


def field_desc_prompt(fields: list[str], field: str) -> str:
    # one field per call: a 3B model did not follow a multi-field tagged list (0 parsed on Kaggle)
    return (f"A document data-extraction schema has these fields: {', '.join(fields)}. In one sentence, describe "
            f"the information that the field '{field}' holds on a document. Output only: <desc>DESCRIPTION</desc>")


def parse_desc(text: str, field: str) -> str:
    """Text inside <desc> (closing tag optional); else the first non-empty line of the answer."""
    m = DESC_RE.search(text)
    desc = m.group(1) if m else next((l for l in text.splitlines() if l.strip()), "")
    desc = re.sub(r"<[^>]+>", " ", desc)
    desc = re.sub(rf"^\W*{re.escape(field)}\W*", "", desc.strip(), flags=re.I)  # drop an echoed "field_name:"
    return " ".join(desc.split()).strip("\"' ")[:300]


def schema_fields(dataset: str, docs: list[dict] | None = None) -> list[str]:
    meta = ROOT / "data" / "vrdu" / dataset / "main" / "meta.json"
    if meta.exists():
        return page_level_fields(meta)
    return sorted({f for d in docs or [] if d["dataset"] == dataset for f in d["gold"]["fields"]})


# ---------------------------------------------------------------- manifest
def fresh_sample(root: Path, subcorpus: str, n_per_template: int, seed: int, exclude: set[str]) -> dict[str, str]:
    """filename -> template for a new per-template sample that avoids every filename in ``exclude``."""
    from data.vrdu_loader import list_templates, template_from_filename

    templates = list_templates(root / subcorpus / "few_shot-splits")
    by_t: dict[str, list[str]] = defaultdict(list)
    with gzip.open(root / subcorpus / "main" / "dataset.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            fn = json.loads(line)["filename"]
            t = template_from_filename(fn, templates)
            if t and fn not in exclude:
                by_t[t].append(fn)
    rng = random.Random(seed)
    return {fn: t for t in sorted(by_t) for fn in rng.sample(sorted(by_t[t]), min(n_per_template, len(by_t[t])))}


def vrdu_docs(root: Path, subcorpus: str, n_per_template: int, seed: int, max_pages: int,
              exclude: set[str] | None = None) -> list[dict]:
    """Whole-document records: the documents v1 sampled (same seed and sampling), or, given ``exclude``,
    a fresh sample that avoids those filenames (confirmation runs)."""
    if exclude is not None:
        sampled = fresh_sample(root, subcorpus, n_per_template, seed, exclude)
    else:
        sampled = {p["page_id"].split("/", 1)[1]: p["template"] for p in vrdu_pages(root, subcorpus, n_per_template, seed)}
    docs = []
    with gzip.open(root / subcorpus / "main" / "dataset.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["filename"] not in sampled:
                continue
            n_pages = len(r["ocr"]["pages"])
            read = min(n_pages, max_pages)
            pages = [{"page": i, "lines": [[ln["text"].strip(), ln["bbox"][1], ln["bbox"][2]]
                                           for ln in r["ocr"]["pages"][i]["lines"] if ln["text"].strip()]}
                     for i in range(read)]
            fields: dict[str, list[str]] = defaultdict(list)
            for name, occs in r["annotations"]:
                if not isinstance(name, str):
                    continue  # ad-buy line-item tables: out of scope, as in v1
                for occ in occs:
                    if occ[1] and int(occ[1][0]) < read and occ[0].strip():
                        fields[name].append(occ[0].strip())
            t = sampled[r["filename"]]
            doc = {"doc_id": f"{subcorpus}/{r['filename']}", "dataset": subcorpus, "template": t,
                   "source": str(root / subcorpus / "main" / "pdfs" / r["filename"]), "n_pages": n_pages,
                   "pages": pages, "gold": {"fields": dict(fields)}}
            if t == "Short-Form":
                doc["gold"]["decoy"] = short_form_filer(r["filename"])
            docs.append(doc)
    return docs


def build_manifest(funsd_root: Path | None, vrdu_root: Path, n_per_template: int, seed: int, max_pages: int,
                   subcorpora: tuple[str, ...] = ("registration-form", "ad-buy-form"),
                   exclude: set[str] | None = None) -> list[dict]:
    docs = []
    if funsd_root and funsd_root.exists():
        for p in funsd_pages(funsd_root):
            docs.append({"doc_id": p["page_id"], "dataset": "funsd", "template": "funsd", "source": p["image"],
                         "n_pages": 1, "pages": [{"page": None, "lines": p["lines"]}], "gold": p["gold"]})
    for sub in subcorpora:
        if (vrdu_root / sub / "main").exists():
            docs += vrdu_docs(vrdu_root, sub, n_per_template, seed, max_pages, exclude)
    # deterministic shuffle: a run stopped by the deadline still covers every template evenly
    docs.sort(key=lambda d: d["doc_id"])
    random.Random(seed).shuffle(docs)
    return docs


# ---------------------------------------------------------------- generation
class VLM:
    def __init__(self, model_id: str, device: str, dtype: str, max_pixels: int):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), attn_implementation="sdpa").to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
        self.proc.tokenizer.padding_side = "left"  # batched generation needs left padding

    def batch_text(self, prompts: list[str], max_new_tokens: int, prefill: str = "",
                   batch_size: int = 16) -> list[tuple[str, int]]:
        """Text-only prompts generated in padded batches; returns (prefill + text, generated token count) per prompt.

        ``prefill`` is written as the start of the assistant's answer, so the model continues in the given format.
        """
        import torch

        pad = self.proc.tokenizer.pad_token_id
        out_all = []
        for s in range(0, len(prompts), batch_size):
            texts = [self.proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": p}]}],
                                                   tokenize=False, add_generation_prompt=True) + prefill
                     for p in prompts[s:s + batch_size]]
            inputs = self.proc(text=texts, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            new = out[:, inputs["input_ids"].shape[1]:]
            texts_out = self.proc.batch_decode(new, skip_special_tokens=True)
            out_all += [(prefill + t, int((row != pad).sum())) for t, row in zip(texts_out, new)]
        return out_all

    def __call__(self, prompt: str, image=None, max_new_tokens: int = 1024) -> tuple[str, int]:
        import torch

        content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True)
        kwargs = {"text": [text], "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = [image]
        inputs = self.proc(**kwargs).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.proc.batch_decode(new, skip_special_tokens=True)[0], int(new.shape[1])


def process_unit(gen, doc: dict, page: dict, max_new_tokens: int, naming_max_new_tokens: int) -> dict:
    t0 = time.time()
    row = {"doc_id": doc["doc_id"], "page": page["page"], "raw": "", "pairs": [], "extract_tokens": 0,
           "naming_raw": "", "items": [], "naming_tokens": 0, "error": None}
    try:
        image = load_image({"image": doc["source"], "pdf_page": page["page"]})
        row["raw"], row["extract_tokens"] = gen(EXTRACT_PROMPT, image=image, max_new_tokens=max_new_tokens)
        row["pairs"] = parse_pairs(row["raw"])
        if row["pairs"]:
            named = gen.batch_text([naming_prompt(k, v) for k, v in row["pairs"]], naming_max_new_tokens, NAMING_PREFILL)
            row["naming_raw"] = "\n---\n".join(t for t, _ in named)
            row["naming_tokens"] = max(n for _, n in named)
            row["items"] = [[i, *item] for i, ((k, v), (t, _)) in enumerate(zip(row["pairs"], named), 1)
                            for item in parse_named(t, v)]
    except Exception as e:  # one bad page must not kill a multi-hour run
        row["error"] = f"{type(e).__name__}: {e}"
    row["extract_truncated"] = row["extract_tokens"] >= max_new_tokens
    row["naming_truncated"] = row["naming_tokens"] >= naming_max_new_tokens
    row["seconds"] = round(time.time() - t0, 2)
    return row


def read_rows(paths: list[str]) -> dict[str, list[dict]]:
    by_doc: dict[str, dict] = defaultdict(dict)
    for path in paths:
        if Path(path).exists():
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    by_doc[r["doc_id"]][r["page"]] = r  # last write wins on a resumed page
    return {d: list(pages.values()) for d, pages in by_doc.items()}


# ---------------------------------------------------------------- scoring
def schema_texts(mode: str, fields: list[str], desc: dict[str, str]) -> list[str]:
    """What each schema field is embedded as: its name (key mode) or name + generated description (named mode)."""
    if mode == "key":
        return [f.replace("_", " ") for f in fields]
    return [f"{f.replace('_', ' ')}: {desc.get(f, '')}".rstrip(": ") for f in fields]


def assign_fields(rs: list[dict], mode: str, fields: list[str], field_emb, embed,
                  one_per_field: bool) -> dict[str, list[tuple[float, str, int | None]]]:
    """Map a document's predicted values to schema fields: field -> [(similarity, value, page)].

    Each candidate goes to its most similar field if similarity >= FIELD_SIM_THRESHOLD; with ``one_per_field``
    only the most similar candidate per field is kept (ties go to the earlier one).
    """
    import numpy as np

    if mode == "key":
        cands = [(k, v, r["page"]) for r in rs for k, v in r["pairs"]]
    else:
        cands = [(f"{f.replace('_', ' ')}: {desc}", v, r["page"]) for r in rs for _, f, desc, v in r["items"]]
    assigned: dict[str, list[tuple[float, str, int | None]]] = defaultdict(list)
    if cands:
        sims = embed([c for c, _, _ in cands]) @ field_emb.T
        for (_, v, page), row in zip(cands, sims):
            j = int(np.argmax(row))
            if row[j] >= FIELD_SIM_THRESHOLD:
                assigned[fields[j]].append((float(row[j]), v, page))
    if one_per_field:
        assigned = {f: [max(c, key=lambda x: x[0])] for f, c in assigned.items()}
    return assigned


def assign_both(rs: list[dict], fields: list[str], field_emb: dict, embed,
                one_per_field: bool) -> dict[str, list[tuple[float, str, int | None]]]:
    """Mode-free mapping: candidates from key mode and named mode, each scored in its own mode, pooled per field.

    With ``one_per_field`` the single most similar candidate per field wins regardless of mode (ties: key first).
    Added in study 11 because the key/named choice did not transfer across templates (study 10).
    """
    merged: dict[str, list] = defaultdict(list)
    for mode in ("key", "named"):
        for f, c in assign_fields(rs, mode, fields, field_emb[mode], embed, False).items():
            merged[f] += c
    if one_per_field:
        return {f: [max(c, key=lambda x: x[0])] for f, c in merged.items()}
    return dict(merged)


def score_v2(docs: list[dict], rows: dict[str, list[dict]], schema_desc: dict, embed,
             one_per_field: bool = False) -> dict:
    """``one_per_field``: keep only the highest-similarity candidate per (document, field); ties go to the earlier one.

    Added after the v2 run showed naming lowering precision (many candidates per field), so results with it
    must be reported as a post-hoc decoding rule, applied identically to both modes.
    """
    complete = [d for d in docs if len(rows.get(d["doc_id"], [])) == len(d["pages"])]
    out: dict = {"docs_total": len(docs), "docs_complete": len(complete), "stats": run_stats(docs, rows)}

    funsd_counts = [("funsd", {"pairs": _funsd_counts(d, rows[d["doc_id"]])}) for d in complete if d["dataset"] == "funsd"]
    if funsd_counts:
        tp, n_pred, n_gold = (sum(c["pairs"][i] for _, c in funsd_counts) for i in range(3))
        out["funsd"] = {**prf(tp, n_pred, n_gold), "f1_ci95": bootstrap_f1_ci(funsd_counts)["pairs"]}

    datasets = sorted({d["dataset"] for d in complete if d["dataset"] != "funsd"})
    fields = {ds: schema_fields(ds, docs) for ds in datasets}
    schema_text = {mode: {ds: schema_texts(mode, fields[ds], schema_desc.get(ds, {})) for ds in datasets}
                   for mode in ("key", "named")}
    for mode in ("key", "named"):
        field_emb = {ds: embed(schema_text[mode][ds]) for ds in datasets}
        per_field = defaultdict(Counter)
        doc_counts, decoy = [], Counter()
        for d in complete:
            if d["dataset"] == "funsd":
                continue
            ds, rs = d["dataset"], rows[d["doc_id"]]
            assigned = assign_fields(rs, mode, fields[ds], field_emb[ds], embed, one_per_field)
            all_values = [v for r in rs for _, v in r["pairs"]] + [it[3] for r in rs for it in r["items"]]
            counts, micro = {}, [0, 0, 0]
            for f in fields[ds]:
                gold = list({norm(g): g for g in d["gold"]["fields"].get(f, [])}.values())
                pv = list({norm(v): v for _, v, _ in assigned.get(f, [])}.values())
                tp = greedy_match(pv, gold, tok_f1)
                c = per_field[(ds, d["template"], f)]
                c["tp"] += tp; c["n_pred"] += len(pv); c["n_gold"] += len(gold)
                c["value_found"] += greedy_match(all_values, gold, tok_f1)
                counts[(ds, d["template"], f)] = (tp, len(pv), len(gold))
                micro = [micro[0] + tp, micro[1] + len(pv), micro[2] + len(gold)]
            counts[(ds, d["template"], "_micro")] = tuple(micro)
            doc_counts.append((d["template"], counts))
            if d["template"] == "Short-Form":
                _decoy_check(decoy, d, [v for _, v, _ in assigned.get("registrant_name", [])])
        ci = bootstrap_f1_ci(doc_counts)
        vrdu: dict = {}
        for (ds, t, f), c in sorted(per_field.items()):
            if c["n_gold"] == 0 and c["n_pred"] == 0:
                continue
            row = prf(c["tp"], c["n_pred"], c["n_gold"])
            row["value_recall"] = round(c["value_found"] / c["n_gold"], 4) if c["n_gold"] else None
            row["f1_ci95"] = ci[(ds, t, f)]
            vrdu.setdefault(ds, {}).setdefault(t, {})[f] = row
        for ds, by_t in vrdu.items():
            for t, by_f in by_t.items():
                tp, n_pred, n_gold = (sum(r[k] for r in by_f.values()) for k in ("tp", "n_pred", "n_gold"))
                by_f["_micro"] = {**prf(tp, n_pred, n_gold), "f1_ci95": ci[(ds, t, "_micro")]}
        out[mode] = {"vrdu": vrdu, "short_form_registrant": dict(decoy)}
    return out


def _funsd_counts(doc: dict, rs: list[dict]) -> tuple[int, int, int]:
    pairs, gold = [p for r in rs for p in r["pairs"]], doc["gold"]["pairs"]
    tp = greedy_match(pairs, gold, lambda a, b: min(tok_f1(a[0], b[0]), tok_f1(a[1], b[1])))
    return tp, len(pairs), len(gold)


def _decoy_check(decoy: Counter, doc: dict, reg_preds: list[str]) -> None:
    d, gold_reg = doc["gold"].get("decoy"), doc["gold"]["fields"].get("registrant_name", [])
    if d and any(tok_f1(d, g) >= MATCH_THRESHOLD for g in gold_reg):
        d = None  # an individual registered themself: no decoy in this document
        decoy["filer_is_registrant"] += 1
    decoy["docs"] += 1
    decoy["predicted_registrant"] += bool(reg_preds)
    decoy["correct"] += any(tok_f1(v, g) >= MATCH_THRESHOLD for v in reg_preds for g in gold_reg)
    decoy["picked_decoy"] += bool(d) and any(tok_f1(v, d) >= MATCH_THRESHOLD for v in reg_preds)


def run_stats(docs: list[dict], rows: dict[str, list[dict]]) -> dict:
    stats: dict = defaultdict(Counter)
    secs = defaultdict(list)
    template = {d["doc_id"]: d["template"] for d in docs}
    for doc_id, rs in rows.items():
        t = template.get(doc_id, "?")
        for r in rs:
            s = stats[t]
            s["pages"] += 1
            s["extract_truncated"] += r["extract_truncated"]
            s["naming_truncated"] += r["naming_truncated"]
            s["errors"] += bool(r["error"])
            s["pairs"] += len(r["pairs"])
            s["items"] += len(r["items"])
            secs[t].append(r["seconds"])
    return {t: {**c, "mean_seconds_per_page": round(sum(secs[t]) / len(secs[t]), 1)} for t, c in stats.items()}


# ---------------------------------------------------------------- CLI
def cmd_prepare(args) -> None:
    exclude = None
    if args.exclude_manifest:
        exclude = {d["doc_id"].split("/", 1)[-1]
                   for d in json.loads(Path(args.exclude_manifest).read_text(encoding="utf-8"))}
    funsd_root = Path(args.funsd_root) if args.funsd_root and "funsd" in args.subcorpora.split(",") else None
    subcorpora = tuple(s for s in args.subcorpora.split(",") if s != "funsd")
    docs = build_manifest(funsd_root, Path(args.vrdu_root), args.n_per_template, args.seed, args.max_pages,
                          subcorpora, exclude)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    n_units = sum(len(d["pages"]) for d in docs)
    print(f"[+] {len(docs)} docs / {n_units} page units -> {args.out}: {dict(Counter(d['template'] for d in docs))}")


def cmd_describe_schema(args) -> None:
    gen = VLM(args.model, args.device, args.dtype, args.max_pixels)
    out = {}
    fields_by_ds = (json.loads(Path(args.fields_file).read_text(encoding="utf-8")) if args.fields_file
                    else {ds: schema_fields(ds) for ds in ("registration-form", "ad-buy-form")})
    for ds, fields in fields_by_ds.items():
        if not fields:
            continue  # no known schema for this dataset: nothing to describe
        out[ds] = {}
        for f in fields:
            raw, _ = gen(field_desc_prompt(fields, f), max_new_tokens=128)
            desc = parse_desc(raw, f)
            if desc:
                out[ds][f] = desc
            print(f"    [{ds}] {f}: {desc!r}  | raw: {raw[:150]!r}", flush=True)
        print(f"[{ds}] described {len(out[ds])}/{len(fields)} fields", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_run(args) -> None:
    docs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    units = [(d, p) for d in docs for p in d["pages"]][args.shard::args.num_shards]
    if args.limit:
        units = units[:args.limit]
    out = Path(args.out)
    done = {(r["doc_id"], r["page"]) for rs in read_rows([str(out)]).values() for r in rs}
    todo = [(d, p) for d, p in units if (d["doc_id"], p["page"]) not in done]
    print(f"[shard {args.shard}] {len(todo)} page units to do ({len(units) - len(todo)} already done) on {args.device}", flush=True)
    if not todo:
        return
    deadline = time.time() + args.deadline_hours * 3600 if args.deadline_hours else None
    gen = VLM(args.model, args.device, args.dtype, args.max_pixels)
    with open(out, "a", encoding="utf-8") as fh:
        for k, (doc, page) in enumerate(todo, 1):
            if deadline and time.time() > deadline:
                print(f"[shard {args.shard}] deadline reached after {k - 1} units; stopping cleanly (re-run to resume)", flush=True)
                break
            row = process_unit(gen, doc, page, args.max_new_tokens, args.naming_max_new_tokens)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            flags = (" TRUNC-E" if row["extract_truncated"] else "") + (" TRUNC-N" if row["naming_truncated"] else "")
            print(f"[shard {args.shard}] {k}/{len(todo)} {doc['doc_id'][:55]} p{page['page']} pairs={len(row['pairs'])} "
                  f"items={len(row['items'])} {row['seconds']}s{flags}{' ERR ' + row['error'] if row['error'] else ''}", flush=True)


def cmd_score(args) -> None:
    docs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    schema_desc = json.loads(Path(args.schema_desc).read_text(encoding="utf-8")) if args.schema_desc else {}
    metrics = score_v2(docs, read_rows(args.preds), schema_desc, sbert_embedder(), one_per_field=args.one_per_field)
    metrics["one_per_field"] = args.one_per_field
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("docs_total", "docs_complete", "stats")}, indent=2))


def cmd_selftest(_args) -> None:
    import tempfile
    import zlib

    import numpy as np

    split = ("<field>registrant_name</field><desc>Organization that registers.</desc><value>Acme Corp</value>\n"
             "<field>registrant_address</field><desc>Its address.</desc><value>1 Main St, Springfield</value>")
    assert parse_named(split, "Acme Corp, 1 Main St, Springfield") == [
        ["registrant_name", "Organization that registers.", "Acme Corp"],
        ["registrant_address", "Its address.", "1 Main St, Springfield"]]
    assert parse_named("<field>Start Date</field><desc>Start.</desc>", "01/02/20") == [["start_date", "Start.", "01/02/20"]]
    assert parse_named("invoice_number<desc>Invoice number</desc><value>963706-1</value>", "x") == [
        ["invoice_number", "Invoice number", "963706-1"]]
    # verbatim untagged answers from the Kaggle smoke run (with the "<field>" prefill prepended, as batch_text does)
    kaggle = [("VAR_description_VAR_description\nThe variable's description is provided.", "(SEE EXPLANATION)",
               ["var_description", "The variable's description is provided.", "(SEE EXPLANATION)"]),
              ("received_by_snake_case_name\nReceived By: NSD/FARA Registration Unit\nDate: 09/27/2017", "NSD/FARA ...",
               ["received_by", "Received By: NSD/FARA Registration Unit", "NSD/FARA ..."]),
              ("invoice_number_invoice_number\nInvoice number is the unique identifier for the invoice. \nVALUE 963706-1",
               "963706-1", ["invoice_number", "Invoice number is the unique identifier for the invoice.", "963706-1"])]
    for raw, value, want in kaggle:
        assert parse_named(raw, value) == [want], parse_named(raw, value)
        assert parse_named(NAMING_PREFILL + raw, value) == [want], parse_named(NAMING_PREFILL + raw, value)
    # prefilled answer that follows the format, and one that forgets </field>
    assert parse_named("<field>invoice_date</field><desc>Date of the invoice.</desc><value>10/21/12</value>", "x") == [
        ["invoice_date", "Date of the invoice.", "10/21/12"]]
    assert parse_named("<field>invoice_date\nDate the invoice was issued.", "10/21/12") == [
        ["invoice_date", "Date the invoice was issued.", "10/21/12"]]
    assert parse_named("", "x") == []
    assert parse_desc("<desc>The date the form was filed.</desc>", "file_date") == "The date the form was filed."
    assert parse_desc("Sure! <desc>Date of filing", "file_date") == "Date of filing"            # unclosed tag
    assert parse_desc("\nfile_date: The filing date.\nMore text", "file_date") == "The filing date."  # no tags
    assert parse_desc("**file_date**: <b>Date</b> filed", "file_date") == "Date filed"
    assert parse_desc("", "file_date") == ""

    def embed(texts):  # hashed bag-of-words stand-in for SBERT
        m = np.full((len(texts), 4096), 1e-9)
        for i, t in enumerate(texts):
            for w in norm(t).split():
                m[i, zlib.crc32(w.encode()) % 4096] += 1.0
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    # two-page Short-Form doc: naming maps the long foreign-principal key correctly, raw key text does not
    doc = {"doc_id": "d1", "dataset": "registration-form", "template": "Short-Form", "pages": [{"page": 0}, {"page": 1}],
           "gold": {"fields": {"foreign_principle_name": ["Masdar"], "registrant_name": ["Glover Park Group"]},
                    "decoy": "Doe, Jane"}}
    long_key = "List every foreign principal to whom you will render services in support of the primary registrant."
    rows = {"d1": [
        {"page": 0, "pairs": [["Name", "Jane Doe"], [long_key, "Masdar"]], "error": None, "seconds": 1,
         "extract_truncated": False, "naming_truncated": False,
         "items": [[1, "individual_name", "Person filing.", "Jane Doe"],
                   [2, "foreign_principal_name", "Foreign principal served.", "Masdar"]]},
        {"page": 1, "pairs": [["Primary registrant", "Glover Park Group, LLC 1 Main St"]], "error": None, "seconds": 1,
         "extract_truncated": True, "naming_truncated": False,
         "items": [[1, "registrant_name", "Organization registered.", "Glover Park Group, LLC"]]}]}
    desc = {"registration-form": {"foreign_principle_name": "Foreign principal on whose behalf the registrant acts.",
                                  "registrant_name": "Organization that is registered."}}
    m = score_v2([doc], rows, desc, embed)
    named = m["named"]["vrdu"]["registration-form"]["Short-Form"]
    assert named["foreign_principle_name"]["tp"] == 1 and named["registrant_name"]["tp"] == 1, named
    assert m["named"]["short_form_registrant"] == {"docs": 1, "predicted_registrant": 1, "correct": 1, "picked_decoy": 0}
    assert m["docs_complete"] == 1 and m["stats"]["Short-Form"]["extract_truncated"] == 1
    assert score_v2([doc], {"d1": rows["d1"][:1]}, desc, embed)["docs_complete"] == 0  # incomplete docs are not scored

    # assign_both: the long foreign-principal key fails in key mode alone but is recovered via its named item
    fs = ["foreign_principle_name", "registrant_name", "signer_name"]
    emb = {m: embed(schema_texts(m, fs, desc["registration-form"])) for m in ("key", "named")}
    assert "foreign_principle_name" not in assign_fields(rows["d1"], "key", fs, emb["key"], embed, True)
    both = assign_both(rows["d1"], fs, emb, embed, True)
    assert both["foreign_principle_name"][0][1] == "Masdar", both

    # one_per_field: a vaguely named filer item and the real registrant both map to registrant_name;
    # unconstrained scoring keeps both (decoy picked), the rule keeps only the closer match
    two = {"d1": [dict(rows["d1"][0], items=[[1, "registrant", "", "Jane Doe"]]), rows["d1"][1]]}
    loose_m = score_v2([doc], two, desc, embed)["named"]["short_form_registrant"]
    top1_m = score_v2([doc], two, desc, embed, one_per_field=True)["named"]["short_form_registrant"]
    assert loose_m["picked_decoy"] == 1 and loose_m["correct"] == 1, loose_m
    assert top1_m["picked_decoy"] == 0 and top1_m["correct"] == 1, top1_m

    # end to end with a stub model on real data, if present: manifest, page rendering, naming, resume
    vrdu = ROOT / "data" / "vrdu"
    if (vrdu / "ad-buy-form" / "main" / "meta.json").exists():
        ad = schema_fields("ad-buy-form")
        assert "flight_to" in ad and "program_end_date" not in ad and "channel" not in ad, ad
    if (vrdu / "registration-form" / "main").exists():
        class Stub:
            def __call__(self, prompt, image=None, max_new_tokens=0):
                if image is not None:
                    assert image.size[0] > 100, "page did not render"
                    return "<key>Name of Registrant</key><value>Acme</value>", 10
                raise AssertionError("naming must use batch_text")

            def batch_text(self, prompts, max_new_tokens, prefill=""):
                assert all(p.rstrip().endswith("Answer:") and "Value: " in p for p in prompts) and prefill == NAMING_PREFILL
                return [(prefill + "registrant_name</field><desc>Org.</desc><value>Acme</value>", 12)] * len(prompts)

        docs = [d for d in build_manifest(None, vrdu, 1, 0, 2) if d["dataset"] == "registration-form"][:1]
        assert len(docs[0]["pages"]) >= 1 and docs[0]["gold"]["fields"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for p in docs[0]["pages"]:
                    r = process_unit(Stub(), docs[0], p, 2048, 160)
                    assert r["error"] is None and r["items"] == [[1, "registrant_name", "Org.", "Acme"]], r
                    fh.write(json.dumps(r) + "\n")
            got = read_rows([str(path)])
            assert len(got[docs[0]["doc_id"]]) == len(docs[0]["pages"])
            assert score_v2(docs, got, {}, embed)["docs_complete"] == 1
        print(f"[+] stub end-to-end on {docs[0]['doc_id'][:60]} ({len(docs[0]['pages'])} pages) passed")
    print("[+] zero_shot_kv_v2 selftest passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--funsd-root", default="data/funsd/dataset")
    p.add_argument("--vrdu-root", default="data/vrdu")
    p.add_argument("--n-per-template", type=int, default=100)
    p.add_argument("--max-pages", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subcorpora", default="funsd,registration-form,ad-buy-form")
    p.add_argument("--exclude-manifest", default=None,
                   help="sample fresh documents avoiding every document in this manifest (confirmation runs)")
    p.add_argument("--out", required=True)
    for name in ("describe-schema", "run"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--dtype", default="float16")
        p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
        p.add_argument("--out", required=True)
        if name == "describe-schema":
            p.add_argument("--fields-file", default=None, help='JSON {"dataset": [field names]} instead of the VRDU schemas')
        if name == "run":
            p.add_argument("--manifest", required=True)
            p.add_argument("--max-new-tokens", type=int, default=2048)
            p.add_argument("--naming-max-new-tokens", type=int, default=160)  # per pair
            p.add_argument("--shard", type=int, default=0)
            p.add_argument("--num-shards", type=int, default=1)
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--deadline-hours", type=float, default=0.0)
    p = sub.add_parser("score")
    p.add_argument("--manifest", required=True)
    p.add_argument("--preds", nargs="+", required=True)
    p.add_argument("--schema-desc", default=None)
    p.add_argument("--one-per-field", action="store_true",
                   help="post-hoc decoding rule: keep the best-matching candidate per document and field")
    p.add_argument("--out", required=True)
    sub.add_parser("selftest")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "describe-schema": cmd_describe_schema, "run": cmd_run, "score": cmd_score,
     "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
