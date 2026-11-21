"""Study 34, step 2: the schema-guided grounded recipe extended from one block per field to a table of rows (VRDU ad-buy line items).

One call per page. The model answers {"rows": [{field: "text identifier" | null, ...}, ...]}; every cell cites an OCR piece (lines cut at
wide gaps, `adbuy_rows.split_lines`) and decodes to the set of words of the quoted text. Text-first repair (study 33): a cell whose cited id's
piece does not hold the quoted text is moved to the one piece on the page that does.

    python scripts/adbuy_sg.py manifests
    python scripts/adbuy_sg.py oracle [--n 30]
    python scripts/adbuy_sg.py run --manifest results/adbuy_sg/pilot.json --arm image|text --out rows.jsonl [--shard 0 --num-shards 2] [--stub]
    python scripts/adbuy_sg.py gate --manifest results/adbuy_sg/pilot.json --rows rows.jsonl [--out gate.json]
    python scripts/adbuy_sg.py score --rows-dir DIR   # DIR holds rows_A2_*.jsonl and rows_A1_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
import adbuy_rows as R  # noqa: E402
import schema_guided_kv as sg  # noqa: E402
from lir_metric import cell_counts, f1, lir_counts  # noqa: E402

OUT = sg.ROOT / "results" / "adbuy_sg"
SPLITS = R.DATA.parent.parent / "few_shot-splits"
GAP = 0.5   # study 34 oracle: the cheapest line cut that reaches the plateau
# Written from the field names alone (no ad-buy page was read for them), frozen before any model run.
DESC = {
    "channel": "The TV channel or station on which the line item runs.",
    "program_desc": "The description of the program or time slot of the line item.",
    "program_start_date": "The date on which the line item starts.",
    "program_end_date": "The date on which the line item ends.",
    "sub_amount": "The amount charged for the line item.",
}
EXAMPLE = ('{"rows": [{"channel": "WXYZ 31|12", "program_desc": "Evening News 6p-7p 31|40", "program_start_date": "03/04/20 31|58", '
           '"program_end_date": "03/10/20 31|66", "sub_amount": null}, {"channel": "WXYZ 33|12", "program_desc": null, '
           '"program_start_date": "03/11/20 33|58", "program_end_date": "03/17/20 33|66", "sub_amount": "$1,200.00 33|80"}]}')   # invented values and ids


def build_prompt(lines: list[sg.Line]) -> str:
    doc = "\n".join(f"{ln.text} {ln.id}" for ln in lines)
    descs = "\n".join(f"- {f}: {DESC[f]}" for f in R.LI_FIELDS)
    schema = '{"rows": [{' + ", ".join(f'"{f}": "<text> <identifier>" or null' for f in R.LI_FIELDS) + "}, ...]}"
    return (f"<Document>\n{doc}\n</Document>\n<Task>\nThis page may contain a table of line items. Extract every row of that table. "
            f"Every document line ends with its identifier (two numbers separated by |).\nRules:\n"
            f"1. One JSON object per table row, in reading order. Do not merge, skip or repeat rows. Write the answer compactly: one row object per "
            f"line, no indentation, no blank lines, no comments.\n"
            f"2. Write each cell as the text copied exactly from ONE line, followed by that line's identifier, for example \"WXYZ 31|12\".\n"
            f"3. Copy only the cell's value, not the table header and not other cells on the same line.\n"
            f"4. A cell that spans several lines is written as one 'text identifier' per line, separated by a newline.\n"
            f"5. Use null for a cell that is empty in that row. If the page has no table rows, answer {{\"rows\": []}}.\n"
            f"Field descriptions:\n{descs}\nExample of the answer format (a different document): {EXAMPLE}\n"
            f"Answer with this JSON object:\n{schema}\n</Task>\n<Extraction>\n")


def parse_rows(raw: str) -> list[dict] | None:
    """The row objects of an answer; works on a truncated answer too (every complete flat object after "rows" counts). None = not an answer."""
    s = re.sub(r"```(?:json)?", "", raw)
    k = s.find('"rows"')
    rows = []
    for m in re.finditer(r"\{[^{}]*\}", s[k:] if k >= 0 else s):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and any(f in o for f in R.LI_FIELDS):
            rows.append(o)
    return rows if k >= 0 or rows else None


def _words(text: str, ln: sg.Line, doc: sg.Doc):
    sp = sg._line_span(text, ln, doc) or sg._line_span_lenient(text, ln, doc)
    return set(range(*sp)) if sp else None


def _xy(lid: str | None):
    m = re.match(r"(\d{2})\|(\d{2})", lid or "")
    return (int(m[1]), int(m[2])) if m else None


def _pick(cand: list[sg.Line], lid: str | None, row_y: float | None, repair: str) -> sg.Line | None:
    """Choose among the pieces that hold a cell's text. unique: only if one piece holds it. nearest: closest to the cited id's coordinates.
    row: closest to the vertical position of the row's other (correct) cells, ties broken by the cited coordinates."""
    if not cand:
        return None
    if repair == "unique":
        return cand[0] if len(cand) == 1 else None
    xy = _xy(lid)
    d_cited = lambda c: ((c.box[0] + c.box[2]) * 50 - xy[0]) ** 2 + ((c.box[1] + c.box[3]) * 50 - xy[1]) ** 2 if xy else 0.0
    if repair == "row" and row_y is not None:
        key = lambda c: (round(abs((c.box[1] + c.box[3]) / 2 - row_y), 3), d_cited(c))
    elif repair in ("nearest", "row") and xy:
        key = lambda c: (d_cited(c),)
    else:
        return cand[0] if len(cand) == 1 else None
    ranked = sorted(cand, key=key)
    return ranked[0] if len(ranked) == 1 or key(ranked[0]) != key(ranked[1]) else None


def decode_page(doc: sg.Doc, page: int, raw: str, st: Counter, repair: str | bool = "row") -> list[list[tuple[str, frozenset]]]:
    """Rows of (field, word set) from one page's answer; cells that cannot be grounded are dropped, empty rows too.
    `repair`: none | unique (study 33 rule) | nearest | row (see `_pick`) | row+ (row, and also moves cells that sit off their row's line); True and False mean unique and none."""
    repair = {True: "unique", False: "none"}.get(repair, repair)
    lines = doc.lines.get(page, [])
    by_id = {ln.id: ln for ln in lines}
    st["outputs"] += 1
    rows = parse_rows(raw)
    if rows is None:
        st["unparsed"] += 1
        return []
    out = []
    for obj in rows:
        parsed = {f: sg._segments(obj[f]) for f in R.LI_FIELDS if obj.get(f) not in (None, "", [])}
        parsed = {f: v for f, v in parsed.items() if v}
        direct = {f: [(_words(t, by_id[i], doc) if i in by_id else None) for t, i in segs] for f, segs in parsed.items()}
        ys = sorted((by_id[i].box[1] + by_id[i].box[3]) / 2 for f, segs in parsed.items() for (t, i), g in zip(segs, direct[f]) if g is not None)
        row_y = ys[len(ys) // 2] if ys else None
        cells = []
        for f, segs in parsed.items():
            st["cells"] += 1
            ws: set[int] = set()
            for (text, lid), got in zip(segs, direct[f]):
                st["segments"] += 1
                st["id_valid"] += lid in by_id
                rule = "row" if repair == "row+" else repair
                if got is not None and repair == "row+" and row_y is not None:   # a cell whose piece sits off its row's line: look for a nearer piece
                    ln = by_id[lid]
                    if abs((ln.box[1] + ln.box[3]) / 2 - row_y) > 1.5 * (ln.box[3] - ln.box[1]):
                        c = _pick([c for c in lines if _words(text, c, doc)], lid, row_y, rule)
                        if c is not None and c is not ln:
                            got = _words(text, c, doc)
                            st["moved"] += 1
                if got is None and repair != "none":
                    c = _pick([c for c in lines if _words(text, c, doc)], lid, row_y, rule)
                    got = _words(text, c, doc) if c else None
                    st["repaired"] += c is not None
                if got is None:
                    ws = set()
                    break
                ws |= got
            if ws:
                cells.append((f, frozenset(ws)))
            else:
                st["ungrounded"] += 1
        if cells:
            out.append(cells)
        else:
            st["empty_rows"] += 1
    return out


def make_doc(rec: dict) -> sg.Doc:
    doc = sg.Doc(rec, max_pages=R.MAX_PAGES)
    R.split_lines(rec, doc, GAP)
    return doc


# ------------------------------------------------------------------ manifests
def split_tests() -> dict[tuple[int, int], list[str]]:
    out = {}
    for f in sorted(glob.glob(str(SPLITS / "DeepForm-unk_template-*.json"))):
        d = json.load(open(f))
        out[(int(f.split("train_")[1].split("-")[0]), int(f.split("SD_")[1][0]))] = d["test"]
    return out


def cmd_manifests(_a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tests = split_tests()
    primary = sorted(set.intersection(*map(set, tests.values())))
    outside = sorted({r["filename"] for r in R.load_records()} - set().union(*map(set, tests.values())))
    pilot = sorted(random.Random(5).sample(outside, 30))
    (OUT / "primary.json").write_text(json.dumps(primary), encoding="utf-8")
    (OUT / "pilot.json").write_text(json.dumps(pilot), encoding="utf-8")
    print(f"primary {len(primary)} documents (test document of every one of the {len(tests)} unseen-template splits), pilot {len(pilot)} documents outside every test set")


# ------------------------------------------------------------------ oracle and self-test
def synthetic_answer(doc: sg.Doc, rec: dict, page: int, corrupt: bool = False) -> str:
    ids = [ln.id for ln in doc.lines.get(page, [])]
    rows = []
    for row in R.gold_rows(rec, doc):
        if row and row[0][2] and row[0][2][0][2] == page:
            obj = {}
            for f, _, segs, _ in row:
                cite = [(t, ids[(ids.index(i) + 1) % len(ids)] if corrupt else i) for t, i, _ in segs]
                obj[f] = "\n".join(f"{t} {i}" for t, i in cite) or None
            rows.append(obj)
    return json.dumps({"rows": rows})


def cmd_oracle(a) -> None:
    recs = R.load_records(a.n)
    tot = {k: [0, 0, 0] for k in ("lir", "cell", "corrupt_repair", "corrupt_plain")}
    st: Counter = Counter()
    for rec in recs:
        doc = make_doc(rec)
        gold = [{(f, ws) for f, ws, _, _ in row} for row in R.gold_rows(rec, doc)]
        if not gold:
            continue
        for name, corrupt, repair in (("clean", False, "row"), ("corrupt_repair", True, "row"), ("corrupt_plain", True, "none")):
            pred = [r for p in doc.lines for r in decode_page(doc, p, synthetic_answer(doc, rec, p, corrupt), st if name == "clean" else Counter(), repair)]
            pred = [set(r) for r in pred]
            if name == "clean":
                for k, c in (("lir", lir_counts(pred, gold)), ("cell", cell_counts(pred, gold))):
                    tot[k] = [x + y for x, y in zip(tot[k], c)]
            else:
                tot[name] = [x + y for x, y in zip(tot[name], cell_counts(pred, gold))]
    print(f"oracle through the decode path on {len(recs)} documents: lir_f1 {f1(tot['lir']) * 100:.2f}, cell_f1 {f1(tot['cell']) * 100:.2f}; {dict(st)}")
    print(f"ids shifted to the next piece: cell_f1 {f1(tot['corrupt_plain']) * 100:.2f} without repair, {f1(tot['corrupt_repair']) * 100:.2f} with repair")
    assert f1(tot["lir"]) > 0.95 and f1(tot["corrupt_repair"]) > f1(tot["corrupt_plain"])


# ------------------------------------------------------------------ run, gate, score
def cmd_run(a) -> None:
    ids = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    recs = {r["filename"]: r for r in R.load_records() if r["filename"] in set(ids)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["doc_id"], r["page"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())} if out.exists() and out.stat().st_size else set()
    units = [(fn, p) for fn in ids for p in range(len(recs[fn]["ocr"]["pages"]))]
    units = [u for u in units[a.shard::a.num_shards] if u not in done]
    if a.limit:
        units = units[:a.limit]
    print(f"[shard {a.shard}] {len(units)} pages on {a.device} (arm {a.arm})", flush=True)
    if not units:
        return

    class _Stub:
        def sample(self, prompt, image, k, max_new_tokens, prefill=""):
            return ['{"rows": []}'] * k

    gen = _Stub() if a.stub else sg._vlm_class()(a.model, a.device, a.dtype, a.max_pixels, a.quant4)
    deadline = time.time() + a.deadline_hours * 3600 if a.deadline_hours else None
    cache: dict[str, sg.Doc] = {}
    with open(out, "a", encoding="utf-8") as fh:
        for k, (fn, p) in enumerate(units, 1):
            if deadline and time.time() > deadline:
                print(f"[shard {a.shard}] deadline reached after {k - 1} units; stopping cleanly (re-run to resume)", flush=True)
                break
            row = {"doc_id": fn, "page": p, "arm": a.arm, "samples": [], "error": None}
            t0 = time.time()
            try:
                doc = cache.setdefault(fn, make_doc(recs[fn]))
                lines = doc.lines.get(p, [])
                if lines:
                    prompt = build_prompt(lines)
                    tok = None if a.stub else gen.proc.tokenizer
                    row["prompt_tokens"] = len(tok(prompt).input_ids) if tok else None
                    image = None
                    # Frozen before the run: the real prompt of a numeric page is about 1.7 times the 3-characters-per-token estimate, and a 6k-token
                    # prompt plus the image ran a T4 out of memory in the smoke run, so a page above the cap is answered from the text alone.
                    if a.arm == "image" and not (row["prompt_tokens"] and row["prompt_tokens"] > a.image_cap_tokens):
                        from zero_shot_kv import load_image
                        image = load_image({"image": str(R.DATA.parent / "pdfs" / fn), "pdf_page": p})
                    row["image_dropped"] = a.arm == "image" and image is None
                    row["samples"] = gen.sample(prompt, image, 1, a.max_new_tokens)
                    row["new_tokens"] = len(tok(row["samples"][0]).input_ids) if tok else None
                else:
                    row["samples"] = ['{"rows": []}']
            except Exception as e:  # one bad page must not kill a multi-hour run
                row["error"] = f"{type(e).__name__}: {e}"
            if not a.stub:
                import torch
                torch.cuda.empty_cache()   # the smoke run died on the third page with 1.6 GiB reserved but unallocated
            row["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            rate = f" {row['new_tokens'] / max(row['seconds'], 1e-6):.1f} tok/s" if row.get("new_tokens") else ""
            print(f"[shard {a.shard}] {k}/{len(units)} {fn[:40]} p{p} {row['seconds']}s prompt {row.get('prompt_tokens')} out {row.get('new_tokens')}{rate}"
                  f"{' ERR ' + row['error'] if row['error'] else ''}", flush=True)


def complete(doc: sg.Doc, answers: dict[int, str]) -> bool:
    """True if every page with text has an answer (a run cut by the deadline or by errors leaves documents half done; they are not scored)."""
    return all(p in answers for p, lines in doc.lines.items() if lines)


def read_pages(paths: list[str]) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    for by_doc in [sg.read_rows([p]) for p in paths]:
        for fn, rows in by_doc.items():
            out.setdefault(fn, {}).update({r["page"]: r["samples"][0] for r in rows if r["samples"]})
    return out


def cmd_gate(a) -> None:
    ids = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    recs = {r["filename"]: r for r in R.load_records() if r["filename"] in set(ids)}
    pages = read_pages(a.rows)
    st: Counter = Counter()
    gold_pages = answered = 0
    for fn in ids:
        if fn not in pages:
            continue
        doc = make_doc(recs[fn])
        has = {row[0][2][0][2] for row in R.gold_rows(recs[fn], doc) if row and row[0][2]}
        for p, raw in pages[fn].items():
            rows = decode_page(doc, p, raw, st)
            if p in has:
                gold_pages += 1
                answered += bool(rows)
    parse = 1 - st["unparsed"] / max(1, st["outputs"])
    ids_ok = st["id_valid"] / max(1, st["segments"])
    res = {"outputs": st["outputs"], "parse_rate": round(parse, 4), "cells": st["cells"], "id_valid_rate": round(ids_ok, 4),
           "pages_with_gold_rows": gold_pages, "answered_rate": round(answered / max(1, gold_pages), 4), "repaired": st["repaired"],
           "ungrounded": st["ungrounded"], "decision": "keep" if parse >= 0.9 and ids_ok >= 0.9 and answered >= 0.9 * gold_pages else "stop_and_review"}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def cmd_score(a) -> None:
    ids = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    recs = {r["filename"]: r for r in R.load_records() if r["filename"] in set(ids)}
    report = {}
    for arm in ("A2", "A1"):
        pages = read_pages(sorted(glob.glob(str(Path(a.rows_dir) / f"rows_{arm}_*.jsonl"))))
        if not pages:
            continue
        tot = {k: [0, 0, 0] for k in ("lir", "cell", "vlir", "vcell")}
        st: Counter = Counter()
        n_done = 0
        for fn in ids:
            doc = make_doc(recs[fn])
            if not complete(doc, pages.get(fn, {})):
                continue
            n_done += 1
            gold = R.gold_rows(recs[fn], doc)
            pred = [r for p in sorted(doc.lines) for r in decode_page(doc, p, pages[fn].get(p, '{"rows": []}'), st)]
            if not gold:
                continue
            g, p = [{(f, ws) for f, ws, _, _ in row} for row in gold], [set(r) for r in pred]
            vg = [{(f, t) for f, _, _, t in row} for row in gold]
            vp = [{(f, " ".join(doc.words[w] for w in sorted(ws))) for f, ws in r} for r in pred]
            for k, c in (("lir", lir_counts(p, g)), ("cell", cell_counts(p, g)), ("vlir", lir_counts(vp, vg, R.value_same)), ("vcell", cell_counts(vp, vg, R.value_same))):
                tot[k] = [x + y for x, y in zip(tot[k], c)]
        report[arm] = {"lir_f1": round(f1(tot["lir"]) * 100, 2), "cell_f1": round(f1(tot["cell"]) * 100, 2), "value_lir_f1": round(f1(tot["vlir"]) * 100, 2),
                       "value_cell_f1": round(f1(tot["vcell"]) * 100, 2), "counts_lir": tot["lir"], "decode": dict(st),
                       "documents_scored": n_done, "documents_in_manifest": len(ids)}
        print(arm, json.dumps(report[arm]))
    (Path(a.rows_dir) / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifests")
    sub.add_parser("oracle").add_argument("--n", type=int, default=30)
    p = sub.add_parser("run")
    p.add_argument("--manifest", required=True); p.add_argument("--arm", choices=["image", "text"], required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct"); p.add_argument("--device", default="cuda:0"); p.add_argument("--dtype", default="float16")
    p.add_argument("--max-pixels", type=int, default=640 * 28 * 28); p.add_argument("--quant4", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=2400)
    p.add_argument("--image-cap-tokens", type=int, default=5500)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--deadline-hours", type=float, default=0.0); p.add_argument("--stub", action="store_true")
    p = sub.add_parser("gate")
    p.add_argument("--manifest", required=True); p.add_argument("--rows", nargs="+", required=True); p.add_argument("--out", default=None)
    p = sub.add_parser("score")
    p.add_argument("--manifest", default=str(OUT / "primary.json")); p.add_argument("--rows-dir", required=True)
    a = ap.parse_args()
    {"manifests": cmd_manifests, "oracle": cmd_oracle, "run": cmd_run, "gate": cmd_gate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
