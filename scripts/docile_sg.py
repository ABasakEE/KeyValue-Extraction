"""Study 34, DocILE: the schema-guided grounded recipe on line items (rows of cells), same method as `adbuy_sg.py`.

One call per page; the model answers {"rows": [{field: "text identifier" | null, ...}, ...]} for the 19 DocILE line-item fields, citing OCR pieces
(lines cut at 0.5 line heights, `docile_rows.Doc`); decode with the row-position repair. Cells are word sets (DocILE's pseudo-character-centre rule).

    python scripts/docile_sg.py manifests                      # results/docile_sg/{test,pilot}.json from results/docile_rows/splits.json
    python scripts/docile_sg.py oracle [--n 200]               # gold cells cited by id through the decode path
    python scripts/docile_sg.py run --manifest M --arm image|text --out rows.jsonl [--shard 0 --num-shards 2] [--stub]
    python scripts/docile_sg.py gate --manifest M --rows rows.jsonl [--out gate.json]
    python scripts/docile_sg.py score --manifest M --rows-dir DIR      # DIR holds rows_A2_*.jsonl and rows_A1_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
import docile_rows as D  # noqa: E402
import schema_guided_kv as sg  # noqa: E402
from row_repair import _pick, patch_sdpa  # noqa: E402
from lir_metric import cell_counts, f1, lir_counts  # noqa: E402

OUT = D.ROOT / "results" / "docile_sg"
GAP = 0.5
# Written from the field names alone (no DocILE page was read for them), frozen before any model run.
DESC = {
    "line_item_quantity": "The quantity of the item.",
    "line_item_units_of_measure": "The unit in which the quantity is measured.",
    "line_item_code": "The code or identifier of the item.",
    "line_item_unit_price_gross": "The price of one unit of the item including tax.",
    "line_item_amount_gross": "The total amount of the line item including tax.",
    "line_item_description": "The description or name of the item.",
    "line_item_date": "The date of the line item.",
    "line_item_unit_price_net": "The price of one unit of the item without tax.",
    "line_item_amount_net": "The total amount of the line item without tax.",
    "line_item_position": "The position or row number of the line item in the table.",
    "line_item_person_name": "The name of a person on the line item.",
    "line_item_currency": "The currency of the line item.",
    "line_item_tax": "The tax amount of the line item.",
    "line_item_hts_number": "The harmonized tariff schedule (HTS) number of the item.",
    "line_item_discount_amount": "The discount amount of the line item.",
    "line_item_order_id": "The order identifier of the line item.",
    "line_item_weight": "The weight of the item.",
    "line_item_tax_rate": "The tax rate of the line item.",
    "line_item_discount_rate": "The discount rate of the line item.",
}
FIELDS = list(DESC)
EXAMPLE = ('{"rows": [{"line_item_position": "1 31|12", "line_item_description": "Office chair 31|25", "line_item_quantity": "2 31|60", '
           '"line_item_unit_price_gross": null, "line_item_amount_gross": "$120.00 31|80"}, {"line_item_position": "2 33|12", '
           '"line_item_description": ["Desk lamp 33|25", "LED, black 34|25"], "line_item_quantity": "1 33|60", "line_item_unit_price_gross": "$25.00 33|72", '
           '"line_item_amount_gross": "$25.00 33|80"}]}')   # invented values and ids


def build_prompt(doc: D.Doc, page: int) -> str:
    text = "\n".join(f"{' '.join(doc.words[i][1] for i in pc.words)} {pc.id}" for pc in doc.pieces[page])
    descs = "\n".join(f"- {f}: {DESC[f]}" for f in FIELDS)
    schema = '{"rows": [{' + ", ".join(f'"{f}": "<text> <identifier>" or null' for f in FIELDS) + "}, ...]}  (a list of such strings when a row has several cells of that field)"
    return (f"<Document>\n{text}\n</Document>\n<Task>\nThis page may contain a table of line items. Extract every row of that table. "
            f"Every document line ends with its identifier (two numbers separated by |).\nRules:\n"
            f"1. One JSON object per table row, in reading order. Do not merge, skip or repeat rows.\n"
            f"2. Write each cell as the text copied exactly from ONE line, followed by that line's identifier, for example \"Desk lamp 33|25\".\n"
            f"3. Copy only the cell's value, not the table header and not other cells on the same line. Leave out fields the table does not have.\n"
            f"4. A cell that spans several lines is written as one 'text identifier' per line, separated by a newline.\n"
            f"5. Use null for a cell that is empty in that row. If a row has several cells of the same field, write that field's value as a JSON "
            f"list of cells. If the page has no table rows, answer {{\"rows\": []}}.\n"
            f"Field descriptions:\n{descs}\nExample of the answer format (a different document): {EXAMPLE}\n"
            f"Answer with this JSON object:\n{schema}\n</Task>\n<Extraction>\n")


def parse_rows(raw: str) -> list[dict] | None:
    """The row objects of an answer; works on a truncated answer too. None = not an answer."""
    s = re.sub(r"```(?:json)?", "", raw)
    k = s.find('"rows"')
    rows = []
    for m in re.finditer(r"\{[^{}]*\}", s[k:] if k >= 0 else s):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and any(f in o for f in FIELDS):
            rows.append(o)
    return rows if k >= 0 or rows else None


def _words(text: str, pc: D.Piece, doc: D.Doc):
    toks = text.split()
    ptoks = [doc.words[i][1] for i in pc.words]
    k = next((k for k in range(len(ptoks) - len(toks) + 1) if toks and ptoks[k:k + len(toks)] == toks), None)
    return None if k is None else set(pc.words[k:k + len(toks)])


def decode_page(doc: D.Doc, page: int, raw: str, st: Counter, repair: str = "row") -> list[list[tuple[str, frozenset]]]:
    """Rows of (field, word set) from one page's answer (same logic as `adbuy_sg.decode_page`, pieces instead of OCR lines)."""
    pieces = doc.pieces.get(page, [])
    by_id = {pc.id: pc for pc in pieces}
    st["outputs"] += 1
    rows = parse_rows(raw)
    if rows is None:
        st["unparsed"] += 1
        return []
    out = []
    for obj in rows:
        parsed: dict[str, list] = {}   # field -> cells; each cell is its list of (text, id) segments
        for f in FIELDS:
            v = obj.get(f)
            cs = [sg._segments(x) for x in (v if isinstance(v, list) else [v]) if isinstance(x, str) and x.strip()]
            if any(cs):
                parsed[f] = [c for c in cs if c]
        direct = {f: [[(_words(t, by_id[i], doc) if i in by_id else None) for t, i in segs] for segs in cs] for f, cs in parsed.items()}
        ys = sorted((by_id[i].box[1] + by_id[i].box[3]) / 2 for f, cs in parsed.items() for segs, dg in zip(cs, direct[f])
                    for (t, i), g in zip(segs, dg) if g is not None)
        row_y = ys[len(ys) // 2] if ys else None
        cells = []
        for f, cs in parsed.items():
            for segs, dg in zip(cs, direct[f]):
                st["cells"] += 1
                ws: set[int] = set()
                for (text, lid), got in zip(segs, dg):
                    st["segments"] += 1
                    st["id_valid"] += lid in by_id
                    if got is None and repair != "none":
                        c = _pick([c for c in pieces if _words(text, c, doc)], lid, row_y, repair)
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


def ann_of(i: str) -> dict:
    return json.loads((D.DATA / "annotations" / f"{i}.json").read_text(encoding="utf-8"))


def cmd_manifests(_a) -> None:
    sp = json.loads((D.ROOT / "results" / "docile_rows" / "splits.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("test", "pilot"):
        (OUT / f"{name}.json").write_text(json.dumps(sp[name]), encoding="utf-8")
    print(f"test {len(sp['test'])} documents, pilot {len(sp['pilot'])} documents")


def synthetic_answer(doc: D.Doc, ann: dict, page: int) -> str:
    rows = []
    for row in D.gold_rows(doc, ann):
        if row[0][2] == page:
            obj: dict = {}
            for f, ws, p in row:
                cell = "\n".join(f"{t} {i}" for t, i in D.cite(doc, ws, p))
                obj[f] = cell if f not in obj else (obj[f] if isinstance(obj[f], list) else [obj[f]]) + [cell]
            rows.append(obj)
    return json.dumps({"rows": rows})


def cmd_oracle(a) -> None:
    ids = json.loads((OUT / "test.json").read_text(encoding="utf-8"))[:a.n]
    tl, tc, st = [0, 0, 0], [0, 0, 0], Counter()
    for i in ids:
        ann, doc = ann_of(i), D.Doc(i, GAP)
        gold = [{(f, ws) for f, ws, _ in row} for row in D.gold_rows(doc, ann)]
        if not gold:
            continue
        pred = [set(r) for p in doc.pieces for r in decode_page(doc, p, synthetic_answer(doc, ann, p), st)]
        tl, tc = [x + y for x, y in zip(tl, lir_counts(pred, gold))], [x + y for x, y in zip(tc, cell_counts(pred, gold))]
    print(f"oracle through the decode path on {len(ids)} test documents: lir_f1 {f1(tl) * 100:.2f}, cell_f1 {f1(tc) * 100:.2f}; {dict(st)}")
    assert f1(tl) > 0.97


def cmd_run(a) -> None:
    ids = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["doc_id"], r["page"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines()) if not r.get("error")} if out.exists() and out.stat().st_size else set()   # a page that failed is retried on a re-run
    docs = {i: D.Doc(i, GAP) for i in ids}
    units = [(i, p) for i in ids for p in range(len(docs[i].pieces))]
    units = [u for u in units[a.shard::a.num_shards] if u not in done]
    if a.limit:
        units = units[:a.limit]
    print(f"[shard {a.shard}] {len(units)} pages on {a.device} (arm {a.arm})", flush=True)
    if not units:
        return

    class _Stub:
        def sample(self, prompt, image, k, max_new_tokens, prefill=""):
            return ['{"rows": []}'] * k

    if not a.stub:
        patch_sdpa()
    gen = _Stub() if a.stub else sg._vlm_class()(a.model, a.device, a.dtype, a.max_pixels, a.quant4)
    deadline = time.time() + a.deadline_hours * 3600 if a.deadline_hours else None
    with open(out, "a", encoding="utf-8") as fh:
        for k, (i, p) in enumerate(units, 1):
            if deadline and time.time() > deadline:
                print(f"[shard {a.shard}] deadline reached after {k - 1} units; stopping cleanly (re-run to resume)", flush=True)
                break
            row = {"doc_id": i, "page": p, "arm": a.arm, "samples": [], "error": None}
            t0 = time.time()
            try:
                if docs[i].pieces[p]:
                    prompt = build_prompt(docs[i], p)
                    tok = None if a.stub else gen.proc.tokenizer
                    row["prompt_tokens"] = len(tok(prompt).input_ids) if tok else None
                    image = None
                    if a.arm == "image" and not (row["prompt_tokens"] and row["prompt_tokens"] > a.image_cap_tokens):   # as in adbuy_sg: the cap is effectively off
                        from zero_shot_kv import load_image
                        image = load_image({"image": str(D.DATA / "pdfs" / f"{i}.pdf"), "pdf_page": p})
                    row["image_dropped"] = a.arm == "image" and image is None
                    if not a.stub:
                        import torch
                        torch.cuda.reset_peak_memory_stats(a.device)
                    try:
                        row["samples"] = gen.sample(prompt, image, 1, a.max_new_tokens)
                    except Exception as e:   # out of memory with the image: answer this page from the text alone, and say so in the row
                        if image is None or "out of memory" not in str(e).lower():
                            raise
                        import torch
                        torch.cuda.empty_cache()
                        row["image_dropped"], row["oom_retry"] = True, True
                        row["samples"] = gen.sample(prompt, None, 1, a.max_new_tokens)
                    if not a.stub:
                        row["peak_gb"] = round(torch.cuda.max_memory_allocated(a.device) / 2**30, 2)
                    row["new_tokens"] = len(tok(row["samples"][0]).input_ids) if tok else None
                else:
                    row["samples"] = ['{"rows": []}']
            except Exception as e:  # one bad page must not kill a multi-hour run
                row["error"] = f"{type(e).__name__}: {e}"
            if not a.stub:
                import torch
                torch.cuda.empty_cache()
            row["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            rate = f" {row['new_tokens'] / max(row['seconds'], 1e-6):.1f} tok/s" if row.get("new_tokens") else ""
            print(f"[shard {a.shard}] {k}/{len(units)} {i} p{p} {row['seconds']}s prompt {row.get('prompt_tokens')} out {row.get('new_tokens')}{rate} peak {row.get('peak_gb')} GB"
                  f"{' OOM-RETRY no image' if row.get('oom_retry') else ''}{' ERR ' + row['error'] if row['error'] else ''}", flush=True)


def complete(doc: D.Doc, answers: dict[int, str]) -> bool:
    """True if every page with text has an answer (a run cut by the deadline or by errors leaves documents half done; they are not scored)."""
    return all(p in answers for p, pcs in doc.pieces.items() if pcs)


def read_pages(paths: list[str]) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    for p in paths:
        for fn, rows in sg.read_rows([p]).items():
            out.setdefault(fn, {}).update({r["page"]: r["samples"][0] for r in rows if r["samples"]})
    return out


def cmd_gate(a) -> None:
    ids = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    pages = read_pages(a.rows)
    st: Counter = Counter()
    gold_pages = answered = 0
    for i in ids:
        if i not in pages:
            continue
        ann, doc = ann_of(i), D.Doc(i, GAP)
        has = {row[0][2] for row in D.gold_rows(doc, ann)}
        for p, raw in pages[i].items():
            rows = decode_page(doc, p, raw, st)
            if p in has:
                gold_pages += 1
                answered += bool(rows)
    parse, ids_ok = 1 - st["unparsed"] / max(1, st["outputs"]), st["id_valid"] / max(1, st["segments"])
    res = {"outputs": st["outputs"], "parse_rate": round(parse, 4), "cells": st["cells"], "id_valid_rate": round(ids_ok, 4),
           "pages_with_gold_rows": gold_pages, "answered_rate": round(answered / max(1, gold_pages), 4), "repaired": st["repaired"], "ungrounded": st["ungrounded"],
           "decision": "keep" if parse >= 0.9 and ids_ok >= 0.9 and answered >= 0.9 * gold_pages else "stop_and_review"}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def cmd_score(a) -> None:
    ids = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    report = {}
    for arm in ("A2", "A1"):
        pages = read_pages(sorted(glob.glob(str(Path(a.rows_dir) / f"rows_{arm}_*.jsonl"))))
        if not pages:
            continue
        tot, per_doc, st = {"lir": [0, 0, 0], "cell": [0, 0, 0]}, {}, Counter()
        n_done = 0
        for i in ids:
            ann, doc = ann_of(i), D.Doc(i, GAP)
            if not complete(doc, pages.get(i, {})):
                continue
            n_done += 1
            gold = [{(f, ws) for f, ws, _ in row} for row in D.gold_rows(doc, ann)]
            pred = [set(r) for p in sorted(doc.pieces) for r in decode_page(doc, p, pages[i].get(p, '{"rows": []}'), st)]
            if not gold:
                continue
            c = {"lir": lir_counts(pred, gold), "cell": cell_counts(pred, gold)}
            per_doc[i] = c
            tot = {k: [x + y for x, y in zip(tot[k], c[k])] for k in tot}
        report[arm] = {"lir_f1": round(f1(tot["lir"]) * 100, 2), "cell_f1": round(f1(tot["cell"]) * 100, 2), "counts": tot, "decode": dict(st), "per_doc": per_doc,
                       "documents_scored": n_done, "documents_in_manifest": len(ids)}
        print(arm, {k: report[arm][k] for k in ("lir_f1", "cell_f1", "decode")})
    (Path(a.rows_dir) / "report.json").write_text(json.dumps(report), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifests")
    sub.add_parser("oracle").add_argument("--n", type=int, default=200)
    p = sub.add_parser("run")
    p.add_argument("--manifest", required=True); p.add_argument("--arm", choices=["image", "text"], required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct"); p.add_argument("--device", default="cuda:0"); p.add_argument("--dtype", default="float16")
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28); p.add_argument("--quant4", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=2400)
    p.add_argument("--image-cap-tokens", type=int, default=8500)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--deadline-hours", type=float, default=0.0); p.add_argument("--stub", action="store_true")
    p = sub.add_parser("gate")
    p.add_argument("--manifest", required=True); p.add_argument("--rows", nargs="+", required=True); p.add_argument("--out", default=None)
    p = sub.add_parser("score")
    p.add_argument("--manifest", default=str(OUT / "test.json")); p.add_argument("--rows-dir", required=True)
    a = ap.parse_args()
    {"manifests": cmd_manifests, "oracle": cmd_oracle, "run": cmd_run, "gate": cmd_gate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
