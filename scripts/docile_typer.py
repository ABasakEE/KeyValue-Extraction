"""Study 43 (DocILE): detector and typer. The BROS predictions say where the monetary cells are; a VLM says which field each one is.

    python scripts/docile_typer.py run --tags N1000_S0,Nfull_T21 --out rows.jsonl [--shard 0 --num-shards 2] [--stub bros|gold] [--limit N]
    python scripts/docile_typer.py score --rows rows_0.jsonl rows_1.jsonl [--tags ...] [--out results/docile_typer/scores.json]

Detector: the saved test predictions of a Study 39 / 41 run (`test_entities.jsonl`), unchanged except the field of the monetary spans.
Typer: per page with at least one candidate, one call to Qwen3-VL-8B (page image + the page's OCR lines with their identifiers + the numbered candidates, the BROS field NOT shown);
the answer maps every candidate to one of the four monetary line-item fields. Candidates = predicted spans of line_item_{unit_price,amount}_{gross,net}.
Score: rows from `row_group.group_rows` (tolerance 0.5) on the retyped spans, strict row F1 on the 200 test documents; an unanswered page or an invalid answer keeps the BROS field.
Upper bound ('oracle'): the same retyping with the gold field for every candidate whose extent is exactly a gold span.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src"), str(HERE.parent)]
import docile_bros as B  # noqa: E402
import docile_rows as D  # noqa: E402
import docile_sg as G  # noqa: E402
from docile_column_rules import boot  # noqa: E402
from lir_metric import f1  # noqa: E402
from row_group import group_rows  # noqa: E402

MONEY = ["line_item_unit_price_gross", "line_item_unit_price_net", "line_item_amount_gross", "line_item_amount_net"]
OUT = B.ROOT / "results" / "docile_typer"
RUN_ROOTS = [B.ROOT / "results" / "docile_bros" / "run", B.ROOT / "results" / "docile_bros_full" / "run"]


def pred_path(tag: str) -> Path:
    for root in RUN_ROOTS:
        f = next((root / tag).glob("seed_*/test_entities.jsonl"), None)
        if f:
            return f
    raise FileNotFoundError(tag)


def load_pred(tag: str) -> dict[str, list]:
    return {r["guid"]: r["pred"] for r in map(json.loads, pred_path(tag).read_text(encoding="utf-8").splitlines())}


def candidates(spans: list, pages: list, page: int) -> list[int]:
    """Indices (into the document's span list) of the monetary spans on a page, in list order: the numbering c1, c2, ... of prompt and scorer."""
    return [k for k, sp in enumerate(spans) if sp[0] in MONEY and pages[sp[1]] == page]


def build_prompt(doc: D.Doc, page: int, spans: list, cands: list[int]) -> str:
    lines = "\n".join(f"{' '.join(doc.words[i][1] for i in pc.words)} {pc.id}" for pc in doc.pieces[page])
    piece_of = {i: pc for pc in doc.pieces[page] for i in pc.words}
    items = []
    for n, k in enumerate(cands, 1):
        _, s, e, _ = spans[k]
        items.append(f'c{n}: "{" ".join(doc.words[i][1] for i in range(s, e))}" on the line with identifier {piece_of[s].id if s in piece_of else "?"}')
    descs = "\n".join(f"- {f}: {G.DESC[f]}" for f in MONEY)
    schema = "{" + ", ".join(f'"c{n}": "<field>"' for n in range(1, min(len(cands), 3) + 1)) + (", ..." if len(cands) > 3 else "") + "}"
    return (f"<Document>\n{lines}\n</Document>\n<Task>\nThis page has a table of line items. Another system found the monetary cells listed below; it is sure where "
            f"they are but not which field each one is. Decide the field of every candidate. Every document line ends with its identifier (two numbers separated by |).\n"
            f"Fields:\n{descs}\n"
            f"Use the whole page: the column headers, whether the prices in this document include tax, and the totals and tax lines. A price per unit is not the line total.\n"
            f"Candidates:\n" + "\n".join(items) + f"\nAnswer with one JSON object that gives every candidate exactly one of the four field names, for example {schema}\n</Task>\n<Extraction>\n")


def parse_answer(raw: str, n: int) -> dict[int, str]:
    s = re.sub(r"```(?:json)?", "", raw)
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {}
    try:
        o = json.loads(m.group(0))
    except json.JSONDecodeError:
        o = {}
        for mm in re.finditer(r'"c(\d+)"\s*:\s*"([^"]+)"', m.group(0)):
            o[f"c{mm.group(1)}"] = mm.group(2)
    return {int(k[1:]): v for k, v in o.items() if isinstance(k, str) and re.fullmatch(r"c\d+", k) and v in MONEY and 1 <= int(k[1:]) <= n}


def cmd_run(a) -> None:
    test = json.loads(B.SPLITS.read_text(encoding="utf-8"))["test"]
    tags = a.tags.split(",")
    preds = {t: load_pred(t) for t in tags}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["tag"], r["doc_id"], r["page"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines()) if not r.get("error")} if out.exists() and out.stat().st_size else set()
    docs = {i: D.Doc(i, G.GAP) for i in test}
    units = []
    for t in tags:
        for i in test:
            pages = [w[0] for w in docs[i].words]
            for p in sorted({pages[sp[1]] for sp in preds[t][i] if sp[0] in MONEY}):
                units.append((t, i, p))
    units = [u for u in units[a.shard::a.num_shards] if u not in done]
    if a.limit:
        units = units[:a.limit]
    print(f"[shard {a.shard}] {len(units)} pages on {a.device}", flush=True)
    if not units:
        return
    gen = None
    if not a.stub:
        from row_repair import patch_sdpa
        import schema_guided_kv as sg
        patch_sdpa()
        gen = sg._vlm_class()(a.model, a.device, a.dtype, a.max_pixels, a.quant4)
    deadline = time.time() + a.deadline_hours * 3600 if a.deadline_hours else None
    with open(out, "a", encoding="utf-8") as fh:
        for k, (t, i, p) in enumerate(units, 1):
            if deadline and time.time() > deadline:
                print(f"[shard {a.shard}] deadline reached after {k - 1} units; stopping cleanly (re-run resumes)", flush=True)
                break
            spans, doc = preds[t][i], docs[i]
            pages = [w[0] for w in doc.words]
            cands = candidates(spans, pages, p)
            row = {"tag": t, "doc_id": i, "page": p, "n_cand": len(cands), "samples": [], "error": None}
            t0 = time.time()
            try:
                if a.stub:
                    gold = {(s, e): f for f, s, e, _ in B.spans_of(B.doc_dict(i))}
                    ans = {f"c{n}": (gold.get((spans[c][1], spans[c][2]), spans[c][0]) if a.stub == "gold" else spans[c][0]) for n, c in enumerate(cands, 1)}
                    row["samples"] = [json.dumps(ans)]
                else:
                    import torch
                    from zero_shot_kv import load_image
                    prompt = build_prompt(doc, p, spans, cands)
                    row["prompt_tokens"] = len(gen.proc.tokenizer(prompt).input_ids)
                    image = load_image({"image": str(D.DATA / "pdfs" / f"{i}.pdf"), "pdf_page": p})
                    torch.cuda.reset_peak_memory_stats(a.device)
                    try:
                        row["samples"] = gen.sample(prompt, image, 1, a.max_new_tokens)
                    except Exception as e:   # out of memory with the image: the page is answered from the text alone, and the row says so
                        if "out of memory" not in str(e).lower():
                            raise
                        torch.cuda.empty_cache()
                        row["oom_retry"] = True
                        row["samples"] = gen.sample(prompt, None, 1, a.max_new_tokens)
                    row["peak_gb"] = round(torch.cuda.max_memory_allocated(a.device) / 2**30, 2)
                    torch.cuda.empty_cache()
            except Exception as e:   # one bad page must not kill a multi-hour run
                row["error"] = f"{type(e).__name__}: {e}"
            row["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[shard {a.shard}] {k}/{len(units)} {t} {i[:8]} p{p} cands {len(cands)} {row['seconds']}s{' OOM-RETRY' if row.get('oom_retry') else ''}{' ERR ' + row['error'] if row['error'] else ''}", flush=True)


def retype(spans: list, pages: list, answers: dict[int, dict[int, str]], mode: str, gold: dict | None = None) -> list:
    """Spans with the field of every candidate replaced: by the page's answer (mode 'vlm'; kept when missing) or by the gold field of an exactly matching span (mode 'oracle')."""
    out = [list(sp) for sp in spans]
    if mode == "oracle":
        for k, sp in enumerate(spans):
            if sp[0] in MONEY and (sp[1], sp[2]) in gold:
                out[k][0] = gold[(sp[1], sp[2])]
        return out
    for p, ans in answers.items():
        for n, k in enumerate(candidates(spans, pages, p), 1):
            if n in ans:
                out[k][0] = ans[n]
    return out


def cmd_score(a) -> None:
    test = json.loads(B.SPLITS.read_text(encoding="utf-8"))["test"]
    docs = {g: B.doc_dict(g) for g in test}
    test = [g for g in test if docs[g]["rows"]]
    rows = [json.loads(l) for f in a.rows for l in open(f, encoding="utf-8") if l.strip()]
    tags = a.tags.split(",") if a.tags else sorted({r["tag"] for r in rows})
    stats: Counter = Counter()
    answers: dict[tuple, dict[int, str]] = {}
    for r in rows:
        if r["error"] or not r["samples"]:
            stats[(r["tag"], "page_failed")] += 1
            continue
        parsed = parse_answer(r["samples"][0], r["n_cand"])
        stats[(r["tag"], "pages")] += 1
        stats[(r["tag"], "candidates")] += r["n_cand"]
        stats[(r["tag"], "answered")] += len(parsed)
        answers[(r["tag"], r["doc_id"], r["page"])] = parsed
    report: dict = {"tags": {}}
    per_run: list = []
    for t in tags:
        pred = load_pred(t)
        tot = {arm: {} for arm in ("base", "vlm", "oracle")}
        per_doc = {arm: [] for arm in tot}
        acc = Counter()
        for g in test:
            d = docs[g]
            spans = pred[g]
            gold_f = {(s, e): f for f, s, e, _ in B.spans_of(d)}
            ans = {p: v for (tt, gg, p), v in answers.items() if tt == t and gg == g}
            variants = {"base": spans, "vlm": retype(spans, d["pages"], ans, "vlm"), "oracle": retype(spans, d["pages"], {}, "oracle", gold_f)}
            for arm, sp in variants.items():
                c = B.counts(group_rows(sp, d, 0.5, merge=False), d)
                B.add(tot[arm], c)
                per_doc[arm].append(c["lir"])
            for k, sp in enumerate(spans):
                if sp[0] in MONEY and (sp[1], sp[2]) in gold_f:
                    acc["exact_monetary"] += 1
                    acc["bros_right"] += sp[0] == gold_f[(sp[1], sp[2])]
                    acc["vlm_right"] += variants["vlm"][k][0] == gold_f[(sp[1], sp[2])]
                    acc["vlm_changed"] += variants["vlm"][k][0] != sp[0]
        res = {arm: B.show(v) for arm, v in tot.items()}
        res["delta_vlm_row_f1"] = round(res["vlm"]["row_f1"] - res["base"]["row_f1"], 2)
        res["delta_oracle_row_f1"] = round(res["oracle"]["row_f1"] - res["base"]["row_f1"], 2)
        res["typing_accuracy_on_exact_monetary_spans"] = {"n": acc["exact_monetary"], "bros": round(100 * acc["bros_right"] / max(1, acc["exact_monetary"]), 2),
                                                          "vlm": round(100 * acc["vlm_right"] / max(1, acc["exact_monetary"]), 2), "vlm_changed_the_bros_field": acc["vlm_changed"]}
        res["pages"] = {k[1]: v for k, v in stats.items() if k[0] == t}
        report["tags"][t] = res
        per_run.append((per_doc["base"], per_doc["vlm"]))
        print(t, json.dumps({k: v for k, v in res.items()}))
    if per_run:
        ds = [report["tags"][t]["delta_vlm_row_f1"] for t in tags]
        report["delta_vlm_mean"] = round(st.mean(ds), 2)
        report["delta_vlm_ci95_over_documents"] = boot(per_run, len(test))
        print(f"mean difference (typer - BROS) {report['delta_vlm_mean']:+.2f}, per tag {ds}, 95% interval over documents {report['delta_vlm_ci95_over_documents']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--tags", default="Nfull_T21,N1000_S0"); p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct"); p.add_argument("--device", default="cuda:0"); p.add_argument("--dtype", default="float16")
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28); p.add_argument("--quant4", action="store_true"); p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--deadline-hours", type=float, default=0.0); p.add_argument("--stub", choices=["bros", "gold"], default=None)
    p = sub.add_parser("score")
    p.add_argument("--rows", nargs="+", required=True); p.add_argument("--tags", default=None); p.add_argument("--out", default=None)
    a = ap.parse_args()
    {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
