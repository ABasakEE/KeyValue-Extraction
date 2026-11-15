"""Study 29, Track E: LoRA-tuned grounded VLM on VRDU registration forms (UTL / MTL, N = 200).

Registered in STUDIES.md (study 29) before this code existed.  One call per page: image + numbered OCR words + schema in,
`field | a-b` lines out (src/vlm_sft.py).  Subcommands:

    python scripts/vrdu_vlm_sft.py selftest                                   # CPU, no model
    python scripts/vrdu_vlm_sft.py prepare   --out results/vlm_sft            # eval manifests and reference spans
    python scripts/vrdu_vlm_sft.py train     --regime utl --fold 0 --out results/vlm_sft/utl_fold0 [--smoke]
    python scripts/vrdu_vlm_sft.py train     --regime mtl --out results/vlm_sft/mtl
    python scripts/vrdu_vlm_sft.py gen       --eval utl --fold 0 --adapter DIR --out rows.jsonl      # adapter "none" = zero-shot E0
    python scripts/vrdu_vlm_sft.py score     --eval utl --rows rows_f0.jsonl rows_f1.jsonl rows_f2.jsonl --arm E1 --out metrics.json
    python scripts/vrdu_vlm_sft.py gate      --eval utl --rows ...

Fold k = held-out template FOLDS[k] (the numbering of studies 9 and 12).  Image flags: `--no-image` is the E3 ablation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from vlm_sft import MAX_PAGES, SUBCORPUS, build_prompt, decode_pages, fields_and_desc, gold_spans, page_target, page_units, parse_page  # noqa: E402

FOLDS = {0: "Amendment", 1: "Dissemination_Report", 2: "Short-Form"}
VRDU = ROOT / "data" / "vrdu"
REFS = {"dev": ROOT / "results" / "matched_eval" / "spans.jsonl", "confirm": ROOT / "results" / "confirm_v2" / "matched" / "spans.jsonl"}
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
SEED = 13


# ---------------------------------------------------------------- data
def load_all():
    from data.vrdu_loader import load_vrdu

    examples, _, id2label = load_vrdu(VRDU)
    return {e.guid: e for e in examples}, examples, id2label


def read_ref(parts=("dev", "confirm")) -> list[dict]:
    return [json.loads(l) for p in parts for l in REFS[p].read_text(encoding="utf-8").splitlines() if l.strip()]


def split_for(examples, regime: str, fold: int | None, seed: int = 0):
    from data.vrdu_loader import load_splits

    splits = load_splits(examples, VRDU / SUBCORPUS / "few_shot-splits", regime=regime.upper(), n_train=200, seed=seed, protocol="strict")
    if regime == "utl":
        splits = [s for s in splits if s.held_out_template == FOLDS[fold]]
    assert len(splits) == 1, [s.name for s in splits]
    return splits[0]


def eval_docs(by, examples, kind: str, fold: int | None) -> list:
    """Documents to score: utl = dev+confirm documents of the held-out template, confirm = confirm only, mtl = lv2 test part."""
    if kind == "mtl":
        return split_for(examples, "mtl", None).test
    parts = ("dev", "confirm") if kind == "utl" else ("confirm",)
    refs = [r for r in read_ref(parts) if fold is None or int(r["fold"]) == fold]
    return [by[r["guid"]] for r in refs]


def pdf_image(guid: str, page: int):
    import pymupdf
    from PIL import Image

    with pymupdf.open(VRDU / SUBCORPUS / "main" / "pdfs" / guid) as doc:
        pix = doc[page].get_pixmap(dpi=150)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


# ---------------------------------------------------------------- model
LLM_RE = r"(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"   # the vision MLP reuses these names
VISION_RE = r"visual\.blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.(gate|up|down)_proj)"


def merger_module(model):
    """The vision-to-language projector (`visual.merger`), found through any PEFT wrapper."""
    for n, m in model.named_modules():
        if n.endswith("visual.merger"):
            return m
    raise RuntimeError("no visual.merger module found")


def load_model(adapter: str | None, max_pixels: int, device: str, trainable: bool = False, vision_lora: bool = False):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").to(device)
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
    if trainable:
        from peft import LoraConfig, get_peft_model

        # rank 64 / alpha 64 (scale 1) on the language model; the projector is trained in full; vision adapters only for arm E4
        pattern = f"^({LLM_RE})$" + (f"|^({VISION_RE})$" if vision_lora else "")
        cfg = LoraConfig(r=64, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=pattern)
        model = get_peft_model(model, cfg)
        names = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
        n_llm, n_vis = sum("visual" not in n for n in names), sum("visual.blocks" in n for n in names)
        print(f"LoRA modules: language model {n_llm}, vision blocks {n_vis}", flush=True)
        if n_llm == 0 or (vision_lora and n_vis == 0) or (not vision_lora and n_vis):
            raise RuntimeError(f"target pattern matched the wrong modules (llm {n_llm}, vision {n_vis}); check the layer names")
        for n, p in model.named_parameters():
            if "visual.merger." in n:
                p.requires_grad_(True)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()  # fp32 trainable weights; the fp16 base stays frozen
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    elif adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        mp = Path(adapter) / "merger.pt"
        if mp.exists():
            merger_module(model).load_state_dict(torch.load(mp, map_location=device))
        else:
            print(f"WARNING: no {mp}; the projector stays untuned", flush=True)
    return model.eval() if not trainable else model, proc


def wrap_head(model):
    """Make the LM head compute logits only for the last `keep` positions (None = all).

    transformers 4.51.3 builds full-sequence float32 logits (and a shifted copy) for the loss: over 4,000 tokens x 152k vocabulary
    that is several GB and caused out-of-memory errors on long pages in the smoke test.  Training needs only the answer positions.
    """
    import torch

    class KeepLast(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head, self.keep = head, None

        def forward(self, h):
            return self.head(h if self.keep is None else h[:, -self.keep:])

    inner = model.base_model.model  # PeftModel -> Qwen2_5_VLForConditionalGeneration
    if not isinstance(inner.lm_head, KeepLast):
        inner.lm_head = KeepLast(inner.lm_head)
    return inner.lm_head


def grad_norm(params) -> float:
    import math

    sq = [float((p.grad.float() ** 2).sum()) for p in params if p.grad is not None]
    return math.sqrt(sum(sq)) if sq else 0.0


def quick_check(model, proc, docs, fields, desc, id2label, args) -> dict:
    """Collapse check: greedy-decode a few validation documents with the model in memory and score them.

    Validation documents come from the strict split (no held-out-template document); a tuned model that scores near 0 here
    has collapsed (v1 of Graph-BROS did, silently).
    """
    import torch
    from matched_eval import f1_of, span_counts

    model.eval()
    model.config.use_cache = True
    tot, sample = [0, 0, 0], ""
    for ex in docs:
        answers = {}
        for u in page_units(ex):
            image = None if args.no_image else pdf_image(ex.guid, u.page)
            enc, _ = encode(proc, build_prompt(u, fields, desc), image, None)
            enc = enc.to(args.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                gen = model.generate(**enc, max_new_tokens=160, do_sample=False)
            answers[u.page] = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            sample = sample or answers[u.page]
        pred, _ = decode_pages(page_units(ex), answers, fields, ex.words)
        c = span_counts(pred, gold_spans(ex, id2label))["_micro"]
        tot = [tot[i] + c[i] for i in range(3)]
    return {"docs": len(docs), "f1": round(f1_of(tot), 4), "tp_pred_gold": tot, "first_answer": sample}


CHECK_MIN_F1 = 0.10


def encode(proc, prompt: str, image, target: str | None):
    content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    if target is None:
        return proc(text=[text], images=[image] if image is not None else None, return_tensors="pt"), 0
    tail = target + "<|im_end|>\n"
    enc = proc(text=[text + tail], images=[image] if image is not None else None, return_tensors="pt")
    return enc, len(proc.tokenizer(tail, add_special_tokens=False).input_ids)


# ---------------------------------------------------------------- train
def build_samples(train_docs, id2label, fields, desc, use_image: bool) -> list[dict]:
    out = []
    for ex in train_docs:
        gold = gold_spans(ex, id2label)
        for u in page_units(ex):
            out.append({"guid": ex.guid, "page": u.page, "prompt": build_prompt(u, fields, desc), "target": page_target(gold, u, fields)})
    return out


def cmd_train(args) -> None:
    import math

    import torch

    by, examples, id2label = load_all()
    fields, desc = fields_and_desc()
    sp = split_for(examples, args.regime, args.fold, args.split_seed)
    print(sp.describe(), flush=True)
    samples = build_samples(sp.train, id2label, fields, desc, not args.no_image)
    out = Path(args.out)
    if args.smoke:  # results/ is untracked, so a rerun would otherwise skip the training and reuse the old adapter
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train_done.json").exists() and (out / "adapter").exists():  # train_done.json is pushed, the adapter is not
        print("already trained, skipping", out)
        return
    seed = SEED + args.split_seed  # replication seed: split seed 1 and training seed 14
    random.Random(seed).shuffle(samples)
    if args.smoke:
        samples = samples[:6]
    torch.manual_seed(seed)
    torch.cuda.set_device(args.device)  # memory stats, the scaler and empty_cache refer to the current device
    model, proc = load_model(None, args.max_pixels, args.device, trainable=True, vision_lora=args.vision_lora)
    model.config.use_cache = False
    head = wrap_head(model)
    params = [p for p in model.parameters() if p.requires_grad]
    mparams = [p for n, p in model.named_parameters() if p.requires_grad and "visual.merger." in n]
    vparams = [p for n, p in model.named_parameters() if p.requires_grad and "visual.blocks" in n]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = max(1, math.ceil(len(samples) * args.epochs / args.accum))
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda")
    log, skipped, t0, n, step, gns, mgns, vgns = [], 0, time.time(), 0, 0, [], [], []
    model.train()
    for ep in range(args.epochs):
        random.Random(seed + ep).shuffle(samples)
        run = []
        for i, s in enumerate(samples):
            try:
                image = pdf_image(s["guid"], s["page"]) if not args.no_image else None
                enc, n_tail = encode(proc, s["prompt"], image, s["target"])
                if enc["input_ids"].shape[1] > args.max_len:
                    skipped += 1
                    continue
                enc = enc.to(args.device)
                head.keep = n_tail + 1  # the position before the first answer token predicts it
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**enc).logits
                loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), enc["input_ids"][0, -n_tail:]) / args.accum
                scaler.scale(loss).backward()
                run.append(loss.item() * args.accum)
                n += 1
            except torch.cuda.OutOfMemoryError:
                skipped += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"OOM on {s['guid'][:40]} p{s['page']}; skipped", flush=True)
                continue
            if n % args.accum == 0:
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                gns.append(gn)
                mgns.append(grad_norm(mparams))
                vgns.append(grad_norm(vparams))
                # PEFT issue 2880: with gradient checkpointing a VLM can train with zero gradients; the scaler's early inf steps are normal
                for what, vals, present in (("LoRA", gns, True), ("projector", mgns, bool(mparams)), ("vision adapter", vgns, bool(vparams))):
                    if step == 2 and present and not any(math.isfinite(g) and g > 0 for g in vals):
                        raise RuntimeError(f"no gradient reached the {what} weights in the first 3 optimizer steps: {vals}")
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 10 == 0 or args.smoke:
                    print(f"ep {ep} step {step}/{steps} loss {sum(run[-args.accum:]) / len(run[-args.accum:]):.4f} {time.time() - t0:.0f}s "
                          f"gradnorm {gns[-1]:.3g} (projector {mgns[-1]:.3g}, vision {vgns[-1]:.3g}) peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        if not run or not all(math.isfinite(x) for x in run):
            raise RuntimeError("non-finite or empty training loss; per the registration, change the recipe only by a dated amendment")
        log.append({"epoch": ep, "mean_loss": sum(run) / len(run), "samples": len(run), "skipped_so_far": skipped, "seconds": round(time.time() - t0)})
        print(log[-1], flush=True)
        model.save_pretrained(out / "adapter")
        torch.save(merger_module(model).state_dict(), out / "adapter" / "merger.pt")
    head.keep = None
    # collapse check on validation documents (strict split); the done marker is written only if it passes
    chk_docs = sp.val[:]
    random.Random(seed).shuffle(chk_docs)
    chk = quick_check(model, proc, chk_docs[:2 if args.smoke else args.check_docs], fields, desc, id2label, args)
    chk["min_f1"] = CHECK_MIN_F1
    chk["passed"] = chk["f1"] >= CHECK_MIN_F1
    (out / "collapse_check.json").write_text(json.dumps(chk, indent=2), encoding="utf-8")
    print("collapse check:", chk, flush=True)
    if not chk["passed"] and not args.smoke:
        raise RuntimeError(f"collapse check failed: validation F1 {chk['f1']} < {CHECK_MIN_F1}; the adapter is kept for inspection")
    (out / "train_done.json").write_text(json.dumps({"log": log, "skipped": skipped, "skipped_fraction": round(skipped / max(1, len(samples) * args.epochs), 4), "optimizer_steps": step, "train_samples": len(samples),
                                                      "peak_gb": torch.cuda.max_memory_allocated() / 2**30, "split": sp.name,
                                                      "vision_lora": args.vision_lora, "collapse_check_f1": chk["f1"]}, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- generate
def cmd_gen(args) -> None:
    import torch

    by, examples, _ = load_all()
    fields, desc = fields_and_desc()
    docs = eval_docs(by, examples, args.eval, args.fold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["guid"], r["page"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())} if out.exists() and out.stat().st_size else set()
    units = [(ex, u) for ex in docs for u in page_units(ex) if (ex.guid, u.page) not in done]
    if args.limit:
        units = units[:args.limit]
    print(f"{len(units)} pages to do ({len(done)} done)", flush=True)
    if not units:
        return
    torch.cuda.set_device(args.device)
    model, proc = load_model(None if args.adapter == "none" else args.adapter, args.max_pixels, args.device)
    deadline = time.time() + args.deadline_hours * 3600 if args.deadline_hours else None
    with open(out, "a", encoding="utf-8") as fh:
        for k, (ex, u) in enumerate(units, 1):
            if deadline and time.time() > deadline:
                print("deadline reached; stopping cleanly (re-run to resume)", flush=True)
                break
            row, t0 = {"guid": ex.guid, "doc_id": f"{SUBCORPUS}/{ex.guid}", "page": u.page, "samples": [], "error": None}, time.time()
            try:
                image = None if args.no_image else pdf_image(ex.guid, u.page)
                enc, _ = encode(proc, build_prompt(u, fields, desc), image, None)
                enc = enc.to(args.device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
                row["samples"] = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            except Exception as e:  # one bad page must not kill a multi-hour run
                row["error"] = f"{type(e).__name__}: {e}"
                torch.cuda.empty_cache()
            row["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if k % 20 == 0 or k == len(units):
                print(f"{k}/{len(units)} {row['seconds']}s{' ERR ' + row['error'] if row['error'] else ''}", flush=True)


# ---------------------------------------------------------------- decode, gate, score
def read_rows(paths) -> dict[str, dict[int, str]]:
    by: dict[str, dict[int, str]] = {}
    for p in paths:
        if Path(p).exists():
            for l in Path(p).read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    if not r.get("error") and r["samples"]:
                        by.setdefault(r["guid"], {})[r["page"]] = r["samples"][0]
    return by


def cmd_gate(args) -> None:
    by, examples, _ = load_all()
    fields, _ = fields_and_desc()
    rows = read_rows(args.rows)
    pages = ok = errors = 0
    for p in args.rows:
        for l in Path(p).read_text(encoding="utf-8").splitlines() if Path(p).exists() else []:
            r = json.loads(l)
            pages += 1
            errors += bool(r.get("error"))
            if not r.get("error"):
                _, st = parse_page(r["samples"][0], 10**6, fields)
                ok += st.get("rows", 0) >= len(fields) // 2  # a page parses if at least half of the field rows are well formed
    res = {"pages": pages, "parse_rate": round(ok / max(1, pages), 4), "error_rate": round(errors / max(1, pages), 4)}
    res["decision"] = "pass" if res["parse_rate"] >= 0.9 and res["error_rate"] < 0.05 else "fail: report as the finding"
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


def cmd_prepare(args) -> None:
    by, examples, _ = load_all()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = read_ref()
    manifest = [{"doc_id": f"{SUBCORPUS}/{r['guid']}", "dataset": SUBCORPUS, "template": r["template"], "n_pages": by[r["guid"]].n_pages,
                 "source": f"data/vrdu/{SUBCORPUS}/main/pdfs/{r['guid']}", "pages": [], "gold": {"fields": {}}} for r in ref]
    (out / "manifest_utl.json").write_text(json.dumps(manifest), encoding="utf-8")
    (out / "ref_utl.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ref), encoding="utf-8")
    print(f"{len(ref)} UTL documents; per fold {Counter(int(r['fold']) for r in ref)}")


def cmd_score(args) -> None:
    import numpy as np

    from matched_eval import ci, f1_of, paired_bootstrap, span_counts

    by, examples, id2label = load_all()
    fields, _ = fields_and_desc()
    rows = read_rows(args.rows)
    name = f"vlm_{args.arm}"
    if args.eval == "mtl":
        docs = [ex for ex in eval_docs(by, examples, "mtl", None) if len({u.page for u in page_units(ex)} & set(rows.get(ex.guid, {}))) == len(page_units(ex))]
        per = [span_counts(decode_pages(page_units(ex), rows[ex.guid], fields, ex.words)[0], gold_spans(ex, id2label)) for ex in docs]
        tot = [sum(x["_micro"][i] for x in per) for i in range(3)]
        boot = paired_bootstrap({name: per}, "_micro")[name]
        rep = {"arm": args.arm, "eval": "mtl", "n_docs": len(docs), "f1": round(f1_of(tot), 4), "f1_ci95": ci(boot),
               "per_field_f1": {f: round(f1_of([sum(x.get(f, (0, 0, 0))[i] for x in per) for i in range(3)]), 4) for f in fields}}
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(json.dumps(rep, indent=1))
        return
    import schema_guided_kv as sg

    ns = argparse.Namespace(manifest=args.manifest, ref=args.ref, rows=args.rows, arm=args.arm, format="span", reference="vlm_both", out=args.out)
    sg.cmd_score(ns)


# ---------------------------------------------------------------- self-test (no model)
def cmd_selftest(_args) -> None:
    import tempfile

    by, examples, id2label = load_all()
    fields, desc = fields_and_desc()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cmd_prepare(argparse.Namespace(out=str(td)))
        # oracle rows: the gold answer of every page of the first 12 UTL documents, written like `gen` would
        ref = read_ref()
        docs = [by[r["guid"]] for r in ref[:12]]
        with open(td / "rows.jsonl", "w", encoding="utf-8") as fh:
            for ex in docs:
                g = gold_spans(ex, id2label)
                for u in page_units(ex):
                    fh.write(json.dumps({"guid": ex.guid, "doc_id": f"{SUBCORPUS}/{ex.guid}", "page": u.page, "samples": [page_target(g, u, fields)], "error": None}) + "\n")
        got = read_rows([td / "rows.jsonl"])
        assert len(got) == 12
        from matched_eval import f1_of, span_counts

        tot = [0, 0, 0]
        for ex, r in zip(docs, ref):
            pred, _ = decode_pages(page_units(ex), got[ex.guid], fields, ex.words)
            c = span_counts(pred, r["gold"])["_micro"]
            tot = [tot[i] + c[i] for i in range(3)]
        print("oracle F1 on 12 documents:", round(100 * f1_of(tot), 2), tot)
        assert f1_of(tot) > 0.9
        sg_out = td / "metrics.json"
        cmd_score(argparse.Namespace(eval="utl", rows=[str(td / "rows.jsonl")], arm="oracle", out=str(sg_out),
                                     manifest=str(td / "manifest_utl.json"), ref=str(td / "ref_utl.jsonl")))
        m = json.loads(sg_out.read_text(encoding="utf-8"))
        assert m["n_docs"] == 12 and m["pooled"]["f1"]["vlm_oracle"] > 0.9, m["pooled"]["f1"]
        # a page prompt and a training pair, shown once
        ex = docs[0]
        u = page_units(ex)[0]
        pr = build_prompt(u, fields, desc)
        print(pr[:700], f"\n[... {len(pr) - 900} more characters, display only; the real prompt is complete ...]\n", pr[-200:],
              "\n--- target ---\n" + page_target(gold_spans(ex, id2label), u, fields))
        # prompt sizes (words) per page: the tokens are measured on the GPU in the smoke cell
        sizes = sorted(len(u.words) for e in examples[:300] for u in page_units(e))
        print("page words: median", sizes[len(sizes) // 2], "p95", sizes[int(0.95 * len(sizes))], "max", sizes[-1])
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("prepare"); p.add_argument("--out", default="results/vlm_sft")
    p = sub.add_parser("train")
    p.add_argument("--regime", choices=["utl", "mtl"], required=True); p.add_argument("--fold", type=int, default=None)
    p.add_argument("--out", required=True); p.add_argument("--device", default="cuda:0"); p.add_argument("--no-image", action="store_true")
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--accum", type=int, default=8); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-pixels", type=int, default=640 * 28 * 28); p.add_argument("--max-len", type=int, default=6144)
    p.add_argument("--smoke", action="store_true"); p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--vision-lora", action="store_true"); p.add_argument("--check-docs", type=int, default=20)
    p = sub.add_parser("gen")
    p.add_argument("--eval", choices=["utl", "confirm", "mtl"], required=True); p.add_argument("--fold", type=int, default=None)
    p.add_argument("--adapter", required=True); p.add_argument("--out", required=True); p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-image", action="store_true"); p.add_argument("--max-pixels", type=int, default=640 * 28 * 28)
    p.add_argument("--max-new-tokens", type=int, default=160); p.add_argument("--split-seed", type=int, default=0); p.add_argument("--limit", type=int, default=0); p.add_argument("--deadline-hours", type=float, default=0.0)
    p = sub.add_parser("gate"); p.add_argument("--rows", nargs="+", required=True); p.add_argument("--out", default=None)
    p = sub.add_parser("score")
    p.add_argument("--eval", choices=["utl", "mtl"], required=True); p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--arm", required=True); p.add_argument("--out", required=True)
    p.add_argument("--manifest", default="results/vlm_sft/manifest_utl.json"); p.add_argument("--ref", default="results/vlm_sft/ref_utl.jsonl")
    a = ap.parse_args()
    {"selftest": cmd_selftest, "prepare": cmd_prepare, "train": cmd_train, "gen": cmd_gen, "gate": cmd_gate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
