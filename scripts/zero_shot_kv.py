"""Zero-shot open-schema key-value extraction baseline (no training, no field list).

A small instruction model reads a form page and lists every key-value pair as
<pair><key>...</key><value>...</value></pair>. Two arms share the prompt:

* ``vlm``: the page image (Qwen2.5-VL).
* ``lm``: the page's OCR lines with [x,y] positions on a 0-1000 grid (Qwen2.5 text model).

Scoring needs no hand-written field descriptions:

* FUNSD: predicted pairs vs gold question->answer links.
* VRDU: each predicted key is mapped to a VRDU field by SBERT similarity to the
  field name split into words ("foreign_principle_name" -> "foreign principle name").
  Only page 1 (index 0) is used, and only gold values that appear on it.

Subcommands:
  prepare   build a deterministic page manifest
  run       generate predictions for one arm (shardable across GPUs, resumable)
  score     compute metrics from prediction files
  selftest  check parser and scorer without models
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PROMPT = (
    "List every field on this form page as key-value pairs. A key is a label or question printed "
    "on the form; its value is the information filled in for it. Copy text exactly as it appears. "
    "Skip keys that have no filled-in value. Output only lines of the form:\n"
    "<pair><key>KEY TEXT</key><value>VALUE TEXT</value></pair>"
)
LM_SUFFIX = "\n\nPage text, one line per row, with the [x,y] position of each line on a 0-1000 grid:\n"
PAIR_RE = re.compile(r"<pair>\s*<key>(.*?)</key>\s*<value>(.*?)</value>\s*</pair>", re.S)
MATCH_THRESHOLD = 0.5  # token F1 needed for two strings to count as the same text
FIELD_SIM_THRESHOLD = 0.4  # min SBERT cosine for a predicted key to map to a VRDU field; fixed a priori


# ---------------------------------------------------------------- text utils
def norm(s: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s.lower()).split())


def tok_f1(a: str, b: str) -> float:
    ta, tb = Counter(norm(a).split()), Counter(norm(b).split())
    common = sum((ta & tb).values())
    if not common:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)


def parse_pairs(text: str) -> list[list[str]]:
    pairs = [[k.strip(), v.strip()] for k, v in PAIR_RE.findall(text)]
    return [p for p in pairs if p[0] and p[1]]


def greedy_match(preds: list, golds: list, score) -> int:
    """One-to-one greedy matching; returns number of matched pairs scoring >= MATCH_THRESHOLD."""
    cands = sorted(((score(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(golds)), reverse=True)
    used_p, used_g, n = set(), set(), 0
    for s, i, j in cands:
        if s < MATCH_THRESHOLD:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); n += 1
    return n


def prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


# ---------------------------------------------------------------- manifest
def group_lines(words: list[tuple[str, list[float]]]) -> list[list]:
    """Group (text, [x0,y0,x1,y1]) words into rows by vertical overlap; returns [text, x0, y0]."""
    rows: list[list] = []
    for text, b in sorted(words, key=lambda w: (w[1][1] + w[1][3]) / 2):
        cy, h = (b[1] + b[3]) / 2, max(b[3] - b[1], 1e-6)
        if rows and abs(cy - rows[-1][0]) < 0.5 * h:
            rows[-1][1].append((text, b))
        else:
            rows.append([cy, [(text, b)]])
    out = []
    for _, ws in rows:
        ws.sort(key=lambda w: w[1][0])
        out.append([" ".join(t for t, _ in ws), min(b[0] for _, b in ws), min(b[1] for _, b in ws)])
    return out


def funsd_pages(root: Path) -> list[dict]:
    from PIL import Image

    pages = []
    for ann_path in sorted((root / "testing_data" / "annotations").glob("*.json")):
        img_path = root / "testing_data" / "images" / f"{ann_path.stem}.png"
        w, h = Image.open(img_path).size
        form = json.loads(ann_path.read_text(encoding="utf-8"))["form"]
        by_id = {e["id"]: e for e in form}
        pairs = set()
        for e in form:
            for a, b in e.get("linking", []):
                q, ans = by_id.get(a), by_id.get(b)
                if q and ans and q["label"] == "question" and ans["label"] == "answer" and q["text"].strip() and ans["text"].strip():
                    pairs.add((q["text"].strip(), ans["text"].strip()))
        words = [(wd["text"], [wd["box"][0] / w, wd["box"][1] / h, wd["box"][2] / w, wd["box"][3] / h])
                 for e in form for wd in e.get("words", []) if wd["text"].strip()]
        pages.append({"page_id": f"funsd/{ann_path.stem}", "dataset": "funsd", "template": "funsd",
                      "image": str(img_path), "pdf_page": None, "lines": group_lines(words),
                      "gold": {"pairs": sorted(pairs)}})
    return pages


def vrdu_pages(root: Path, subcorpus: str, n_per_template: int, seed: int) -> list[dict]:
    from data.vrdu_loader import list_templates, template_from_filename

    templates = list_templates(root / subcorpus / "few_shot-splits") if subcorpus == "registration-form" else []
    by_t: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(root / subcorpus / "main" / "dataset.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            t = template_from_filename(r["filename"], templates) if templates else "ad-buy"
            if t is None or not r["ocr"]["pages"]:
                continue
            p0 = r["ocr"]["pages"][0]
            lines = [[ln["text"].strip(), ln["bbox"][1], ln["bbox"][2]] for ln in p0["lines"] if ln["text"].strip()]
            fields: dict[str, list[str]] = defaultdict(list)
            for name, occs in r["annotations"]:
                if not isinstance(name, str):
                    continue  # ad-buy line-item tables: out of scope for a page-level baseline
                for occ in occs:
                    if occ[1] and int(occ[1][0]) == 0 and occ[0].strip():
                        fields[name].append(occ[0].strip())
            page = {"page_id": f"{subcorpus}/{r['filename']}", "dataset": subcorpus, "template": t,
                    "image": str(root / subcorpus / "main" / "pdfs" / r["filename"]), "pdf_page": 0,
                    "lines": lines, "gold": {"fields": dict(fields)}}
            if t == "Short-Form":
                page["gold"]["decoy"] = short_form_filer(r["filename"])
            by_t[t].append(page)
    rng = random.Random(seed)
    out = []
    for t in sorted(by_t):
        out += sorted(rng.sample(by_t[t], min(n_per_template, len(by_t[t]))), key=lambda p: p["page_id"])
    return out


def short_form_filer(filename: str) -> str:
    """The individual filer (the item-1 decoy), named in the filename: '<date>_<registrant>_<Last, First>_Short-Form.pdf'.

    Used for scoring only; the model never sees filenames.
    """
    return filename.rsplit("_", 2)[-2]


# ---------------------------------------------------------------- models
def load_image(page: dict):
    from PIL import Image

    if page["pdf_page"] is None:
        return Image.open(page["image"]).convert("RGB")
    import fitz  # PyMuPDF

    with fitz.open(page["image"]) as doc:
        pix = doc[page["pdf_page"]].get_pixmap(dpi=150)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def lm_prompt(page: dict) -> str:
    rows = sorted(page["lines"], key=lambda ln: (round(ln[2], 3), ln[1]))
    return PROMPT + LM_SUFFIX + "\n".join(f"[{int(1000 * x)},{int(1000 * y)}] {t}" for t, x, y in rows)


class Generator:
    def __init__(self, arm: str, model_id: str, device: str, dtype: str, max_pixels: int):
        import torch

        self.arm, self.device = arm, device
        torch_dtype = getattr(torch, dtype)
        if arm == "vlm":
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch_dtype, attn_implementation="sdpa").to(device)
            self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, attn_implementation="sdpa").to(device)
            self.proc = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()

    def __call__(self, page: dict, max_new_tokens: int) -> str:
        import torch

        if self.arm == "vlm":
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.proc(text=[text], images=[load_image(page)], return_tensors="pt")
        else:
            msgs = [{"role": "user", "content": lm_prompt(page)}]
            text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.proc([text], return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def cmd_run(args) -> None:
    pages = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pages = pages[args.shard::args.num_shards]
    if args.limit:
        pages = pages[:args.limit]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["page_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [p for p in pages if p["page_id"] not in done]
    print(f"[{args.arm} shard {args.shard}] {len(todo)} pages to do ({len(done)} already done) on {args.device}", flush=True)
    if not todo:
        return
    gen = Generator(args.arm, args.model, args.device, args.dtype, args.max_pixels)
    with open(out, "a", encoding="utf-8") as fh:
        for k, page in enumerate(todo, 1):
            t0 = time.time()
            try:
                raw, err = gen(page, args.max_new_tokens), None
            except Exception as e:  # one bad page must not kill a multi-hour run
                raw, err = "", f"{type(e).__name__}: {e}"
            row = {"page_id": page["page_id"], "arm": args.arm, "model": args.model, "raw": raw,
                   "pairs": parse_pairs(raw), "seconds": round(time.time() - t0, 2), "error": err}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{args.arm} shard {args.shard}] {k}/{len(todo)} {page['page_id'][:60]} "
                  f"pairs={len(row['pairs'])} {row['seconds']}s{' ERR ' + err if err else ''}", flush=True)


# ---------------------------------------------------------------- scoring
def sbert_embedder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return lambda texts: model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def score(pages: list[dict], preds: dict[str, dict], embed) -> dict:
    import numpy as np

    funsd = Counter()
    fields_by_ds: dict[str, list[str]] = defaultdict(list)
    for p in pages:
        if p["dataset"] != "funsd":
            for f in p["gold"]["fields"]:
                if f not in fields_by_ds[p["dataset"]]:
                    fields_by_ds[p["dataset"]].append(f)
    meta_fields = {}
    for ds in fields_by_ds:
        meta = ROOT / "data" / "vrdu" / ds / "main" / "meta.json"
        names = list(json.loads(meta.read_text())["entity_name_to_match_func"]) if meta.exists() else fields_by_ds[ds]
        meta_fields[ds] = [n for n in names if isinstance(n, str)]
    field_emb = {ds: embed([f.replace("_", " ") for f in fs]) for ds, fs in meta_fields.items()}

    per_field = defaultdict(Counter)  # (dataset, template, field) -> tp/n_pred/n_gold/value_found
    decoy = Counter()
    missing, seconds = 0, []
    for p in pages:
        pr = preds.get(p["page_id"])
        if pr is None:
            missing += 1
            continue
        seconds.append(pr.get("seconds", 0.0))
        pairs = pr["pairs"]
        if p["dataset"] == "funsd":
            gold = p["gold"]["pairs"]
            funsd["tp"] += greedy_match(pairs, gold, lambda a, b: min(tok_f1(a[0], b[0]), tok_f1(a[1], b[1])))
            funsd["value_tp"] += greedy_match([v for _, v in pairs], [v for _, v in gold], tok_f1)
            funsd["n_pred"] += len(pairs); funsd["n_gold"] += len(gold)
            continue
        ds, fields = p["dataset"], meta_fields[p["dataset"]]
        assigned: dict[str, list[str]] = defaultdict(list)
        if pairs:
            sims = embed([k for k, _ in pairs]) @ field_emb[ds].T
            for (k, v), row in zip(pairs, sims):
                j = int(np.argmax(row))
                if row[j] >= FIELD_SIM_THRESHOLD:
                    assigned[fields[j]].append(v)
        all_values = [v for _, v in pairs]
        for f in fields:
            gold = list({norm(g): g for g in p["gold"]["fields"].get(f, [])}.values())
            pv = list({norm(v): v for v in assigned.get(f, [])}.values())
            c = per_field[(ds, p["template"], f)]
            c["tp"] += greedy_match(pv, gold, tok_f1)
            c["n_pred"] += len(pv); c["n_gold"] += len(gold)
            c["value_found"] += greedy_match(all_values, gold, tok_f1)
        if p["template"] == "Short-Form":
            d, gold_reg = p["gold"].get("decoy"), p["gold"]["fields"].get("registrant_name", [])
            reg_preds = assigned.get("registrant_name", [])
            if d and any(tok_f1(d, g) >= MATCH_THRESHOLD for g in gold_reg):
                d = None  # an individual registered themself: no decoy on this page
                decoy["filer_is_registrant"] += 1
            decoy["pages"] += 1
            decoy["predicted_registrant"] += bool(reg_preds)
            decoy["correct"] += any(tok_f1(v, g) >= MATCH_THRESHOLD for v in reg_preds for g in gold_reg)
            decoy["picked_decoy"] += bool(d) and any(tok_f1(v, d) >= MATCH_THRESHOLD for v in reg_preds)

    out: dict = {"missing_pages": missing, "pages_scored": len(seconds),
                 "mean_seconds_per_page": round(sum(seconds) / len(seconds), 2) if seconds else None}
    if funsd["n_gold"]:
        out["funsd"] = {"pairs": prf(funsd["tp"], funsd["n_pred"], funsd["n_gold"]),
                        "value_recall": round(funsd["value_tp"] / funsd["n_gold"], 4)}
    vrdu: dict = {}
    for (ds, t, f), c in sorted(per_field.items()):
        if c["n_gold"] == 0 and c["n_pred"] == 0:
            continue
        row = prf(c["tp"], c["n_pred"], c["n_gold"])
        row["value_recall"] = round(c["value_found"] / c["n_gold"], 4) if c["n_gold"] else None
        vrdu.setdefault(ds, {}).setdefault(t, {})[f] = row
    for ds, by_t in vrdu.items():
        for t, by_f in by_t.items():
            tp, npred, ngold = (sum(r[k] for r in by_f.values()) for k in ("tp", "n_pred", "n_gold"))
            by_f["_micro"] = prf(tp, npred, ngold)
    out["vrdu"] = vrdu
    if decoy:
        out["short_form_registrant"] = dict(decoy)
    return out


def read_preds(paths: list[str]) -> dict[str, dict]:
    preds = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                preds[row["page_id"]] = row
    return preds


# ---------------------------------------------------------------- CLI
def cmd_prepare(args) -> None:
    pages = []
    if args.funsd_root and Path(args.funsd_root).exists():
        pages += funsd_pages(Path(args.funsd_root))
    for sub in ("registration-form", "ad-buy-form"):
        if (Path(args.vrdu_root) / sub / "main").exists():
            pages += vrdu_pages(Path(args.vrdu_root), sub, args.n_per_template, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    counts = Counter((p["dataset"], p["template"]) for p in pages)
    print(f"[+] {len(pages)} pages -> {args.out}: {dict(counts)}")


def cmd_score(args) -> None:
    pages = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    metrics = score(pages, read_preds(args.preds), sbert_embedder())
    Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "vrdu"}, indent=2))


def cmd_selftest(_args) -> None:
    import numpy as np

    raw = ("Sure!\n<pair><key>Name of Registrant</key><value>DLA Piper LLP</value></pair>\n"
           "<pair><key>Date</key><value></value></pair><pair> <key>Registration No.</key> <value>3712</value> </pair>")
    assert parse_pairs(raw) == [["Name of Registrant", "DLA Piper LLP"], ["Registration No.", "3712"]], parse_pairs(raw)
    assert tok_f1("DLA Piper US LLP", "DLA Piper LLP") > 0.8 and tok_f1("a", "b") == 0.0
    assert greedy_match(["x y", "z"], ["z", "x y", "w"], tok_f1) == 2
    lines = group_lines([("Name:", [0.1, 0.1, 0.2, 0.12]), ("Bob", [0.3, 0.101, 0.4, 0.121]), ("Date", [0.1, 0.3, 0.2, 0.32])])
    assert [l[0] for l in lines] == ["Name: Bob", "Date"], lines
    filer = short_form_filer("20080204_Singapore Economic Development Board_Wong, Yau-Chung_Short-Form.pdf")
    assert filer == "Wong, Yau-Chung" and tok_f1("Yau Chung Wong", filer) == 1.0

    import zlib

    def embed(texts):  # hashed bag-of-words stand-in for SBERT
        m = np.full((len(texts), 512), 1e-9)
        for i, t in enumerate(texts):
            for w in norm(t).split():
                m[i, zlib.crc32(w.encode()) % 512] += 1.0
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    pages = [{"page_id": "a", "dataset": "registration-form", "template": "Short-Form", "lines": [],
              "gold": {"fields": {"registrant_name": ["Singapore EDB"], "registration_num": ["2003"]}, "decoy": "Yau Chung Wong"}},
             {"page_id": "f", "dataset": "funsd", "template": "funsd", "lines": [],
              "gold": {"pairs": [["Date:", "12/1/98"], ["To:", "Bob"]]}}]
    preds = {"a": {"pairs": [["Name", "Yau Chung Wong"], ["Name of Registrant", "Yau Chung Wong"], ["Registration No.", "2003"]]},
             "f": {"pairs": [["Date", "12/1/98"], ["From", "Ann"]]}}
    m = score(pages, preds, embed)
    reg = m["vrdu"]["registration-form"]["Short-Form"]
    assert reg["registration_num"]["tp"] == 1 and reg["registrant_name"]["tp"] == 0, reg
    assert m["short_form_registrant"]["picked_decoy"] == 1 and m["short_form_registrant"]["correct"] == 0
    assert m["funsd"]["pairs"]["tp"] == 1 and m["funsd"]["pairs"]["n_gold"] == 2
    print("[+] zero_shot_kv selftest passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--funsd-root", default="data/funsd/dataset")
    p.add_argument("--vrdu-root", default="data/vrdu")
    p.add_argument("--n-per-template", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--manifest", required=True)
    p.add_argument("--arm", choices=["vlm", "lm"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("--manifest", required=True)
    p.add_argument("--preds", nargs="+", required=True)
    p.add_argument("--out", required=True)
    sub.add_parser("selftest")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "score": cmd_score, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
