"""Study 34, step 3: BROS tagger on the five VRDU ad-buy line-item fields plus row grouping, scored with the row metric.

    python scripts/adbuy_bros.py oracle [--tau 0.5 --merge 0]  # CPU: tags -> spans -> rows round trip with gold cells, row-grouping ceiling
    python scripts/adbuy_bros.py train --out DIR [--smoke] [--device cuda:0] [--shard 0 --num-shards 2]   # GPU (Kaggle)
    python scripts/adbuy_bros.py score --runs DIR              # CPU: rows from the runs' test_entities.jsonl, scored on the 143 primary documents

A cell is the set of words of one gold annotation. BIO tagging cannot hold a cell whose words are not consecutive (a cell on two OCR lines
with other columns in between), so every run of consecutive words is tagged as its own span, and the oracle reports how much that costs.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import statistics as st
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(ROOT)]
from lir_metric import cell_counts, f1, lir_counts  # noqa: E402

DATA = ROOT / "data" / "vrdu" / "ad-buy-form"
FIELDS = ["channel", "program_desc", "program_end_date", "program_start_date", "sub_amount"]   # sorted, as the loader sorts labels
LABELS = ["O"] + [f"{p}-{f}" for f in FIELDS for p in "BI"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
RULES = json.loads((DATA / "main" / "meta.json").read_text(encoding="utf-8"))["entity_name_to_match_func"]
PRIMARY = ROOT / "results" / "adbuy_sg" / "primary.json"


def load_docs(filenames: set[str] | None = None) -> dict[str, dict]:
    """Per document: words, scaled boxes (0-1000), float boxes, pages, gold rows (list of [(field, frozenset words, text)])."""
    from data.vrdu_loader import _scale_box, _token_char_range

    out = {}
    with gzip.open(DATA / "main" / "dataset.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if filenames is not None and rec["filename"] not in filenames:
                continue
            words, boxes, fboxes, pages, ranges = [], [], [], [], []
            for page in rec["ocr"]["pages"]:
                for t in page.get("tokens", []):
                    text = (t.get("text") or "").strip()
                    if text:
                        p, b = _scale_box(t["bbox"])
                        words.append(text); boxes.append(b); pages.append(p); ranges.append(_token_char_range(t))
                        fboxes.append(t["bbox"][1:5])
            starts = [r[0] for r in ranges]
            order = sorted(range(len(words)), key=lambda i: ranges[i][0])   # char order, to find the words under a character span quickly
            keys = [ranges[i][0] for i in order]
            rows = []
            for ent in rec["annotations"]:
                if not isinstance(ent[0], list):
                    continue
                for row in ent[1]:
                    cells = []
                    for name, cell in zip(ent[0], row):
                        ws = set()
                        for cs, ce in (cell[2] if len(cell) > 2 else []):
                            # words starting before ce whose end is after cs; every such word starts after cs - (longest word), so scan back a little
                            hi = bisect.bisect_left(keys, ce)
                            for i in order[max(0, bisect.bisect_left(keys, cs) - 40):hi]:
                                a, b = ranges[i]
                                if a >= 0 and a < ce and b > cs:
                                    ws.add(i)
                        if ws and name in FIELDS:
                            cells.append((name, frozenset(ws), cell[0].strip()))
                    if cells:
                        rows.append(cells)
            out[rec["filename"]] = {"words": words, "boxes": boxes, "fboxes": fboxes, "pages": pages, "rows": rows}
    return out


def runs(ws) -> list[tuple[int, int]]:
    out, ws = [], sorted(ws)
    for i in ws:
        if out and i == out[-1][1]:
            out[-1] = (out[-1][0], i + 1)
        else:
            out.append((i, i + 1))
    return out


def tags_of(doc: dict) -> list[int]:
    tags = [0] * len(doc["words"])
    for row in doc["rows"]:
        for f, ws, _ in row:
            for s, e in runs(ws):
                if all(tags[i] == 0 for i in range(s, e)):   # a word claimed by an earlier cell keeps its label
                    tags[s] = LABEL2ID[f"B-{f}"]
                    for i in range(s + 1, e):
                        tags[i] = LABEL2ID[f"I-{f}"]
    return tags


def extract_spans(names: list[str]) -> list[tuple[str, int, int]]:
    """(field, start, end) chunks of a strict IOB2 tag sequence, as src/evaluate._extract_spans (not imported: it needs seqeval)."""
    out, cur, start = [], None, 0
    for i, t in enumerate(names + ["O"]):
        pre, _, typ = t.partition("-")
        if cur is not None and (pre != "I" or typ != cur):
            out.append((cur, start, i))
            cur = None
        if pre == "B":
            cur, start = typ, i
    return out


def to_example(guid: str, doc: dict):
    from data.vrdu_loader import VrduExample

    return VrduExample(guid=guid, words=doc["words"], boxes=doc["boxes"], ner_tags=tags_of(doc), template_id="ad-buy-form",
                       page_ids=doc["pages"], n_pages=max(doc["pages"], default=0) + 1)


# ------------------------------------------------------------------ rows from spans
def group_rows(spans: list[list], doc: dict, tau: float = 0.5, merge: bool = True) -> list[set]:
    """Rows of (field, word set) from predicted spans [field, start, end, text]: per page, anchor field = the one with most spans;
    every span goes to the closest anchor at or above it (within `tau` span heights below counts as 'at'), else the closest below."""
    fb, pg = doc["fboxes"], doc["pages"]
    by_page = defaultdict(list)
    for f, s, e, _ in spans:
        y0, y1 = min(fb[i][1] for i in range(s, e)), max(fb[i][3] for i in range(s, e))
        by_page[pg[s]].append((f, frozenset(range(s, e)), (y0 + y1) / 2, y1 - y0))
    rows = []
    for p, items in by_page.items():
        counts = defaultdict(int)
        for f, _, _, _ in items:
            counts[f] += 1
        anchor = max(counts, key=lambda f: (counts[f], f))
        anchors = sorted(((y, k) for k, (f, _, y, _) in enumerate(items) if f == anchor))
        cluster: dict[int, set] = {k: set() for _, k in anchors}
        for k, (f, ws, y, h) in enumerate(items):
            above = [(y - ya, ka) for ya, ka in anchors if ya <= y + tau * h]
            below = [(ya - y, ka) for ya, ka in anchors if ya > y + tau * h]
            _, target = min(above) if above else min(below)
            cluster[target].add((f, ws))
        for cells in cluster.values():
            if merge:   # a row holds at most one cell per field (checked on the gold: no row repeats a field), so same-field pieces in one row are one cell
                byf: dict[str, frozenset] = {}
                for f, ws in cells:
                    byf[f] = byf.get(f, frozenset()) | ws
                cells = set(byf.items())
            rows.append(cells)
    return rows


def gold_sets(doc: dict) -> list[set]:
    return [{(f, ws) for f, ws, _ in row} for row in doc["rows"]]


def run_counts(pred: list[set], doc: dict) -> dict:
    g = gold_sets(doc)
    vg = [{(f, t) for f, _, t in row} for row in doc["rows"]]
    vp = [{(f, " ".join(doc["words"][i] for i in sorted(ws))) for f, ws in r} for r in pred]
    from adbuy_rows import value_same
    return {"lir": lir_counts(pred, g), "cell": cell_counts(pred, g), "vlir": lir_counts(vp, vg, value_same), "vcell": cell_counts(vp, vg, value_same)}


def add(tot: dict, c: dict) -> None:
    for k, v in c.items():
        tot[k] = [a + b for a, b in zip(tot.get(k, [0, 0, 0]), v)]


def show(tot: dict) -> dict:
    return {"lir_f1": round(f1(tot["lir"]) * 100, 2), "cell_f1": round(f1(tot["cell"]) * 100, 2),
            "value_lir_f1": round(f1(tot["vlir"]) * 100, 2), "value_cell_f1": round(f1(tot["vcell"]) * 100, 2)}


# ------------------------------------------------------------------ commands
def cmd_oracle(a) -> None:
    docs = load_docs()
    split = json.loads(next((DATA / "few_shot-splits").glob("DeepForm-unk_template-train_100-*-SD_0.json")).read_text(encoding="utf-8"))
    sets = {"all 641 documents": list(docs), "train documents of split 100/SD_0": [d for d in split["train"] if d in docs],
            "primary (143)": json.loads(PRIMARY.read_text(encoding="utf-8"))}
    for name, ids in sets.items():
        tot_tag, tot_rows, tot_gaps = {}, {}, {}
        n_cells = n_split = 0
        for g in ids:
            d = docs[g]
            spans = sorted([t, s, e, " ".join(d["words"][s:e])] for t, s, e in extract_spans([ID2LABEL[i] for i in tags_of(d)]))
            n_cells += sum(len(r) for r in d["rows"])
            n_split += sum(len(runs(ws)) > 1 for r in d["rows"] for _, ws, _ in r)
            if not d["rows"]:
                continue
            # (1) tagging round trip with the true row of every span: ceiling of the tagger decode
            true_row = {(f, i): k for k, r in enumerate(d["rows"]) for f, ws, _ in r for i in ws}
            byrow = defaultdict(set)
            for f, s, e, _ in spans:
                byrow[true_row.get((f, s), -1)].add((f, frozenset(range(s, e))))
            add(tot_tag, run_counts(list(byrow.values()), d))
            # (2) perfect spans, rows by clustering: cost of the grouping step alone
            add(tot_rows, run_counts(group_rows(spans, d, a.tau, a.merge), d))
        print(f"{name}: {len(ids)} documents, {n_cells} gold cells, {n_split} ({100 * n_split / max(1, n_cells):.1f}%) not consecutive words\n"
              f"   tags -> spans with true rows : {show(tot_tag)}\n   tags -> spans -> clustered rows: {show(tot_rows)}")


def split_file(n: int, sd: int) -> dict:
    return json.loads(next((DATA / "few_shot-splits").glob(f"DeepForm-unk_template-train_{n}-*-SD_{sd}.json")).read_text(encoding="utf-8"))


def cmd_train(a) -> None:
    """One BROS run per (training documents N, split SD): train on the split's N unseen-template training documents, early-stop on its
    validation documents, predict the 143 primary documents (all of them are in every split's test set)."""
    import torch
    from train import TrainConfig, train_single_seed

    config = TrainConfig.from_yaml(ROOT / a.config)
    seed = config.seeds[0]
    out = Path(a.out)
    primary = json.loads(PRIMARY.read_text(encoding="utf-8"))
    tasks = [(n, sd) for n in [int(x) for x in a.sizes.split(",")] for sd in (0, 1, 2)]
    if a.smoke:
        out, tasks = Path(tempfile.mkdtemp()) / "smoke", tasks[:1]
        config.epochs, config.batch_size = 1, 2
    tasks = tasks[a.shard::a.num_shards]
    print(f"[shard {a.shard}] {len(tasks)} runs on {a.device}: {tasks}", flush=True)
    if not tasks:
        return
    docs = load_docs()
    device = torch.device(a.device)
    deadline = time.time() + a.deadline_hours * 3600 if a.deadline_hours else None
    for n, sd in tasks:
        if deadline and time.time() > deadline:
            print(f"[shard {a.shard}] deadline reached; stopping cleanly (re-run resumes: finished runs are skipped)", flush=True)
            break
        sp = split_file(n, sd)
        assert set(primary) <= set(sp["test"]), "primary documents must be in every split's test set"
        pick = lambda ids: [to_example(g, docs[g]) for g in ids if g in docs]
        train_ex, val_ex, test_ex = pick(sp["train"]), pick(sp["valid"]), pick(primary)
        if a.smoke:
            train_ex, val_ex, test_ex = train_ex[:12], val_ex[:4], test_ex[:6]
        print(f"[shard {a.shard}] N={n} SD={sd}: train {len(train_ex)} val {len(val_ex)} test {len(test_ex)}", flush=True)
        res = train_single_seed(seed, config, train_ex, val_ex, test_ex, LABELS, ID2LABEL, device, out / f"N{n}_SD{sd}", log_prefix=f"[N{n} SD{sd}]")
        print(f"[shard {a.shard}] N={n} SD={sd}: test entity F1 {res['test']['f1']:.4f}", flush=True)
    if a.smoke:
        run_dir = out / f"N{tasks[0][0]}_SD{tasks[0][1]}" / f"seed_{seed}"
        assert (run_dir / "result.json").exists() and (run_dir / "test_entities.jsonl").exists(), list(run_dir.glob("*"))
        print("[+] adbuy_bros smoke passed:", sorted(p.name for p in run_dir.glob("*")))


def cmd_score(a) -> None:
    primary = json.loads(PRIMARY.read_text(encoding="utf-8"))
    docs = load_docs(set(primary))
    report: dict = {"tau": a.tau, "runs": {}, "per_size": {}}
    for run in sorted(Path(a.runs).glob("N*_SD*")):
        f = next(run.glob("seed_*/test_entities.jsonl"), None)
        if f is None:
            continue
        pred = {r["guid"]: r["pred"] for r in map(json.loads, f.read_text(encoding="utf-8").splitlines())}
        tot, per_doc = {}, {}
        for g in primary:
            d = docs[g]
            if not d["rows"]:
                continue
            c = run_counts(group_rows(pred.get(g, []), d, a.tau, merge=False), d)
            add(tot, c)
            per_doc[g] = c
        report["runs"][run.name] = {**show(tot), "counts": tot, "per_doc": per_doc}
        print(run.name, show(tot))
    for n in sorted({int(k.split("_")[0][1:]) for k in report["runs"]}):
        rs = [v for k, v in report["runs"].items() if int(k.split("_")[0][1:]) == n]
        report["per_size"][str(n)] = {m: round(st.mean(r[m] for r in rs), 2) for m in ("lir_f1", "cell_f1", "value_lir_f1", "value_cell_f1")} | {"splits": len(rs)}
        print(f"N={n}: mean over {len(rs)} splits", report["per_size"][str(n)])
    Path(a.runs, "report.json").write_text(json.dumps(report), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("oracle")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--merge", type=int, default=0)
    p = sub.add_parser("train")
    p.add_argument("--config", default="configs/adbuy_rows_bros.yaml"); p.add_argument("--out", default="results/adbuy_bros/run")
    p.add_argument("--sizes", default="10,50,100,200"); p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1); p.add_argument("--smoke", action="store_true")
    p.add_argument("--deadline-hours", type=float, default=0.0)
    p = sub.add_parser("score")
    p.add_argument("--runs", required=True); p.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()
    {"oracle": cmd_oracle, "train": cmd_train, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
