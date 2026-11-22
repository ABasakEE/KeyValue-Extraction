"""Study 34, DocILE: the BROS line-item tagger plus row grouping (training, scoring) and its ceilings (CPU oracle).

    python scripts/docile_bros.py oracle                                  # CPU: tags -> spans -> rows round trip with gold cells
    python scripts/docile_bros.py train --out DIR [--sizes 10,50,200,1000] [--seeds 0,1,2] [--smoke] [--device cuda:0] [--shard 0 --num-shards 2]   # GPU
    python scripts/docile_bros.py score --runs DIR                         # CPU: rows from the runs' test_entities.jsonl, scored on the 200 test documents

Same method as the ad-buy baseline (`adbuy_bros.py`): Track-1 BROS (unchanged recipe, one training seed) tags the 19 line-item fields (a run of consecutive words of a cell is one span);
spans are grouped into rows by the anchor rule (`row_group.group_rows`, tolerance 0.5). Data: the layout-disjoint splits of `results/docile_rows/splits.json`: the training set of size N for
split seed s is the first N documents of its order (nested in N), early stopping on the 100 validation documents, prediction on the 200 test documents. A row may hold several cells of a field.
"""
from __future__ import annotations

import argparse
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
import docile_rows as D  # noqa: E402
import docile_sg as G  # noqa: E402
from lir_metric import cell_counts, f1, lir_counts  # noqa: E402
from row_group import extract_spans, group_rows, runs  # noqa: E402

FIELDS = sorted(G.FIELDS)
LABELS = ["O"] + [f"{p}-{f}" for f in FIELDS for p in "BI"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
SPLITS = ROOT / "results" / "docile_rows" / "splits.json"
POOL = ROOT / "results" / "docile_rows" / "pool.json"   # all training-pool documents (3,108), written by docile_splits.py (study 41)


def doc_dict(i: str) -> dict:
    """Words, boxes (float 0-1 and scaled 0-1000), pages and gold rows of one document; words that are empty become '_' so that the tokenizer keeps them."""
    doc = D.Doc(i, None)
    ann = G.ann_of(i)
    rows = [[(f, ws, "") for f, ws, _ in row] for row in D.gold_rows(doc, ann)]
    scale = lambda b: [max(0, min(1000, int(round(1000 * v)))) for v in b]
    return {"words": [w[1] for w in doc.words], "ex_words": [w[1] if w[1].strip() else "_" for w in doc.words], "fboxes": [w[2] for w in doc.words],
            "boxes": [scale(w[2]) for w in doc.words], "pages": [w[0] for w in doc.words], "rows": rows}


def tag_names(d: dict) -> list[str]:
    tags = ["O"] * len(d["words"])
    for row in d["rows"]:
        for f, ws, _ in row:
            for s, e in runs(ws):
                if all(tags[i] == "O" for i in range(s, e)):   # a word claimed by an earlier cell keeps its label
                    tags[s] = f"B-{f}"
                    for i in range(s + 1, e):
                        tags[i] = f"I-{f}"
    return tags


def spans_of(d: dict) -> list[list]:
    return sorted([t, s, e, ""] for t, s, e in extract_spans(tag_names(d)))


def to_example(guid: str, d: dict):
    from data.vrdu_loader import VrduExample

    return VrduExample(guid=guid, words=d["ex_words"], boxes=d["boxes"], ner_tags=[LABEL2ID[t] for t in tag_names(d)], template_id="docile",
                       page_ids=d["pages"], n_pages=max(d["pages"], default=0) + 1)


def counts(pred: list[set], d: dict) -> dict:
    g = [{(f, ws) for f, ws, _ in row} for row in d["rows"]]
    return {"lir": lir_counts(pred, g), "cell": cell_counts(pred, g)}


def add(tot: dict, c: dict) -> None:
    for k, v in c.items():
        tot[k] = [a + b for a, b in zip(tot.get(k, [0, 0, 0]), v)]


def show(t: dict) -> dict:
    return {"row_f1": round(f1(t["lir"]) * 100, 2), "cell_f1": round(f1(t["cell"]) * 100, 2)}


# ------------------------------------------------------------------ commands
def cmd_oracle(_a) -> None:
    sp = json.loads(SPLITS.read_text(encoding="utf-8"))
    sets = {"test (200 documents)": sp["test"], "training documents, seed 0, N = 200": sp["train"]["0"][:200]}
    for name, ids in sets.items():
        tot_true, tot_clu = {}, {}
        n_cells = n_split = n_dup = 0
        for i in ids:
            d = doc_dict(i)
            if not d["rows"]:
                continue
            n_cells += sum(len(r) for r in d["rows"])
            n_split += sum(len(runs(ws)) > 1 for r in d["rows"] for _, ws, _ in r)
            n_dup += sum(len(r) - len({f for f, _, _ in r}) for r in d["rows"])
            spans = spans_of(d)
            true_row = {(f, i2): k for k, r in enumerate(d["rows"]) for f, ws, _ in r for i2 in ws}
            byrow = defaultdict(set)
            for f, s, e, _ in spans:
                byrow[true_row.get((f, s), -1)].add((f, frozenset(range(s, e))))
            add(tot_true, counts(list(byrow.values()), d))
            add(tot_clu, counts(group_rows(spans, d, 0.5, merge=False), d))
        print(f"{name}: {n_cells} gold cells, {100 * n_split / n_cells:.1f}% not consecutive words, {100 * n_dup / n_cells:.1f}% repeat a field already in their row\n"
              f"   tags -> spans with true rows : {show(tot_true)}\n   tags -> spans -> clustered rows: {show(tot_clu)}")


def cmd_train(a) -> None:
    """One BROS run per (training documents N, split seed s); resumable (a finished run is skipped by train_single_seed)."""
    import torch
    from train import TrainConfig, train_single_seed

    config = TrainConfig.from_yaml(ROOT / a.config)
    out = Path(a.out)
    sp = json.loads(SPLITS.read_text(encoding="utf-8"))
    if a.full_pool:   # study 41: the whole pool, one run per training seed (run directory NfullT<seed>)
        tasks = [("full", None, int(x)) for x in a.train_seeds.split(",")]
    else:
        tasks = [(n, s, config.seeds[0]) for n in [int(x) for x in a.sizes.split(",")] for s in [int(x) for x in a.seeds.split(",")]]
    if a.smoke:
        out, tasks = Path(tempfile.mkdtemp()) / "smoke", tasks[:1]
        config.epochs, config.batch_size = 1, 2
    tasks = tasks[a.shard::a.num_shards]
    print(f"[shard {a.shard}] {len(tasks)} runs on {a.device}: {tasks}", flush=True)
    if not tasks:
        return
    cache: dict[str, object] = {}
    pick = lambda ids: [cache.setdefault(g, to_example(g, doc_dict(g))) for g in ids]
    device = torch.device(a.device)
    deadline = time.time() + a.deadline_hours * 3600 if a.deadline_hours else None
    for n, s, seed in tasks:
        if deadline and time.time() > deadline:
            print(f"[shard {a.shard}] deadline reached; stopping cleanly (re-run resumes: finished runs are skipped)", flush=True)
            break
        train_ids = json.loads(POOL.read_text(encoding="utf-8")) if n == "full" else sp["train"][str(s)][:n]
        name = f"Nfull_T{seed}" if n == "full" else f"N{n}_S{s}"
        train_ex, val_ex, test_ex = pick(train_ids), pick(sp["val"]), pick(sp["test"])
        if a.smoke:
            train_ex, val_ex, test_ex = train_ex[:6], val_ex[:2], test_ex[:3]
        print(f"[shard {a.shard}] {name}: train {len(train_ex)} val {len(val_ex)} test {len(test_ex)}", flush=True)
        res = train_single_seed(seed, config, train_ex, val_ex, test_ex, LABELS, ID2LABEL, device, out / name, log_prefix=f"[{name}]")
        print(f"[shard {a.shard}] {name}: test entity F1 {res['test']['f1']:.4f}", flush=True)
    if a.smoke:
        run_dir = out / name / f"seed_{seed}"
        assert (run_dir / "result.json").exists() and (run_dir / "test_entities.jsonl").exists(), list(run_dir.glob("*"))
        print("[+] docile_bros smoke passed:", sorted(p.name for p in run_dir.glob("*")), "| run dir", run_dir.parent)


def cmd_score(a) -> None:
    test = json.loads(SPLITS.read_text(encoding="utf-8"))["test"]
    docs = {g: doc_dict(g) for g in test}
    report: dict = {"tau": a.tau, "runs": {}, "per_size": {}}
    for run in sorted(Path(a.runs).glob("N*_[ST]*")):
        f = next(run.glob("seed_*/test_entities.jsonl"), None)
        if f is None:
            continue
        pred = {r["guid"]: r["pred"] for r in map(json.loads, f.read_text(encoding="utf-8").splitlines())}
        tot, per_doc = {}, {}
        for g in test:
            d = docs[g]
            if not d["rows"] or g not in pred:
                continue
            c = counts(group_rows(pred[g], d, a.tau, merge=False), d)
            add(tot, c)
            per_doc[g] = c
        report["runs"][run.name] = {**show(tot), "counts": tot, "documents": len(per_doc), "per_doc": per_doc}
        print(run.name, show(tot), f"({len(per_doc)} documents)")
    for n in sorted({k.split("_")[0][1:] for k in report["runs"]}, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x)):
        rs = [v for k, v in report["runs"].items() if k.split("_")[0][1:] == n]
        report["per_size"][str(n)] = {m: round(st.mean(r[m] for r in rs), 2) for m in ("row_f1", "cell_f1")} | {"splits": len(rs)}
        print(f"N={n}: mean over {len(rs)} splits", report["per_size"][str(n)])
    Path(a.runs, a.report).write_text(json.dumps(report), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("oracle")
    p = sub.add_parser("train")
    p.add_argument("--config", default="configs/docile_rows_bros.yaml"); p.add_argument("--out", default="results/docile_bros/run")
    p.add_argument("--sizes", default="10,50,200,1000"); p.add_argument("--seeds", default="0,1,2"); p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1); p.add_argument("--smoke", action="store_true")
    p.add_argument("--deadline-hours", type=float, default=0.0)
    p.add_argument("--full-pool", action="store_true"); p.add_argument("--train-seeds", default="13,21")
    p = sub.add_parser("score")
    p.add_argument("--runs", required=True); p.add_argument("--tau", type=float, default=0.5); p.add_argument("--report", default="report.json")
    a = ap.parse_args()
    {"oracle": cmd_oracle, "train": cmd_train, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
