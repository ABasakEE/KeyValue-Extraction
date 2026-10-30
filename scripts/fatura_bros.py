"""Track-1 BROS on FATURA, one run per (held-out fold, seed): train on 200 documents of the other templates, test on
every document of the held-out templates. Writes <out>/fold<f>/seed_<s>/{result.json,test_entities.jsonl,...}.

Study 15 adds --k-list 1,2,5,10 (train on only k seen templates, --per-template invoices each) and --test-set eval
(test only the 200 VLM-evaluated invoices per fold); outputs then go to <out>/K<k>/fold<f>/seed_<s>/.

Usage (one process per GPU):
  python scripts/fatura_bros.py --out results/fatura_v1/bros --device cuda:0 --shard 0 --num-shards 2
  python scripts/fatura_bros.py --smoke        # 1 epoch on a handful of documents, writes to a temp dir
Runs already finished (result.json present) are skipped, so a stopped job resumes.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data import fatura_loader as fl  # noqa: E402


def build_split(docs: dict, fold: int, seed: int, held_out: list[int], smoke: bool, k: int = 0,
                per_template: int = 10, test_set: str = "all"):
    if k:   # study 15: BROS trained on only k seen templates
        train_ids, val_ids = fl.starved_sample(held_out, seed, fold, k, per_template)
    else:
        train_ids, val_ids = fl.train_val_sample(held_out, seed, fold)
    if test_set == "eval":   # only the VLM-evaluated invoices (dev + confirm) of the held-out templates
        sample = fl.eval_sample()
        test_ids = sorted(g for t in held_out for names in sample[t].values() for g in names)
    else:
        test_ids = sorted(g for g, d in docs.items() if d["template"] in set(held_out))
    if smoke:
        train_ids, val_ids, test_ids = train_ids[:12], val_ids[:4], test_ids[:6]
    pick = lambda ids: [fl.to_example(docs[g]) for g in ids]
    return pick(train_ids), pick(val_ids), pick(test_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fatura_bros.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k-list", default="", help="study 15: comma-separated numbers of seen templates to train on, e.g. 1,2,5,10")
    ap.add_argument("--per-template", type=int, default=10)
    ap.add_argument("--test-set", choices=["all", "eval"], default="all")
    args = ap.parse_args()

    import torch
    from train import TrainConfig, train_single_seed

    config = TrainConfig.from_yaml(ROOT / args.config)
    out = Path(args.out or config.output_dir)
    held = fl.folds()
    ks = [int(k) for k in args.k_list.split(",") if k.strip()] or [0]
    tasks = [(k, f, s) for k in ks for f in range(len(held)) for s in config.seeds]
    if args.smoke:
        out, tasks = Path(tempfile.mkdtemp()) / "smoke", [(ks[0], 0, config.seeds[0])]
        config.epochs, config.batch_size = 1, 2
    tasks = tasks[args.shard::args.num_shards]
    print(f"[shard {args.shard}] {len(tasks)} runs on {args.device}: {tasks}", flush=True)
    if not tasks:
        return

    docs = fl.load_docs(ROOT / config.data_root)
    device = torch.device(args.device)
    for k, fold, seed in tasks:
        train_ex, val_ex, test_ex = build_split(docs, fold, seed, held[fold], args.smoke, k, args.per_template, args.test_set)
        run_out = out / f"K{k}" if k else out
        print(f"[shard {args.shard}] K={k or 'all'} fold {fold} seed {seed}: train {len(train_ex)} val {len(val_ex)} test {len(test_ex)} "
              f"(held-out templates {held[fold]})", flush=True)
        res = train_single_seed(seed, config, train_ex, val_ex, test_ex, fl.LABELS, fl.ID2LABEL, device,
                                run_out / f"fold{fold}", log_prefix=f"[K{k} f{fold} s{seed}]")
        print(f"[shard {args.shard}] K={k or 'all'} fold {fold} seed {seed}: test F1 {res['test']['f1']:.4f}", flush=True)
    if args.smoke:
        run_dir = (out / f"K{ks[0]}" if ks[0] else out) / "fold0" / f"seed_{config.seeds[0]}"
        assert (run_dir / "result.json").exists() and (run_dir / "test_entities.jsonl").exists(), list(run_dir.glob("*"))
        print("[+] fatura_bros smoke passed:", sorted(p.name for p in run_dir.glob("*")))


if __name__ == "__main__":
    main()
