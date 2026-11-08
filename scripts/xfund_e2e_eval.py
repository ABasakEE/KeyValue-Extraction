"""End-to-end entity and link F1 of joint-LiLT checkpoints on XFUND val: entities from the model's own tags.

    python scripts/xfund_e2e_eval.py --ckpt-dir results/xfund_joint_lilt --out results/xfund_e2e/baseline.json

Per seed (seed_*/best_model.pt + metrics.json in --ckpt-dir) and for two readings of a document: ``single`` (first 512
tokens, as trained before) and ``windowed`` (entity-safe 510-token windows, stride --stride).  Gold is every annotated
link (words cut off by the window count as misses); links touching an untagged ("other") entity can never be found.
Subsets: all val documents, and those with no train sibling (see xfund_link_eval.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    sys.path.insert(0, p)

from data.xfund_loader import ID2LABEL, LABEL_LIST, load_xfund_split  # noqa: E402
from data.xfund_windows import word_windows  # noqa: E402
from e2e_link import add_counts, gold_qa_links, gold_structure, predict_document, prf, qa_pred_links  # noqa: E402
from train import build_model  # noqa: E402
from data.funsd_loader import load_funsd_split  # noqa: E402
from xfund_link_errors import direction, entity_boxes, rank_of  # noqa: E402
from xfund_link_eval import LANGS, sibling_free  # noqa: E402

RANK_GROUPS = ("rank_0", "rank_1", "rank_2_4", "rank_5plus")
GROUPS = ("below", "same_line", "above") + RANK_GROUPS + ("q_to_q",)


def link_groups(ex) -> dict:
    """Gold links (word-range form, as gold_structure) split by geometry, partner rank and question->question.

    below / same_line / above partition the link set, as do the four rank groups; q_to_q is a subset.
    """
    rng = {i: (min(e["token_indices"]), max(e["token_indices"]) + 1) for i, e in ex.entities.items() if e.get("token_indices")}
    boxes = entity_boxes(ex)
    types = {i: e["label"].lower() for i, e in ex.entities.items()}
    out = {k: set() for k in GROUPS}
    for s, d in ex.links:
        if not (s in rng and d in rng and s != d):
            continue
        link = (rng[s], rng[d])
        dr = direction(boxes[s], boxes[d])
        out["same_line" if dr.startswith("same line") else "below" if dr == "partner below" else "above"].add(link)
        r = rank_of(s, d, boxes, types)
        out["rank_0" if r == 0 else "rank_1" if r == 1 else "rank_2_4" if r <= 4 else "rank_5plus"].add(link)
        if types[s] == "question" and types[d] == "question":
            out["q_to_q"].add(link)
    return out


def evaluate_seed(model, docs, keep: set[str], stride: int) -> dict:
    counts: dict[str, list[int]] = {}
    untagged = total = 0
    for ex in docs:
        g_ents, g_links = gold_structure(ex)
        g_qa = gold_qa_links(ex)
        groups = link_groups(ex)
        ranges = {(a, b) for _, a, b in g_ents}
        total += len(g_links)
        untagged += sum(1 for s, d in g_links if s not in ranges or d not in ranges)
        wins = word_windows(model.tokenizer, ex.words, 510, stride)
        for mode, w in (("single", None), ("windowed", wins)):
            ents, links = predict_document(model, ex, w)
            for sub in ("all", "sibling_free"):
                if sub == "sibling_free" and (not keep or ex.guid not in keep):
                    continue
                add_counts(counts.setdefault(f"{sub}|{mode}|entity", [0, 0, 0]), ents, g_ents)
                add_counts(counts.setdefault(f"{sub}|{mode}|link", [0, 0, 0]), links, g_links)
                add_counts(counts.setdefault(f"{sub}|{mode}|link_qa", [0, 0, 0]), qa_pred_links(ents, links), g_qa)
                for name, grp in groups.items():  # only recall is meaningful: gold is the group, pred is restricted to gold
                    add_counts(counts.setdefault(f"{sub}|{mode}|group|{name}", [0, 0, 0]), links & grp, grp)
    out = {k: prf(*v) for k, v in counts.items()}
    out["gold_links_total"], out["gold_links_with_untagged_endpoint"] = total, untagged
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data", choices=("xfund", "synthetic"), default="xfund")
    ap.add_argument("--data-root", default=None, help="default data/xfund for xfund; required for synthetic (FUNSD-format root)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    ckpt = Path(a.ckpt_dir)
    cfg = SimpleNamespace(**json.loads((ckpt / "metrics.json").read_text(encoding="utf-8"))["config"])
    seeds = a.seeds or list(cfg.seeds)
    if a.data == "synthetic":
        if not a.data_root:
            ap.error("--data synthetic needs --data-root")
        val, keep = load_funsd_split(a.data_root, "test"), set()
    else:
        root = a.data_root or "data/xfund"
        val, train = load_xfund_split(root, "val", LANGS), load_xfund_split(root, "train", LANGS)
        keep = sibling_free(val, train)
    if a.limit:
        val = val[: a.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"stride": a.stride, "n_val": len(val), "n_sibling_free": sum(v.guid in keep for v in val), "seeds": {}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        if not (ckpt / f"seed_{seed}" / "best_model.pt").exists():
            print(f"WARNING: seed {seed}: no best_model.pt in {ckpt}, skipped", flush=True)
            continue
        model = build_model(cfg, len(LABEL_LIST), ID2LABEL)
        sd = torch.load(ckpt / f"seed_{seed}" / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
        model.to(device).eval()
        res["seeds"][seed] = evaluate_seed(model, val, keep, a.stride)
        print(seed, {k: v["f1"] for k, v in res["seeds"][seed].items() if isinstance(v, dict)}, flush=True)
        keys = [k for k in res["seeds"][seed] if isinstance(res["seeds"][seed][k], dict)]
        res["mean_f1"] = {k: round(sum(res["seeds"][s][k]["f1"] for s in res["seeds"]) / len(res["seeds"]), 4) for k in keys}
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")  # a crash on a later seed keeps the earlier ones
        del model


if __name__ == "__main__":
    main()
