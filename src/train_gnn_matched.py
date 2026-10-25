"""Matched-Size Protocol (MTL vs UTL) for Track 2 RSA-GNN.

Applies the exact same template-disjoint protocol as Track 1 (N=200, MTL vs UTL,
3 folds x multi-seeds) to scientifically measure the genuine cross-template
generalization gap of the Relational Spatial-Aware GNN.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tabulate import tabulate

from src.data.splits import assert_no_template_leakage, matched_size_protocol, Split
from src.evaluate import entity_f1, cross_check
from src.models.edge_gat import EdgeGATTokenClassifier
from src.train import set_seed, json_serializable, generalization_gap


@dataclass
class GraphDocProxy:
    """Document proxy for template-aware split construction."""
    guid: str
    template_id: str
    files: list[Path]


class FastCachedGraphDataset(Dataset):
    """In-memory dataset holding graph dicts for a specific fold."""

    def __init__(self, graphs: list[dict]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict:
        return self.graphs[idx]


def collate_disjoint_graphs(batch: list[dict]) -> dict:
    if not batch:
        return {}

    xs, edges, attrs, labels = [], [], [], []
    node_offset = 0
    doc_slices = []

    has_edge_attr = "edge_attr" in batch[0] and batch[0]["edge_attr"] is not None and batch[0]["edge_attr"].numel() > 0
    edge_dim = batch[0]["edge_attr"].size(-1) if has_edge_attr else 0

    for g in batch:
        n_nodes = g["x"].size(0)
        xs.append(g["x"])
        labels.append(g["labels"])

        e_idx = g["edge_index"]
        if e_idx.numel() > 0 and e_idx.size(1) > 0:
            edges.append(e_idx + node_offset)
            if has_edge_attr:
                attrs.append(g["edge_attr"])

        doc_slices.append({
            "doc_guid": g.get("doc_guid", g.get("guid", "unknown")),
            "word_offset": int(g.get("word_offset", 0)),
            "total_doc_words": int(g.get("total_doc_words", n_nodes)),
            "start": node_offset,
            "end": node_offset + n_nodes,
        })
        node_offset += n_nodes

    return {
        "x": torch.cat(xs, dim=0),
        "labels": torch.cat(labels, dim=0),
        "edge_index": torch.cat(edges, dim=1) if edges else torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.cat(attrs, dim=0) if attrs else torch.zeros((0, edge_dim), dtype=torch.float32),
        "doc_slices": doc_slices,
    }


def evaluate_gnn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    id2label: dict[int, str],
) -> dict:
    model.eval()
    all_preds_by_doc: dict[str, list[dict]] = {}
    all_refs_by_doc: dict[str, list[dict]] = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device) if batch["edge_attr"].numel() > 0 else None
            labels = batch["labels"]

            logits = model(x, edge_index, edge_attr)
            preds = logits.argmax(dim=-1).cpu()

            for s in batch["doc_slices"]:
                d_guid = s["doc_guid"]
                st, en = s["start"], s["end"]
                all_preds_by_doc.setdefault(d_guid, []).append({
                    "offset": s["word_offset"],
                    "total": s["total_doc_words"],
                    "preds": preds[st:en].tolist(),
                })
                all_refs_by_doc.setdefault(d_guid, []).append({
                    "offset": s["word_offset"],
                    "total": s["total_doc_words"],
                    "labels": labels[st:en].tolist(),
                })

    pred_sequences = []
    ref_sequences = []
    for d_guid in sorted(all_preds_by_doc.keys()):
        p_chunks = sorted(all_preds_by_doc[d_guid], key=lambda c: c["offset"])
        r_chunks = sorted(all_refs_by_doc[d_guid], key=lambda c: c["offset"])
        tot = p_chunks[0]["total"]

        doc_p = [0] * tot
        doc_r = [0] * tot
        for c in p_chunks:
            off = c["offset"]
            for idx, val in enumerate(c["preds"]):
                if off + idx < tot:
                    doc_p[off + idx] = val
        for c in r_chunks:
            off = c["offset"]
            for idx, val in enumerate(c["labels"]):
                if off + idx < tot:
                    doc_r[off + idx] = val

        pred_sequences.append([id2label.get(v, "O") for v in doc_p])
        ref_sequences.append([id2label.get(v, "O") for v in doc_r])

    res = entity_f1(pred_sequences, ref_sequences, id2label)
    return {
        "f1": round(res.f1, 4),
        "precision": round(res.precision, 4),
        "recall": round(res.recall, 4),
        "per_type": res.per_type,
    }


def train_one_matched_run(
    regime: str,
    fold: int,
    seed: int,
    train_graphs: list[dict],
    val_graphs: list[dict],
    test_graphs: list[dict],
    num_labels: int,
    id2label: dict[int, str],
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    num_layers: int = 3,
    edge_dim: int = 16,
) -> dict:
    set_seed(seed)
    tag = f"[{device} | {regime}/fold{fold}/seed{seed}]"

    train_ds = FastCachedGraphDataset(train_graphs)
    val_ds = FastCachedGraphDataset(val_graphs)
    test_ds = FastCachedGraphDataset(test_graphs)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_disjoint_graphs)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_disjoint_graphs)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_disjoint_graphs)

    model = EdgeGATTokenClassifier(
        in_dim=train_graphs[0]["x"].size(-1),
        hidden_dim=256,
        num_labels=num_labels,
        num_layers=num_layers,
        edge_dim=edge_dim,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device) if batch["edge_attr"].numel() > 0 else None
            labels = batch["labels"].to(device)

            logits = model(x, edge_index, edge_attr)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        val_res = evaluate_gnn(model, val_loader, device, id2label)
        val_f1 = val_res["f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_res = evaluate_gnn(model, test_loader, device, id2label)
    print(f"  {tag} -> Test F1: {test_res['f1']*100:.2f}% (val: {best_val_f1*100:.2f}%)", flush=True)

    return {
        "regime": regime,
        "fold": fold,
        "seed": seed,
        "best_val_f1": best_val_f1,
        "test": test_res,
    }


def main():
    parser = argparse.ArgumentParser(description="Run matched MTL vs UTL protocol on RSA-GNN.")
    parser.add_argument("--cache-dir", default="feature_cache/registration-form")
    parser.add_argument("--output-dir", default="results/gnn_matched")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--seeds", default="13,42")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--edge-dim", type=int, default=16)
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    all_files = sorted(cache_path.glob("*.pt"))
    if not all_files:
        raise FileNotFoundError(f"No .pt files found in {cache_path}")

    # Load metadata
    meta_path = cache_path / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in meta["id2label"].items()}
        num_labels = len(meta["labels"])
    else:
        # Fallback to VRDU FARA 6 entity BIO labels
        id2label = {
            0: "O",
            1: "B-file_date", 2: "I-file_date",
            3: "B-foreign_principle_name", 4: "I-foreign_principle_name",
            5: "B-registrant_name", 6: "I-registrant_name",
            7: "B-registration_num", 8: "I-registration_num",
            9: "B-signer_name", 10: "I-signer_name",
            11: "B-signer_title", 12: "I-signer_title",
        }
        num_labels = len(id2label)

    print(f"[*] Preloading {len(all_files)} graph tensors into memory...", flush=True)
    loaded_cache: dict[Path, dict] = {}
    by_doc: dict[str, list[dict]] = {}
    doc_templates: dict[str, str] = {}

    for p in all_files:
        g = torch.load(p, weights_only=False)
        loaded_cache[p] = g
        d_guid = g.get("doc_guid", g.get("guid", "unknown")).split("#chunk")[0]
        by_doc.setdefault(d_guid, []).append(g)
        if "template_id" in g:
            doc_templates[d_guid] = g["template_id"]

    print(f"[+] Loaded {len(by_doc)} unique documents across {len(set(doc_templates.values()))} templates: {sorted(set(doc_templates.values()))}")

    # Create document proxies for matched_size_protocol
    doc_proxies = [
        GraphDocProxy(guid=d, template_id=doc_templates.get(d, "unknown"), files=[])
        for d in sorted(by_doc.keys())
    ]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    protocol = matched_size_protocol(doc_proxies, n_train=args.n_train, seed=seeds[0])

    print("\n" + "=" * 75)
    print(f"[*] PROTOCOL VERIFIED: Matched N={args.n_train} across MTL and UTL")
    print("=" * 75)
    for regime in ("MTL", "UTL"):
        for fold_idx, split in enumerate(protocol[regime]):
            proof = assert_no_template_leakage(split)
            print(f"\n{proof}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Training RSA-GNN ({args.num_layers}-Layer, {args.edge_dim}-D Edges) on {device}...\n")

    all_results = {"MTL": [], "UTL": []}

    for regime in ("MTL", "UTL"):
        print(f"\n{'='*75}\n[*] RUNNING REGIME: {regime}\n{'='*75}", flush=True)
        for fold_idx, split in enumerate(protocol[regime]):
            train_graphs = [g for doc in split.train for g in by_doc[doc.guid]]
            val_graphs = [g for doc in split.val for g in by_doc[doc.guid]]
            test_graphs = [g for doc in split.test for g in by_doc[doc.guid]]

            for s in seeds:
                res = train_one_matched_run(
                    regime=regime,
                    fold=fold_idx,
                    seed=s,
                    train_graphs=train_graphs,
                    val_graphs=val_graphs,
                    test_graphs=test_graphs,
                    num_labels=num_labels,
                    id2label=id2label,
                    device=device,
                    epochs=args.epochs,
                    num_layers=args.num_layers,
                    edge_dim=args.edge_dim,
                )
                all_results[regime].append(res)

    # Compute official generalization gap
    mtl_f1s = [r["test"]["f1"] for r in all_results["MTL"]]
    utl_f1s = [r["test"]["f1"] for r in all_results["UTL"]]
    gap_data = generalization_gap(mtl_f1s, utl_f1s)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_summary = {
        "n_train": args.n_train,
        "seeds": seeds,
        "num_layers": args.num_layers,
        "edge_dim": args.edge_dim,
        "gap": gap_data,
        "MTL": all_results["MTL"],
        "UTL": all_results["UTL"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_summary, indent=2, default=json_serializable), encoding="utf-8")

    mtl_mean = gap_data["mtl_mean_f1"] * 100
    mtl_std = gap_data["mtl_std"] * 100
    utl_mean = gap_data["utl_mean_f1"] * 100
    utl_std = gap_data["utl_std"] * 100
    diff = gap_data["gap_f1"] * 100
    drop = gap_data["relative_drop_pct"]

    # Comparative Summary Table with LiLT
    comparison = [
        ["Standard LiLT Baseline (Track 1)", "91.56% ± 1.1%", "64.24% ± 12.7%", "27.31%", "29.8% drop"],
        [f"RSA-GNN {args.num_layers}-Layer 16-D (Track 2)", f"{mtl_mean:.2f}% ± {mtl_std:.2f}%", f"{utl_mean:.2f}% ± {utl_std:.2f}%", f"{diff:.2f}%", f"{drop:.1f}% drop"],
    ]

    print("\n" + "=" * 80)
    print("FINAL HEAD-TO-HEAD: GENERALIZATION GAP CLOSURE ON VRDU FARA (N=200)")
    print("=" * 80)
    print(tabulate(comparison, headers=["Architecture", "MTL F1 (Seen)", "UTL F1 (Unseen)", "Gap (Δ)", "Relative Degradation"], tablefmt="github"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
