"""Matched-Size Protocol (MTL vs UTL) for Track 3 NER-Fused Joint RSA-GNN.

Runs the strict template-disjoint protocol on the dual-head joint GNN (776-D fused
node features, 16-D asymmetric spatial edges) across dual GPUs (Worker 0 on cuda:0,
Worker 1 on cuda:1). Measures genuine cross-template generalization for Key->Value linking.
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
from src.models.edge_gat_joint import EdgeGATJointExtractor, focal_binary_loss
from src.train import set_seed, json_serializable, generalization_gap
from src.train_typed_gnn import collate_fused_graphs, evaluate_fused_gnn


@dataclass
class GraphDocProxy:
    guid: str
    template_id: str
    files: list[Path]


class FastCachedFusedDataset(Dataset):
    def __init__(self, graphs: list[dict]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict:
        return self.graphs[idx]


def train_one_matched_fused_run(
    regime: str,
    fold: int,
    seed: int,
    train_graphs: list[dict],
    val_graphs: list[dict],
    test_graphs: list[dict],
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    num_layers: int = 3,
    edge_dim: int = 16,
) -> dict:
    set_seed(seed)
    tag = f"[{device} | {regime}/fold{fold}/seed{seed}]"

    train_ds = FastCachedFusedDataset(train_graphs)
    val_ds = FastCachedFusedDataset(val_graphs)
    test_ds = FastCachedFusedDataset(test_graphs)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fused_graphs)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fused_graphs)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fused_graphs)

    model = EdgeGATJointExtractor(
        in_dim=train_graphs[0]["x"].size(-1),
        hidden_dim=256,
        num_node_labels=4,
        num_layers=num_layers,
        edge_dim=edge_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            node_labels = batch["node_labels"].to(device)
            edge_labels = batch["edge_labels"].to(device)

            node_logits, edge_logits = model(x, edge_index, edge_attr)

            loss_node = torch.nn.functional.cross_entropy(node_logits, node_labels)
            loss_edge = focal_binary_loss(edge_logits, edge_labels)
            loss = loss_node + 1.5 * loss_edge

            opt.zero_grad()
            loss.backward()
            opt.step()

        val_metrics = evaluate_fused_gnn(model, val_loader, device)
        val_score = 0.5 * (val_metrics["node_acc"] + val_metrics["link_f1"])

        if val_score > best_val_f1:
            best_val_f1 = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_fused_gnn(model, test_loader, device)
    print(
        f"  {tag} -> Link F1: {test_metrics['link_f1']*100:.2f}% | "
        f"Prec: {test_metrics['link_precision']*100:.2f}% | "
        f"Rec: {test_metrics['link_recall']*100:.2f}% | "
        f"Node Acc: {test_metrics['node_acc']*100:.2f}%",
        flush=True,
    )

    return {
        "regime": regime,
        "fold": fold,
        "seed": seed,
        "best_val_f1": best_val_f1,
        "test": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Run matched MTL vs UTL protocol on NER-Fused Joint RSA-GNN.")
    parser.add_argument("--cache-dir", default="feature_cache_fused/registration-form")
    parser.add_argument("--output-dir", default="results/typed_gnn_matched")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--seeds", default="13,42")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--edge-dim", type=int, default=16)
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    all_files = sorted(cache_path.glob("*.pt"))
    if not all_files:
        raise FileNotFoundError(f"No .pt files found in {cache_path}. Run cache_fused_features.py first.")

    print(f"[*] Preloading {len(all_files)} fused graph tensors into memory...", flush=True)
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

    doc_proxies = [
        GraphDocProxy(guid=d, template_id=doc_templates.get(d, "unknown"), files=[])
        for d in sorted(by_doc.keys())
    ]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    protocol = matched_size_protocol(doc_proxies, n_train=args.n_train, seed=seeds[0])

    print("\n" + "=" * 75)
    print(f"[*] PROTOCOL VERIFIED: Matched N={args.n_train} across MTL and UTL (NER-Fused)")
    print("=" * 75)
    for regime in ("MTL", "UTL"):
        for fold_idx, split in enumerate(protocol[regime]):
            proof = assert_no_template_leakage(split)
            print(f"\n{proof}")

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device("cuda:0" if num_gpus > 0 else "cpu")

    all_results = {"MTL": [], "UTL": []}

    for regime in ("MTL", "UTL"):
        print(f"\n{'='*75}\n[*] RUNNING REGIME: {regime} (NER-Fused)\n{'='*75}", flush=True)
        for fold_idx, split in enumerate(protocol[regime]):
            train_graphs = [g for doc in split.train for g in by_doc[doc.guid]]
            val_graphs = [g for doc in split.val for g in by_doc[doc.guid]]
            test_graphs = [g for doc in split.test for g in by_doc[doc.guid]]

            if num_gpus >= 2 and len(seeds) >= 2:
                import concurrent.futures

                def _worker(item):
                    idx, s = item
                    assigned_dev = torch.device(f"cuda:{idx % num_gpus}")
                    return train_one_matched_fused_run(
                        regime=regime,
                        fold=fold_idx,
                        seed=s,
                        train_graphs=train_graphs,
                        val_graphs=val_graphs,
                        test_graphs=test_graphs,
                        device=assigned_dev,
                        epochs=args.epochs,
                        num_layers=args.num_layers,
                        edge_dim=args.edge_dim,
                    )

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_gpus, len(seeds))) as ex:
                    res_list = list(ex.map(_worker, enumerate(seeds)))
                    all_results[regime].extend(res_list)
            else:
                for s in seeds:
                    res = train_one_matched_fused_run(
                        regime=regime,
                        fold=fold_idx,
                        seed=s,
                        train_graphs=train_graphs,
                        val_graphs=val_graphs,
                        test_graphs=test_graphs,
                        device=device,
                        epochs=args.epochs,
                        num_layers=args.num_layers,
                        edge_dim=args.edge_dim,
                    )
                    all_results[regime].append(res)

    # Compute Link F1 gap
    mtl_f1s = [r["test"]["link_f1"] for r in all_results["MTL"]]
    utl_f1s = [r["test"]["link_f1"] for r in all_results["UTL"]]
    gap_data = generalization_gap(mtl_f1s, utl_f1s)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_train": args.n_train,
        "seeds": seeds,
        "gap": gap_data,
        "MTL": all_results["MTL"],
        "UTL": all_results["UTL"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=json_serializable), encoding="utf-8")

    mtl_mean = gap_data["mtl_mean_f1"] * 100
    mtl_std = gap_data["mtl_std"] * 100
    utl_mean = gap_data["utl_mean_f1"] * 100
    utl_std = gap_data["utl_std"] * 100
    diff = gap_data["gap_f1"] * 100
    drop = gap_data["relative_drop_pct"]

    print("\n" + "=" * 80)
    print("NER-FUSED RSA-GNN: LINKING GENERALIZATION GAP ON VRDU FARA (N=200)")
    print("=" * 80)
    print(f"MTL Link F1 (Seen Templates)   : {mtl_mean:.2f}% ± {mtl_std:.2f}%")
    print(f"UTL Link F1 (Unseen Templates) : {utl_mean:.2f}% ± {utl_std:.2f}%")
    print(f"Generalization Gap (Δ)         : {diff:.2f}% ({drop:.1f}% relative drop)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
