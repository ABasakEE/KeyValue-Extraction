"""Training Entrypoint for Dual-Head Edge-GAT with Greedy Bipartite Linking & S-BERT Schema Mapping.

Fast in-memory training over fused (LiLT 768-D + NER 8-D) representations and Delaunay planar graphs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
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
try:
    from utils.progress import FlushProgressBar as tqdm
except ImportError:
    from src.utils.progress import FlushProgressBar as tqdm

from src.models.edge_gat_joint import EdgeGATJointExtractor, focal_binary_loss
from src.train import set_seed, json_serializable


class CachedFusedDataset(Dataset):
    """Loads precomputed fused graph dicts (.pt)."""

    def __init__(self, file_paths: list[Path]):
        self.file_paths = file_paths
        print(f"[*] Preloading {len(file_paths)} fused graph tensors into memory...", flush=True)
        self.graphs = [torch.load(p, weights_only=False) for p in file_paths]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict:
        return self.graphs[idx]


def collate_fused_graphs(batch: list[dict]) -> dict:
    """Collates graphs into a single disjoint block-diagonal batch."""
    xs = []
    edges = []
    attrs = []
    node_labels = []
    edge_labels = []
    node_offset = 0

    edge_dim = 16

    for g in batch:
        n_nodes = g["x"].size(0)
        xs.append(g["x"])
        node_labels.append(g["node_labels"])

        e_idx = g["edge_index"]
        if e_idx.numel() > 0 and e_idx.size(1) > 0:
            edges.append(e_idx + node_offset)
            attrs.append(g["edge_attr"])
            edge_labels.append(g["edge_labels"])

        node_offset += n_nodes

    batched_x = torch.cat(xs, dim=0)
    batched_node_labels = torch.cat(node_labels, dim=0)
    batched_edge_index = torch.cat(edges, dim=1) if edges else torch.zeros((2, 0), dtype=torch.long)
    batched_edge_attr = torch.cat(attrs, dim=0) if attrs else torch.zeros((0, edge_dim), dtype=torch.float32)
    batched_edge_labels = torch.cat(edge_labels, dim=0) if edge_labels else torch.zeros((0,), dtype=torch.float32)

    return {
        "x": batched_x,
        "edge_index": batched_edge_index,
        "edge_attr": batched_edge_attr,
        "node_labels": batched_node_labels,
        "edge_labels": batched_edge_labels,
    }


def evaluate_fused_gnn(
    model: EdgeGATJointExtractor,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    all_node_preds = []
    all_node_targets = []
    all_edge_preds = []
    all_edge_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)

            node_logits, edge_logits = model(x, edge_index, edge_attr)

            node_preds = node_logits.argmax(dim=-1).cpu().numpy()
            node_targs = batch["node_labels"].numpy()
            all_node_preds.extend(node_preds)
            all_node_targets.extend(node_targs)

            if edge_logits.numel() > 0:
                edge_probs = torch.sigmoid(edge_logits).cpu().numpy()
                edge_preds = (edge_probs >= threshold).astype(float)
                edge_targs = batch["edge_labels"].numpy()
                all_edge_preds.extend(edge_preds)
                all_edge_targets.extend(edge_targs)

    # Compute node accuracy
    node_preds_arr = np.array(all_node_preds)
    node_targs_arr = np.array(all_node_targets)
    node_acc = float((node_preds_arr == node_targs_arr).mean())

    # Compute edge linking precision, recall, F1
    if all_edge_preds:
        p_arr = np.array(all_edge_preds)
        t_arr = np.array(all_edge_targets)
        tp = float(((p_arr == 1.0) & (t_arr == 1.0)).sum())
        fp = float(((p_arr == 1.0) & (t_arr == 0.0)).sum())
        fn = float(((p_arr == 0.0) & (t_arr == 1.0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        link_f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    else:
        prec, rec, link_f1 = 0.0, 0.0, 0.0

    return {
        "node_acc": round(node_acc, 4),
        "link_precision": round(prec, 4),
        "link_recall": round(rec, 4),
        "link_f1": round(link_f1, 4),
    }


def train_fused_gnn_one_seed(
    seed: int,
    train_ds: CachedFusedDataset,
    val_ds: CachedFusedDataset,
    test_ds: CachedFusedDataset,
    epochs: int = 25,
    lr: float = 1e-3,
    output_dir: Path = Path("results/typed_gnn"),
    device_str: str = "auto",
) -> dict:
    set_seed(seed)
    device = torch.device(device_str if device_str != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"\n[*] Training Fused GNN Seed {seed} on {device}...", flush=True)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fused_graphs)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fused_graphs)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fused_graphs)

    model = EdgeGATJointExtractor(
        in_dim=776,
        hidden_dim=256,
        num_node_labels=4,
        num_layers=3,
        edge_dim=16,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_f1 = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate_fused_gnn(model, val_loader, device)
        val_score = 0.5 * (val_metrics["node_acc"] + val_metrics["link_f1"])

        if val_score > best_val_f1:
            best_val_f1 = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  [Seed {seed} on {device}] Ep {ep+1:>2}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
            f"Val Node Acc: {val_metrics['node_acc']:.4f} | Link F1: {val_metrics['link_f1']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_fused_gnn(model, test_loader, device)
    print(f"\n[+] Seed {seed} on {device} Test Results: {test_metrics}", flush=True)

    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, run_dir / "best_model.pt")

    result = {
        "seed": seed,
        "device": str(device),
        "val_score": best_val_f1,
        "test": test_metrics,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="feature_cache_fused/registration-form")
    parser.add_argument("--output-dir", type=str, default="results/fused_gnn")
    parser.add_argument("--seeds", type=str, default="13,42")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    all_files = sorted(list(cache_path.glob("*.pt")))
    if not all_files:
        print(f"[-] No cached .pt files found in {cache_path}. Run cache_fused_features.py first.")
        return

    random.seed(42)
    random.shuffle(all_files)
    n_total = len(all_files)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    train_paths = all_files[:n_train]
    val_paths = all_files[n_train : n_train + n_val]
    test_paths = all_files[n_train + n_val :]

    print(f"[*] Dataset Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test", flush=True)

    # Preload datasets once into shared memory
    train_ds = CachedFusedDataset(train_paths)
    val_ds = CachedFusedDataset(val_paths)
    test_ds = CachedFusedDataset(test_paths)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    results = []

    if num_gpus >= 2 and len(seeds) >= 2:
        max_workers = min(num_gpus, len(seeds))
        print(f"\n{'='*75}", flush=True)
        print(f"[*] DUAL GPU DETECTED: Spawning {max_workers} parallel workers on {[f'cuda:{i}' for i in range(max_workers)]}", flush=True)
        print(f"[*] Worker 0 takes Seed {seeds[0]} on cuda:0 | Worker 1 takes Seed {seeds[1]} on cuda:1", flush=True)
        print(f"{'='*75}\n", flush=True)
        import concurrent.futures

        def _worker(item):
            worker_idx, s = item
            assigned_gpu = f"cuda:{worker_idx % num_gpus}"
            return train_fused_gnn_one_seed(
                seed=s,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                epochs=args.epochs,
                output_dir=output_dir,
                device_str=assigned_gpu,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_worker, enumerate(seeds)))
    else:
        for s in seeds:
            res = train_fused_gnn_one_seed(
                seed=s,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                epochs=args.epochs,
                output_dir=output_dir,
                device_str="auto",
            )
            results.append(res)

    print("\n" + "=" * 65)
    print("FUSED RSA-GNN BENCHMARK SUMMARY")
    print("=" * 65)
    avg_node_acc = np.mean([r["test"]["node_acc"] for r in results])
    avg_link_f1 = np.mean([r["test"]["link_f1"] for r in results])
    print(f"Mean Node Accuracy : {avg_node_acc*100:.2f}%")
    print(f"Mean Link F1 Score  : {avg_link_f1*100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
