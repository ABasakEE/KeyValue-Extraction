"""Master Training Entrypoint for Track 2 Edge-Augmented Graph Attention Networks (Edge-GAT).

Trains relational message-passing directly on precomputed LiLT node embeddings and
16-D Delaunay visibility graphs, accelerating training to ~3s/epoch with <1 GB VRAM.
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
from tqdm.auto import tqdm

from src.evaluate import entity_f1, cross_check, malformed_tag_rate
from src.models.edge_gat import EdgeGATTokenClassifier
from src.train import TrainConfig, set_seed, json_serializable


class CachedGraphDataset(Dataset):
    """Dataset loading precomputed graph dicts (.pt) from feature_cache/."""

    def __init__(self, file_paths: list[Path]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.file_paths[idx], weights_only=False)


def collate_graphs(batch: list[dict]) -> list[dict]:
    """Collates a list of graph dictionaries."""
    return batch


@torch.no_grad()
def evaluate_gnn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    id2label: dict[int, str],
    desc: str = "Evaluating GNN",
) -> tuple[dict, list[list[int]], list[list[int]]]:
    model.eval()
    doc_scores: dict[str, torch.Tensor] = {}
    doc_counts: dict[str, torch.Tensor] = {}
    doc_refs: dict[str, list[int]] = {}
    doc_order: list[str] = []

    for batch in loader:
        for g in batch:
            x = g["x"].to(device)
            edge_index = g["edge_index"].to(device)
            edge_attr = g["edge_attr"].to(device)
            labels = g["labels"]
            doc_guid = g.get("doc_guid", g["guid"])
            offset = g.get("word_offset", 0)
            total_words = g.get("total_doc_words", len(labels))

            logits = model(x, edge_index, edge_attr).cpu()  # (N, num_classes)
            probs = torch.softmax(logits, dim=-1)

            if doc_guid not in doc_scores:
                num_classes = logits.size(-1)
                doc_scores[doc_guid] = torch.zeros((total_words, num_classes), dtype=torch.float32)
                doc_counts[doc_guid] = torch.zeros(total_words, dtype=torch.float32)
                doc_refs[doc_guid] = [-1] * total_words
                doc_order.append(doc_guid)

            for w_idx in range(len(labels)):
                target_idx = offset + w_idx
                if target_idx < total_words:
                    doc_refs[doc_guid][target_idx] = int(labels[w_idx].item())
                    doc_scores[doc_guid][target_idx] += probs[w_idx]
                    doc_counts[doc_guid][target_idx] += 1.0

    predictions: list[list[int]] = []
    references: list[list[int]] = []
    for doc_guid in doc_order:
        scores = doc_scores[doc_guid]
        counts = doc_counts[doc_guid]
        refs = doc_refs[doc_guid]
        pred_tags = scores.argmax(dim=-1).tolist()

        clean_preds = []
        clean_refs = []
        for w in range(len(counts)):
            if refs[w] != -1:
                clean_preds.append(pred_tags[w] if counts[w] > 0 else 0)
                clean_refs.append(refs[w])

        if clean_refs:
            predictions.append(clean_preds)
            references.append(clean_refs)

    scores = entity_f1(predictions, references, id2label)
    return scores.as_dict(), predictions, references


def train_gnn_one_seed(
    seed: int,
    train_files: list[Path],
    val_files: list[Path],
    test_files: list[Path],
    num_labels: int,
    id2label: dict[int, str],
    output_dir: Path,
    epochs: int = 30,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    num_layers: int = 2,
    edge_dim: int = 16,
    num_heads: int = 4,
    device_str: str = "auto",
) -> dict:
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and device_str == "auto" else (device_str if device_str != "auto" else "cpu"))
    print(f"\n[*] Training GNN Seed {seed} on {device} ({len(train_files)} train, {len(val_files)} val, {len(test_files)} test)...")

    train_loader = DataLoader(CachedGraphDataset(train_files), batch_size=4, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(CachedGraphDataset(val_files), batch_size=4, shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(CachedGraphDataset(test_files), batch_size=4, shuffle=False, collate_fn=collate_graphs)

    model = EdgeGATTokenClassifier(
        in_dim=768,
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        num_layers=num_layers,
        edge_dim=edge_dim,
        num_heads=num_heads,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_f1 = -1.0
    best_state = None
    patience = 5
    patience_cnt = 0
    history = []

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            for g in batch:
                x = g["x"].to(device)
                edge_index = g["edge_index"].to(device)
                edge_attr = g["edge_attr"].to(device)
                labels = g["labels"].to(device)

                logits = model(x, edge_index, edge_attr)
                loss = criterion(logits, labels)
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / max(1, len(batch))
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += batch_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        val_res, _, _ = evaluate_gnn(model, val_loader, device, id2label)
        val_f1 = val_res.get("f1", 0.0)

        history.append({"epoch": ep + 1, "train_loss": round(avg_loss, 4), "val_f1": round(val_f1, 4)})
        print(f"  [Ep {ep+1:>2}/{epochs}] Train Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} (best: {max(best_val_f1, val_f1):.4f})")

        if val_f1 > best_val_f1 + 0.001:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  [Early Stop] Reached plateau at epoch {ep+1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_res, test_preds, test_refs = evaluate_gnn(model, test_loader, device, id2label)
    result = {
        "seed": seed,
        "best_val_f1": round(best_val_f1, 4),
        "test": test_res,
        "history": history,
        "cross_check": cross_check(test_preds, test_refs, id2label),
        "malformed_tags": malformed_tag_rate(test_preds, id2label),
    }

    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=json_serializable), encoding="utf-8")
    if best_state is not None:
        torch.save(best_state, run_dir / "best_model.pt")

    print(f"[+] Seed {seed} Test F1: {test_res.get('f1', 0):.4f} | {result['cross_check']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train Edge-Augmented GAT directly on cached graphs.")
    parser.add_argument("--cache-dir", default="feature_cache/registration-form")
    parser.add_argument("--output-dir", default="results/vrdu_gnn_lilt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--edge-dim", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seeds", type=str, default="13,42")
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    meta_path = cache_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Run src/cache_features.py first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    num_labels = len(id2label)

    all_files = sorted(cache_path.glob("*.pt"))
    if not all_files:
        raise FileNotFoundError(f"No .pt graph files found in {cache_path}")

    # Group chunks by parent document GUID to prevent cross-chunk document leakage
    by_doc: dict[str, list[Path]] = {}
    for p in all_files:
        doc_key = p.stem.split("#chunk")[0]
        by_doc.setdefault(doc_key, []).append(p)

    doc_keys = sorted(by_doc.keys())
    rng = random.Random(42)
    rng.shuffle(doc_keys)

    n_total = len(doc_keys)
    n_train = max(1, int(0.8 * n_total))
    n_val = max(1, int(0.1 * n_total))

    train_docs = set(doc_keys[:n_train])
    val_docs = set(doc_keys[n_train:n_train + n_val])
    test_docs = set(doc_keys[n_train + n_val:])

    train_files = [f for d in train_docs for f in by_doc[d]]
    val_files = [f for d in val_docs for f in by_doc[d]]
    test_files = [f for d in test_docs for f in by_doc[d]]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for s in seeds:
        res = train_gnn_one_seed(
            seed=s,
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            num_labels=num_labels,
            id2label=id2label,
            output_dir=output_dir,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            edge_dim=args.edge_dim,
            num_heads=args.num_heads,
        )
        results.append(res)

    f1s = [r["test"]["f1"] for r in results]
    summary = {
        "model": "EdgeGATTokenClassifier",
        "mean_f1": round(sum(f1s) / len(f1s), 4),
        "std_f1": round(float(np.std(f1s)), 4) if len(f1s) > 1 else 0.0,
        "runs": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=json_serializable), encoding="utf-8")
    print(f"\n{'='*70}\n[+] Track 2 GNN Completed: Mean Test F1 = {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}\n{'='*70}")


if __name__ == "__main__":
    main()
