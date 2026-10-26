"""Offline Feature Caching for Track 2 RSA-GNN.

Extracts (N, 768) node embeddings from the LiLT backbone once and precomputes
the Delaunay visibility graph with 16-D structural edge vectors.
Saved tensors accelerate GNN training to ~3s/epoch with <1 GB VRAM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
try:
    from utils.progress import FlushProgressBar as tqdm
except ImportError:
    from src.utils.progress import FlushProgressBar as tqdm

from src.data.vrdu_loader import VrduExample, chunk_vrdu_examples, load_vrdu
from src.models.delaunay_graph import build_delaunay_graph
from src.models.token_clf import TokenClassifier
from src.train import TrainConfig


def _cache_worker_shard(
    gpu_id: int,
    shard_examples: list,
    config,
    label_list: list,
    id2label: dict,
    cache_root: Path,
    graph_mode: str,
    k_neighbors: int,
    prune_visual_lines: bool,
):
    """Processes a shard of documents on a dedicated GPU."""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[GPU {gpu_id}] Initializing LiLT on {device} for {len(shard_examples)} document chunks...", flush=True)

    model_wrapper = TokenClassifier(
        backbone=config.backbone,
        num_labels=len(label_list),
        id2label=id2label,
        max_length=config.max_length,
    )
    model = model_wrapper.model.to(device)
    model.eval()

    with torch.no_grad():
        pbar = tqdm(shard_examples, desc=f"GPU {gpu_id} Caching", position=gpu_id, leave=True)
        for ex in pbar:
            safe_name = ex.guid.replace("/", "_").replace("\\", "_")
            target_path = cache_root / f"{safe_name}.pt"
            if target_path.exists():
                continue

            # 1. Encode with LiLT
            encoded = model_wrapper.encode([ex.words], [ex.boxes], [ex.ner_tags])
            inputs = {k: v.to(device) for k, v in encoded.inputs.items() if k != "labels"}

            # Extract last hidden states (1, seq_len, 768)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][0].cpu()

            word_ids = encoded.word_ids[0] if isinstance(encoded.word_ids, list) else encoded.word_ids(0)
            n_words = len(ex.words)
            node_feats = torch.zeros((n_words, hidden_states.size(-1)), dtype=torch.float32)

            previous = None
            for pos, wid in enumerate(word_ids):
                if wid is not None and wid != previous and wid < n_words:
                    node_feats[wid] = hidden_states[pos]
                previous = wid

            # 2. Build Delaunay Visibility Graph
            edge_index, edge_attr = build_delaunay_graph(
                boxes=ex.boxes,
                image=getattr(ex, "image", None),
                mode=graph_mode,
                k_neighbors=k_neighbors,
                prune_visual_lines=prune_visual_lines,
                add_self_loops=False,
            )

            # 3. Save to disk
            payload = {
                "x": node_feats,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "boxes": torch.tensor(ex.boxes, dtype=torch.long),
                "labels": torch.tensor(ex.ner_tags, dtype=torch.long),
                "guid": ex.guid,
                "doc_guid": getattr(ex, "doc_guid", ex.guid),
                "word_offset": getattr(ex, "word_offset", 0),
                "total_doc_words": getattr(ex, "total_doc_words", n_words),
                "template_id": ex.template_id,
            }
            torch.save(payload, target_path)


@torch.no_grad()
def extract_and_cache(
    config_path: str,
    output_dir: str = "feature_cache",
    device_str: str = "auto",
    graph_mode: str = "delaunay",
    k_neighbors: int = 5,
    prune_visual_lines: bool = True,
):
    config = TrainConfig.from_yaml(config_path)
    cache_root = Path(output_dir) / config.subcorpus
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading dataset: {config.dataset} ({config.subcorpus})...")
    examples, label_list, id2label = load_vrdu(config.data_root, config.subcorpus)
    print(f"[*] Loaded {len(examples)} document(s).")

    # Apply sliding window chunking if enabled
    if getattr(config, "sliding_window", True):
        w_size = getattr(config, "window_size", 384)
        w_overlap = getattr(config, "window_overlap", 128)
        print(f"[*] Applying sliding window chunking (window={w_size}, overlap={w_overlap})...")
        examples = chunk_vrdu_examples(examples, window_size=w_size, overlap=w_overlap)
        print(f"[*] Total chunked examples: {len(examples)}")

    # Save metadata
    meta = {
        "dataset": config.dataset,
        "subcorpus": config.subcorpus,
        "num_examples": len(examples),
        "labels": label_list,
        "id2label": id2label,
        "graph_mode": graph_mode,
        "k_neighbors": k_neighbors,
        "prune_visual_lines": prune_visual_lines,
    }
    (cache_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if device_str != "auto":
        target_gid = int(device_str.split(":")[-1]) if ":" in device_str else 0
        _cache_worker_shard(
            target_gid,
            examples,
            config,
            label_list,
            id2label,
            cache_root,
            graph_mode,
            k_neighbors,
            prune_visual_lines,
        )
    elif num_gpus >= 2:
        print(f"\n{'='*75}")
        print(f"[*] DUAL GPU ACCELERATION ACTIVE: Sharding {len(examples)} examples across {num_gpus} GPUs")
        print(f"[*] Worker 0 handles GPU 0 (cuda:0) | Worker 1 handles GPU 1 (cuda:1)")
        print(f"{'='*75}\n", flush=True)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = [
                executor.submit(
                    _cache_worker_shard,
                    gid,
                    examples[gid::num_gpus],
                    config,
                    label_list,
                    id2label,
                    cache_root,
                    graph_mode,
                    k_neighbors,
                    prune_visual_lines,
                )
                for gid in range(num_gpus)
            ]
            concurrent.futures.wait(futures)
    else:
        # Single GPU or CPU execution
        _cache_worker_shard(
            0,
            examples,
            config,
            label_list,
            id2label,
            cache_root,
            graph_mode,
            k_neighbors,
            prune_visual_lines,
        )

    cached_count = len(list(cache_root.glob("*.pt")))
    print(f"\n[+] Successfully cached {cached_count} graphs to {cache_root}")


def main():
    parser = argparse.ArgumentParser(description="Cache LiLT node features and Delaunay graphs to disk.")
    parser.add_argument("--config", default="configs/vrdu_track1_lilt.yaml")
    parser.add_argument("--output-dir", default="feature_cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--graph-mode", choices=["delaunay", "knn", "fully_connected"], default="delaunay")
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--no-prune-lines", action="store_true")
    args = parser.parse_args()

    extract_and_cache(
        config_path=args.config,
        output_dir=args.output_dir,
        device_str=args.device,
        graph_mode=args.graph_mode,
        k_neighbors=args.k_neighbors,
        prune_visual_lines=not args.no_prune_lines,
    )


if __name__ == "__main__":
    main()
