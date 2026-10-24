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
from tqdm.auto import tqdm

from src.data.vrdu_loader import VrduExample, chunk_vrdu_examples, load_vrdu
from src.models.delaunay_graph import build_delaunay_graph
from src.models.token_clf import TokenClassifier
from src.train import TrainConfig


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

    if device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

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

    print(f"[*] Initializing LiLT backbone encoder on {device}...")
    model_wrapper = TokenClassifier(
        backbone=config.backbone,
        num_labels=len(label_list),
        id2label=id2label,
        max_length=config.max_length,
    )
    model = model_wrapper.model.to(device)
    model.eval()

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

    cached_count = 0
    pbar = tqdm(examples, desc="Extracting & Caching Features", ncols=88)
    for ex in pbar:
        # Sanitize filename for disk
        safe_name = ex.guid.replace("/", "_").replace("\\", "_")
        target_path = cache_root / f"{safe_name}.pt"
        if target_path.exists():
            cached_count += 1
            continue

        # 1. Encode with LiLT
        encoded = model_wrapper.encode([ex.words], [ex.boxes], [ex.ner_tags])
        inputs = {k: v.to(device) for k, v in encoded.inputs.items() if k != "labels"}

        # Extract last hidden states (1, seq_len, 768)
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][0].cpu()  # (seq_len, 768)

        # Pull first-subword representations for each word
        word_ids = encoded.word_ids[0]
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
            image=None,
            mode=graph_mode,
            k_neighbors=k_neighbors,
            prune_visual_lines=prune_visual_lines,
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
        cached_count += 1

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
