"""Feature Caching with Linguistic Entity Fusion & Delaunay Edge Ground Truth.

Extracts:
1. LiLT 768-D contextual token embeddings.
2. 8-D Linguistic Entity one-hot vector (DATE, MONEY, ORG, PERSON, ID, LOC, TEXT, OTHER).
3. 776-D fused node representations.
4. Planar Delaunay visibility graph with OpenCV border pruning & 16-D asymmetric spatial edges.
5. Dual node & edge ground truth targets for fast GNN training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
from tqdm.auto import tqdm

from src.data.vrdu_loader import VrduExample, chunk_vrdu_examples, load_vrdu
from src.models.delaunay_graph import build_delaunay_graph
from src.models.token_clf import TokenClassifier
from src.train import TrainConfig


def classify_text_semantic_type(text: str) -> int:
    """Classifies a text token into one of 8 canonical linguistic types.
    
    0: OTHER
    1: DATE
    2: MONEY / AMOUNT
    3: NUMERIC ID / CODE
    4: PERSON / ORG NAME
    5: LOCATION / ADDRESS
    6: GENERAL TEXT
    7: PUNCT / SEPARATOR
    """
    t = text.strip()
    if not t:
        return 0
    if t in {":", "-", "/", ",", ".", "#", "$", "(", ")"}:
        return 7

    lower = t.lower()

    # 1. Date detection
    if re.search(r"\b(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\b", lower):
        return 1

    # 2. Currency / Money detection
    if re.search(r"[\$€£¥]\s*[\d,]+(?:\.\d+)?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\b", t):
        return 2

    # 3. Numeric ID / Code
    if re.search(r"\b[A-Za-z0-9-]{4,}\b", t) and any(c.isdigit() for c in t):
        return 3

    # 4. Location keywords
    if any(k in lower for k in ["ave", "st", "blvd", "rd", "suite", "box", "street", "city", "state", "usa"]):
        return 5

    # 5. Name / Proper Noun heuristics
    if t.istitle() and len(t) > 2 and t.isalpha():
        return 4

    return 6


def get_semantic_one_hot(words: list[str]) -> torch.Tensor:
    """Generates (N, 8) one-hot semantic type matrix."""
    one_hots = torch.zeros((len(words), 8), dtype=torch.float32)
    for i, w in enumerate(words):
        cat = classify_text_semantic_type(w)
        one_hots[i, cat] = 1.0
    return one_hots


@torch.no_grad()
def extract_and_cache_fused(
    config_path: str,
    output_dir: str = "feature_cache_fused",
    device_str: str = "auto",
):
    config = TrainConfig.from_yaml(config_path)
    cache_root = Path(output_dir) / config.subcorpus
    cache_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if (device_str == "auto" and torch.cuda.is_available()) else "cpu")

    print(f"[*] Loading dataset: {config.dataset} ({config.subcorpus})...")
    examples, label_list, id2label = load_vrdu(config.data_root, config.subcorpus)
    print(f"[*] Loaded {len(examples)} documents.")

    if getattr(config, "sliding_window", True):
        w_size = getattr(config, "window_size", 384)
        w_overlap = getattr(config, "window_overlap", 128)
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

    pbar = tqdm(examples, desc="Extracting & Fusing Features", ncols=88)
    for ex in pbar:
        safe_name = ex.guid.replace("/", "_").replace("\\", "_")
        target_path = cache_root / f"{safe_name}.pt"
        if target_path.exists():
            continue

        # 1. LiLT contextual embeddings
        encoded = model_wrapper.encode([ex.words], [ex.boxes], [ex.ner_tags])
        inputs = {k: v.to(device) for k, v in encoded.inputs.items() if k != "labels"}
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][0].cpu()  # (seq_len, 768)

        # Map word tokens to first subword embedding
        word_ids = encoded.word_ids(0)
        num_words = len(ex.words)
        token_embeddings = torch.zeros((num_words, 768), dtype=torch.float32)
        seen_words = set()

        for s_idx, wid in enumerate(word_ids):
            if wid is not None and wid not in seen_words and wid < num_words:
                seen_words.add(wid)
                token_embeddings[wid] = hidden_states[s_idx]

        # 2. Extract linguistic entity one-hot features (N, 8)
        semantic_one_hot = get_semantic_one_hot(ex.words)

        # 3. Fuse: (N, 776)
        fused_x = torch.cat([token_embeddings, semantic_one_hot], dim=-1)

        # 4. Construct Delaunay Visibility Graph with 16-D edge attributes
        edge_index, edge_attr = build_delaunay_graph(
            boxes=ex.boxes,
            image=getattr(ex, "image", None),
            mode="delaunay",
            prune_visual_lines=True,
            add_self_loops=False,
        )

        # 5. Derive Node Roles & Edge Linking Ground Truth
        # Node roles: 0: OTHER, 1: KEY, 2: VALUE, 3: HEADER
        node_roles = torch.zeros((num_words,), dtype=torch.long)
        for i, tag in enumerate(ex.ner_tags):
            tag_name = id2label.get(tag, "O")
            if tag_name != "O":
                node_roles[i] = 2  # VALUE
            elif ex.words[i].endswith(":") or (i + 1 < num_words and id2label.get(ex.ner_tags[i + 1], "O") != "O"):
                node_roles[i] = 1  # KEY

        # Edge labels: 1 if edge is (KEY -> VALUE) within same field
        E = edge_index.size(1) if edge_index.numel() > 0 else 0
        edge_labels = torch.zeros((E,), dtype=torch.float32)
        if E > 0:
            for e in range(E):
                src, dst = int(edge_index[0, e]), int(edge_index[1, e])
                if node_roles[src] == 1 and node_roles[dst] == 2:
                    # Key sits to left or above Value
                    if ex.boxes[dst][0] >= ex.boxes[src][0] - 10:
                        edge_labels[e] = 1.0

        graph_dict = {
            "guid": ex.guid,
            "x": fused_x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "node_labels": node_roles,
            "edge_labels": edge_labels,
            "words": ex.words,
            "boxes": ex.boxes,
            "ner_tags": ex.ner_tags,
            "template_id": getattr(ex, "template_id", "unknown"),
        }
        torch.save(graph_dict, target_path)

    print(f"[+] Successfully cached fused features to {cache_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vrdu_track1_lilt.yaml")
    parser.add_argument("--output-dir", type=str, default="feature_cache_fused")
    args = parser.parse_args()
    extract_and_cache_fused(args.config, args.output_dir)
