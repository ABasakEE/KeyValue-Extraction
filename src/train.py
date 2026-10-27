"""Config-driven fine-tuning for the token-classification track.

Two entry points matching the plan's two pilot tracks::

    # Track A — pipeline sanity on FUNSD's standard split
    python src/train.py --config configs/funsd_lilt.yaml

    # Track B — the headline, MTL vs UTL at matched training-set size
    python src/train.py --config configs/vrdu_lilt.yaml --protocol matched

Track A's number validates the code path and nothing more. FUNSD's train and
test sets share 16% of their templates (Laatiri et al., ICDAR 2023,
arXiv:2304.14936) and its block-level annotation leaks entity boundaries through
shared coordinates (Zhang et al., arXiv:2402.02379), so a high FUNSD F1 is not
evidence of cross-template generalization and must not be presented as such.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import time
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from torch.utils.data import DataLoader, Dataset

from evaluate import cross_check, entity_f1, generalization_gap, malformed_tag_rate, linking_f1
from models.token_clf import TokenClassifier
from models.joint_extractor import JointFormExtractor, prepare_entity_batch
try:
    from utils.progress import FlushProgressBar as tqdm
except ImportError:
    from src.utils.progress import FlushProgressBar as tqdm

TQDM_KWARGS = {
    "file": sys.stdout,
    "ncols": 88,
    "mininterval": 0.2,
}


def json_serializable(obj):
    """Fallback serializer for json.dumps to handle numpy/torch scalars."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    backbone: str = "lilt"
    dataset: str = "funsd"
    data_root: str = "data/funsd/dataset"
    task: str = "tagging"                  # "tagging" or "joint"
    linking_loss_weight: float = 1.0
    pos_weight: float = 10.0
    use_spatial_relations: bool = True
    spatial_in_features: int = 16
    use_geometry: bool = True
    use_containment: bool = True
    use_line_of_sight: bool = True
    use_typography: bool = True
    loss_type: str = "focal"              # "focal" or "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    masking_mode: str = "soft"            # "soft", "hard", or "none"
    k_neighbors: int = 0                  # 0 = all O(N^2) pairs, > 0 = KNN spatial filtering
    fallback_top_k: int = 0
    decode_one_to_one: bool = False       # True = 1-to-1 argmax matching per value entity
    link_threshold: float = 0.5
    languages: tuple[str, ...] | list[str] = ("es", "fr", "de", "it", "pt")
    curriculum: bool = False              # True = 3-stage decoupled warmup
    patience_tolerance: int = 3           # Early stopping patience (epochs without validation improvement)
    min_delta: float = 0.001              # Minimum improvement threshold to reset patience
    phase1_epochs: int = 10               # Phase 1: Tagging warmup (lambda_link = 0)
    phase2_epochs: int = 8                # Phase 2: Teacher-forced relation head warmup
    phase3_epochs: int = 10               # Phase 3: Joint fine-tuning
    max_phase1_epochs: int = 10           # Alias/cap for Phase 1
    max_phase2_epochs: int = 8            # Alias/cap for Phase 2
    max_phase3_epochs: int = 10           # Alias/cap for Phase 3
    epochs: int = 20
    batch_size: int = 4
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    seeds: tuple[int, ...] = (13, 21, 42)
    subcorpus: str = "registration-form"
    n_train: int | None = None
    output_dir: str = "results/run"
    sliding_window: bool = False
    window_size: int = 384
    window_overlap: int = 128
    coordinate_jitter: bool = False
    jitter_shift: int = 80
    jitter_scale_min: float = 0.85
    jitter_scale_max: float = 1.15
    relative_coords: bool = False
    anchor_mode: str = "content_box"
    adversarial_dann: bool = False
    dann_weight: float = 0.1
    dann_gamma: float = 10.0
    dann_hidden_dim: int = 256
    num_templates: int = 32
    save_weights: bool = True
    use_graph_bros: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    beta_skeleton: float = 1.0
    graph_eps: float = 1e-9
    merge_spans: bool = True
    gradient_checkpointing: bool = False
    max_folds: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if "seeds" in raw:
            raw["seeds"] = tuple(raw["seeds"])
        if "languages" in raw and isinstance(raw["languages"], (list, tuple)):
            raw["languages"] = tuple(raw["languages"])
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}. Known keys: {sorted(known)}")
        return cls(**raw)


def jitter_boxes(
    boxes: list[list[int]],
    shift_range: int = 80,
    scale_range: tuple[float, float] = (0.85, 1.15),
) -> list[list[int]]:
    """Track 1 Coordinate Jitter: random translation and scaling to break spatial memorization."""
    if not boxes:
        return boxes
    dx = random.randint(-shift_range, shift_range)
    dy = random.randint(-shift_range, shift_range)
    sx = random.uniform(scale_range[0], scale_range[1])
    sy = random.uniform(scale_range[0], scale_range[1])

    all_x = [b[0] for b in boxes] + [b[2] for b in boxes]
    all_y = [b[1] for b in boxes] + [b[3] for b in boxes]
    cx = (min(all_x) + max(all_x)) / 2.0
    cy = (min(all_y) + max(all_y)) / 2.0

    new_boxes = []
    for x0, y0, x1, y1 in boxes:
        nx0 = int(round(cx + (x0 - cx) * sx + dx))
        ny0 = int(round(cy + (y0 - cy) * sy + dy))
        nx1 = int(round(cx + (x1 - cx) * sx + dx))
        ny1 = int(round(cy + (y1 - cy) * sy + dy))
        nx0, nx1 = min(nx0, nx1), max(nx0, nx1)
        ny0, ny1 = min(ny0, ny1), max(ny0, ny1)
        nx0 = max(0, min(1000, nx0))
        ny0 = max(0, min(1000, ny0))
        nx1 = max(0, min(1000, nx1))
        ny1 = max(0, min(1000, ny1))
        new_boxes.append([nx0, ny0, nx1, ny1])
    return new_boxes


def make_relative_boxes(boxes: list[list[int]]) -> list[list[int]]:
    """Track 1 Relative Coordinate Injection: center document centroid at (500, 500)."""
    if not boxes:
        return boxes
    all_x = [b[0] for b in boxes] + [b[2] for b in boxes]
    all_y = [b[1] for b in boxes] + [b[3] for b in boxes]
    cx = (min(all_x) + max(all_x)) / 2.0
    cy = (min(all_y) + max(all_y)) / 2.0

    new_boxes = []
    for x0, y0, x1, y1 in boxes:
        nx0 = int(round(500 + (x0 - cx)))
        ny0 = int(round(500 + (y0 - cy)))
        nx1 = int(round(500 + (x1 - cx)))
        ny1 = int(round(500 + (y1 - cy)))
        nx0, nx1 = min(nx0, nx1), max(nx0, nx1)
        ny0, ny1 = min(ny0, ny1), max(ny0, ny1)
        nx0 = max(0, min(1000, nx0))
        ny0 = max(0, min(1000, ny0))
        nx1 = max(0, min(1000, nx1))
        ny1 = max(0, min(1000, ny1))
        new_boxes.append([nx0, ny0, nx1, ny1])
    return new_boxes


class FormDataset(Dataset):
    """Wraps parsed documents; tokenization happens in the collator."""

    def __init__(self, examples: list):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


def make_collator(
    model: TokenClassifier | JointFormExtractor,
    is_joint: bool = False,
    is_train: bool = False,
    coordinate_jitter: bool = False,
    relative_coords: bool = False,
    anchor_mode: str = "none",
    is_dann: bool = False,
    template_to_idx: dict[str, int] | None = None,
):
    def collate(batch: list):
        boxes_batch = [ex.boxes for ex in batch]
        if is_train and coordinate_jitter:
            boxes_batch = [jitter_boxes(b) for b in boxes_batch]
        mode = anchor_mode
        if mode == "none" and relative_coords:
            mode = "content_box"
        if mode != "none":
            try:
                from data.transforms import anchor_boxes
            except ImportError:
                from src.data.transforms import anchor_boxes
            boxes_batch = [
                anchor_boxes(b, mode=mode, active_envelope=getattr(ex, "doc_envelope", None))
                for b, ex in zip(boxes_batch, batch)
            ]

        encoded = model.encode(
            [ex.words for ex in batch],
            boxes_batch,
            [ex.ner_tags for ex in batch],
        )
        if is_joint:
            spans, links, types = prepare_entity_batch(batch, encoded.word_ids)
            return encoded, batch, spans, links, types
        if is_dann and template_to_idx is not None:
            t_indices = [template_to_idx.get(getattr(ex, "template_id", "unk"), 0) for ex in batch]
            template_targets = torch.tensor(t_indices, dtype=torch.long)
            return encoded, batch, template_targets
        return encoded, batch

    return collate


@torch.no_grad()
def predict(
    classifier: TokenClassifier,
    loader: DataLoader,
    device: torch.device,
    desc: str | None = None,
):
    """Predict token labels with multi-page sliding window soft-voting aggregation."""
    classifier.model.eval()
    doc_scores: dict[str, torch.Tensor] = {}
    doc_counts: dict[str, torch.Tensor] = {}
    doc_refs: dict[str, list[int]] = {}
    doc_order: list[str] = []

    it = tqdm(loader, desc=desc, leave=False, **TQDM_KWARGS) if desc else loader
    for batch_data in it:
        if len(batch_data) == 3:
            encoded, batch, _ = batch_data
        else:
            encoded, batch = batch_data
        inputs = {k: v.to(device) for k, v in encoded.inputs.items() if k != "labels"}
        out = classifier.model(**inputs)
        logits = out.token_logits.cpu() if hasattr(out, "token_logits") else out.logits.cpu()
        probs = torch.softmax(logits, dim=-1)

        for i, example in enumerate(batch):
            doc_guid = getattr(example, "doc_guid", None) or getattr(example, "guid", f"doc_{i}")
            word_offset = getattr(example, "word_offset", 0)
            total_words = getattr(example, "total_doc_words", None) or len(example.words)

            if doc_guid not in doc_scores:
                num_classes = logits.shape[-1]
                doc_scores[doc_guid] = torch.zeros((total_words, num_classes), dtype=torch.float32)
                doc_counts[doc_guid] = torch.zeros(total_words, dtype=torch.float32)
                doc_refs[doc_guid] = [-1] * total_words
                doc_order.append(doc_guid)

            for w_idx, tag in enumerate(example.ner_tags):
                target_idx = word_offset + w_idx
                if target_idx < total_words:
                    doc_refs[doc_guid][target_idx] = tag

            word_ids = encoded.word_ids[i]
            previous = None
            for pos, wid in enumerate(word_ids):
                if wid is not None and wid != previous and wid < len(example.words):
                    target_idx = word_offset + wid
                    if target_idx < total_words:
                        doc_scores[doc_guid][target_idx] += probs[i, pos]
                        doc_counts[doc_guid][target_idx] += 1.0
                previous = wid

    predictions: list[list[int]] = []
    references: list[list[int]] = []
    for doc_guid in doc_order:
        scores = doc_scores[doc_guid]
        counts = doc_counts[doc_guid]
        pred_tags = scores.argmax(dim=-1).tolist()
        for w in range(len(counts)):
            if counts[w] == 0:
                pred_tags[w] = 0
        predictions.append(pred_tags)
        references.append(doc_refs[doc_guid])

    return predictions, references


@torch.no_grad()
def predict_joint(
    model: JointFormExtractor,
    loader: DataLoader,
    device: torch.device,
    desc: str | None = None,
):
    model.eval()
    predictions: list[list[int]] = []
    references: list[list[int]] = []
    all_pred_links: list[set[tuple[int, int]]] = []
    all_gold_links: list[set[tuple[int, int]]] = []

    it = tqdm(loader, desc=desc, leave=False, **TQDM_KWARGS) if desc else loader
    for encoded, batch, entity_spans, gold_links, entity_types in it:
        inputs = {k: v.to(device) for k, v in encoded.inputs.items() if k != "labels"}
        outputs = model(
            inputs=inputs,
            entity_spans=entity_spans,
            gold_links=None,
            entity_types=entity_types,
        )
        logits = outputs.token_logits.cpu()
        for i, example in enumerate(batch):
            predictions.append(
                model.first_subword_predictions(logits[i], encoded.word_ids[i], len(example.words))
            )
            references.append(example.ner_tags)

        all_pred_links.extend(outputs.predicted_links)
        for g_links in gold_links:
            all_gold_links.append(set(g_links))

    return predictions, references, all_pred_links, all_gold_links


@torch.no_grad()
def predict_phase1_tagging(
    model: JointFormExtractor,
    loader: DataLoader,
    device: torch.device,
    desc: str | None = None,
):
    """Phase 1 validation: ONLY evaluate token classification (tagging) with soft voting."""
    model.eval()
    doc_scores: dict[str, torch.Tensor] = {}
    doc_counts: dict[str, torch.Tensor] = {}
    doc_refs: dict[str, list[int]] = {}
    doc_order: list[str] = []

    it = tqdm(loader, desc=desc, leave=False, **TQDM_KWARGS) if desc else loader
    for batch_data in it:
        encoded, batch = batch_data[0], batch_data[1]
        inputs = {k: v.to(device) for k, v in encoded.inputs.items() if k != "labels"}
        logits = model.encoder(**inputs).logits.cpu()
        probs = torch.softmax(logits, dim=-1)

        for i, example in enumerate(batch):
            doc_guid = getattr(example, "doc_guid", None) or getattr(example, "guid", f"doc_{i}")
            word_offset = getattr(example, "word_offset", 0)
            total_words = getattr(example, "total_doc_words", None) or len(example.words)

            if doc_guid not in doc_scores:
                num_classes = logits.shape[-1]
                doc_scores[doc_guid] = torch.zeros((total_words, num_classes), dtype=torch.float32)
                doc_counts[doc_guid] = torch.zeros(total_words, dtype=torch.float32)
                doc_refs[doc_guid] = [-1] * total_words
                doc_order.append(doc_guid)

            for w_idx, tag in enumerate(example.ner_tags):
                target_idx = word_offset + w_idx
                if target_idx < total_words:
                    doc_refs[doc_guid][target_idx] = tag

            word_ids = encoded.word_ids[i]
            previous = None
            for pos, wid in enumerate(word_ids):
                if wid is not None and wid != previous and wid < len(example.words):
                    target_idx = word_offset + wid
                    if target_idx < total_words:
                        doc_scores[doc_guid][target_idx] += probs[i, pos]
                        doc_counts[doc_guid][target_idx] += 1.0
                previous = wid

    predictions: list[list[int]] = []
    references: list[list[int]] = []
    for doc_guid in doc_order:
        scores = doc_scores[doc_guid]
        counts = doc_counts[doc_guid]
        pred_tags = scores.argmax(dim=-1).tolist()
        for w in range(len(counts)):
            if counts[w] == 0:
                pred_tags[w] = 0
        predictions.append(pred_tags)
        references.append(doc_refs[doc_guid])

    return predictions, references


@torch.no_grad()
def predict_phase2_linking(
    model: JointFormExtractor,
    loader: DataLoader,
    device: torch.device,
    desc: str | None = None,
):
    """Phase 2 validation: ONLY evaluate key-value linking under Teacher Forcing.

    Evaluates relation linking performance with ground-truth entity spans/types
    without re-evaluating the frozen token encoder.
    """
    model.eval()
    all_pred_links: list[set[tuple[int, int]]] = []
    all_gold_links: list[set[tuple[int, int]]] = []
    it = tqdm(loader, desc=desc, leave=False, **TQDM_KWARGS) if desc else loader
    for encoded, batch, entity_spans, gold_links, entity_types in it:
        inputs = {k: v.to(device) for k, v in encoded.inputs.items() if k != "labels"}
        outputs = model(
            inputs=inputs,
            entity_spans=entity_spans,
            gold_links=None,
            entity_types=entity_types,
        )
        all_pred_links.extend(outputs.predicted_links)
        for g_links in gold_links:
            all_gold_links.append(set(g_links))
    return all_pred_links, all_gold_links


def train_one(
    model: TokenClassifier | JointFormExtractor,
    train_examples: list,
    val_examples: list,
    test_examples: list,
    config: TrainConfig,
    device: torch.device,
    checkpoint_dir: Path | None = None,
    log_prefix: str = "",
) -> dict:
    """Fine-tune once and return test metrics for the best-validation checkpoint."""
    pfx = f"{log_prefix} " if log_prefix else ""
    leave_bar = not bool(log_prefix)

    is_joint = config.task == "joint"
    if getattr(config, "sliding_window", False):
        try:
            from data.vrdu_loader import chunk_vrdu_examples
        except ImportError:
            from src.data.vrdu_loader import chunk_vrdu_examples
        w_size = getattr(config, "window_size", 384)
        w_overlap = getattr(config, "window_overlap", 128)
        train_examples = chunk_vrdu_examples(train_examples, window_size=w_size, overlap=w_overlap)
        val_examples = chunk_vrdu_examples(val_examples, window_size=w_size, overlap=w_overlap)
        test_examples = chunk_vrdu_examples(test_examples, window_size=w_size, overlap=w_overlap)

    anc_mode = getattr(config, "anchor_mode", "content_box" if getattr(config, "relative_coords", False) else "none")
    is_dann = getattr(config, "adversarial_dann", False)
    template_to_idx = None
    if is_dann:
        unique_templates = sorted(list({getattr(ex, "template_id", "unk") for ex in train_examples}))
        template_to_idx = {t: idx for idx, t in enumerate(unique_templates)}

    train_collate = make_collator(
        model,
        is_joint=is_joint,
        is_train=True,
        coordinate_jitter=getattr(config, "coordinate_jitter", False),
        relative_coords=getattr(config, "relative_coords", False),
        anchor_mode=anc_mode,
        is_dann=is_dann,
        template_to_idx=template_to_idx,
    )
    eval_collate = make_collator(
        model,
        is_joint=is_joint,
        is_train=False,
        relative_coords=getattr(config, "relative_coords", False),
        anchor_mode=anc_mode,
        is_dann=is_dann,
        template_to_idx=template_to_idx,
    )
    train_loader = DataLoader(FormDataset(train_examples), batch_size=config.batch_size, shuffle=True, collate_fn=train_collate)
    val_loader = DataLoader(FormDataset(val_examples), batch_size=config.batch_size, collate_fn=eval_collate)
    test_loader = DataLoader(FormDataset(test_examples), batch_size=config.batch_size, collate_fn=eval_collate)

    model_to_optimize = model if is_joint else model.model
    model_to_optimize.to(device)

    best_val_score, best_state, history = -1.0, None, []
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    patience = getattr(config, "patience_tolerance", 3)
    min_delta = getattr(config, "min_delta", 0.001)

    if is_joint and getattr(config, "curriculum", False):
        # =========================================================================
        # DECOUPLED 3-STAGE CURRICULUM WITH PER-PHASE VALIDATION & EARLY STOPPING
        # =========================================================================

        # -------------------------------------------------------------------------
        # Phase 1: Pure Token Classification Warmup (lambda_linking = 0.0)
        # Validation subset: ONLY Entity Tagging (predict_phase1_tagging)
        # -------------------------------------------------------------------------
        max_p1 = getattr(config, "max_phase1_epochs", getattr(config, "phase1_epochs", 10))
        print(f"\n{pfx}{'='*70}", flush=True)
        print(f"{pfx}[*] [Phase 1/3] Token Classification Warmup (max {max_p1} eps, patience={patience})", flush=True)
        print(f"{pfx}[*] Validating strictly on Entity F1 (no relation linking)", flush=True)
        print(f"{pfx}{'='*70}", flush=True)

        model.linking_loss_weight = 0.0
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.relation_head.parameters():
            p.requires_grad = False

        opt_p1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        sched_p1 = torch.optim.lr_scheduler.OneCycleLR(
            opt_p1, max_lr=config.lr, total_steps=max(1, len(train_loader) * max_p1), pct_start=config.warmup_ratio, anneal_strategy="linear"
        )

        best_p1_ent_f1, best_p1_state, p1_patience = -1.0, None, 0
        for ep in range(max_p1):
            model.train()
            ep_loss = 0.0
            pbar = tqdm(train_loader, desc=f"{pfx}Phase 1 Epoch {ep+1:>2}/{max_p1}".strip(), leave=leave_bar, unit="b", **TQDM_KWARGS)
            for batch_data in pbar:
                encoded, _, spans, gold_links, entity_types = batch_data
                inputs = {k: v.to(device) for k, v in encoded.inputs.items()}
                out = model(inputs=inputs, entity_spans=spans, gold_links=gold_links, entity_types=entity_types)
                loss = out.tagging_loss if out.tagging_loss is not None else out.total_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_p1.step()
                sched_p1.step()
                opt_p1.zero_grad()
                ep_loss += loss.item()
                pbar.set_postfix(tag_loss=f"{loss.item():.3f}")
            pbar.close()

            # Validation: ONLY evaluate Entity Tagging
            val_pred, val_ref = predict_phase1_tagging(model, val_loader, device)
            val_ent_f1 = entity_f1(val_pred, val_ref, model.id2label).f1
            avg_loss = ep_loss / max(1, len(train_loader))
            print(f"  {pfx}[Phase 1 Ep {ep+1}] Train Loss: {avg_loss:.4f} | Val Entity F1: {val_ent_f1:.4f} (best: {max(best_p1_ent_f1, val_ent_f1):.4f})", flush=True)

            history.append({
                "phase": 1,
                "epoch": ep + 1,
                "train_loss": round(avg_loss, 4),
                "val_entity_f1": round(val_ent_f1, 4),
            })

            if val_ent_f1 > best_p1_ent_f1 + min_delta:
                best_p1_ent_f1 = val_ent_f1
                best_p1_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                p1_patience = 0
            else:
                p1_patience += 1
                if p1_patience >= patience:
                    print(f"  {pfx}[Phase 1 Early Stop] Plateau reached ({patience} eps without gain). Transitioning to Phase 2 with best Val Entity F1={best_p1_ent_f1:.4f}\n", flush=True)
                    break

        # Restore peak Phase 1 weights before Phase 2
        if best_p1_state is not None:
            model.load_state_dict(best_p1_state)

        # -------------------------------------------------------------------------
        # Phase 2: Teacher-Forced Relation Head Warmup (Frozen LiLT Backbone)
        # Validation subset: ONLY Key-Value Linking (predict_phase2_linking)
        # -------------------------------------------------------------------------
        max_p2 = getattr(config, "max_phase2_epochs", getattr(config, "phase2_epochs", 8))
        print(f"\n{pfx}{'='*70}", flush=True)
        print(f"{pfx}[*] [Phase 2/3] Spatial Relation Head Warmup (max {max_p2} eps, patience={patience})", flush=True)
        print(f"{pfx}[*] Backbone frozen. Validating strictly on Linking F1 under Teacher Forcing", flush=True)
        print(f"{pfx}{'='*70}", flush=True)

        model.linking_loss_weight = config.linking_loss_weight
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.relation_head.parameters():
            p.requires_grad = True

        opt_p2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        sched_p2 = torch.optim.lr_scheduler.OneCycleLR(
            opt_p2, max_lr=config.lr, total_steps=max(1, len(train_loader) * max_p2), pct_start=config.warmup_ratio, anneal_strategy="linear"
        )

        best_p2_link_f1, best_p2_state, p2_patience = -1.0, None, 0
        for ep in range(max_p2):
            model.train()
            ep_loss = 0.0
            pbar = tqdm(train_loader, desc=f"{pfx}Phase 2 Epoch {ep+1:>2}/{max_p2}".strip(), leave=leave_bar, unit="b", **TQDM_KWARGS)
            for batch_data in pbar:
                encoded, _, spans, gold_links, entity_types = batch_data
                inputs = {k: v.to(device) for k, v in encoded.inputs.items()}
                out = model(inputs=inputs, entity_spans=spans, gold_links=gold_links, entity_types=entity_types)
                loss = out.linking_loss if out.linking_loss is not None else out.total_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.relation_head.parameters(), 1.0)
                opt_p2.step()
                sched_p2.step()
                opt_p2.zero_grad()
                ep_loss += loss.item()
                pbar.set_postfix(link_loss=f"{loss.item():.3f}")
            pbar.close()

            # Validation: ONLY evaluate Key-Value Linking
            val_pred_links, val_gold_links = predict_phase2_linking(model, val_loader, device)
            val_link_scores = linking_f1(val_pred_links, val_gold_links)
            val_link_f1 = val_link_scores["f1"]
            avg_loss = ep_loss / max(1, len(train_loader))
            print(f"  {pfx}[Phase 2 Ep {ep+1}] Train Loss: {avg_loss:.4f} | Val Linking F1: {val_link_f1:.4f} (best: {max(best_p2_link_f1, val_link_f1):.4f})", flush=True)

            history.append({
                "phase": 2,
                "epoch": ep + 1,
                "train_loss": round(avg_loss, 4),
                "val_linking_f1": round(val_link_f1, 4),
            })

            if val_link_f1 > best_p2_link_f1 + min_delta:
                best_p2_link_f1 = val_link_f1
                best_p2_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                p2_patience = 0
            else:
                p2_patience += 1
                if p2_patience >= patience:
                    print(f"  {pfx}[Phase 2 Early Stop] Plateau reached ({patience} eps without gain). Transitioning to Phase 3 with best Val Linking F1={best_p2_link_f1:.4f}\n", flush=True)
                    break

        # Restore peak Phase 2 weights before Phase 3
        if best_p2_state is not None:
            model.load_state_dict(best_p2_state)

        # -------------------------------------------------------------------------
        # Phase 3: Joint End-to-End Fine-Tuning
        # Validation: FULL Joint Evaluation (Entity F1 + Linking F1)
        # -------------------------------------------------------------------------
        max_p3 = getattr(config, "max_phase3_epochs", getattr(config, "phase3_epochs", 10))
        print(f"\n{pfx}{'='*70}", flush=True)
        print(f"{pfx}[*] [Phase 3/3] Joint End-to-End Fine-Tuning (max {max_p3} eps, patience={patience})", flush=True)
        print(f"{pfx}[*] All weights unfrozen. Validating on Joint Score = 0.5 * (Entity F1 + Linking F1)", flush=True)
        print(f"{pfx}{'='*70}", flush=True)

        model.linking_loss_weight = config.linking_loss_weight
        for p in model.parameters():
            p.requires_grad = True

        opt_p3 = torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": config.lr * 0.2},
                {"params": model.relation_head.parameters(), "lr": config.lr},
            ],
            weight_decay=config.weight_decay,
        )
        sched_p3 = torch.optim.lr_scheduler.OneCycleLR(
            opt_p3,
            max_lr=[config.lr * 0.2, config.lr],
            total_steps=max(1, len(train_loader) * max_p3),
            pct_start=config.warmup_ratio,
            anneal_strategy="linear",
        )

        best_val_score, best_state, p3_patience = -1.0, None, 0
        for ep in range(max_p3):
            model.train()
            ep_loss = 0.0
            pbar = tqdm(train_loader, desc=f"{pfx}Phase 3 Epoch {ep+1:>2}/{max_p3}".strip(), leave=leave_bar, unit="b", **TQDM_KWARGS)
            for batch_data in pbar:
                encoded, _, spans, gold_links, entity_types = batch_data
                inputs = {k: v.to(device) for k, v in encoded.inputs.items()}
                out = model(inputs=inputs, entity_spans=spans, gold_links=gold_links, entity_types=entity_types)
                loss = out.total_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_p3.step()
                sched_p3.step()
                opt_p3.zero_grad()
                ep_loss += loss.item()
                pbar.set_postfix(total_loss=f"{loss.item():.3f}")
            pbar.close()

            # Validation: FULL Joint Evaluation
            val_pred, val_ref, val_pred_links, val_gold_links = predict_joint(model, val_loader, device)
            val_ent_f1 = entity_f1(val_pred, val_ref, model.id2label).f1
            val_link_scores = linking_f1(val_pred_links, val_gold_links)
            val_link_f1 = val_link_scores["f1"]
            val_score = 0.5 * (val_ent_f1 + val_link_f1)
            avg_loss = ep_loss / max(1, len(train_loader))
            print(f"  {pfx}[Phase 3 Ep {ep+1}] Train Loss: {avg_loss:.4f} | Val Ent F1: {val_ent_f1:.4f} | Val Link F1: {val_link_f1:.4f} | Joint Val: {val_score:.4f} (best: {max(best_val_score, val_score):.4f})", flush=True)

            history.append({
                "phase": 3,
                "epoch": ep + 1,
                "train_loss": round(avg_loss, 4),
                "val_entity_f1": round(val_ent_f1, 4),
                "val_linking_f1": round(val_link_f1, 4),
                "val_score": round(val_score, 4),
            })

            if val_score > best_val_score + min_delta:
                best_val_score = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if checkpoint_dir is not None:
                    if getattr(config, "save_weights", True):
                        try:
                            torch.save(best_state, checkpoint_dir / "best_model.pt")
                        except Exception as e:
                            print(f"  {pfx}[Warning] Failed to save best_model.pt ({e}); continuing in-memory.", flush=True)
                    meta = {
                        "phase": 3,
                        "epoch": ep + 1,
                        "best_val_score": round(best_val_score, 4),
                        "val_entity_f1": round(val_ent_f1, 4),
                        "val_linking_f1": round(val_link_f1, 4),
                        "train_loss": round(avg_loss, 4),
                    }
                    (checkpoint_dir / "best_meta.json").write_text(
                        json.dumps(meta, indent=2, default=json_serializable), encoding="utf-8"
                    )
                p3_patience = 0
            else:
                p3_patience += 1
                if p3_patience >= patience:
                    print(f"  {pfx}[Phase 3 Early Stop] Plateau reached ({patience} eps without gain). Concluding training with best Joint Val={best_val_score:.4f}\n", flush=True)
                    break

    else:
        # =========================================================================
        # STANDARD SINGLE-STAGE TRAINING WITH PATIENCE EARLY STOPPING
        # =========================================================================
        trainable_params = [p for p in model_to_optimize.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
        total_steps = max(1, len(train_loader) * config.epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr, total_steps=total_steps, pct_start=config.warmup_ratio, anneal_strategy="linear"
        )
        patience_cnt = 0
        for epoch in range(config.epochs):
            model_to_optimize.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"{pfx}Epoch {epoch+1:>2}/{config.epochs}".strip(), leave=leave_bar, unit="b", **TQDM_KWARGS)
            for batch_data in pbar:
                if is_joint:
                    encoded, _, spans, gold_links, entity_types = batch_data
                    inputs = {k: v.to(device) for k, v in encoded.inputs.items()}
                    output = model(inputs=inputs, entity_spans=spans, gold_links=gold_links, entity_types=entity_types)
                    loss = output.total_loss
                elif is_dann:
                    encoded, _, template_targets = batch_data
                    inputs = {k: v.to(device) for k, v in encoded.inputs.items()}
                    try:
                        from models.dann import compute_dann_alpha
                    except ImportError:
                        from src.models.dann import compute_dann_alpha
                    step = epoch * len(train_loader) + (pbar.n if hasattr(pbar, "n") else 0)
                    cur_alpha = compute_dann_alpha(step, total_steps, gamma=getattr(config, "dann_gamma", 10.0))
                    out = model_to_optimize(inputs=inputs, template_targets=template_targets.to(device), alpha=cur_alpha)
                    loss = out.total_loss
                else:
                    encoded, _ = batch_data
                    inputs = {k: v.to(device) for k, v in encoded.inputs.items()}
                    loss = model.model(**inputs).loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.3f}")

            avg_loss = epoch_loss / max(1, len(train_loader))
            if is_joint:
                val_pred, val_ref, val_pred_links, val_gold_links = predict_joint(model, val_loader, device)
                val_ent_f1 = entity_f1(val_pred, val_ref, model.id2label).f1
                val_link_scores = linking_f1(val_pred_links, val_gold_links)
                val_link_f1 = val_link_scores["f1"]
                val_score = 0.5 * (val_ent_f1 + val_link_f1)
                history.append({
                    "epoch": epoch + 1,
                    "train_loss": round(avg_loss, 4),
                    "val_entity_f1": round(val_ent_f1, 4),
                    "val_linking_f1": round(val_link_f1, 4),
                    "val_score": round(val_score, 4),
                })
            else:
                val_pred, val_ref = predict(model, val_loader, device)
                val_score = entity_f1(val_pred, val_ref, model.model.config.id2label).f1
                history.append({"epoch": epoch + 1, "train_loss": round(avg_loss, 4), "val_f1": round(val_score, 4)})

            pbar.close()

            if val_score > best_val_score + min_delta:
                best_val_score = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model_to_optimize.state_dict().items()}
                if checkpoint_dir is not None:
                    if getattr(config, "save_weights", True):
                        try:
                            torch.save(best_state, checkpoint_dir / "best_model.pt")
                        except Exception as e:
                            print(f"  {pfx}[Warning] Failed to save best_model.pt ({e}); continuing in-memory.", flush=True)
                    meta = {
                        "epoch": epoch + 1,
                        "best_val_score": round(best_val_score, 4),
                        "train_loss": round(avg_loss, 4),
                    }
                    (checkpoint_dir / "best_meta.json").write_text(
                        json.dumps(meta, indent=2, default=json_serializable), encoding="utf-8"
                    )
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"  {pfx}[Early Stop] Early stopping triggered after {patience} epochs without improvement.", flush=True)
                    break

    if best_state is not None:
        model_to_optimize.load_state_dict(best_state)

    if is_joint:
        test_pred, test_ref, test_pred_links, test_gold_links = predict_joint(
            model, test_loader, device, desc=f"{pfx}Test eval".strip()
        )
        id2label = model.id2label
        scores = entity_f1(test_pred, test_ref, id2label)
        link_scores = linking_f1(test_pred_links, test_gold_links)
        res = {
            "test": scores.as_dict(),
            "test_linking": link_scores,
            "best_val_score": round(best_val_score, 4),
            "history": history,
            "malformed_tags": malformed_tag_rate(test_pred, id2label),
            "cross_check": cross_check(test_pred, test_ref, id2label),
        }
    else:
        test_pred, test_ref = predict(model, test_loader, device, desc=f"{pfx}Test eval".strip())
        id2label = model.model.config.id2label
        scores = entity_f1(test_pred, test_ref, id2label)
        res = {
            "test": scores.as_dict(),
            "best_val_f1": round(best_val_score, 4),
            "history": history,
            "malformed_tags": malformed_tag_rate(test_pred, id2label),
            "cross_check": cross_check(test_pred, test_ref, id2label),
        }

    if checkpoint_dir is not None:
        (checkpoint_dir / "result.json").write_text(
            json.dumps(res, indent=2, default=json_serializable), encoding="utf-8"
        )
        (checkpoint_dir / "history.json").write_text(
            json.dumps(history, indent=2, default=json_serializable), encoding="utf-8"
        )

    return res


def load_dataset(config: TrainConfig):
    if config.dataset == "funsd":
        from data.funsd_loader import ID2LABEL, LABEL_LIST, load_funsd_split, summarize

        train = load_funsd_split(config.data_root, "train")
        test = load_funsd_split(config.data_root, "test")
        print("FUNSD train:", json.dumps(summarize(train)))
        print("FUNSD test: ", json.dumps(summarize(test)))
        # FUNSD ships no dev set; carve one out of train so checkpoint selection
        # never touches test.
        rng = random.Random(13)
        shuffled = list(train)
        rng.shuffle(shuffled)
        n_val = max(1, int(0.15 * len(shuffled)))
        return shuffled[n_val:], shuffled[:n_val], test, LABEL_LIST, ID2LABEL

    if config.dataset == "vrdu":
        from data.vrdu_loader import (
            assert_no_template_leakage as vrdu_leak_check,
            load_splits,
            load_vrdu,
            summarize as vrdu_summarize,
            verify_template_recovery,
        )

        root = Path(config.data_root)
        examples, label_list, id2label = load_vrdu(root, config.subcorpus)
        print("VRDU:", json.dumps(vrdu_summarize(examples)))

        splits_dir = root / config.subcorpus / "few_shot-splits"
        print(verify_template_recovery(splits_dir))

        # For the "standard" protocol path, use the first official UTL split
        # as a convenience default.  The "matched" protocol path in main()
        # handles the full MTL-vs-UTL comparison via splits.matched_size_protocol.
        n = config.n_train or 200
        official = load_splits(
            examples, splits_dir, regime="UTL", n_train=n, seed=0, protocol="strict",
        )
        if not official:
            raise FileNotFoundError(
                f"No UTL split files found in {splits_dir} with n_train={n}, seed=0. "
                f"Check that the VRDU few_shot-splits directory is populated."
            )
        split = official[0]
        print(split.describe())
        print(vrdu_leak_check(split))
        return split.train, split.val, split.test, label_list, id2label

    if config.dataset == "xfund":
        from data.xfund_loader import ID2LABEL, LABEL_LIST, load_xfund_dataset, summarize

        langs = list(getattr(config, "languages", ["es", "fr", "de", "it", "pt"]))
        train, val, test = load_xfund_dataset(config.data_root, languages=langs)
        print("XFUND train:", json.dumps(summarize(train)))
        print("XFUND val:  ", json.dumps(summarize(val)))
        print("XFUND test: ", json.dumps(summarize(test)))
        return train, val, test, LABEL_LIST, ID2LABEL

    raise NotImplementedError(f"Unknown dataset {config.dataset!r}. Supported: 'funsd', 'vrdu', 'xfund'.")


def build_model(config: TrainConfig, num_labels: int, id2label: dict[int, str], num_templates: int | None = None):
    if config.task == "joint":
        return JointFormExtractor(
            backbone=config.backbone,
            num_labels=num_labels,
            id2label=id2label,
            max_length=config.max_length,
            linking_loss_weight=config.linking_loss_weight,
            pos_weight=config.pos_weight,
            use_spatial_relations=config.use_spatial_relations,
            spatial_in_features=getattr(config, "spatial_in_features", 16),
            use_geometry=getattr(config, "use_geometry", True),
            use_containment=getattr(config, "use_containment", True),
            use_line_of_sight=getattr(config, "use_line_of_sight", True),
            use_typography=getattr(config, "use_typography", True),
            loss_type=getattr(config, "loss_type", "focal"),
            focal_gamma=getattr(config, "focal_gamma", 2.0),
            focal_alpha=getattr(config, "focal_alpha", 0.75),
            masking_mode=getattr(config, "masking_mode", "soft"),
            k_neighbors=getattr(config, "k_neighbors", 0),
            fallback_top_k=getattr(config, "fallback_top_k", 0),
            decode_one_to_one=getattr(config, "decode_one_to_one", False),
            link_threshold=getattr(config, "link_threshold", 0.5),
        )
    base = TokenClassifier(config.backbone, num_labels, id2label, config.max_length)
    if getattr(config, "use_graph_bros", False):
        try:
            from models.graph_bros import GraphBrosForTokenClassification
        except ImportError:
            from src.models.graph_bros import GraphBrosForTokenClassification
        base = GraphBrosForTokenClassification(
            base_classifier=base,
            lora_r=getattr(config, "lora_r", 8),
            lora_alpha=getattr(config, "lora_alpha", 16.0),
            lora_dropout=getattr(config, "lora_dropout", 0.05),
            beta=getattr(config, "beta_skeleton", 1.0),
            eps=getattr(config, "graph_eps", 1e-9),
            merge_spans=getattr(config, "merge_spans", True),
            loss_type=getattr(config, "loss_type", "focal"),
            focal_gamma=getattr(config, "focal_gamma", 2.0),
            gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        )
    if getattr(config, "adversarial_dann", False):
        try:
            from models.dann import DomainAdversarialModel
        except ImportError:
            from src.models.dann import DomainAdversarialModel
        n_templates = num_templates if num_templates is not None else getattr(config, "num_templates", 32)
        return DomainAdversarialModel(
            base_classifier=base,
            num_templates=n_templates,
            hidden_dim=768,
            discriminator_hidden_dim=getattr(config, "dann_hidden_dim", 256),
            domain_loss_weight=getattr(config, "dann_weight", 0.1),
            gamma=getattr(config, "dann_gamma", 10.0),
        )
    return base


def save_summary(runs: list[dict], config: TrainConfig, output_dir: Path) -> dict:
    """Save metrics.json summary across all completed seeds."""
    if not runs:
        return {}
    f1s = [r["test"]["f1"] for r in runs]
    summary = {
        "config": vars(config) | {"seeds": list(config.seeds)},
        "runs": runs,
        "mean_f1": round(sum(f1s) / len(f1s), 4),
        "std_f1": round((sum((x - sum(f1s) / len(f1s)) ** 2 for x in f1s) / max(1, len(f1s) - 1)) ** 0.5, 4) if len(f1s) > 1 else 0.0,
    }
    if config.task == "joint":
        link_f1s = [r["test_linking"]["f1"] for r in runs]
        summary["mean_linking_f1"] = round(sum(link_f1s) / len(link_f1s), 4)
        summary["std_linking_f1"] = round((sum((x - sum(link_f1s) / len(link_f1s)) ** 2 for x in link_f1s) / max(1, len(link_f1s) - 1)) ** 0.5, 4) if len(link_f1s) > 1 else 0.0
    out = output_dir / "metrics.json"
    out.write_text(json.dumps(summary, indent=2, default=json_serializable), encoding="utf-8")
    return summary


def train_single_seed(
    seed: int,
    config: TrainConfig,
    train_ex: list,
    val_ex: list,
    test_ex: list,
    label_list: list[str],
    id2label: dict[int, str],
    device: torch.device,
    output_dir: Path,
    log_prefix: str = "",
) -> dict:
    """Train and evaluate one single seed, resuming from checkpoint if available."""
    run_dir = output_dir / f"seed_{seed}"
    result_file = run_dir / "result.json"

    if result_file.exists():
        print(f"\n{log_prefix} === Seed {seed} (resuming from checkpoint: {result_file}) ===", flush=True)
        try:
            cached = json.loads(result_file.read_text(encoding="utf-8"))
            msg = f"{log_prefix}   Loaded existing result: Test F1 {cached['test']['f1']:.4f}"
            if "test_linking" in cached:
                msg += f" | Linking F1 {cached['test_linking']['f1']:.4f}"
            print(msg, flush=True)
            return cached
        except Exception as e:
            print(f"{log_prefix}   Warning: Failed reading {result_file} ({e}), re-running Seed {seed}...", flush=True)

    print(f"\n{log_prefix} === Seed {seed} ===", flush=True)
    set_seed(seed)
    model = build_model(config, len(label_list), id2label)
    result = train_one(model, train_ex, val_ex, test_ex, config, device, checkpoint_dir=run_dir, log_prefix=log_prefix)
    result["seed"] = seed
    if config.task == "joint":
        print(
            f"{log_prefix} Seed {seed} -> Test Entity F1: {result['test']['f1']:.4f} | "
            f"Test Linking F1: {result['test_linking']['f1']:.4f} | {result['cross_check']}",
            flush=True,
        )
    else:
        print(f"{log_prefix} Seed {seed} -> Test F1: {result['test']['f1']:.4f}  |  {result['cross_check']}", flush=True)
    return result


def _mp_seed_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    config_input: TrainConfig | str,
    output_dir_str: str,
    n_train_override: int | None = None,
) -> None:
    """Worker process targeting a specific GPU (e.g. cuda:0 or cuda:1)."""
    try:
        for _p in (str(ROOT_DIR), str(SRC_DIR)):
            if _p not in sys.path:
                sys.path.insert(0, _p)

        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        if isinstance(config_input, str):
            config = TrainConfig.from_yaml(config_input)
        else:
            config = config_input
        if n_train_override is not None:
            config.n_train = n_train_override
        output_dir = Path(output_dir_str)

        train_ex, val_ex, test_ex, label_list, id2label = load_dataset(config)
        if config.n_train:
            train_ex = train_ex[: config.n_train]

        while True:
            seed = task_queue.get()
            if seed is None:
                # Poison pill received: worker terminates cleanly
                break

            exp_name = Path(config.output_dir).name if config.output_dir else 'run'
            log_prefix = f"[{exp_name} | GPU {gpu_id} | Seed {seed}]"
            try:
                res = train_single_seed(
                    seed=seed,
                    config=config,
                    train_ex=train_ex,
                    val_ex=val_ex,
                    test_ex=test_ex,
                    label_list=label_list,
                    id2label=id2label,
                    device=device,
                    output_dir=output_dir,
                    log_prefix=log_prefix,
                )
                result_queue.put((seed, res, None))
            except Exception as e:
                import traceback
                err = traceback.format_exc()
                print(f"{log_prefix} Training failed:\n{err}", flush=True)
                result_queue.put((seed, None, err))
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"[GPU {gpu_id} Worker Initialization Error]\n{err}", flush=True)
        result_queue.put((-1, None, err))


def _mp_matched_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    config: TrainConfig,
    protocol: dict,
    label_list: list,
    id2label: dict,
    output_dir_str: str,
) -> None:
    """Worker process for matched protocol targeting a specific GPU."""
    try:
        for _p in (str(ROOT_DIR), str(SRC_DIR)):
            if _p not in sys.path:
                sys.path.insert(0, _p)

        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        output_dir = Path(output_dir_str)

        while True:
            task = task_queue.get()
            if task is None:
                # Poison pill received: worker terminates cleanly
                break

            regime, fold_idx, seed = task
            tag = f"{regime}/fold{fold_idx}/seed{seed}"
            exp_name = Path(config.output_dir).name if config.output_dir else 'run'
            log_prefix = f"[{exp_name} | GPU {gpu_id} | {tag}]"
            run_dir = output_dir / f"{regime}_fold{fold_idx}_seed{seed}"
            result_file = run_dir / "result.json"

            if result_file.exists():
                try:
                    cached = json.loads(result_file.read_text(encoding="utf-8"))
                    print(f"{log_prefix} Resuming from checkpoint: {result_file} (Test F1 {cached['test']['f1']:.4f})", flush=True)
                    result_queue.put((regime, fold_idx, seed, cached, None))
                    continue
                except Exception as e:
                    print(f"{log_prefix} Warning: Failed reading {result_file} ({e}), re-running...", flush=True)

            try:
                split = protocol[regime][fold_idx]
                print(f"\n{log_prefix} Starting training ({len(split.train)} train, {len(split.val)} val, {len(split.test)} test)...", flush=True)
                set_seed(seed)

                model = build_model(config, len(label_list), id2label)
                result = train_one(
                    model=model,
                    train_examples=split.train,
                    val_examples=split.val,
                    test_examples=split.test,
                    config=config,
                    device=device,
                    checkpoint_dir=run_dir,
                    log_prefix=log_prefix,
                )
                result["seed"] = seed
                result["fold"] = fold_idx
                result["regime"] = regime
                result["held_out"] = list(split.held_out_templates)

                if config.task == "joint":
                    print(
                        f"{log_prefix} -> Test Entity F1: {result['test']['f1']:.4f} | "
                        f"Linking F1: {result['test_linking']['f1']:.4f} | {result['cross_check']}",
                        flush=True,
                    )
                else:
                    print(f"{log_prefix} -> Test F1: {result['test']['f1']:.4f}  |  {result['cross_check']}", flush=True)

                result_queue.put((regime, fold_idx, seed, result, None))
            except Exception as e:
                import traceback
                err = traceback.format_exc()
                print(f"{log_prefix} Training failed:\n{err}", flush=True)
                result_queue.put((regime, fold_idx, seed, None, err))
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"[GPU {gpu_id} Matched Worker Initialization Error]\n{err}", flush=True)
        result_queue.put(("ERROR", -1, -1, None, err))


def get_device_list(gpu_arg: str, no_parallel: bool) -> list[int]:
    """Parse device indices for parallel or single execution."""
    if no_parallel or not torch.cuda.is_available():
        return []
    count = torch.cuda.device_count()
    if count == 0:
        return []
    if gpu_arg == "auto":
        return list(range(count))
    try:
        gpus = [int(x.strip()) for x in gpu_arg.split(",") if x.strip()]
        return [g for g in gpus if 0 <= g < count]
    except Exception as e:
        print(f"Warning: Failed to parse --gpus '{gpu_arg}' ({e}), defaulting to [0] if available")
        return [0] if count > 0 else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", choices=["standard", "matched"], default="standard",
                        help="standard = the dataset's own split; matched = MTL vs UTL at equal train size")
    parser.add_argument("--gpus", default="auto",
                        help="Comma-separated GPU indices (e.g. '0,1') or 'auto' to use all available CUDA devices")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Force single-process execution even if multiple GPUs are available")
    parser.add_argument("--seed", "--seeds", type=str, default=None,
                        help="Run specific seed(s), e.g. '42' or '13,42' (enables multi-GPU seed parallelism)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    parser.add_argument("--no-line-of-sight", action="store_true",
                        help="Ablation: disable Line-of-Sight visibility ray-casting")
    parser.add_argument("--no-typography", action="store_true",
                        help="Ablation: disable typographical scale and prominence features")
    parser.add_argument("--no-containment", action="store_true",
                        help="Ablation: disable axial containment overlap-over-minimum features")
    parser.add_argument("--geometry-only", action="store_true",
                        help="Ablation: use only continuous geometry features (5 dims)")
    parser.add_argument("--loss-type", choices=["focal", "bce"], default=None,
                        help="Ablation: override loss function ('focal' or 'bce')")
    parser.add_argument("--no-one-to-one", action="store_true",
                        help="Ablation: disable one-to-one argmax matching during decoding")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--sliding-window", action="store_true",
                        help="Track 1: enable multi-page sliding window chunking with soft voting")
    parser.add_argument("--coordinate-jitter", action="store_true",
                        help="Track 1: enable aggressive random translation, scaling and jittering during training")
    parser.add_argument("--relative-coords", action="store_true",
                        help="Track 1: inject centroid-relative coordinates instead of absolute [0, 1000]")
    parser.add_argument("--window-size", type=int, default=None,
                        help="Sliding window size (default: 384)")
    parser.add_argument("--window-overlap", type=int, default=None,
                        help="Sliding window overlap (default: 128)")
    parser.add_argument("--anchor-mode", choices=["none", "centroid", "content_box", "cluster"], default=None, help="Track 1 ablation: anchoring strategy")
    parser.add_argument("--save-weights", action="store_true", default=False,
                        help="Force saving best_model.pt weights (~500MB per fold)")
    parser.add_argument("--no-save-weights", action="store_true", default=False,
                        help="Disable saving best_model.pt weights to disk to conserve disk space on Kaggle")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Fast-track option: limit number of folds per regime in matched protocol (e.g. 2)")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)

    # Apply command-line overrides and ablation flags
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_line_of_sight:
        config.use_line_of_sight = False
    if args.no_typography:
        config.use_typography = False
    if args.no_containment:
        config.use_containment = False
    if args.geometry_only:
        config.use_line_of_sight = False
        config.use_typography = False
        config.use_containment = False
    if args.loss_type:
        config.loss_type = args.loss_type
    if args.no_one_to_one:
        config.decode_one_to_one = False
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.sliding_window:
        config.sliding_window = True
    if args.coordinate_jitter:
        config.coordinate_jitter = True
    if args.relative_coords:
        config.relative_coords = True
    if args.window_size is not None:
        config.window_size = args.window_size
    if args.window_overlap is not None:
        config.window_overlap = args.window_overlap
    if args.anchor_mode is not None:
        config.anchor_mode = args.anchor_mode
        config.relative_coords = (args.anchor_mode != "none")
    if args.no_save_weights:
        config.save_weights = False
    elif args.save_weights:
        config.save_weights = True
    if args.max_folds is not None:
        config.max_folds = args.max_folds

    # Dynamically compute spatial_in_features to match active ablation groups
    if config.task == "joint" and config.use_spatial_relations:
        dim = 0
        if config.use_geometry: dim += 5
        if config.use_containment: dim += 5
        if config.use_line_of_sight: dim += 2
        if config.use_typography: dim += 4
        config.spatial_in_features = dim if dim > 0 else 5

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = get_device_list(args.gpus, args.no_parallel)
    if not torch.cuda.is_available():
        print("WARNING: no GPU visible. Fine-tuning on CPU will take hours; this is a Kaggle/Colab job.")
    else:
        print(f"[*] Detected {len(gpu_ids)} GPU(s): {gpu_ids} (total visible: {torch.cuda.device_count()})")
        for gid in gpu_ids:
            prop = torch.cuda.get_device_properties(gid)
            print(f"    GPU {gid}: {prop.name} ({prop.total_memory / 1e9:.1f} GB VRAM)")

    if args.seed is not None:
        seeds = [int(s.strip()) for s in str(args.seed).split(",") if s.strip()]
    else:
        seeds = list(config.seeds)

    exp_name = Path(config.output_dir).name if config.output_dir else Path(args.config).stem
    print(f"\n{'='*75}", flush=True)
    print(f"[*] ACTIVE EXPERIMENT / ABLATION: {exp_name}", flush=True)
    print(f"[*] Config: {args.config} | Protocol: {args.protocol} | Output: {config.output_dir}", flush=True)
    print(f"[*] Backbone: {config.backbone} | Graph-BROS: {getattr(config, 'use_graph_bros', False)} | DANN: {getattr(config, 'adversarial_dann', False)} | Sliding Window: {getattr(config, 'sliding_window', False)} | Jitter: {getattr(config, 'coordinate_jitter', False)}", flush=True)
    print(f"[*] Task: {config.task} | Seeds: {seeds} | Max Folds: {getattr(config, 'max_folds', 'all')}", flush=True)
    print(f"{'='*75}\n", flush=True)

    try:
        Path("current_run.txt").write_text(
            f"Active Experiment: {exp_name}\n"
            f"Config: {args.config}\n"
            f"Output Dir: {config.output_dir}\n"
            f"Protocol: {args.protocol}\n"
            f"Backbone: {config.backbone}\n"
            f"Graph-BROS: {getattr(config, 'use_graph_bros', False)}\n"
            f"DANN: {getattr(config, 'adversarial_dann', False)}\n"
            f"Sliding Window: {getattr(config, 'sliding_window', False)}\n"
            f"Coordinate Jitter: {getattr(config, 'coordinate_jitter', False)}\n"
            f"Max Folds: {getattr(config, 'max_folds', 'all')}\n"
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            encoding="utf-8",
        )
    except Exception:
        pass
    if config.task == "joint":
        print(f"[*] Spatial Features: {config.spatial_in_features}D (Geo: {config.use_geometry}, Contain: {config.use_containment}, LOS: {config.use_line_of_sight}, Typo: {config.use_typography}) | Loss: {config.loss_type} | 1-to-1: {config.decode_one_to_one}")

    if args.protocol == "standard":
        can_parallel = torch.cuda.is_available() and len(gpu_ids) >= 2 and len(seeds) > 1 and not args.no_parallel

        if can_parallel:
            num_workers = min(len(gpu_ids), len(seeds))
            active_gpus = gpu_ids[:num_workers]
            print(f"\n{'='*75}")
            print(f"[*] MULTI-GPU SEED PARALLELISM ACTIVE")
            print(f"[*] Spawning {num_workers} parallel workers on GPUs {active_gpus} for seeds {seeds}")
            print(f"{'='*75}\n", flush=True)

            task_queue = mp.Queue()
            result_queue = mp.Queue()

            # Enqueue all seed tasks
            for s in seeds:
                task_queue.put(s)
            # Enqueue poison pills so workers terminate when queue is exhausted
            for _ in active_gpus:
                task_queue.put(None)

            workers = []
            for gid in active_gpus:
                p = mp.Process(
                    target=_mp_seed_worker,
                    args=(
                        gid,
                        task_queue,
                        result_queue,
                        config,
                        str(output_dir),
                        config.n_train,
                    ),
                )
                p.start()
                workers.append(p)

            runs_dict = {}
            for _ in range(len(seeds)):
                seed, res, err = result_queue.get()
                if err is not None:
                    print(f"\n[FATAL] Worker failed for seed {seed}:\n{err}", flush=True)
                    for p in workers:
                        if p.is_alive():
                            p.terminate()
                    raise RuntimeError(f"Multi-GPU worker failed: {err}")

                runs_dict[seed] = res
                current_runs = [runs_dict[s] for s in seeds if s in runs_dict]
                save_summary(current_runs, config, output_dir)

            for p in workers:
                p.join()

            runs = [runs_dict[s] for s in seeds]

        else:
            device = torch.device(f"cuda:{gpu_ids[0]}" if (gpu_ids and torch.cuda.is_available()) else "cpu")
            print(f"[*] Running sequentially on device: {device}", flush=True)
            train_ex, val_ex, test_ex, label_list, id2label = load_dataset(config)
            if config.n_train:
                train_ex = train_ex[: config.n_train]

            runs = []
            for seed in seeds:
                res = train_single_seed(
                    seed=seed,
                    config=config,
                    train_ex=train_ex,
                    val_ex=val_ex,
                    test_ex=test_ex,
                    label_list=label_list,
                    id2label=id2label,
                    device=device,
                    output_dir=output_dir,
                    log_prefix=f"[{exp_name} | Seed {seed}]",
                )
                runs.append(res)
                save_summary(runs, config, output_dir)

        summary = save_summary(runs, config, output_dir)
        out = output_dir / "metrics.json"
        if config.task == "joint":
            print(f"\nmean test Entity F1 {summary['mean_f1']:.4f} +/- {summary['std_f1']:.4f} | Linking F1 {summary['mean_linking_f1']:.4f} +/- {summary['std_linking_f1']:.4f}  ->  {out}")
        else:
            print(f"\nmean test F1 {summary['mean_f1']:.4f} +/- {summary['std_f1']:.4f}  ->  {out}")

    else:
        # --protocol matched: the headline experiment.
        # Train under both MTL and UTL at identical training-set size, measure
        # the generalization gap.
        if config.dataset != "vrdu":
            raise ValueError(
                f"--protocol matched requires a template-aware dataset (vrdu), "
                f"got {config.dataset!r}."
            )

        from data.splits import assert_no_template_leakage, matched_size_protocol
        from data.vrdu_loader import load_vrdu, summarize as vrdu_summarize

        root = Path(config.data_root)
        examples, label_list, id2label = load_vrdu(root, config.subcorpus)
        print("VRDU:", json.dumps(vrdu_summarize(examples)))

        n = config.n_train or 200
        if config.subcorpus == "ad-buy-form":
            from data.vrdu_loader import load_deepform_protocol
            splits_dir = root / config.subcorpus / "few_shot-splits"
            protocol = load_deepform_protocol(examples, splits_dir, n_train=n)
        else:
            protocol = matched_size_protocol(examples, n_train=n, seed=seeds[0])

        if getattr(config, "max_folds", None) is not None and config.max_folds > 0:
            for regime in ("MTL", "UTL"):
                protocol[regime] = protocol[regime][: config.max_folds]
            print(f"[*] Fast-Track Protocol active: limiting to {config.max_folds} fold(s) per regime.", flush=True)

        all_results: dict[str, list[dict]] = {"MTL": [], "UTL": []}

        # Print split proofs and descriptions upfront
        for regime in ("MTL", "UTL"):
            folds = protocol[regime]
            for fold_idx, split in enumerate(folds):
                if config.subcorpus != "ad-buy-form":
                    proof = assert_no_template_leakage(split)
                    print(f"\n{proof}", flush=True)
                else:
                    print(f"\n[Official DeepForm Split] {regime} Fold {fold_idx} ({len(split.train)} train, {len(split.val)} val, {len(split.test)} test)", flush=True)

        tasks = []
        for regime in ("MTL", "UTL"):
            for fold_idx in range(len(protocol[regime])):
                for seed in seeds:
                    tasks.append((regime, fold_idx, seed))

        # Check existing checkpoints upfront
        pending_tasks = []
        for t in tasks:
            regime, fold_idx, seed = t
            run_dir = output_dir / f"{regime}_fold{fold_idx}_seed{seed}"
            result_file = run_dir / "result.json"
            if result_file.exists():
                try:
                    cached = json.loads(result_file.read_text(encoding="utf-8"))
                    print(f"[*] Resuming from checkpoint: {regime}/fold{fold_idx}/seed{seed} ({result_file}) -> Test F1 {cached['test']['f1']:.4f}", flush=True)
                    all_results[regime].append(cached)
                    continue
                except Exception as e:
                    print(f"[*] Warning: Failed reading {result_file} ({e}), will re-run {regime}/fold{fold_idx}/seed{seed}", flush=True)
            pending_tasks.append(t)

        def _update_and_save_matched_summary():
            all_results["MTL"].sort(key=lambda r: (r.get("fold", 0), r.get("seed", 0)))
            all_results["UTL"].sort(key=lambda r: (r.get("fold", 0), r.get("seed", 0)))
            mtl_f1s = [r["test"]["f1"] for r in all_results["MTL"]]
            utl_f1s = [r["test"]["f1"] for r in all_results["UTL"]]
            gap = generalization_gap(mtl_f1s, utl_f1s) if (mtl_f1s and utl_f1s) else {}

            summary = {
                "config": vars(config) | {"seeds": list(config.seeds)},
                "protocol": "matched",
                "n_train": n,
                "MTL": all_results["MTL"],
                "UTL": all_results["UTL"],
                "gap": gap,
            }
            out = output_dir / "metrics.json"
            out.write_text(json.dumps(summary, indent=2, default=json_serializable), encoding="utf-8")
            return summary, gap, out

        if not pending_tasks:
            print("\n[*] All tasks already completed from checkpoints!", flush=True)
            summary, gap, out = _update_and_save_matched_summary()
        else:
            can_parallel = torch.cuda.is_available() and len(gpu_ids) >= 2 and not args.no_parallel
            if can_parallel and len(pending_tasks) > 1:
                num_workers = min(len(gpu_ids), len(pending_tasks))
                active_gpus = gpu_ids[:num_workers]
                print(f"\n{'='*75}")
                print(f"[*] MULTI-GPU MATCHED PROTOCOL PARALLELISM ACTIVE")
                print(f"[*] Spawning {num_workers} parallel workers on GPUs {active_gpus} for {len(pending_tasks)} pending tasks")
                print(f"{'='*75}\n", flush=True)

                task_queue = mp.Queue()
                result_queue = mp.Queue()

                for t in pending_tasks:
                    task_queue.put(t)
                for _ in active_gpus:
                    task_queue.put(None)

                workers = []
                for gid in active_gpus:
                    p = mp.Process(
                        target=_mp_matched_worker,
                        args=(
                            gid,
                            task_queue,
                            result_queue,
                            config,
                            protocol,
                            label_list,
                            id2label,
                            str(output_dir),
                        ),
                    )
                    p.start()
                    workers.append(p)

                for _ in range(len(pending_tasks)):
                    regime, fold_idx, seed, res, err = result_queue.get()
                    if err is not None:
                        print(f"\n[FATAL] Worker failed for {regime} fold {fold_idx} seed {seed}:\n{err}", flush=True)
                        for p in workers:
                            if p.is_alive():
                                p.terminate()
                        raise RuntimeError(f"Multi-GPU worker failed: {err}")

                    all_results[regime].append(res)
                    _update_and_save_matched_summary()

                for p in workers:
                    p.join()

            else:
                device = torch.device(f"cuda:{gpu_ids[0]}" if (gpu_ids and torch.cuda.is_available()) else "cpu")
                print(f"[*] Running sequentially on device: {device} ({len(pending_tasks)} task(s))", flush=True)

                for regime, fold_idx, seed in pending_tasks:
                    tag = f"{regime}/fold{fold_idx}/seed{seed}"
                    split = protocol[regime][fold_idx]
                    run_dir = output_dir / f"{regime}_fold{fold_idx}_seed{seed}"

                    print(f"\n=== {tag} ===", flush=True)
                    print(split.describe(), flush=True)
                    set_seed(seed)

                    model = build_model(config, len(label_list), id2label)
                    log_prefix = f"[{exp_name} | {tag}]"
                    result = train_one(
                        model, split.train, split.val, split.test, config, device, checkpoint_dir=run_dir, log_prefix=log_prefix
                    )
                    result["seed"] = seed
                    result["fold"] = fold_idx
                    result["regime"] = regime
                    result["held_out"] = list(split.held_out_templates)
                    if config.task == "joint":
                        print(
                            f"{tag} -> Test Entity F1: {result['test']['f1']:.4f} | "
                            f"Linking F1: {result['test_linking']['f1']:.4f} | {result['cross_check']}",
                            flush=True,
                        )
                    else:
                        print(f"{tag} -> Test F1: {result['test']['f1']:.4f}  |  {result['cross_check']}", flush=True)
                    all_results[regime].append(result)
                    _update_and_save_matched_summary()

        summary, gap, out = _update_and_save_matched_summary()
        print(f"\n{'=' * 60}")
        print(f"MTL mean F1: {gap.get('mtl_mean_f1', 0.0):.4f} +/- {gap.get('mtl_std', 0.0):.4f}")
        print(f"UTL mean F1: {gap.get('utl_mean_f1', 0.0):.4f} +/- {gap.get('utl_std', 0.0):.4f}")
        print(f"GAP (MTL - UTL): {gap.get('gap_f1', 0.0):.4f}  ({gap.get('relative_drop_pct', 0.0):.1f}% relative drop)")
        print(f"-> {out}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

