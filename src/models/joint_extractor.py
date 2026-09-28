"""Unified Joint Model for Entity Tagging and Key-Value Linking.

Combines a layout-aware transformer encoder (LiLT / LayoutLMv3 / BROS) with:
1. Token Classification Head for field type extraction (Question, Answer, Header, Other).
2. Spatial Biaffine Relation Head for key-value linking prediction (k -> v).

This enables joint end-to-end multi-task learning where representations are shaped by both
entity semantics and relational layout structure:
    L_total = L_tagging + lambda_link * L_linking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

try:
    from models.linking_head import (
        BiaffineRelationClassifier,
        build_type_compatibility_mask,
        pool_entity_representations,
    )
    from models.token_clf import BACKBONES, IGNORE_INDEX, EncodedBatch, TokenClassifier
except ImportError:
    from src.models.linking_head import (
        BiaffineRelationClassifier,
        build_type_compatibility_mask,
        pool_entity_representations,
    )
    from src.models.token_clf import BACKBONES, IGNORE_INDEX, EncodedBatch, TokenClassifier


@dataclass
class JointModelOutput:
    """Outputs for a single document or batch from JointFormExtractor."""

    total_loss: torch.Tensor | None
    tagging_loss: torch.Tensor | None
    linking_loss: torch.Tensor | None
    token_logits: torch.Tensor               # (B, L, num_labels)
    relation_logits: list[torch.Tensor]     # per document: (M, M) pairwise logits
    predicted_links: list[set[tuple[int, int]]]


class JointFormExtractor(nn.Module):
    """Joint Document Key-Value Information Extraction model."""

    def __init__(
        self,
        backbone: str,
        num_labels: int,
        id2label: dict[int, str],
        max_length: int = 512,
        hidden_dim: int = 768,
        relation_proj_dim: int = 128,
        relation_geo_dim: int = 64,
        linking_loss_weight: float = 1.0,
        pos_weight: float = 10.0,
        use_spatial_relations: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")

        checkpoint, tokenizer_checkpoint, needs_image = BACKBONES[backbone]
        self.backbone = backbone
        self.needs_image = needs_image
        self.num_labels = num_labels
        self.id2label = id2label
        self.max_length = max_length
        self.linking_loss_weight = linking_loss_weight
        self.pos_weight = pos_weight

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, use_fast=True)

        # 1. Backbone with token classification head
        config = AutoConfig.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
        )
        self.encoder = AutoModelForTokenClassification.from_pretrained(
            checkpoint, config=config, ignore_mismatched_sizes=True
        )

        # 2. Key-Value Linking Head
        self.relation_head = BiaffineRelationClassifier(
            hidden_dim=hidden_dim,
            proj_dim=relation_proj_dim,
            geo_dim=relation_geo_dim,
            dropout=dropout,
            use_spatial=use_spatial_relations,
        )

    def encode(
        self,
        words_batch: list[list[str]],
        boxes_batch: list[list[list[int]]],
        labels_batch: list[list[int]] | None = None,
    ) -> EncodedBatch:
        return TokenClassifier.encode(self, words_batch, boxes_batch, labels_batch)

    first_subword_predictions = staticmethod(TokenClassifier.first_subword_predictions)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        entity_spans: list[list[list[int]]] | None = None,
        gold_links: list[list[tuple[int, int]]] | None = None,
        entity_types: list[list[str]] | None = None,
        link_threshold: float = 0.5,
    ) -> JointModelOutput:
        """Forward pass computing token logits, relation matrices, and joint loss.

        Args:
            inputs: Dictionary containing input_ids, bbox, attention_mask, labels, etc.
            entity_spans: For each document in batch, a list of token-index lists per entity.
            gold_links: For each document in batch, a list of (src_entity_idx, dst_entity_idx) tuples.
            entity_types: Optional list of entity type strings for domain constraint masking.
            link_threshold: Probability threshold for predicting active links.
        """
        # Forward pass through backbone with hidden states requested
        encoder_outputs = self.encoder(**inputs, output_hidden_states=True)

        token_logits = encoder_outputs.logits                       # (B, L, num_labels)
        tagging_loss = encoder_outputs.loss                         # scalar or None
        hidden_states = encoder_outputs.hidden_states[-1]           # (B, L, hidden_dim)

        batch_size = hidden_states.size(0)
        token_boxes = inputs.get("bbox", torch.zeros_like(hidden_states[..., :4]))

        relation_logits_list: list[torch.Tensor] = []
        predicted_links_list: list[set[tuple[int, int]]] = []
        linking_losses: list[torch.Tensor] = []

        if entity_spans is not None:
            for b in range(batch_size):
                doc_spans = entity_spans[b]
                doc_hidden = hidden_states[b]
                doc_boxes = token_boxes[b]

                if not doc_spans:
                    relation_logits_list.append(torch.zeros((0, 0), device=hidden_states.device))
                    predicted_links_list.append(set())
                    continue

                # Pool token embeddings to entity spans
                entity_reps, entity_boxes = pool_entity_representations(
                    doc_hidden, doc_boxes, doc_spans
                )

                # Optional type compatibility mask
                type_mask = None
                if entity_types is not None and b < len(entity_types) and entity_types[b]:
                    type_mask = build_type_compatibility_mask(entity_types[b]).to(hidden_states.device)

                # Compute pairwise relation logits
                rel_logits = self.relation_head(entity_reps, entity_boxes, type_mask=type_mask)
                relation_logits_list.append(rel_logits)

                # Decode link predictions
                preds = BiaffineRelationClassifier.decode_predictions(
                    rel_logits, threshold=link_threshold
                )
                predicted_links_list.append(preds)

                # Compute relation extraction loss if gold links are provided
                if gold_links is not None and b < len(gold_links):
                    doc_gold = gold_links[b]
                    link_loss = BiaffineRelationClassifier.compute_loss(
                        rel_logits, doc_gold, pos_weight=self.pos_weight
                    )
                    linking_losses.append(link_loss)

        # Aggregate losses
        linking_loss = None
        if linking_losses:
            linking_loss = torch.stack(linking_losses).mean()

        total_loss = None
        if tagging_loss is not None and linking_loss is not None:
            total_loss = tagging_loss + self.linking_loss_weight * linking_loss
        elif tagging_loss is not None:
            total_loss = tagging_loss
        elif linking_loss is not None:
            total_loss = linking_loss

        return JointModelOutput(
            total_loss=total_loss,
            tagging_loss=tagging_loss,
            linking_loss=linking_loss,
            token_logits=token_logits,
            relation_logits=relation_logits_list,
            predicted_links=predicted_links_list,
        )


def prepare_entity_batch(
    batch: list,
    word_ids_batch: list[list[int | None]],
) -> tuple[list[list[list[int]]], list[list[tuple[int, int]]], list[list[str]]]:
    """Map raw FormExamples with word-level entities and links to subword index spans.

    Args:
        batch: List of FormExample objects.
        word_ids_batch: List of word_ids mappings per document (from EncodedBatch).

    Returns:
        entity_spans: For each document, list of subword indices per entity.
        gold_links: For each document, list of (src_entity_idx, dst_entity_idx) tuples.
        entity_types: For each document, list of entity label strings.
    """
    batch_spans: list[list[list[int]]] = []
    batch_gold_links: list[list[tuple[int, int]]] = []
    batch_types: list[list[str]] = []

    for b, example in enumerate(batch):
        if not hasattr(example, "entities") or not example.entities:
            batch_spans.append([])
            batch_gold_links.append([])
            batch_types.append([])
            continue

        # Map each source word index to its first subword token index
        word_ids = word_ids_batch[b]
        word_to_subword: dict[int, int] = {}
        for subword_idx, wid in enumerate(word_ids):
            if wid is not None and wid not in word_to_subword:
                word_to_subword[wid] = subword_idx

        # Map each valid entity to a contiguous index 0..M-1
        entity_id_to_idx: dict[int, int] = {}
        doc_spans: list[list[int]] = []
        doc_types: list[str] = []

        # Sort entity IDs for determinism
        sorted_ent_ids = sorted(example.entities.keys())
        for ent_id in sorted_ent_ids:
            ent = example.entities[ent_id]
            w_indices = ent.get("token_indices", [])
            subword_indices = [
                word_to_subword[w] for w in w_indices if w in word_to_subword
            ]
            if subword_indices:
                idx = len(doc_spans)
                entity_id_to_idx[ent_id] = idx
                doc_spans.append(subword_indices)
                doc_types.append(ent.get("label", "other"))

        # Map gold links from dataset entity IDs to contiguous indices
        doc_links: list[tuple[int, int]] = []
        if hasattr(example, "links") and example.links:
            for s, d in example.links:
                if s in entity_id_to_idx and d in entity_id_to_idx and s != d:
                    doc_links.append((entity_id_to_idx[s], entity_id_to_idx[d]))

        batch_spans.append(doc_spans)
        batch_gold_links.append(doc_links)
        batch_types.append(doc_types)

    return batch_spans, batch_gold_links, batch_types

