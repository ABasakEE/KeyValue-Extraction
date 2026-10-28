"""Phase 3, 4 & 5: Graph-BROS with LoRA Q/K/V Adaptation, Differential LR, and Isotropic Beta-Skeleton Attention.

Implements:
- LoRALinear: Low-Rank Adaptation wrapper on Query, Key, and Value projections. Supports both
  hybrid adaptation (pretrained base weights fine-tuned at a gentle 5e-5 LR + LoRA/Graph weights
  at 2e-4 LR) and strict frozen-base adaptation (where LayerNorms, attention output dense, and
  bbox projections remain trainable so representations never bottleneck).
- GraphBrosSelfAttention: Overrides BrosSelfAttention to inject LayerNorm-stabilized 3D isotropic
  edge features (W_g * LN(e_ij)) and bounded topological Beta-Skeleton log-mask log(A_ij + eps):
      Attention_ij = (q_i^T k_j + q_i^T PE(Delta p_ij)) / sqrt(d) + W_g * LN(e_ij) + gate_h * log(A_ij + eps)
  Initialized near identity at Step 0 (W_g = 0, gate_h = 0.02 on local heads and 0.0 on global heads)
  to prevent catastrophic destruction of pretrained BROS attention in Epoch 1.
- Class-balanced Focal Loss and standard Cross-Entropy objectives for BIO token tagging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.spatial_graph import build_isotropic_spatial_graph
except ImportError:
    from src.models.spatial_graph import build_isotropic_spatial_graph


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for an nn.Linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, r)

        for param in self.base_layer.parameters():
            param.requires_grad = not freeze_base

        if r > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
            self.lora_A = nn.Parameter(torch.empty((r, self.in_features)))
            self.lora_B = nn.Parameter(torch.zeros((self.out_features, r)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
            return base_out + lora_out * self.scaling
        return base_out


class GraphBrosSelfAttention(nn.Module):
    """BROS Self-Attention augmented with LoRA Q/K/V and Direction-Agnostic Graph Bias.

    Computes:
        Attention_ij = (q_i^T k_j + q_i^T PE(Delta p_ij)) / sqrt(d)
                       + W_g * LN(e_ij)
                       + ReLU(gate_h) * log(A_ij + eps)
    """

    def __init__(
        self,
        base_self_attn: nn.Module,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        eps: float = 1e-2,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = base_self_attn.num_attention_heads
        self.attention_head_size = base_self_attn.attention_head_size
        self.all_head_size = base_self_attn.all_head_size
        self.is_decoder = getattr(base_self_attn, "is_decoder", False)
        self.eps = eps

        # Apply LoRA to Query, Key, and Value projections
        self.query = LoRALinear(
            base_self_attn.query, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, freeze_base=freeze_base
        )
        self.key = LoRALinear(
            base_self_attn.key, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, freeze_base=freeze_base
        )
        self.value = LoRALinear(
            base_self_attn.value, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, freeze_base=freeze_base
        )

        self.dropout = base_self_attn.dropout

        # Normalize 3-D isotropic edge features e_ij before projecting with zero-initialized W_g
        self.edge_norm = nn.LayerNorm(3)
        self.W_g = nn.Linear(3, self.num_attention_heads, bias=True)
        nn.init.zeros_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)

        # Per-head topological gate modulating log(A_ij + eps):
        # Initialized small (0.02 on even heads, 0.0 on odd heads) so Step 0 preserves
        # pretrained BROS attention without a -17.0 mask shock across all heads.
        init_gates = torch.tensor(
            [0.02 if h % 2 == 0 else 0.0 for h in range(self.num_attention_heads)],
            dtype=torch.float32,
        )
        self.head_graph_gate = nn.Parameter(init_gates)

        # Transient graph cache populated once per batch before the encoder forward pass
        self._cached_edge_features: torch.Tensor | None = None
        self._cached_log_adjacency: torch.Tensor | None = None

    def set_graph_context(self, edge_features: torch.Tensor | None, log_adjacency: torch.Tensor | None) -> None:
        self._cached_edge_features = edge_features
        self._cached_log_adjacency = log_adjacency

    def clear_graph_context(self) -> None:
        self._cached_edge_features = None
        self._cached_log_adjacency = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        bbox_pos_emb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, -1, self.num_attention_heads, self.attention_head_size)

        query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)

        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_layer = self.key(encoder_hidden_states).view(hidden_shape).transpose(1, 2)
            value_layer = self.value(encoder_hidden_states).view(hidden_shape).transpose(1, 2)
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
            value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        # 1. Semantic dot-product: q_i^T k_j
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 2. Native BROS 2D relative positional encoding: q_i^T PE(Delta p_ij)
        _, n_head, _, d_head = query_layer.shape
        bbox_pos_emb_reshaped = bbox_pos_emb.view(seq_length, seq_length, batch_size, d_head)
        bbox_pos_emb_reshaped = bbox_pos_emb_reshaped.permute([2, 0, 1, 3])
        bbox_pos_scores = torch.einsum("bnid,bijd->bnij", (query_layer, bbox_pos_emb_reshaped))

        attention_scores = (attention_scores + bbox_pos_scores) / math.sqrt(self.attention_head_size)

        # 3. Direction-Agnostic Graph Bias: W_g * LN(e_ij) + ReLU(gate_h) * log(A_ij + eps)
        if self._cached_edge_features is not None and not is_cross_attention:
            normed_edges = self.edge_norm(self._cached_edge_features)
            edge_bias = self.W_g(normed_edges).permute(0, 3, 1, 2)
            attention_scores = attention_scores + edge_bias

        if self._cached_log_adjacency is not None and not is_cross_attention:
            gates = F.relu(self.head_graph_gate).view(1, self.num_attention_heads, 1, 1)
            attention_scores = attention_scores + gates * self._cached_log_adjacency

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs


@dataclass
class GraphBrosOutput:
    """Output container matching Hugging Face TokenClassifierOutput."""

    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


def focal_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    entity_weight: float = 2.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Class-balanced Multi-class Focal Loss over valid token positions.

    Up-weights non-background entity tokens (label > 0) by `entity_weight` so the
    97% background ('O', label == 0) class cannot suppress entity recall during early epochs.
    """
    num_classes = logits.shape[-1]
    flat_logits = logits.view(-1, num_classes)
    flat_labels = labels.view(-1)

    valid_mask = flat_labels != ignore_index
    if not valid_mask.any():
        return flat_logits.sum() * 0.0

    valid_logits = flat_logits[valid_mask]
    valid_targets = flat_labels[valid_mask]

    ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction="none")
    pt = torch.exp(-ce_loss)
    alpha_t = torch.where(valid_targets > 0, entity_weight, 1.0).to(ce_loss.dtype)
    focal_loss = alpha_t * ((1.0 - pt) ** gamma) * ce_loss
    return focal_loss.mean()


class GraphBrosForTokenClassification(nn.Module):
    """Wraps BrosForTokenClassification with Beta-Skeleton Graph Attention, LoRA, and Differential LR."""

    def __init__(
        self,
        base_classifier: Any,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        beta: float = 1.0,
        eps: float = 1e-2,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        use_focal_loss: bool | None = None,
        merge_spans: bool = True,
        freeze_base: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.base_classifier = base_classifier
        self.bros_for_token_clf = base_classifier.model
        self.beta = beta
        self.eps = eps
        self.focal_gamma = focal_gamma
        self.use_focal_loss = (loss_type == "focal") if use_focal_loss is None else use_focal_loss
        self.merge_spans = merge_spans
        self.freeze_base = freeze_base

        # 1. Configure base parameter trainability
        for name, param in self.bros_for_token_clf.named_parameters():
            if not freeze_base:
                param.requires_grad = True
            else:
                # Even in frozen-base mode, keep LayerNorms, attention output dense,
                # bbox_embeddings projection, and classifier trainable so adapted Q/K/V
                # representations can propagate cleanly through the transformer.
                keep_trainable = any(
                    k in name
                    for k in ("LayerNorm", "layer_norm", "attention.output", "bbox_projection", "classifier")
                )
                param.requires_grad = keep_trainable

        # 2. Replace BrosSelfAttention in every encoder layer with GraphBrosSelfAttention + LoRA
        self.graph_attn_layers = nn.ModuleList()
        encoder_layers = self.bros_for_token_clf.bros.encoder.layer
        for layer in encoder_layers:
            orig_self_attn = layer.attention.self
            graph_self_attn = GraphBrosSelfAttention(
                orig_self_attn,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                eps=eps,
                freeze_base=freeze_base,
            )
            layer.attention.self = graph_self_attn
            self.graph_attn_layers.append(graph_self_attn)

        # 3. Always ensure classification head is trainable
        for param in self.bros_for_token_clf.classifier.parameters():
            param.requires_grad = True

        if gradient_checkpointing and hasattr(self.bros_for_token_clf, "gradient_checkpointing_enable"):
            try:
                self.bros_for_token_clf.gradient_checkpointing_enable()
            except Exception:
                pass

    def get_optimizer_param_groups(
        self,
        base_lr: float = 5e-5,
        graph_lr: float = 2.5e-4,
        weight_decay: float = 0.01,
    ) -> list[dict[str, Any]]:
        """Separate pretrained BROS weights (5e-5) from newly injected Graph/LoRA/Head weights (2.5e-4)."""
        fast_params = []
        backbone_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_fast = any(
                k in name
                for k in ("W_g", "edge_norm", "head_graph_gate", "lora_A", "lora_B", "classifier")
            )
            if is_fast:
                fast_params.append(param)
            else:
                backbone_params.append(param)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay})
        if fast_params:
            groups.append({"params": fast_params, "lr": graph_lr, "weight_decay": weight_decay})
        return groups

    @property
    def model(self):
        return self

    @property
    def config(self):
        return self.bros_for_token_clf.config

    @property
    def id2label(self):
        return getattr(self.base_classifier, "id2label", None) or self.config.id2label

    def encode(self, *args, **kwargs):
        return self.base_classifier.encode(*args, **kwargs)

    def first_subword_predictions(self, *args, **kwargs):
        return self.base_classifier.first_subword_predictions(*args, **kwargs)

    def trainable_parameter_summary(self) -> dict[str, int | float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = (100.0 * trainable / total) if total > 0 else 0.0
        return {"total_params": total, "trainable_params": trainable, "trainable_pct": round(pct, 3)}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        bbox: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> GraphBrosOutput:
        if bbox is None:
            raise ValueError("GraphBrosForTokenClassification requires 'bbox' tensor.")

        # 1. Construct direction-agnostic Beta-Skeleton graph & 3D isotropic edge features once per batch
        with torch.no_grad():
            edge_features, adjacency = build_isotropic_spatial_graph(
                bbox=bbox,
                attention_mask=attention_mask,
                beta=self.beta,
                merge_spans=self.merge_spans,
            )
            # Bounded topological log-mask: 0 where A_ij=1, bounded to [-5.0, 0.0] where A_ij=0
            log_adjacency = (
                torch.log(adjacency.clamp(min=0.0, max=1.0) + self.eps)
                .clamp(min=-5.0, max=0.0)
                .unsqueeze(1)
            )

        # 2. Populate graph context into all GraphBrosSelfAttention layers
        for attn_layer in self.graph_attn_layers:
            attn_layer.set_graph_context(edge_features, log_adjacency)

        try:
            outputs = self.bros_for_token_clf(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )
        finally:
            # Always clear transient graph tensors to free VRAM immediately
            for attn_layer in self.graph_attn_layers:
                attn_layer.clear_graph_context()

        logits = outputs.logits
        loss = None
        if labels is not None:
            if self.use_focal_loss:
                loss = focal_cross_entropy_loss(logits, labels, gamma=self.focal_gamma)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)

        return GraphBrosOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
