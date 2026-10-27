"""Phase 3, 4 & 5: Graph-BROS with LoRA Query/Key Adaptation and Isotropic Beta-Skeleton Attention.

Implements:
- LoRALinear: Surgical low-rank adaptation (rank r=8/16) on Query and Key projections while
  freezing the 110M base BROS parameters (reducing optimizer state VRAM by >98%).
- GraphBrosSelfAttention: Overrides BrosSelfAttention to inject 3D isotropic edge features
  (W_g * e_ij) and the topological Beta-Skeleton log-mask log(A_ij + eps):
      Attention_ij = (q_i^T k_j + q_i^T PE(Delta p_ij)) / sqrt(d) + W_g * e_ij + gamma_h * log(A_ij + eps)
- Focal Loss token classification objective to handle background ("O") vs entity token imbalance.
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
    """Low-Rank Adaptation wrapper for a frozen nn.Linear layer."""

    def __init__(self, base_layer: nn.Linear, r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, r)

        # Freeze original pretrained weights
        for param in self.base_layer.parameters():
            param.requires_grad = False

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
    """BROS Self-Attention augmented with LoRA Q/K and Direction-Agnostic Graph Bias.

    Computes:
        Attention_ij = (q_i^T k_j + q_i^T PE(Delta p_ij)) / sqrt(d)
                       + W_g * e_ij
                       + gate_h * log(A_ij + eps)
    """

    def __init__(
        self,
        base_self_attn: nn.Module,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.num_attention_heads = base_self_attn.num_attention_heads
        self.attention_head_size = base_self_attn.attention_head_size
        self.all_head_size = base_self_attn.all_head_size
        self.is_decoder = getattr(base_self_attn, "is_decoder", False)
        self.eps = eps

        # Surgical LoRA on Query and Key projections; freeze Value projection
        self.query = LoRALinear(base_self_attn.query, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.key = LoRALinear(base_self_attn.key, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.value = base_self_attn.value
        for param in self.value.parameters():
            param.requires_grad = False

        self.dropout = base_self_attn.dropout

        # Learned linear projection W_g mapping 3-D isotropic edge features e_ij to per-head bias
        self.W_g = nn.Linear(3, self.num_attention_heads, bias=True)
        nn.init.zeros_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)

        # Per-head topological gate modulating log(A_ij + eps)
        # Half of the heads focus locally on the beta-skeleton graph; half maintain broader context
        init_gates = torch.tensor(
            [0.25 if h % 2 == 0 else 0.05 for h in range(self.num_attention_heads)],
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

        # 3. Direction-Agnostic Graph Bias: W_g * e_ij + gate_h * log(A_ij + eps)
        if self._cached_edge_features is not None and not is_cross_attention:
            # W_g(e_ij): (B, L, L, 3) -> (B, L, L, H) -> (B, H, L, L)
            edge_bias = self.W_g(self._cached_edge_features).permute(0, 3, 1, 2)
            attention_scores = attention_scores + edge_bias

        if self._cached_log_adjacency is not None and not is_cross_attention:
            # _cached_log_adjacency is log(A_ij + eps) of shape (B, 1, L, L)
            gates = F.softplus(self.head_graph_gate).view(1, self.num_attention_heads, 1, 1)
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
    ignore_index: int = -100,
) -> torch.Tensor:
    """Multi-class Focal Loss over valid token positions."""
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
    focal_loss = ((1.0 - pt) ** gamma) * ce_loss
    return focal_loss.mean()


class GraphBrosForTokenClassification(nn.Module):
    """Wraps BrosForTokenClassification with Beta-Skeleton Graph Attention, LoRA, and Focal Loss."""

    def __init__(
        self,
        base_classifier: Any,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        beta: float = 1.0,
        eps: float = 1e-9,
        loss_type: str = "focal",
        focal_gamma: float = 2.0,
        use_focal_loss: bool | None = None,
        merge_spans: bool = True,
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

        # 1. Freeze all base BROS parameters first
        for param in self.bros_for_token_clf.parameters():
            param.requires_grad = False

        # Wrap frozen bbox_embeddings and word embeddings in torch.no_grad() to save activation VRAM/RAM
        orig_bbox_emb_forward = self.bros_for_token_clf.bros.bbox_embeddings.forward
        self.bros_for_token_clf.bros.bbox_embeddings.forward = torch.no_grad()(orig_bbox_emb_forward)

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
            )
            layer.attention.self = graph_self_attn
            self.graph_attn_layers.append(graph_self_attn)

        # 3. Unfreeze classification head and LayerNorms so output projection adapts cleanly
        for param in self.bros_for_token_clf.classifier.parameters():
            param.requires_grad = True

        if gradient_checkpointing and hasattr(self.bros_for_token_clf, "gradient_checkpointing_enable"):
            try:
                self.bros_for_token_clf.gradient_checkpointing_enable()
            except Exception:
                pass

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
            # Compute topological log-mask: log(A_ij + eps) -> 0 where A_ij=1, log(eps) where A_ij=0
            log_adjacency = torch.log(adjacency.clamp(min=0.0, max=1.0) + self.eps).unsqueeze(1)

        # 2. Populate graph context into all GraphBrosSelfAttention layers
        for attn_layer in self.graph_attn_layers:
            attn_layer.set_graph_context(edge_features, log_adjacency)

        try:
            outputs = self.bros_for_token_clf(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None,  # We compute Focal Loss or CE below
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
