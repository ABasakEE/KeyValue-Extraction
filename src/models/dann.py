"""Domain-Adversarial Neural Network (DANN) for Template-Invariant Document Representation.

Reference:
    Yaroslav Ganin et al.
    Domain-Adversarial Training of Neural Networks. JMLR 2016.
    https://arxiv.org/abs/1505.07818

In Cross-Template Visually-Rich Document Understanding (VrDU), models often suffer
from severe layout overfitting, memorizing template-specific absolute coordinates.
DANN introduces a Template Discriminator connected to the multimodal transformer
backbone via a Gradient Reversal Layer (GRL).

During backpropagation:
    1. The Template Discriminator minimizes domain cross-entropy (learning to identify form templates).
    2. The Backbone is updated with REVERSED gradients (-alpha * grad), forcing it to learn
       representations from which form templates CANNOT be discriminated (template-invariance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Autograd function that passes input forward unchanged and scales gradients by -alpha during backward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Reverse gradient and scale by alpha: dx = -alpha * grad_output
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """Module wrapper for GradientReversalFunction with optional dynamic scheduling."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
        a = float(alpha) if alpha is not None else self.alpha
        return GradientReversalFunction.apply(x, a)


def compute_dann_alpha(current_step: int, total_steps: int, gamma: float = 10.0) -> float:
    """Standard Ganin & Lempitsky dynamic schedule for GRL adaptation factor alpha.

    alpha(p) = 2 / (1 + exp(-gamma * p)) - 1
    where p in [0, 1] is the training progress.
    At p=0, alpha=0 (focus on learning the primary task).
    At p=1, alpha=1 (full adversarial pressure for domain invariance).
    """
    if total_steps <= 0:
        return 1.0
    p = float(current_step) / float(total_steps)
    p = max(0.0, min(1.0, p))
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


class TemplateDiscriminator(nn.Module):
    """Multi-layer MLP discriminator predicting document template class from pooled embeddings."""

    def __init__(
        self,
        in_features: int = 768,
        hidden_dim: int = 256,
        num_templates: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_templates),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: (batch_size, in_features) document-level representation.

        Returns: (batch_size, num_templates) unnormalized template logits.
        """
        return self.net(x)


@dataclass
class DANNModelOutput:
    """Output bundle for Domain-Adversarial Document Models."""

    total_loss: torch.Tensor | None
    task_loss: torch.Tensor | None
    domain_loss: torch.Tensor | None
    token_logits: torch.Tensor                 # (batch_size, seq_len, num_labels)
    domain_logits: torch.Tensor | None         # (batch_size, num_templates)
    alpha: float


class DomainAdversarialModel(nn.Module):
    """Wraps a token classification model (LiLT or BROS) with a Template Discriminator and GRL.

    Architecture:
        Inputs -> Backbone (LiLT / BROS) -> Token Hidden States (B, L, D)
                    ├──> Token Classifier Head -> Token Logits (B, L, C) -> Task Loss
                    └──> Masked Sequence Pooling -> Document Vector (B, D)
                          └──> GRL (-alpha) -> Template Discriminator -> Domain Loss

        Total Loss = Task Loss + lambda_domain * Domain Loss
    """

    def __init__(
        self,
        base_classifier: Any,
        num_templates: int = 4,
        hidden_dim: int = 768,
        discriminator_hidden_dim: int = 256,
        domain_loss_weight: float = 0.1,
        gamma: float = 10.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.base_classifier = base_classifier
        if hasattr(base_classifier, "model") and isinstance(base_classifier.model, nn.Module):
            self.base_model = base_classifier.model
        elif isinstance(base_classifier, nn.Module):
            self.base_model = base_classifier
        else:
            self.base_model = None

        self.num_templates = num_templates
        self.hidden_dim = hidden_dim
        self.domain_loss_weight = domain_loss_weight
        self.gamma = gamma

        self.grl = GradientReversalLayer(alpha=1.0)
        self.discriminator = TemplateDiscriminator(
            in_features=hidden_dim,
            hidden_dim=discriminator_hidden_dim,
            num_templates=num_templates,
            dropout=dropout,
        )

    def to(self, *args, **kwargs):
        if hasattr(self.base_classifier, "to") and callable(self.base_classifier.to):
            self.base_classifier.to(*args, **kwargs)
        elif hasattr(self.base_classifier, "model") and hasattr(self.base_classifier.model, "to"):
            self.base_classifier.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @property
    def model(self):
        return self

    @property
    def config(self):
        return getattr(self.base_classifier.model, "config", None)

    @property
    def id2label(self):
        return getattr(self.base_classifier, "id2label", None) or self.config.id2label

    def encode(self, *args, **kwargs):
        return self.base_classifier.encode(*args, **kwargs)

    def first_subword_predictions(self, *args, **kwargs):
        return self.base_classifier.first_subword_predictions(*args, **kwargs)

    def forward(
        self,
        inputs: dict[str, torch.Tensor] | None = None,
        template_targets: torch.Tensor | None = None,
        alpha: float | None = None,
        **kwargs,
    ) -> DANNModelOutput:
        if inputs is None:
            inputs = kwargs
        elif kwargs:
            inputs = {**inputs, **kwargs}
        """Forward pass computing token labeling task loss and template domain adversarial loss."""
        # 1. Forward through transformer backbone requesting hidden states
        base_model = getattr(self, "base_model", None) or getattr(self.base_classifier, "model", self.base_classifier)
        outputs = base_model(**inputs, output_hidden_states=True)

        token_logits = outputs.logits
        task_loss = outputs.loss

        # 2. Extract final hidden states
        # Shape: (batch_size, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states[-1]

        # 3. Masked Sequence Pooling to get document-level representation
        # Use attention_mask to ignore padding tokens
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            pooled_doc = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled_doc = hidden_states.mean(dim=1)  # (B, D)

        # 4. GRL and Template Discrimination
        cur_alpha = float(alpha) if alpha is not None else 1.0
        reversed_features = self.grl(pooled_doc, cur_alpha)
        domain_logits = self.discriminator(reversed_features)

        # 5. Domain Loss
        domain_loss = None
        total_loss = task_loss
        if template_targets is not None:
            # Defensive check: ensure template_targets matches domain_logits device
            if template_targets.device != domain_logits.device:
                template_targets = template_targets.to(domain_logits.device)
            # Cross-entropy over template classes
            domain_loss = nn.functional.cross_entropy(domain_logits, template_targets)
            if task_loss is not None:
                total_loss = task_loss + self.domain_loss_weight * domain_loss
            else:
                total_loss = self.domain_loss_weight * domain_loss

        return DANNModelOutput(
            total_loss=total_loss,
            task_loss=task_loss,
            domain_loss=domain_loss,
            token_logits=token_logits,
            domain_logits=domain_logits,
            alpha=cur_alpha,
        )
