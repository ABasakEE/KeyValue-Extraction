"""Unit tests for Template-Adversarial Representation Learning (DANN)."""

import sys
from pathlib import Path
import math
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.models.dann import (
    GradientReversalFunction,
    GradientReversalLayer,
    TemplateDiscriminator,
    DomainAdversarialModel,
    compute_dann_alpha,
)


def test_gradient_reversal_layer():
    """Verify forward identity and backward gradient reversal."""
    grl = GradientReversalLayer(alpha=0.75)
    x = torch.randn(4, 16, requires_grad=True)

    # Forward should be exact identity
    y = grl(x)
    assert torch.equal(x, y), "GRL forward pass must be identity"

    # Backward: d(loss)/dx should be -alpha * d(loss)/dy
    loss = (y * 2.0).sum()
    loss.backward()

    expected_grad = -0.75 * 2.0 * torch.ones_like(x)
    assert torch.allclose(x.grad, expected_grad), f"Expected {expected_grad}, got {x.grad}"

    # Test dynamic alpha override
    x2 = torch.randn(4, 16, requires_grad=True)
    y2 = grl(x2, alpha=1.5)
    (y2 * 3.0).sum().backward()
    expected_grad2 = -1.5 * 3.0 * torch.ones_like(x2)
    assert torch.allclose(x2.grad, expected_grad2), f"Expected {expected_grad2}, got {x2.grad}"
    print("[PASS] test_gradient_reversal_layer")


def test_dann_alpha_schedule():
    """Verify Ganin & Lempitsky schedule bounds."""
    a_0 = compute_dann_alpha(0, 1000, gamma=10.0)
    assert math.isclose(a_0, 0.0, abs_tol=1e-5), f"At step 0, alpha should be 0.0, got {a_0}"

    a_half = compute_dann_alpha(500, 1000, gamma=10.0)
    assert 0.98 <= a_half <= 1.0, f"At step 500/1000, alpha should be ~0.986, got {a_half}"

    a_end = compute_dann_alpha(1000, 1000, gamma=10.0)
    assert 0.999 <= a_end <= 1.0, f"At step 1000, alpha should be ~1.0, got {a_end}"
    print("[PASS] test_dann_alpha_schedule")


def test_template_discriminator():
    """Verify discriminator shapes and forward execution."""
    disc = TemplateDiscriminator(in_features=64, hidden_dim=32, num_templates=5)
    x = torch.randn(4, 64)
    out = disc(x)
    assert out.shape == (4, 5), f"Expected shape (4, 5), got {out.shape}"
    print("[PASS] test_template_discriminator")


def test_domain_adversarial_wrapper():
    """Verify joint forward pass, dual loss computation, and gradient flow."""
    class MockOutput:
        def __init__(self, logits, loss, hidden_states):
            self.logits = logits
            self.loss = loss
            self.hidden_states = hidden_states

    class MockBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 4)
            self.config = type("Config", (), {"id2label": {0: "O", 1: "KEY"}})()

        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
            B, L = input_ids.shape
            hidden = torch.randn(B, L, 32, requires_grad=True)
            # make linear depend on hidden so gradients flow
            logits = self.linear(hidden)
            loss = torch.tensor(1.23, requires_grad=True)
            return MockOutput(logits=logits, loss=loss, hidden_states=[hidden])

    class MockClassifier:
        def __init__(self):
            self.model = MockBaseModel()
            self.id2label = {0: "O", 1: "KEY"}

        def encode(self, words, boxes, labels=None):
            return None

        def first_subword_predictions(self, *args):
            return [0]

    base_clf = MockClassifier()
    dann_model = DomainAdversarialModel(
        base_classifier=base_clf,
        num_templates=3,
        hidden_dim=32,
        discriminator_hidden_dim=16,
        domain_loss_weight=0.2,
    )

    B, L = 2, 8
    inputs = {
        "input_ids": torch.randint(0, 100, (B, L)),
        "attention_mask": torch.ones((B, L), dtype=torch.long),
    }
    template_targets = torch.tensor([0, 2], dtype=torch.long)

    out = dann_model(inputs=inputs, template_targets=template_targets, alpha=0.5)

    assert out.token_logits.shape == (B, L, 4)
    assert out.domain_logits.shape == (B, 3)
    assert out.task_loss is not None
    assert out.domain_loss is not None
    assert out.total_loss is not None
    assert torch.isclose(out.total_loss, out.task_loss + 0.2 * out.domain_loss)

    # Test backward pass
    out.total_loss.backward()
    for p in dann_model.discriminator.parameters():
        if p.requires_grad:
            assert p.grad is not None, "Discriminator parameters must receive gradients"

    print("[PASS] test_domain_adversarial_wrapper")


import tempfile
from dataclasses import dataclass
from src.train import TrainConfig, train_one, make_collator
from src.models.token_clf import EncodedBatch

@dataclass
class DummyDannExample:
    guid: str
    words: list[str]
    boxes: list[list[int]]
    ner_tags: list[int]
    template_id: str


def test_dann_training_pipeline():
    """Verify that train_one with adversarial_dann=True executes cleanly."""
    class DummyBertModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 3)
            self.loss_fn = nn.CrossEntropyLoss()
            self.config = type("Config", (), {"id2label": {0: "O", 1: "B-KEY", 2: "B-VAL"}})()

        def forward(self, input_ids=None, bbox=None, attention_mask=None, labels=None, output_hidden_states=False, **kwargs):
            B, L = input_ids.shape
            hidden = torch.randn(B, L, 32, requires_grad=True)
            logits = self.linear(hidden)
            loss = None
            if labels is not None:
                loss = self.loss_fn(logits.view(-1, 3), labels.view(-1).clamp(min=0))
            return type("Out", (), {"logits": logits, "loss": loss, "hidden_states": [hidden]})()

    class DummyClassifier:
        def __init__(self):
            self.model = DummyBertModel()
            self.id2label = {0: "O", 1: "B-KEY", 2: "B-VAL"}
            self.backbone = "lilt"

        def encode(self, words_batch, boxes_batch, labels_batch=None):
            B = len(words_batch)
            L = 16
            wids = [list(range(len(words_batch[b]))) + [None] * (L - len(words_batch[b])) for b in range(B)]
            labels_tensor = None
            if labels_batch is not None:
                labels_tensor = torch.zeros((B, L), dtype=torch.long)
                for b in range(B):
                    for idx, t in enumerate(labels_batch[b]):
                        labels_tensor[b, idx] = t
            return EncodedBatch(
                inputs={
                    "input_ids": torch.randint(0, 50, (B, L)),
                    "attention_mask": torch.ones((B, L), dtype=torch.long),
                    "bbox": torch.randint(0, 1000, (B, L, 4)),
                    "labels": labels_tensor,
                },
                word_ids=wids,
            )

        def first_subword_predictions(self, logits, word_ids, n_words):
            return [logits[i].argmax().item() for i in range(min(n_words, logits.size(0)))]

    base = DummyClassifier()
    dann_model = DomainAdversarialModel(
        base_classifier=base,
        num_templates=3,
        hidden_dim=32,
        discriminator_hidden_dim=16,
        domain_loss_weight=0.1,
    )

    examples = [
        DummyDannExample(guid=f"doc_{i}", words=["Total", "$50"], boxes=[[10, 10, 50, 50], [60, 10, 100, 50]], ner_tags=[1, 2], template_id=f"T{i%3}")
        for i in range(6)
    ]

    cfg = TrainConfig(
        dataset="vrdu",
        backbone="lilt",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        seeds=(42,),
        adversarial_dann=True,
        dann_weight=0.1,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        res = train_one(
            model=dann_model,
            train_examples=examples[:4],
            val_examples=examples[4:],
            test_examples=examples[4:],
            config=cfg,
            device=torch.device("cpu"),
            checkpoint_dir=Path(tmp_dir),
        )

    assert "test" in res
    assert "f1" in res["test"]
    print("[PASS] test_dann_training_pipeline")

if __name__ == "__main__":
    test_gradient_reversal_layer()
    test_dann_alpha_schedule()
    test_template_discriminator()
    test_domain_adversarial_wrapper()
    test_dann_training_pipeline()
    print("ALL DANN ADVERSARIAL TESTS PASSED SUCCESSFULLY!")
