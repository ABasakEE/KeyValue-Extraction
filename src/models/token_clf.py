"""One interface over the three token-classification backbones in the plan.

Each backbone wants slightly different tensors, and the differences are exactly
where silent bugs live:

* **LiLT** (``SCUT-DLVCLab/lilt-roberta-en-base``, MIT) — text + 4-value boxes,
  no image. Ships with no tokenizer of its own; the checkpoint is a RoBERTa text
  tower fused with a layout tower, so it pairs with the LayoutLMv3 tokenizer
  (same BPE vocabulary, and it already knows how to carry ``boxes``).
* **LayoutLMv3** (``microsoft/layoutlmv3-base``, CC-BY-NC-SA-4.0 — non-commercial,
  fine for coursework) — text + boxes + ViT image patches, so it additionally
  needs ``pixel_values``.
* **BROS** (``jinho8345/bros-base-uncased``, Apache-2.0) — text + boxes, no image,
  BERT-style WordPiece vocabulary.

Every one of them expects boxes normalized to 0-1000; see
``data.funsd_loader.normalize_box``.

Subword alignment: a word becomes several subword tokens, but an entity label
belongs to the word. We label the first subword and set the rest to -100 so the
loss ignores them, which is also what makes ``seqeval`` chunk evaluation line up
with word-level ground truth. Labelling every subword instead inflates F1,
because long words contribute more correct predictions than short ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

IGNORE_INDEX = -100

# name -> (checkpoint, tokenizer checkpoint, needs_image)
BACKBONES: dict[str, tuple[str, str, bool]] = {
    "lilt": ("SCUT-DLVCLab/lilt-roberta-en-base", "microsoft/layoutlmv3-base", False),
    "layoutlmv3": ("microsoft/layoutlmv3-base", "microsoft/layoutlmv3-base", True),
    "bros": ("jinho8345/bros-base-uncased", "jinho8345/bros-base-uncased", False),
}


@dataclass
class EncodedBatch:
    """Tensors plus the bookkeeping evaluation needs to get back to words."""

    inputs: dict[str, torch.Tensor]
    # per example, the index of the source word for each subword position
    # (None for special tokens) — lets predictions be mapped back to words
    word_ids: list[list[int | None]]


class TokenClassifier:
    def __init__(self, backbone: str, num_labels: int, id2label: dict[int, str], max_length: int = 512):
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")

        checkpoint, tokenizer_checkpoint, needs_image = BACKBONES[backbone]
        self.backbone = backbone
        self.needs_image = needs_image
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, use_fast=True)

        config = AutoConfig.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            checkpoint, config=config, ignore_mismatched_sizes=True
        )

    def encode(
        self,
        words_batch: list[list[str]],
        boxes_batch: list[list[list[int]]],
        labels_batch: list[list[int]] | None = None,
    ) -> EncodedBatch:
        """Tokenize a batch of documents, propagating boxes and labels to subwords."""
        for words, boxes in zip(words_batch, boxes_batch):
            if len(words) != len(boxes):
                raise ValueError(f"{len(words)} words but {len(boxes)} boxes — these must correspond 1:1")

        tokenize_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        if boxes_batch is not None:
            tokenize_kwargs["boxes"] = boxes_batch

        # LayoutLMv3Tokenizer natively accepts list[list[str]] when boxes is provided and
        # raises TypeError if is_split_into_words is passed. Other tokenizers (e.g. BROS) require it.
        try:
            encoding = self.tokenizer(
                words_batch,
                is_split_into_words=True,
                **tokenize_kwargs,
            )
        except TypeError:
            encoding = self.tokenizer(
                words_batch,
                **tokenize_kwargs,
            )

        all_word_ids: list[list[int | None]] = []
        aligned_labels: list[list[int]] = []

        for i in range(len(words_batch)):
            word_ids = encoding.word_ids(batch_index=i)
            all_word_ids.append(word_ids)

            if labels_batch is None:
                continue

            source = labels_batch[i]
            row, previous = [], None
            for wid in word_ids:
                if wid is None:
                    row.append(IGNORE_INDEX)          # special token
                elif wid != previous:
                    row.append(source[wid])            # first subword carries the label
                else:
                    row.append(IGNORE_INDEX)           # continuation subword
                previous = wid
            aligned_labels.append(row)

        inputs = dict(encoding)
        inputs.pop("offset_mapping", None)
        # The BROS tokenizer emits token_type_ids that BrosForTokenClassification
        # accepts, but LiLT's RoBERTa-derived tower does not take them.
        if self.backbone == "lilt":
            inputs.pop("token_type_ids", None)
        if labels_batch is not None:
            inputs["labels"] = torch.tensor(aligned_labels, dtype=torch.long)

        return EncodedBatch(inputs=inputs, word_ids=all_word_ids)

    @staticmethod
    def first_subword_predictions(
        logits: torch.Tensor,
        word_ids: list[int | None],
        n_words: int,
    ) -> list[int]:
        """Reduce subword logits to one prediction per source word.

        Takes the first subword of each word, matching how labels were assigned.
        Words truncated away by the 512-token cap get no prediction; they are
        returned as ``-1`` so the caller can count them rather than silently
        scoring them as correct.
        """
        predictions = [-1] * n_words
        argmax = logits.argmax(dim=-1).tolist()
        previous = None
        for position, wid in enumerate(word_ids):
            if wid is not None and wid != previous and wid < n_words:
                predictions[wid] = argmax[position]
            previous = wid
        return predictions
