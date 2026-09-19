"""FUNSD loader.

FUNSD ships one JSON per form alongside the scanned PNG. The JSON looks like::

    {"form": [{"id": 0,
               "text": "Registration No.",
               "box": [94, 169, 191, 186],
               "label": "question",
               "words": [{"text": "Registration", "box": [...]}, ...],
               "linking": [[0, 1]]},
              ...]}

We parse the raw release rather than the ``nielsr/funsd`` HuggingFace mirror
because the mirror keeps only words/boxes/ner_tags and drops ``linking`` — and
the linking edges are exactly the key->value annotation this project is about.

Two caveats that belong in the write-up, not buried in a docstring:

* FUNSD annotates at block level, so every token inside an entity shares the
  entity's coordinates. Zhang et al. (EC-FUNSD, arXiv:2402.02379) show models
  exploit this as a proxy for entity boundaries rather than learning semantics.
* Lin et al. (PEneo, ACM MM 2024, arXiv:2401.03472) re-annotated FUNSD into
  RFUND because the original linking ground truth is too noisy to benchmark
  against. Prefer RFUND for any reported linking number.

Reference: Guillaume Jaume, Hazim Kemal Ekenel, Jean-Philippe Thiran. FUNSD: A
Dataset for Form Understanding in Noisy Scanned Documents, ICDAR-OST, 2019.
https://arxiv.org/abs/1905.13538
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

# FUNSD's four semantic entity classes. "other" is the outside class and never
# gets a B-/I- prefix.
ENTITY_LABELS = ("header", "question", "answer")
LABEL_LIST = ["O"] + [f"{prefix}-{lab.upper()}" for lab in ENTITY_LABELS for prefix in ("B", "I")]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


@dataclass
class FormExample:
    """One annotated form, flattened to the token sequence a tagger consumes."""

    guid: str
    words: list[str]
    boxes: list[list[int]]          # normalized to 0-1000, [x0, y0, x1, y1]
    ner_tags: list[int]
    image_path: str
    image_size: tuple[int, int]
    # entity_id per token, so linking edges can be mapped back onto tokens
    entity_ids: list[int] = field(default_factory=list)
    # (key_entity_id, value_entity_id) pairs straight from the annotation
    links: list[tuple[int, int]] = field(default_factory=list)
    # entity_id -> (label, [token indices]) for entity-level evaluation
    entities: dict[int, dict] = field(default_factory=dict)


def normalize_box(box: list[int], width: int, height: int) -> list[int]:
    """Scale an absolute pixel box to the 0-1000 range every LayoutLM-family
    model expects.

    Boxes are clamped because a handful of FUNSD annotations extend a pixel or
    two past the image edge, which trips the embedding lookup (position ids
    above 1000 index out of range).
    """
    x0, y0, x1, y1 = box
    scaled = [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]
    return [max(0, min(1000, v)) for v in scaled]


def _parse_form(annotation_path: Path, image_path: Path) -> FormExample:
    with annotation_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    with Image.open(image_path) as img:
        width, height = img.size

    words: list[str] = []
    boxes: list[list[int]] = []
    ner_tags: list[int] = []
    entity_ids: list[int] = []
    entities: dict[int, dict] = {}
    links: set[tuple[int, int]] = set()

    for entity in data["form"]:
        label = entity["label"].lower()
        # FUNSD includes empty entities (no recognized words); skip them so
        # they cannot become zero-length spans in the tag sequence.
        entity_words = [w for w in entity["words"] if w["text"].strip()]
        if not entity_words:
            continue

        token_start = len(words)
        for position, word in enumerate(entity_words):
            words.append(word["text"])
            boxes.append(normalize_box(word["box"], width, height))
            entity_ids.append(entity["id"])
            if label == "other":
                ner_tags.append(LABEL2ID["O"])
            else:
                prefix = "B" if position == 0 else "I"
                ner_tags.append(LABEL2ID[f"{prefix}-{label.upper()}"])

        entities[entity["id"]] = {
            "label": label,
            "token_indices": list(range(token_start, len(words))),
            "text": entity["text"],
        }

        # linking is a list of [from_id, to_id]; FUNSD records each edge on both
        # endpoints, so dedupe via a set.
        for src, dst in entity.get("linking", []):
            links.add((src, dst))

    # Drop edges pointing at entities we skipped as empty, otherwise the
    # linking evaluation would count unreachable gold pairs as recall misses.
    valid = set(entities)
    kept_links = sorted((s, d) for s, d in links if s in valid and d in valid)

    return FormExample(
        guid=annotation_path.stem,
        words=words,
        boxes=boxes,
        ner_tags=ner_tags,
        image_path=str(image_path),
        image_size=(width, height),
        entity_ids=entity_ids,
        links=kept_links,
        entities=entities,
    )


def load_funsd_split(root: str | Path, split: str) -> list[FormExample]:
    """Load one FUNSD split from the raw release.

    ``root`` is the directory holding ``training_data/`` and ``testing_data/``
    (i.e. the extracted ``dataset/`` folder of the official zip). ``split`` is
    ``"train"`` or ``"test"``.
    """
    root = Path(root)
    folder = {"train": "training_data", "test": "testing_data"}[split]
    ann_dir = root / folder / "annotations"
    img_dir = root / folder / "images"

    if not ann_dir.is_dir():
        raise FileNotFoundError(
            f"No annotations at {ann_dir}. Download FUNSD from "
            "https://guillaumejaume.github.io/FUNSD/ and extract so that "
            f"{root}/training_data/annotations/*.json exists."
        )

    examples = []
    for ann_path in sorted(ann_dir.glob("*.json")):
        img_path = img_dir / f"{ann_path.stem}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Annotation {ann_path.name} has no matching image at {img_path}")
        examples.append(_parse_form(ann_path, img_path))

    return examples


def summarize(examples: list[FormExample]) -> dict:
    """Corpus statistics, for the dataset slide and as a load-time sanity check."""
    n_links = sum(len(e.links) for e in examples)
    n_entities = sum(len(e.entities) for e in examples)
    label_counts: dict[str, int] = {}
    for ex in examples:
        for ent in ex.entities.values():
            label_counts[ent["label"]] = label_counts.get(ent["label"], 0) + 1
    return {
        "documents": len(examples),
        "words": sum(len(e.words) for e in examples),
        "entities": n_entities,
        "links": n_links,
        "entities_per_label": label_counts,
        "mean_words_per_doc": round(sum(len(e.words) for e in examples) / max(1, len(examples)), 1),
    }
