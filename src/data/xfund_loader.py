"""XFUND Multilingual Form Understanding Dataset Loader.

Reference:
    Guillaume Jaume, Hazim Kemal Ekenel, Jean-Philippe Thiran.
    XFUND: A Benchmark Dataset for Multilingual Visually Rich Form Understanding,
    ACL Findings 2022. https://github.com/doc-analysis/XFUND

XFUND provides human-annotated forms across 7 languages:
    Chinese (zh), Japanese (ja), Spanish (es), French (fr),
    Italian (it), German (de), and Portuguese (pt).

Each split is packaged as a single JSON file (<lang>.train.json, <lang>.val.json):
    {
      "documents": [
        {
          "uid": "es_001",
          "img": {"fname": "es_001.jpg", "width": 1000, "height": 1000},
          "document": [
            {
              "id": 0,
              "text": "Factura No.:",
              "box": [115, 120, 150, 210],
              "label": "question",
              "words": [{"text": "Factura", "box": [...]}, ...],
              "linking": [[0, 1]]
            }
          ]
        }
      ]
    }

Converts into the standard FormExample dataclass, ensuring 100% compatibility with
the KeyValue-Extraction joint extraction pipeline.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

try:
    from data.funsd_loader import (
        ENTITY_LABELS,
        FormExample,
        ID2LABEL,
        LABEL2ID,
        LABEL_LIST,
        normalize_box,
        summarize,
    )
except ImportError:
    try:
        from src.data.funsd_loader import (
            ENTITY_LABELS,
            FormExample,
            ID2LABEL,
            LABEL2ID,
            LABEL_LIST,
            normalize_box,
            summarize,
        )
    except ImportError:
        from .funsd_loader import (
            ENTITY_LABELS,
            FormExample,
            ID2LABEL,
            LABEL2ID,
            LABEL_LIST,
            normalize_box,
            summarize,
        )

SUPPORTED_LANGUAGES = ("es", "fr", "de", "it", "pt", "zh", "ja")
LATIN_LANGUAGES = ("es", "fr", "de", "it", "pt")


def _parse_xfund_doc(doc: dict[str, Any], lang: str) -> FormExample | None:
    """Parse one XFUND document dictionary into FormExample."""
    uid = doc.get("uid", doc.get("id", f"{lang}_{random.randint(1000, 9999)}"))
    img_meta = doc.get("img", {})
    width = img_meta.get("width", 1000) or 1000
    height = img_meta.get("height", 1000) or 1000
    img_path = img_meta.get("fname", f"{uid}.jpg")

    elements = doc.get("document", doc.get("form", []))
    if not elements:
        return None

    words: list[str] = []
    boxes: list[list[int]] = []
    ner_tags: list[int] = []
    entity_ids: list[int] = []
    entities: dict[int, dict] = {}
    links: set[tuple[int, int]] = set()

    for item in elements:
        label = item.get("label", "other").lower()
        item_words = item.get("words", [])
        clean_words = [w for w in item_words if w.get("text", "").strip()]
        if not clean_words:
            continue

        token_start = len(words)
        for pos, w in enumerate(clean_words):
            w_text = w["text"].strip()
            w_box = w["box"]
            norm_b = normalize_box(w_box, width, height)

            words.append(w_text)
            boxes.append(norm_b)
            entity_ids.append(item["id"])

            if label == "other":
                ner_tags.append(LABEL2ID["O"])
            else:
                prefix = "B" if pos == 0 else "I"
                tag_name = f"{prefix}-{label.upper()}"
                ner_tags.append(LABEL2ID.get(tag_name, LABEL2ID["O"]))

        entities[item["id"]] = {
            "label": label,
            "token_indices": list(range(token_start, len(words))),
            "text": item.get("text", ""),
        }

        for pair in item.get("linking", []):
            if len(pair) == 2:
                links.add((pair[0], pair[1]))

    valid_ent_ids = set(entities)
    kept_links = sorted((s, d) for s, d in links if s in valid_ent_ids and d in valid_ent_ids)

    if not words:
        return None

    return FormExample(
        guid=f"{lang}_{uid}",
        words=words,
        boxes=boxes,
        ner_tags=ner_tags,
        image_path=str(img_path),
        image_size=(width, height),
        entity_ids=entity_ids,
        links=kept_links,
        entities=entities,
    )


def load_xfund_split(
    data_root: str | Path,
    split: str = "train",
    languages: list[str] | tuple[str, ...] = LATIN_LANGUAGES,
) -> list[FormExample]:
    """Load one split across requested languages.

    Args:
        data_root: Directory containing XFUND JSON files (<lang>.train.json, <lang>.val.json).
        split: "train" or "val" / "test".
        languages: List of language codes to load (e.g. ['es', 'fr', 'de']).
    """
    root = Path(data_root)
    examples: list[FormExample] = []

    file_split = "val" if split in ("val", "test") else "train"

    for lang in languages:
        json_path = root / f"{lang}.{file_split}.json"
        if not json_path.exists():
            # Try finding under a language subdirectory: root / lang / <lang>.<split>.json
            alt = root / lang / f"{lang}.{file_split}.json"
            if alt.exists():
                json_path = alt
            else:
                print(f"[!] Warning: XFUND file not found at {json_path}. Skipping language '{lang}'.")
                continue

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        docs = data.get("documents", [data] if "document" in data or "form" in data else [])
        parsed_count = 0
        for doc in docs:
            ex = _parse_xfund_doc(doc, lang)
            if ex is not None:
                examples.append(ex)
                parsed_count += 1

        print(f"[*] Loaded {parsed_count} documents for language '{lang}' ({file_split}).")

    return examples


def load_xfund_dataset(
    data_root: str | Path,
    languages: list[str] | tuple[str, ...] = LATIN_LANGUAGES,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[FormExample], list[FormExample], list[FormExample]]:
    """Load train, validation, and test splits for XFUND.

    Returns:
        (train_examples, val_examples, test_examples)
    """
    train_docs = load_xfund_split(data_root, split="train", languages=languages)
    test_docs = load_xfund_split(data_root, split="val", languages=languages)

    if not train_docs:
        raise FileNotFoundError(
            f"No XFUND training documents found under {data_root}. "
            f"Please verify downloaded JSON files: {list(languages)}"
        )

    # Carve validation split from train
    rng = random.Random(seed)
    shuffled = list(train_docs)
    rng.shuffle(shuffled)
    n_val = max(1, int(val_ratio * len(shuffled)))
    val_split = shuffled[:n_val]
    train_split = shuffled[n_val:]

    return train_split, val_split, test_docs
