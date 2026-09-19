"""VRDU loader.

Reference: Zilong Wang, Yichao Zhou, Wei Wei, Chen-Yu Lee and Sandeep Tata.
VRDU: A Benchmark for Visually-rich Document Understanding, KDD, 2023.
https://arxiv.org/abs/2211.15421 — data at
https://github.com/google-research-datasets/vrdu

On-disk layout (per sub-corpus, e.g. ``registration-form/``)::

    main/dataset.jsonl.gz    one JSON record per document
    main/meta.json           {"entity_name_to_match_func": {...}}
    few_shot-splits/*.json   {"train": [...], "valid": [...], "test": [...]}

Record schema, confirmed by inspecting the released file::

    {"filename": "19410222_DLA Piper US LLP_Amendment_Amendment.pdf",
     "ocr": {"text": "<full document text>",
             "pages": [{"page_id": 0,
                        "dimension": {"height": 2273, "width": 1759},
                        "tokens": [{"bbox": [page, x0, y0, x1, y1],   # 0-1 floats
                                    "segments": [[char_start, char_end]],
                                    "text": "3712\\n"}, ...]}]},
     "annotations": [["registration_num",
                      [["3712\\n", [0, 0.463, 0.328, 0.5, 0.344], [[2380, 2385]]]]]]}

Entity occurrences are located by **character offsets into ``ocr.text``**, and
every OCR token carries its own character range, so annotations are aligned to
tokens by character-range overlap rather than by string matching.

VRDU ships no per-document template field. Two independent ways to recover one
are implemented below, and ``verify_template_recovery`` cross-checks them.

----------------------------------------------------------------------------
IMPORTANT — VRDU's official UTL splits are NOT validation-disjoint
----------------------------------------------------------------------------
Auditing the released ``lv3`` (Unseen Template Learning) splits for
registration-form shows the held-out template appears in **validation** on all
three folds, while being correctly absent from training::

    lv3-unk_Amendment             valid: 88/100 Amendment
    lv3-unk_Dissemination_Report  valid: 100/100 Dissemination_Report
    lv3-unk_Short-Form            valid: 79/100 Short-Form

This is VRDU's design — their UTL condition constrains the *training* set, and
validation is drawn from the target template for checkpoint selection. It is not
a defect in their benchmark.

It is, however, incompatible with this project's stated contract, which forbids
*any* use of held-out-template data including model selection. ``load_splits``
therefore offers both, and makes the choice explicit rather than silent:

* ``protocol="official"`` — VRDU's splits verbatim. Comparable to published
  numbers; violates our contract.
* ``protocol="strict"``  — validation rebuilt from training templates only.
  Honours our contract; not directly comparable to published UTL numbers.

Reporting both, and the difference between them, quantifies what target-template
checkpoint selection is worth.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SUBCORPORA = {
    "registration-form": "FARA",
    "ad-buy-form": "DeepForm",
}

# Split filename grammar, e.g.
#   FARA-lv3-unk_Amendment-train_200-test_300-valid_100-SD_0.json
SPLIT_RE = re.compile(
    r"^(?P<corpus>[A-Za-z]+)-lv(?P<level>[123])-"
    r"(?P<condition>single|mixed_template|unk)(?:_(?P<template>.+?))?-"
    r"train_(?P<n_train>\d+)-test_(?P<n_test>\d+)-valid_(?P<n_valid>\d+)-SD_(?P<seed>\d+)\.json$"
)

LEVEL_TO_REGIME = {"1": "STL", "2": "MTL", "3": "UTL"}


@dataclass
class VrduExample:
    guid: str                       # the PDF filename, which is the split key
    words: list[str]
    boxes: list[list[int]]          # 0-1000, [x0, y0, x1, y1]
    ner_tags: list[int]
    template_id: str
    page_ids: list[int] = field(default_factory=list)
    n_pages: int = 1
    unaligned_entities: int = 0     # annotation occurrences that matched no token


def normalize_template(name: str) -> str:
    """Canonicalise a template name so spaces and underscores agree.

    The corpus writes the same template both ways — ``Dissemination Report`` in
    document filenames, ``Dissemination_Report`` in split filenames.
    """
    return re.sub(r"[\s_]+", "_", name.strip())


def template_from_filename(filename: str, known_templates: list[str]) -> str | None:
    """Recover the template from the document filename's trailing component.

    Validated against the ``lv1`` single-template splits: 1600/1600 documents
    agree, 0 disagree, once spaces are normalised to underscores.
    """
    stem = filename[:-4] if filename.lower().endswith(".pdf") else filename
    normalized = normalize_template(stem)
    # Longest first so "Short-Form" cannot be shadowed by a shorter suffix.
    for template in sorted(known_templates, key=len, reverse=True):
        if normalized.endswith("_" + normalize_template(template)):
            return normalize_template(template)
    return None


def templates_from_lv1_splits(splits_dir: str | Path) -> dict[str, str]:
    """Authoritative filename -> template map, read from the lv1 splits.

    Each ``lv1-single_<T>`` split contains documents of template ``T`` only, so
    the union over all of them labels every document those splits mention.
    """
    splits_dir = Path(splits_dir)
    labels: dict[str, str] = {}
    conflicts: list[str] = []

    for path in sorted(splits_dir.glob("*-lv1-single_*.json")):
        match = SPLIT_RE.match(path.name)
        if not match:
            continue
        template = normalize_template(match.group("template"))
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        for part in ("train", "valid", "test"):
            for filename in payload.get(part, []):
                if labels.get(filename, template) != template:
                    conflicts.append(filename)
                labels[filename] = template

    if conflicts:
        raise ValueError(
            f"{len(conflicts)} document(s) are claimed by more than one lv1 template "
            f"split, so lv1 cannot be treated as authoritative: {conflicts[:3]}"
        )
    return labels


def list_templates(splits_dir: str | Path) -> list[str]:
    templates = set()
    for path in Path(splits_dir).glob("*-lv1-single_*.json"):
        match = SPLIT_RE.match(path.name)
        if match:
            templates.add(normalize_template(match.group("template")))
    return sorted(templates)


def verify_template_recovery(splits_dir: str | Path) -> str:
    """Cross-check the filename heuristic against the lv1 ground truth."""
    truth = templates_from_lv1_splits(splits_dir)
    templates = list_templates(splits_dir)

    agree = disagree = unresolved = 0
    examples: list[str] = []
    for filename, template in truth.items():
        guess = template_from_filename(filename, templates)
        if guess is None:
            unresolved += 1
            if len(examples) < 3:
                examples.append(f"unresolved: {filename}")
        elif guess == template:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 3:
                examples.append(f"truth={template} guess={guess}: {filename}")

    if disagree or unresolved:
        raise ValueError(
            "Filename-based template recovery is unreliable "
            f"({agree} agree, {disagree} disagree, {unresolved} unresolved). "
            f"Examples: {examples}"
        )
    return (
        f"TEMPLATE RECOVERY VERIFIED: {agree}/{len(truth)} documents agree between the "
        f"lv1 ground truth and the filename heuristic, 0 disagreements.\n"
        f"  templates: {templates}"
    )


def _entity_labels(meta_path: str | Path) -> list[str]:
    with Path(meta_path).open(encoding="utf-8") as fh:
        meta = json.load(fh)
    return sorted(meta["entity_name_to_match_func"])


def build_label_maps(meta_path: str | Path) -> tuple[list[str], dict[str, int], dict[int, str]]:
    labels = ["O"]
    for entity in _entity_labels(meta_path):
        labels += [f"B-{entity}", f"I-{entity}"]
    label2id = {label: i for i, label in enumerate(labels)}
    return labels, label2id, {i: label for label, i in label2id.items()}


def _token_char_range(token: dict) -> tuple[int, int]:
    segments = token.get("segments") or []
    if not segments:
        return (-1, -1)
    starts = [s for s, _ in segments]
    ends = [e for _, e in segments]
    return (min(starts), max(ends))


def _scale_box(bbox: list[float]) -> tuple[int, list[int]]:
    """VRDU boxes are ``[page, x0, y0, x1, y1]`` with coordinates in 0-1.

    Returns the page index and the box scaled to the 0-1000 range every
    LayoutLM-family model expects.
    """
    page = int(bbox[0])
    x0, y0, x1, y1 = bbox[1:5]
    scaled = [int(round(1000 * v)) for v in (x0, y0, x1, y1)]
    scaled = [max(0, min(1000, v)) for v in scaled]
    # Guard against inverted boxes, which the embedding lookup would accept
    # silently while encoding nonsense geometry.
    if scaled[2] < scaled[0]:
        scaled[0], scaled[2] = scaled[2], scaled[0]
    if scaled[3] < scaled[1]:
        scaled[1], scaled[3] = scaled[3], scaled[1]
    return page, scaled


def _parse_record(record: dict, label2id: dict[str, int], templates: list[str]) -> VrduExample | None:
    words: list[str] = []
    boxes: list[list[int]] = []
    page_ids: list[int] = []
    char_ranges: list[tuple[int, int]] = []

    for page in record["ocr"]["pages"]:
        for token in page.get("tokens", []):
            text = (token.get("text") or "").strip()
            if not text:
                continue
            page_index, box = _scale_box(token["bbox"])
            words.append(text)
            boxes.append(box)
            page_ids.append(page_index)
            char_ranges.append(_token_char_range(token))

    if not words:
        return None

    tags = [label2id["O"]] * len(words)
    unaligned = 0

    for entity_name, occurrences in record.get("annotations", []):
        begin_id = label2id.get(f"B-{entity_name}")
        inside_id = label2id.get(f"I-{entity_name}")
        if begin_id is None:
            continue  # entity absent from meta.json; ignore rather than crash

        for occurrence in occurrences:
            # occurrence == [text, bbox, [[char_start, char_end], ...]]
            spans = occurrence[2] if len(occurrence) > 2 else []
            matched: list[int] = []
            for span in spans:
                start, end = span[0], span[1]
                for index, (t_start, t_end) in enumerate(char_ranges):
                    if t_start < 0:
                        continue
                    if t_start < end and t_end > start:   # half-open overlap
                        matched.append(index)

            if not matched:
                unaligned += 1
                continue

            # Assign in document order, first token B- and the rest I-.
            # Tokens already claimed by another entity keep their first label:
            # overlapping gold entities are rare but must not silently reassign.
            for position, index in enumerate(sorted(set(matched))):
                if tags[index] != label2id["O"]:
                    continue
                tags[index] = begin_id if position == 0 else inside_id

    template = template_from_filename(record["filename"], templates)
    if template is None:
        return None

    return VrduExample(
        guid=record["filename"],
        words=words,
        boxes=boxes,
        ner_tags=tags,
        template_id=template,
        page_ids=page_ids,
        n_pages=len(record["ocr"]["pages"]),
        unaligned_entities=unaligned,
    )


def load_vrdu(root: str | Path, subcorpus: str = "registration-form") -> tuple[list[VrduExample], list[str], dict[int, str]]:
    """Load one VRDU sub-corpus.

    ``root`` is the checkout of the vrdu repository, so that
    ``<root>/<subcorpus>/main/dataset.jsonl.gz`` exists.
    """
    root = Path(root)
    base = root / subcorpus
    data_path = base / "main" / "dataset.jsonl.gz"
    meta_path = base / "main" / "meta.json"
    splits_dir = base / "few_shot-splits"

    for path in (data_path, meta_path, splits_dir):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Clone https://github.com/google-research-datasets/vrdu "
                f"so that {base}/main/ and {base}/few_shot-splits/ exist."
            )

    labels, label2id, id2label = build_label_maps(meta_path)
    templates = list_templates(splits_dir)

    examples: list[VrduExample] = []
    skipped = 0
    with gzip.open(data_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            example = _parse_record(json.loads(line), label2id, templates)
            if example is None:
                skipped += 1
            else:
                examples.append(example)

    if skipped:
        print(f"  note: skipped {skipped} record(s) with no usable tokens or no recoverable template")
    return examples, labels, id2label


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

@dataclass
class OfficialSplit:
    name: str
    regime: str                     # STL | MTL | UTL
    held_out_template: str | None
    n_train: int
    seed: int
    train: list[VrduExample]
    val: list[VrduExample]
    test: list[VrduExample]
    protocol: str = "official"
    moved_to_train_pool: int = 0    # validation docs dropped by the strict protocol

    def template_sets(self) -> dict[str, set[str]]:
        return {
            "train": {e.template_id for e in self.train},
            "val": {e.template_id for e in self.val},
            "test": {e.template_id for e in self.test},
        }

    def describe(self) -> str:
        s = self.template_sets()
        lines = [f"[{self.regime}/{self.protocol}] {self.name}"]
        for part, items in (("train", self.train), ("val", self.val), ("test", self.test)):
            lines.append(f"  {part:<5} n={len(items):>4}  templates={sorted(s[part])}")
        if self.protocol == "strict" and self.moved_to_train_pool:
            lines.append(f"  (strict protocol removed {self.moved_to_train_pool} held-out-template doc(s) from validation)")
        return "\n".join(lines)


def find_split_files(
    splits_dir: str | Path,
    *,
    regime: str,
    n_train: int,
    seed: int,
    held_out: str | None = None,
) -> list[Path]:
    """Locate official split files matching a regime, training size and seed."""
    wanted_level = {v: k for k, v in LEVEL_TO_REGIME.items()}[regime]
    found = []
    for path in sorted(Path(splits_dir).glob("*.json")):
        match = SPLIT_RE.match(path.name)
        if not match or match.group("level") != wanted_level:
            continue
        if int(match.group("n_train")) != n_train or int(match.group("seed")) != seed:
            continue
        template = match.group("template")
        if held_out is not None and (template is None or normalize_template(template) != normalize_template(held_out)):
            continue
        found.append(path)
    return found


def load_splits(
    examples: list[VrduExample],
    splits_dir: str | Path,
    *,
    regime: str,
    n_train: int,
    seed: int,
    protocol: str = "strict",
) -> list[OfficialSplit]:
    """Materialise official split files into example lists.

    ``protocol="strict"`` additionally removes any held-out-template document
    from validation, honouring this project's zero-same-template contract. See
    the module docstring for why that differs from VRDU's own definition.
    """
    if protocol not in {"official", "strict"}:
        raise ValueError(f"protocol must be 'official' or 'strict', got {protocol!r}")

    by_id = {e.guid: e for e in examples}
    splits: list[OfficialSplit] = []

    for path in find_split_files(splits_dir, regime=regime, n_train=n_train, seed=seed):
        match = SPLIT_RE.match(path.name)
        held_out = match.group("template")
        held_out = normalize_template(held_out) if held_out else None

        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)

        parts = {
            part: [by_id[name] for name in payload.get(part, []) if name in by_id]
            for part in ("train", "valid", "test")
        }

        moved = 0
        if protocol == "strict" and regime == "UTL" and held_out is not None:
            kept = [e for e in parts["valid"] if e.template_id != held_out]
            moved = len(parts["valid"]) - len(kept)
            if not kept:
                # Every validation document belonged to the held-out template, so
                # carve a replacement out of train rather than run without one.
                carve = max(1, len(parts["train"]) // 10)
                kept = parts["train"][:carve]
                parts["train"] = parts["train"][carve:]
            parts["valid"] = kept

        splits.append(
            OfficialSplit(
                name=path.stem,
                regime=regime,
                held_out_template=held_out,
                n_train=n_train,
                seed=seed,
                train=parts["train"],
                val=parts["valid"],
                test=parts["test"],
                protocol=protocol,
                moved_to_train_pool=moved,
            )
        )

    return splits


def assert_no_template_leakage(split: OfficialSplit, *, strict: bool = True) -> str:
    """Verify the UTL contract on a materialised official split."""
    sets = split.template_sets()
    problems: list[str] = []

    train_ids = {e.guid for e in split.train}
    val_ids = {e.guid for e in split.val}
    test_ids = {e.guid for e in split.test}
    for a, b, name_a, name_b in (
        (train_ids, test_ids, "train", "test"),
        (train_ids, val_ids, "train", "val"),
        (val_ids, test_ids, "val", "test"),
    ):
        shared = a & b
        if shared:
            problems.append(f"{len(shared)} document(s) in both {name_a} and {name_b}")

    if split.regime == "UTL":
        if sets["train"] & sets["test"]:
            problems.append(f"train shares templates with test: {sorted(sets['train'] & sets['test'])}")
        if sets["val"] & sets["test"]:
            problems.append(
                f"val shares templates with test (checkpoint-selection leakage): "
                f"{sorted(sets['val'] & sets['test'])}"
            )

    if problems:
        message = f"TEMPLATE LEAKAGE [{split.name}]:\n  - " + "\n  - ".join(problems)
        if strict:
            raise AssertionError(message)
        return message

    return (
        f"LEAKAGE CHECK PASSED [{split.regime}/{split.protocol}] {split.name}\n"
        f"  train templates: {sorted(sets['train'])}\n"
        f"  val   templates: {sorted(sets['val'])}\n"
        f"  test  templates: {sorted(sets['test'])}\n"
        f"  (train | val) INTERSECT test = "
        f"{sorted((sets['train'] | sets['val']) & sets['test'])}  <- must be empty for UTL"
    )


def summarize(examples: list[VrduExample]) -> dict:
    by_template: dict[str, int] = {}
    for e in examples:
        by_template[e.template_id] = by_template.get(e.template_id, 0) + 1
    total_words = sum(len(e.words) for e in examples)
    return {
        "documents": len(examples),
        "templates": by_template,
        "words": total_words,
        "mean_words_per_doc": round(total_words / max(1, len(examples)), 1),
        "multi_page_docs": sum(1 for e in examples if e.n_pages > 1),
        "docs_with_unaligned_entities": sum(1 for e in examples if e.unaligned_entities),
        "total_unaligned_entity_occurrences": sum(e.unaligned_entities for e in examples),
    }
