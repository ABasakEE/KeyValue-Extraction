"""Four-bucket failure taxonomy for cross-template key-value extraction.

This module classifies every prediction error into one of four categories,
following the taxonomy in UNIKIE-BENCH (arXiv:2602.07038) and From Pixels to
Pairs (arXiv:2609.17538):

1. **OCR / recognition errors** — gold entity exists but the underlying token
   text is garbled, fragmented, or contains non-alphabetic noise that makes the
   entity unrecognisable regardless of the model.  Measured by checking whether
   false-negative spans contain tokens with anomalous character distributions.

2. **Layout-association errors** — the model extracted text that *is* a gold
   entity somewhere in the document, but tagged it with the wrong type.  The
   spatial association between key and value failed; the recognition did not.
   This is the "right text, wrong key" bucket.

3. **Field-type / schema errors** — the model predicted an entity type that has
   no gold instances in the document (hallucinated schema), or missed an entity
   type entirely (missing schema).  These are type-level rather than
   span-level errors and point to the model's failure to adapt its label space
   to an unseen template.

4. **Repeated-structure errors** — errors occurring among entity types that
   appear more than once in the document (e.g. multiple "amount" fields in a
   table).  These isolate failures caused by spatial ambiguity within a repeated
   layout pattern, as opposed to the other buckets which cover simpler cases.

Usage::

    from errors import classify_errors, ErrorBreakdown, summarize_errors

    breakdown = classify_errors(
        predictions=[...],      # list[list[int]]  (word-level pred ids)
        references=[...],       # list[list[int]]  (word-level gold ids)
        words=[...],            # list[list[str]]  (word tokens per doc)
        boxes=[...],            # list[list[list[int]]]  (boxes per doc)
        id2label={0: "O", ...},
    )
    print(summarize_errors(breakdown))

The ``ErrorBreakdown`` returned by ``classify_errors`` carries both aggregate
counts and per-document, per-span examples that can be rendered on a slide or
fed into a qualitative analysis.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Span extraction (reused from evaluate.py logic, kept self-contained here
# to avoid circular imports and to add word-index tracking)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Span:
    """An entity mention: type, start index (inclusive), end index (exclusive)."""
    entity_type: str
    start: int
    end: int

    @property
    def indices(self) -> range:
        return range(self.start, self.end)


def _extract_spans_from_tags(tags: list[str]) -> list[Span]:
    """Parse strict IOB2 spans from a tag sequence.

    Returns a list (not a set) to preserve document order, which matters for
    the repeated-structure bucket.
    """
    spans: list[Span] = []
    current_type: str | None = None
    start = 0

    for index, tag in enumerate(list(tags) + ["O"]):
        if tag == "O":
            prefix, tag_type = "O", None
        else:
            prefix, _, tag_type = tag.partition("-")

        if prefix == "B":
            if current_type is not None:
                spans.append(Span(current_type, start, index))
            current_type, start = tag_type, index
        elif prefix == "I":
            if current_type is None or tag_type != current_type:
                if current_type is not None:
                    spans.append(Span(current_type, start, index))
                current_type = None  # strict: discard orphan I-
            # valid continuation: extend
        else:
            if current_type is not None:
                spans.append(Span(current_type, start, index))
            current_type = None

    return spans


# ---------------------------------------------------------------------------
# OCR noise detection
# ---------------------------------------------------------------------------

_ALPHA_RE = re.compile(r"[A-Za-z]")


def _ocr_noise_score(tokens: list[str]) -> float:
    """Heuristic: fraction of characters in `tokens` that are non-alphabetic
    and not common punctuation.

    A high score (>0.5) on a span that was supposed to be a named field
    strongly suggests OCR corruption.  Common punctuation (.,;:/-) is excluded
    because it is legitimate in addresses, dates, and amounts.
    """
    if not tokens:
        return 0.0
    text = " ".join(tokens)
    if not text:
        return 0.0
    noise = 0
    for ch in text:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        # L* = letters, Nd = digits, Pd/Ps/Pe/Po for common punct
        if cat.startswith("L") or cat == "Nd":
            continue
        if ch in ".,;:/-()#@&'\"":
            continue
        noise += 1
    non_space = sum(1 for ch in text if not ch.isspace())
    return noise / max(1, non_space)


def _is_fragmented(tokens: list[str]) -> bool:
    """Detect token fragmentation: many very short tokens that look like
    OCR split a word into individual characters."""
    if len(tokens) < 3:
        return False
    single_char = sum(1 for t in tokens if len(t) == 1)
    return single_char / len(tokens) > 0.6


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

@dataclass
class SpanError:
    """One misclassified span with its bucket assignment."""
    doc_index: int
    span: Span
    kind: str          # "fp" or "fn"
    bucket: str        # one of the four taxonomy buckets
    tokens: list[str]
    detail: str = ""   # human-readable explanation


@dataclass
class ErrorBreakdown:
    """Aggregate and per-instance error classification."""

    # Per-bucket counts (each entry: {"fp": n, "fn": n, "total": n})
    ocr_recognition: dict[str, int] = field(default_factory=lambda: {"fp": 0, "fn": 0, "total": 0})
    layout_association: dict[str, int] = field(default_factory=lambda: {"fp": 0, "fn": 0, "total": 0})
    field_type_schema: dict[str, int] = field(default_factory=lambda: {"fp": 0, "fn": 0, "total": 0})
    repeated_structure: dict[str, int] = field(default_factory=lambda: {"fp": 0, "fn": 0, "total": 0})

    # Total TP / FP / FN for reference
    tp: int = 0
    fp: int = 0
    fn: int = 0

    # Per-instance examples (capped for memory)
    examples: list[SpanError] = field(default_factory=list)
    max_examples: int = 200

    @property
    def total_errors(self) -> int:
        return self.fp + self.fn

    def bucket_counts(self) -> dict[str, dict[str, int]]:
        return {
            "ocr_recognition": dict(self.ocr_recognition),
            "layout_association": dict(self.layout_association),
            "field_type_schema": dict(self.field_type_schema),
            "repeated_structure": dict(self.repeated_structure),
        }

    def bucket_fractions(self) -> dict[str, float]:
        """Fraction of total errors in each bucket."""
        total = self.total_errors or 1
        return {
            name: counts["total"] / total
            for name, counts in self.bucket_counts().items()
        }

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "total_errors": self.total_errors,
            "buckets": self.bucket_counts(),
            "bucket_fractions": {
                k: round(v, 4) for k, v in self.bucket_fractions().items()
            },
        }


def _add_error(breakdown: ErrorBreakdown, error: SpanError) -> None:
    bucket = getattr(breakdown, error.bucket)
    bucket[error.kind] += 1
    bucket["total"] += 1
    if len(breakdown.examples) < breakdown.max_examples:
        breakdown.examples.append(error)


def _spans_overlap(a: Span, b: Span) -> bool:
    """True if two spans share at least one token index."""
    return a.start < b.end and b.start < a.end


def _span_text(span: Span, words: list[str]) -> str:
    return " ".join(words[i] for i in span.indices if i < len(words))


def _classify_fn(
    gold_span: Span,
    pred_spans: list[Span],
    gold_spans: list[Span],
    words: list[str],
    doc_index: int,
) -> SpanError:
    """Classify a false negative (missed gold span) into a bucket."""
    tokens = [words[i] for i in gold_span.indices if i < len(words)]

    # Bucket 1: OCR / recognition — text is garbled
    noise = _ocr_noise_score(tokens)
    fragmented = _is_fragmented(tokens)
    if noise > 0.4 or fragmented:
        detail = f"noise_score={noise:.2f}, fragmented={fragmented}"
        return SpanError(doc_index, gold_span, "fn", "ocr_recognition", tokens, detail)

    # Bucket 2: Layout-association — the model predicted these tokens, but
    # with a different type (right text, wrong key)
    gold_text = _span_text(gold_span, words)
    for ps in pred_spans:
        if _spans_overlap(gold_span, ps) and ps.entity_type != gold_span.entity_type:
            detail = f"gold={gold_span.entity_type}, predicted_as={ps.entity_type}"
            return SpanError(doc_index, gold_span, "fn", "layout_association", tokens, detail)
    # Also check if the gold text appears verbatim in a prediction of a
    # different type (non-overlapping but same words elsewhere)
    for ps in pred_spans:
        if ps.entity_type != gold_span.entity_type:
            pred_text = _span_text(ps, words)
            if pred_text and pred_text == gold_text:
                detail = f"gold={gold_span.entity_type}, predicted_as={ps.entity_type} (text match, different position)"
                return SpanError(doc_index, gold_span, "fn", "layout_association", tokens, detail)

    # Bucket 4: Repeated-structure — this entity type appears >1 time in the
    # document's gold spans, indicating a repeated layout pattern
    same_type_gold = [g for g in gold_spans if g.entity_type == gold_span.entity_type]
    if len(same_type_gold) > 1:
        detail = f"{len(same_type_gold)} instances of {gold_span.entity_type!r} in document"
        return SpanError(doc_index, gold_span, "fn", "repeated_structure", tokens, detail)

    # Bucket 3: Field-type / schema — the model produced no predictions of
    # this type at all (missing schema), or this is a clean miss
    preds_of_type = [ps for ps in pred_spans if ps.entity_type == gold_span.entity_type]
    if not preds_of_type:
        detail = f"no predictions of type {gold_span.entity_type!r} in document"
        return SpanError(doc_index, gold_span, "fn", "field_type_schema", tokens, detail)

    # Default to field_type_schema for remaining clean misses
    detail = f"missed span of type {gold_span.entity_type!r}"
    return SpanError(doc_index, gold_span, "fn", "field_type_schema", tokens, detail)


def _classify_fp(
    pred_span: Span,
    gold_spans: list[Span],
    pred_spans: list[Span],
    words: list[str],
    doc_index: int,
) -> SpanError:
    """Classify a false positive (spurious prediction) into a bucket."""
    tokens = [words[i] for i in pred_span.indices if i < len(words)]

    # Bucket 1: OCR / recognition — predicted from noisy tokens
    noise = _ocr_noise_score(tokens)
    fragmented = _is_fragmented(tokens)
    if noise > 0.4 or fragmented:
        detail = f"noise_score={noise:.2f}, fragmented={fragmented}"
        return SpanError(doc_index, pred_span, "fp", "ocr_recognition", tokens, detail)

    # Bucket 2: Layout-association — overlaps a gold span of a different type
    pred_text = _span_text(pred_span, words)
    for gs in gold_spans:
        if _spans_overlap(pred_span, gs) and gs.entity_type != pred_span.entity_type:
            detail = f"predicted={pred_span.entity_type}, gold={gs.entity_type}"
            return SpanError(doc_index, pred_span, "fp", "layout_association", tokens, detail)
    # Also text-match against gold spans of a different type
    for gs in gold_spans:
        if gs.entity_type != pred_span.entity_type:
            gold_text = _span_text(gs, words)
            if gold_text and gold_text == pred_text:
                detail = f"predicted={pred_span.entity_type}, gold={gs.entity_type} (text match)"
                return SpanError(doc_index, pred_span, "fp", "layout_association", tokens, detail)

    # Bucket 4: Repeated-structure — this predicted type has >1 prediction,
    # suggesting confusion among repeated instances
    same_type_preds = [ps for ps in pred_spans if ps.entity_type == pred_span.entity_type]
    same_type_gold = [gs for gs in gold_spans if gs.entity_type == pred_span.entity_type]
    if len(same_type_preds) > 1 and len(same_type_gold) >= 1:
        detail = f"{len(same_type_preds)} predictions vs {len(same_type_gold)} gold of type {pred_span.entity_type!r}"
        return SpanError(doc_index, pred_span, "fp", "repeated_structure", tokens, detail)

    # Bucket 3: Field-type / schema — predicted a type with no gold instances
    gold_of_type = [gs for gs in gold_spans if gs.entity_type == pred_span.entity_type]
    if not gold_of_type:
        detail = f"hallucinated type {pred_span.entity_type!r} (no gold instances)"
        return SpanError(doc_index, pred_span, "fp", "field_type_schema", tokens, detail)

    # Default: schema error for remaining boundary mismatches
    detail = f"spurious span of type {pred_span.entity_type!r}"
    return SpanError(doc_index, pred_span, "fp", "field_type_schema", tokens, detail)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def classify_errors(
    predictions: list[list[int]],
    references: list[list[int]],
    words: list[list[str]],
    id2label: dict[int, str],
    boxes: list[list[list[int]]] | None = None,
) -> ErrorBreakdown:
    """Classify every prediction error across a corpus into four buckets.

    Parameters
    ----------
    predictions : list of list of int
        Word-level predicted label ids (one list per document). Truncated
        positions use ``-1`` and are skipped.
    references : list of list of int
        Word-level gold label ids (one list per document).
    words : list of list of str
        Tokenised words per document, aligned with ``predictions``/``references``.
    id2label : dict
        Mapping from integer label id to string tag (e.g. ``{0: "O", 1: "B-HEADER", ...}``).
    boxes : optional list of list of [x0, y0, x1, y1]
        Bounding boxes per word, per document.  Reserved for future spatial
        analysis; not used in the current heuristics.

    Returns
    -------
    ErrorBreakdown
        Aggregate bucket counts, per-instance examples, and TP/FP/FN totals.
    """
    breakdown = ErrorBreakdown()

    for doc_idx, (preds, refs, doc_words) in enumerate(zip(predictions, references, words)):
        # Convert ids to tag strings, skipping truncated positions
        pred_tags = [
            id2label.get(p, "O") if p != -1 else "O"
            for p in preds
        ]
        ref_tags = [id2label.get(r, "O") for r in refs]

        # Ensure equal length (truncated predictions may be shorter)
        min_len = min(len(pred_tags), len(ref_tags), len(doc_words))
        pred_tags = pred_tags[:min_len]
        ref_tags = ref_tags[:min_len]
        trimmed_words = doc_words[:min_len]

        pred_spans = _extract_spans_from_tags(pred_tags)
        gold_spans = _extract_spans_from_tags(ref_tags)

        pred_set = set((s.entity_type, s.start, s.end) for s in pred_spans)
        gold_set = set((s.entity_type, s.start, s.end) for s in gold_spans)

        tp_set = pred_set & gold_set
        fp_set = pred_set - gold_set
        fn_set = gold_set - pred_set

        breakdown.tp += len(tp_set)
        breakdown.fp += len(fp_set)
        breakdown.fn += len(fn_set)

        # Classify each false negative
        for et, s, e in fn_set:
            span = Span(et, s, e)
            error = _classify_fn(span, pred_spans, gold_spans, trimmed_words, doc_idx)
            _add_error(breakdown, error)

        # Classify each false positive
        for et, s, e in fp_set:
            span = Span(et, s, e)
            error = _classify_fp(span, gold_spans, pred_spans, trimmed_words, doc_idx)
            _add_error(breakdown, error)

    return breakdown


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize_errors(breakdown: ErrorBreakdown) -> str:
    """Human-readable summary for printing or including on a slide."""
    lines = [
        "ERROR TAXONOMY BREAKDOWN",
        f"  TP: {breakdown.tp}  FP: {breakdown.fp}  FN: {breakdown.fn}  "
        f"total errors: {breakdown.total_errors}",
        "",
    ]

    fracs = breakdown.bucket_fractions()
    for bucket_name, counts in breakdown.bucket_counts().items():
        label = bucket_name.replace("_", " ").title()
        pct = fracs[bucket_name] * 100
        lines.append(
            f"  {label:<25s}  FP: {counts['fp']:>4}  FN: {counts['fn']:>4}  "
            f"total: {counts['total']:>4}  ({pct:5.1f}%)"
        )

    lines.append("")

    # Top examples per bucket
    by_bucket: dict[str, list[SpanError]] = defaultdict(list)
    for ex in breakdown.examples:
        by_bucket[ex.bucket].append(ex)

    for bucket_name in ("ocr_recognition", "layout_association", "field_type_schema", "repeated_structure"):
        examples = by_bucket.get(bucket_name, [])
        if not examples:
            continue
        label = bucket_name.replace("_", " ").title()
        lines.append(f"  {label} examples (up to 3):")
        for ex in examples[:3]:
            text = " ".join(ex.tokens[:10])
            if len(ex.tokens) > 10:
                text += " ..."
            lines.append(
                f"    [{ex.kind.upper()}] doc={ex.doc_index} "
                f"type={ex.span.entity_type!r} [{ex.span.start}:{ex.span.end}] "
                f"text={text!r}"
            )
            if ex.detail:
                lines.append(f"          {ex.detail}")
        lines.append("")

    return "\n".join(lines)


def error_table(breakdown: ErrorBreakdown) -> list[dict]:
    """Return bucket data as a list of dicts suitable for ``pandas.DataFrame``.

    Example::

        import pandas as pd
        df = pd.DataFrame(error_table(breakdown))
    """
    rows = []
    fracs = breakdown.bucket_fractions()
    for name, counts in breakdown.bucket_counts().items():
        rows.append({
            "bucket": name.replace("_", " ").title(),
            "fp": counts["fp"],
            "fn": counts["fn"],
            "total": counts["total"],
            "fraction": round(fracs[name], 4),
        })
    return rows

