"""Metrics: entity F1, linking F1, and the seen-vs-unseen gap.

Entity F1 follows the convention every LayoutLM-family paper uses — strict,
entity-level, computed over BIO chunks with ``seqeval``. A prediction counts
only if the span boundaries *and* the type both match exactly; no partial
credit.

``entity_f1_independent`` reimplements the same metric from scratch. Running
both on one fold and confirming they agree is the plan's metric-sanity check:
``seqeval`` silently tolerates malformed tag sequences (an I- tag with no
preceding B-), and a disagreement between the two implementations is usually the
first sign that subword alignment is wrong.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


@dataclass
class EntityScores:
    precision: float
    recall: float
    f1: float
    support: int
    per_type: dict[str, dict] = field(default_factory=dict)
    truncated_words: int = 0

    def as_dict(self) -> dict:
        return {
            "precision": round(float(self.precision), 4),
            "recall": round(float(self.recall), 4),
            "f1": round(float(self.f1), 4),
            "support": int(self.support),
            "truncated_words": int(self.truncated_words),
            "per_type": {
                k: {
                    "precision": round(float(v["precision"]), 4),
                    "recall": round(float(v["recall"]), 4),
                    "f1": round(float(v["f1"]), 4),
                    "support": int(v["support"]),
                }
                for k, v in self.per_type.items()
            },
        }


def _to_tag_sequences(
    predictions: list[list[int]],
    references: list[list[int]],
    id2label: dict[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Map label ids to tag strings, dropping positions with no prediction.

    A word can lack a prediction when the 512-token cap truncated it. Scoring
    such a word as "O" would quietly reward the model for text it never saw, so
    those positions are excluded from both sequences and counted separately.
    """
    pred_tags, ref_tags = [], []
    for pred_row, ref_row in zip(predictions, references):
        p_row, r_row = [], []
        for p, r in zip(pred_row, ref_row):
            if p == -1:
                continue
            p_row.append(id2label[p])
            r_row.append(id2label[r])
        pred_tags.append(p_row)
        ref_tags.append(r_row)
    return pred_tags, ref_tags


def count_truncated(predictions: list[list[int]]) -> int:
    return sum(1 for row in predictions for p in row if p == -1)


def entity_f1(
    predictions: list[list[int]],
    references: list[list[int]],
    id2label: dict[int, str],
) -> EntityScores:
    """Strict entity-level micro F1 via seqeval, in IOB2 mode."""
    pred_tags, ref_tags = _to_tag_sequences(predictions, references, id2label)

    report = classification_report(
        ref_tags, pred_tags, scheme=IOB2, mode="strict", output_dict=True, zero_division=0
    )
    per_type = {
        key: {
            "precision": round(float(value["precision"]), 4),
            "recall": round(float(value["recall"]), 4),
            "f1": round(float(value["f1-score"]), 4),
            "support": int(value["support"]),
        }
        for key, value in report.items()
        if key not in {"micro avg", "macro avg", "weighted avg"}
    }

    return EntityScores(
        precision=float(precision_score(ref_tags, pred_tags, scheme=IOB2, mode="strict", zero_division=0)),
        recall=float(recall_score(ref_tags, pred_tags, scheme=IOB2, mode="strict", zero_division=0)),
        f1=float(f1_score(ref_tags, pred_tags, scheme=IOB2, mode="strict", zero_division=0)),
        support=int(sum(len(row) for row in ref_tags)),
        per_type=per_type,
        truncated_words=int(count_truncated(predictions)),
    )


def _extract_spans(tags: list[str], *, strict: bool = True) -> set[tuple[str, int, int]]:
    """Pull (type, start, end_exclusive) chunks out of a BIO sequence.

    Two readings, and the difference matters:

    * ``strict=True`` (IOB2, the default, and what ``seqeval`` does in
      ``mode="strict"``): a chunk may only open with ``B-``. An ``I-X`` with no
      compatible open chunk is malformed — it closes any open chunk and belongs
      to no entity.
    * ``strict=False`` (the forgiving CoNLL reading): a stray ``I-X`` opens a new
      chunk of type X.

    Gold sequences from ``funsd_loader`` are always well-formed, so the two agree
    on references. They diverge only on model output, which makes their
    difference a free diagnostic — see ``malformed_tag_rate``.
    """
    spans: set[tuple[str, int, int]] = set()
    current_type: str | None = None
    start = 0

    for index, tag in enumerate(list(tags) + ["O"]):
        if tag == "O":
            prefix, tag_type = "O", None
        else:
            prefix, _, tag_type = tag.partition("-")

        if prefix == "B":
            if current_type is not None:
                spans.add((current_type, start, index))
            current_type, start = tag_type, index
        elif prefix == "I":
            if current_type is None or tag_type != current_type:
                # Malformed continuation.
                if current_type is not None:
                    spans.add((current_type, start, index))
                current_type = None if strict else tag_type
                if not strict:
                    start = index
            # A valid continuation simply extends the open chunk.
        else:  # "O"
            if current_type is not None:
                spans.add((current_type, start, index))
            current_type = None

    return spans


def malformed_tag_rate(
    predictions: list[list[int]],
    id2label: dict[int, str],
) -> dict:
    """How often the model emits an ``I-`` tag that opens a chunk illegally.

    A model that has learned the tagging scheme produces almost none of these.
    A rising rate on held-out templates is evidence the model is guessing at
    entity boundaries rather than recognizing them — one of the concrete signals
    the failure analysis looks for.
    """
    invalid = total = 0
    for row in predictions:
        tags = [id2label[p] for p in row if p != -1]
        current_type: str | None = None
        for tag in tags:
            if tag == "O":
                prefix, tag_type = "O", None
            else:
                prefix, _, tag_type = tag.partition("-")
            if prefix == "I":
                total += 1
                if current_type is None or tag_type != current_type:
                    invalid += 1
            current_type = tag_type if prefix in {"B", "I"} else None
    return {
        "invalid_i_tags": invalid,
        "total_i_tags": total,
        "rate": round(invalid / total, 4) if total else 0.0,
    }


def entity_f1_independent(
    predictions: list[list[int]],
    references: list[list[int]],
    id2label: dict[int, str],
    *,
    strict: bool = True,
) -> EntityScores:
    """The same metric, implemented from scratch, to cross-check seqeval."""
    pred_tags, ref_tags = _to_tag_sequences(predictions, references, id2label)

    tp = fp = fn = 0
    per_type_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for p_row, r_row in zip(pred_tags, ref_tags):
        predicted = _extract_spans(p_row, strict=strict)
        gold = _extract_spans(r_row, strict=strict)
        for span in predicted & gold:
            tp += 1
            per_type_counts[span[0]][0] += 1
        for span in predicted - gold:
            fp += 1
            per_type_counts[span[0]][1] += 1
        for span in gold - predicted:
            fn += 1
            per_type_counts[span[0]][2] += 1

    def prf(t: int, f_pos: int, f_neg: int) -> tuple[float, float, float]:
        precision = t / (t + f_pos) if t + f_pos else 0.0
        recall = t / (t + f_neg) if t + f_neg else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    precision, recall, f1 = prf(tp, fp, fn)
    per_type = {}
    for name, (t, f_pos, f_neg) in per_type_counts.items():
        p, r, f = prf(t, f_pos, f_neg)
        per_type[name] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": t + f_neg}

    return EntityScores(
        precision=precision,
        recall=recall,
        f1=f1,
        support=sum(len(row) for row in ref_tags),
        per_type=per_type,
        truncated_words=count_truncated(predictions),
    )


def cross_check(
    predictions: list[list[int]],
    references: list[list[int]],
    id2label: dict[int, str],
    tolerance: float = 1e-6,
) -> str:
    """Run both implementations and report agreement. Raises on divergence.

    Both sides use the strict IOB2 reading, so this verifies the span-extraction
    and alignment logic rather than a tagging convention. The lenient score is
    reported alongside: a large strict/lenient spread means the model is emitting
    malformed tag sequences.
    """
    a = entity_f1(predictions, references, id2label)
    b = entity_f1_independent(predictions, references, id2label)
    delta = abs(a.f1 - b.f1)
    if delta > tolerance:
        raise AssertionError(
            f"Metric implementations disagree: seqeval F1={a.f1:.6f}, independent F1={b.f1:.6f} "
            f"(delta {delta:.2e}). Usually means subword alignment is wrong, or the two sides "
            f"are using different IOB2 readings (strict vs lenient)."
        )
    lenient = entity_f1_independent(predictions, references, id2label, strict=False)
    malformed = malformed_tag_rate(predictions, id2label)
    return (
        f"METRIC CROSS-CHECK PASSED: seqeval F1={a.f1:.4f} == independent F1={b.f1:.4f} "
        f"(delta {delta:.2e})\n"
        f"  lenient-IOB2 F1={lenient.f1:.4f} (spread {abs(lenient.f1 - a.f1):.4f})\n"
        f"  malformed I- tags: {malformed['invalid_i_tags']}/{malformed['total_i_tags']} "
        f"({malformed['rate']:.1%})"
    )


def linking_f1(
    predicted_links: list[set[tuple[int, int]]],
    gold_links: list[set[tuple[int, int]]],
) -> dict:
    """Exact-match F1 over (key_entity_id, value_entity_id) pairs.

    Strictly harder than entity F1: both endpoints must already be correctly
    segmented and typed before a link can be right. The gap between entity F1
    and linking F1 separates recognition failure from spatial-association
    failure, which is the distinction the project's failure taxonomy turns on.
    """
    tp = fp = fn = 0
    for predicted, gold in zip(predicted_links, gold_links):
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gold_links": tp + fn,
        "predicted_links": tp + fp,
    }


def generalization_gap(mtl_f1: list[float], utl_f1: list[float]) -> dict:
    """The headline number: mean F1 over seen-template folds minus unseen.

    Reported with per-regime standard deviation because a single-seed gap over
    three templates is noise. VRDU reports 13-17 points on Registration Forms
    (FormNet at 200 training docs: MTL 90.51 vs UTL 77.29, a 13.22-point gap) —
    that is the calibration our own number should be read against.
    """
    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def stdev(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        mu = mean(xs)
        return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    mtl_mean, utl_mean = mean(mtl_f1), mean(utl_f1)
    return {
        "mtl_mean_f1": round(mtl_mean, 4),
        "mtl_std": round(stdev(mtl_f1), 4),
        "mtl_folds": len(mtl_f1),
        "utl_mean_f1": round(utl_mean, 4),
        "utl_std": round(stdev(utl_f1), 4),
        "utl_folds": len(utl_f1),
        "gap_f1": round(mtl_mean - utl_mean, 4),
        "relative_drop_pct": round(100 * (mtl_mean - utl_mean) / mtl_mean, 2) if mtl_mean else 0.0,
    }
