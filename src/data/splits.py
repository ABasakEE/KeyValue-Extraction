"""Template-disjoint split construction.

The project's whole claim rests on one property: no document from a held-out
template may influence training in any way — not as a labelled example, not as
an unlabelled one, not through a validation set used for early stopping. This
module builds the splits and, more importantly, *proves* the property holds and
prints the proof so it can go straight onto a slide.

Two regimes, following the VRDU benchmark (Wang, Zhou, Wei, Lee, Tata. VRDU: A
Benchmark for Visually-rich Document Understanding, KDD, 2023.
https://arxiv.org/abs/2211.15421):

* **MTL** (Mixed Template Learning) — train and test drawn from the same
  template pool. This is the "seen" condition.
* **UTL** (Unseen Template Learning) — train and test template sets are
  disjoint. This is the "unseen" condition.

The reported generalization gap is micro-F1(MTL) - micro-F1(UTL), which is only
meaningful if both regimes use the same number of training documents. VRDU
reports gaps of 13-17 points; FormNet at 200 training documents on Registration
Forms scores MTL 90.51 vs UTL 77.29.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, Sequence


class HasTemplate(Protocol):
    """Anything with a document id and a template id can be split."""

    guid: str
    template_id: str


class TemplateLeakageError(AssertionError):
    """Raised when a split would let a held-out template influence training."""


@dataclass
class Split:
    train: list
    val: list
    test: list
    regime: str                     # "MTL" or "UTL"
    held_out_templates: tuple[str, ...]
    fold: int = 0

    def template_sets(self) -> dict[str, set[str]]:
        return {
            "train": {ex.template_id for ex in self.train},
            "val": {ex.template_id for ex in self.val},
            "test": {ex.template_id for ex in self.test},
        }

    def describe(self) -> str:
        sets = self.template_sets()
        lines = [
            f"[{self.regime}] fold {self.fold}  held-out: {', '.join(self.held_out_templates) or '(none)'}",
            f"  train: {len(self.train):>4} docs over {len(sets['train'])} templates {sorted(sets['train'])}",
            f"  val:   {len(self.val):>4} docs over {len(sets['val'])} templates {sorted(sets['val'])}",
            f"  test:  {len(self.test):>4} docs over {len(sets['test'])} templates {sorted(sets['test'])}",
        ]
        return "\n".join(lines)


def assert_no_template_leakage(split: Split, *, strict: bool = True) -> str:
    """Verify the UTL contract and return a human-readable proof.

    For a UTL split, the train and validation template sets must both be
    disjoint from the test template set. Validation matters as much as training:
    selecting a checkpoint on a held-out template is leakage even though no
    gradient flowed from it.

    For an MTL split, overlap is expected and is *not* an error — but we still
    check that no individual document appears in two splits, which would inflate
    the seen-condition score and shrink the measured gap.
    """
    sets = split.template_sets()
    problems: list[str] = []

    train_ids = {ex.guid for ex in split.train}
    val_ids = {ex.guid for ex in split.val}
    test_ids = {ex.guid for ex in split.test}
    for a, b, name_a, name_b in (
        (train_ids, test_ids, "train", "test"),
        (train_ids, val_ids, "train", "val"),
        (val_ids, test_ids, "val", "test"),
    ):
        shared = a & b
        if shared:
            problems.append(
                f"{len(shared)} document(s) appear in both {name_a} and {name_b}: "
                f"{sorted(shared)[:5]}{' ...' if len(shared) > 5 else ''}"
            )

    if split.regime == "UTL":
        train_test = sets["train"] & sets["test"]
        val_test = sets["val"] & sets["test"]
        if train_test:
            problems.append(f"train shares templates with test: {sorted(train_test)}")
        if val_test:
            problems.append(f"val shares templates with test (checkpoint selection leakage): {sorted(val_test)}")

    if problems:
        message = "TEMPLATE LEAKAGE:\n  - " + "\n  - ".join(problems)
        if strict:
            raise TemplateLeakageError(message)
        return message

    if split.regime == "UTL":
        proof = (
            f"LEAKAGE CHECK PASSED [{split.regime} fold {split.fold}]\n"
            f"  train templates ({len(sets['train'])}): {sorted(sets['train'])}\n"
            f"  val   templates ({len(sets['val'])}): {sorted(sets['val'])}\n"
            f"  test  templates ({len(sets['test'])}): {sorted(sets['test'])}\n"
            f"  (train | val) INTERSECT test = {sorted((sets['train'] | sets['val']) & sets['test'])}  <- must be empty\n"
            f"  no document id appears in more than one split"
        )
    else:
        proof = (
            f"LEAKAGE CHECK PASSED [{split.regime} fold {split.fold}]\n"
            f"  template overlap is expected in MTL; checked document-level disjointness only\n"
            f"  shared templates train/test: {sorted(sets['train'] & sets['test'])}\n"
            f"  no document id appears in more than one split"
        )
    return proof


def _take(pool: list, n: int | None) -> list:
    return pool if n is None else pool[:n]


def leave_one_template_out(
    examples: Sequence[HasTemplate],
    *,
    n_train: int | None = None,
    val_fraction: float = 0.15,
    seed: int = 13,
) -> list[Split]:
    """Build one UTL split per template: hold that template out entirely.

    ``n_train`` caps the number of training documents so the MTL and UTL
    conditions can be compared at matched training-set size — without this the
    gap conflates template novelty with training-set size.
    """
    by_template: dict[str, list] = {}
    for ex in examples:
        by_template.setdefault(ex.template_id, []).append(ex)

    if len(by_template) < 2:
        raise ValueError(
            f"Need at least 2 templates for a held-out split, found {len(by_template)}: "
            f"{sorted(by_template)}"
        )

    splits: list[Split] = []
    for fold, held_out in enumerate(sorted(by_template)):
        rng = random.Random(seed + fold)

        train_pool = [ex for tmpl, docs in by_template.items() if tmpl != held_out for ex in docs]
        rng.shuffle(train_pool)

        n_val = max(1, int(len(train_pool) * val_fraction))
        val, remaining = train_pool[:n_val], train_pool[n_val:]

        split = Split(
            train=_take(remaining, n_train),
            val=val,
            test=list(by_template[held_out]),
            regime="UTL",
            held_out_templates=(held_out,),
            fold=fold,
        )
        assert_no_template_leakage(split)
        splits.append(split)

    return splits


def mixed_template_split(
    examples: Sequence[HasTemplate],
    *,
    n_train: int | None = None,
    test_fraction: float = 0.2,
    val_fraction: float = 0.15,
    seed: int = 13,
    n_folds: int = 3,
) -> list[Split]:
    """Build the seen-template control condition.

    Every template appears in train and test. Sampling is stratified per
    template so the test set's template mix mirrors the corpus — an unstratified
    shuffle can accidentally produce a near-UTL test set on a small corpus and
    silently shrink the measured gap.
    """
    by_template: dict[str, list] = {}
    for ex in examples:
        by_template.setdefault(ex.template_id, []).append(ex)

    splits: list[Split] = []
    for fold in range(n_folds):
        rng = random.Random(seed + 100 + fold)
        train_pool, val, test = [], [], []

        for docs in by_template.values():
            shuffled = list(docs)
            rng.shuffle(shuffled)
            n_test = max(1, int(len(shuffled) * test_fraction))
            n_val = max(1, int(len(shuffled) * val_fraction))
            test.extend(shuffled[:n_test])
            val.extend(shuffled[n_test:n_test + n_val])
            train_pool.extend(shuffled[n_test + n_val:])

        rng.shuffle(train_pool)
        split = Split(
            train=_take(train_pool, n_train),
            val=val,
            test=test,
            regime="MTL",
            held_out_templates=(),
            fold=fold,
        )
        assert_no_template_leakage(split)
        splits.append(split)

    return splits


def matched_size_protocol(
    examples: Sequence[HasTemplate],
    *,
    n_train: int,
    seed: int = 13,
) -> dict[str, list[Split]]:
    """The full experiment: MTL and UTL at identical training-set size.

    Returns ``{"MTL": [...], "UTL": [...]}``. Report
    ``mean(F1 over UTL folds)`` against ``mean(F1 over MTL folds)``; the
    difference is the generalization gap.
    """
    utl = leave_one_template_out(examples, n_train=n_train, seed=seed)
    mtl = mixed_template_split(examples, n_train=n_train, seed=seed, n_folds=len(utl))

    # A gap measured at different training sizes is not a gap, it is a
    # sample-efficiency curve. Fail loudly rather than silently reporting one.
    sizes = {len(s.train) for s in utl} | {len(s.train) for s in mtl}
    if len(sizes) > 1:
        raise ValueError(
            f"Training-set sizes differ across regimes ({sorted(sizes)}); the gap would "
            f"conflate template novelty with training-set size. Lower n_train to "
            f"{min(sizes)} or supply more documents."
        )

    return {"MTL": mtl, "UTL": utl}
