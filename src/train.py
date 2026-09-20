"""Config-driven fine-tuning for the token-classification track.

Two entry points matching the plan's two pilot tracks::

    # Track A — pipeline sanity on FUNSD's standard split
    python src/train.py --config configs/funsd_lilt.yaml

    # Track B — the headline, MTL vs UTL at matched training-set size
    python src/train.py --config configs/vrdu_lilt.yaml --protocol matched

Track A's number validates the code path and nothing more. FUNSD's train and
test sets share 16% of their templates (Laatiri et al., ICDAR 2023,
arXiv:2304.14936) and its block-level annotation leaks entity boundaries through
shared coordinates (Zhang et al., arXiv:2402.02379), so a high FUNSD F1 is not
evidence of cross-template generalization and must not be presented as such.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from evaluate import cross_check, entity_f1, generalization_gap, malformed_tag_rate
from models.token_clf import TokenClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    backbone: str = "lilt"
    dataset: str = "funsd"
    data_root: str = "data/funsd/dataset"
    epochs: int = 20
    batch_size: int = 4
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    seeds: tuple[int, ...] = (13, 21, 42)
    subcorpus: str = "registration-form"
    n_train: int | None = None
    output_dir: str = "results/run"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if "seeds" in raw:
            raw["seeds"] = tuple(raw["seeds"])
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}. Known keys: {sorted(known)}")
        return cls(**raw)


class FormDataset(Dataset):
    """Wraps parsed documents; tokenization happens in the collator."""

    def __init__(self, examples: list):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


def make_collator(classifier: TokenClassifier):
    def collate(batch: list):
        encoded = classifier.encode(
            [ex.words for ex in batch],
            [ex.boxes for ex in batch],
            [ex.ner_tags for ex in batch],
        )
        return encoded, batch

    return collate


@torch.no_grad()
def predict(classifier: TokenClassifier, loader: DataLoader, device: torch.device):
    classifier.model.eval()
    predictions: list[list[int]] = []
    references: list[list[int]] = []

    for encoded, batch in loader:
        inputs = {k: v.to(device) for k, v in encoded.inputs.items() if k != "labels"}
        logits = classifier.model(**inputs).logits.cpu()
        for i, example in enumerate(batch):
            predictions.append(
                classifier.first_subword_predictions(logits[i], encoded.word_ids[i], len(example.words))
            )
            references.append(example.ner_tags)

    return predictions, references


def train_one(
    classifier: TokenClassifier,
    train_examples: list,
    val_examples: list,
    test_examples: list,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    """Fine-tune once and return test metrics for the best-validation checkpoint."""
    collate = make_collator(classifier)
    train_loader = DataLoader(FormDataset(train_examples), batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(FormDataset(val_examples), batch_size=config.batch_size, collate_fn=collate)
    test_loader = DataLoader(FormDataset(test_examples), batch_size=config.batch_size, collate_fn=collate)

    classifier.model.to(device)
    optimizer = torch.optim.AdamW(classifier.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, total_steps=total_steps, pct_start=config.warmup_ratio, anneal_strategy="linear"
    )

    best_val_f1, best_state, history = -1.0, None, []

    for epoch in range(config.epochs):
        classifier.model.train()
        epoch_loss = 0.0
        for encoded, _ in train_loader:
            inputs = {k: v.to(device) for k, v in encoded.inputs.items()}
            loss = classifier.model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        val_pred, val_ref = predict(classifier, val_loader, device)
        val_f1 = entity_f1(val_pred, val_ref, classifier.model.config.id2label).f1
        history.append({"epoch": epoch, "train_loss": round(epoch_loss / max(1, len(train_loader)), 4), "val_f1": round(val_f1, 4)})
        print(f"  epoch {epoch:>2}  loss {history[-1]['train_loss']:.4f}  val_f1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in classifier.model.state_dict().items()}

    if best_state is not None:
        classifier.model.load_state_dict(best_state)

    test_pred, test_ref = predict(classifier, test_loader, device)
    id2label = classifier.model.config.id2label
    scores = entity_f1(test_pred, test_ref, id2label)

    return {
        "test": scores.as_dict(),
        "best_val_f1": round(best_val_f1, 4),
        "history": history,
        "malformed_tags": malformed_tag_rate(test_pred, id2label),
        "cross_check": cross_check(test_pred, test_ref, id2label),
    }


def load_dataset(config: TrainConfig):
    if config.dataset == "funsd":
        from data.funsd_loader import ID2LABEL, LABEL_LIST, load_funsd_split, summarize

        train = load_funsd_split(config.data_root, "train")
        test = load_funsd_split(config.data_root, "test")
        print("FUNSD train:", json.dumps(summarize(train)))
        print("FUNSD test: ", json.dumps(summarize(test)))
        # FUNSD ships no dev set; carve one out of train so checkpoint selection
        # never touches test.
        rng = random.Random(13)
        shuffled = list(train)
        rng.shuffle(shuffled)
        n_val = max(1, int(0.15 * len(shuffled)))
        return shuffled[n_val:], shuffled[:n_val], test, LABEL_LIST, ID2LABEL

    if config.dataset == "vrdu":
        from data.vrdu_loader import (
            assert_no_template_leakage as vrdu_leak_check,
            load_splits,
            load_vrdu,
            summarize as vrdu_summarize,
            verify_template_recovery,
        )

        root = Path(config.data_root)
        examples, label_list, id2label = load_vrdu(root, config.subcorpus)
        print("VRDU:", json.dumps(vrdu_summarize(examples)))

        splits_dir = root / config.subcorpus / "few_shot-splits"
        print(verify_template_recovery(splits_dir))

        # For the "standard" protocol path, use the first official UTL split
        # as a convenience default.  The "matched" protocol path in main()
        # handles the full MTL-vs-UTL comparison via splits.matched_size_protocol.
        n = config.n_train or 200
        official = load_splits(
            examples, splits_dir, regime="UTL", n_train=n, seed=0, protocol="strict",
        )
        if not official:
            raise FileNotFoundError(
                f"No UTL split files found in {splits_dir} with n_train={n}, seed=0. "
                f"Check that the VRDU few_shot-splits directory is populated."
            )
        split = official[0]
        print(split.describe())
        print(vrdu_leak_check(split))
        return split.train, split.val, split.test, label_list, id2label

    raise NotImplementedError(f"Unknown dataset {config.dataset!r}. Supported: 'funsd', 'vrdu'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", choices=["standard", "matched"], default="standard",
                        help="standard = the dataset's own split; matched = MTL vs UTL at equal train size")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  backbone: {config.backbone}  dataset: {config.dataset}")
    if device.type == "cpu":
        print("WARNING: no GPU visible. Fine-tuning on CPU will take hours; this is a Kaggle/Colab job.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.protocol == "standard":
        train_ex, val_ex, test_ex, label_list, id2label = load_dataset(config)
        if config.n_train:
            train_ex = train_ex[: config.n_train]

        runs = []
        for seed in config.seeds:
            print(f"\n=== seed {seed} ===")
            set_seed(seed)
            classifier = TokenClassifier(config.backbone, len(label_list), id2label, config.max_length)
            result = train_one(classifier, train_ex, val_ex, test_ex, config, device)
            result["seed"] = seed
            print(result["cross_check"])
            print(f"  test F1 {result['test']['f1']:.4f}")
            runs.append(result)

        f1s = [r["test"]["f1"] for r in runs]
        summary = {
            "config": vars(config) | {"seeds": list(config.seeds)},
            "runs": runs,
            "mean_f1": round(sum(f1s) / len(f1s), 4),
            "std_f1": round((sum((x - sum(f1s) / len(f1s)) ** 2 for x in f1s) / max(1, len(f1s) - 1)) ** 0.5, 4),
        }
        out = output_dir / "metrics.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nmean test F1 {summary['mean_f1']:.4f} +/- {summary['std_f1']:.4f}  ->  {out}")
    else:
        # --protocol matched: the headline experiment.
        # Train under both MTL and UTL at identical training-set size, measure
        # the generalization gap.
        if config.dataset != "vrdu":
            raise ValueError(
                f"--protocol matched requires a template-aware dataset (vrdu), "
                f"got {config.dataset!r}."
            )

        from data.splits import assert_no_template_leakage, matched_size_protocol
        from data.vrdu_loader import load_vrdu, summarize as vrdu_summarize

        root = Path(config.data_root)
        examples, label_list, id2label = load_vrdu(root, config.subcorpus)
        print("VRDU:", json.dumps(vrdu_summarize(examples)))

        n = config.n_train or 200
        protocol = matched_size_protocol(examples, n_train=n, seed=config.seeds[0])

        all_results: dict[str, list[dict]] = {"MTL": [], "UTL": []}

        for regime in ("MTL", "UTL"):
            folds = protocol[regime]
            for fold_idx, split in enumerate(folds):
                proof = assert_no_template_leakage(split)
                print(f"\n{proof}")

                for seed in config.seeds:
                    tag = f"{regime}/fold{fold_idx}/seed{seed}"
                    print(f"\n=== {tag} ===")
                    print(split.describe())
                    set_seed(seed)

                    classifier = TokenClassifier(
                        config.backbone, len(label_list), id2label, config.max_length
                    )
                    result = train_one(
                        classifier, split.train, split.val, split.test, config, device,
                    )
                    result["seed"] = seed
                    result["fold"] = fold_idx
                    result["regime"] = regime
                    result["held_out"] = list(split.held_out_templates)
                    print(result["cross_check"])
                    print(f"  test F1 {result['test']['f1']:.4f}")
                    all_results[regime].append(result)

        mtl_f1s = [r["test"]["f1"] for r in all_results["MTL"]]
        utl_f1s = [r["test"]["f1"] for r in all_results["UTL"]]
        gap = generalization_gap(mtl_f1s, utl_f1s)

        summary = {
            "config": vars(config) | {"seeds": list(config.seeds)},
            "protocol": "matched",
            "n_train": n,
            "MTL": all_results["MTL"],
            "UTL": all_results["UTL"],
            "gap": gap,
        }
        out = output_dir / "metrics.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n{'=' * 60}")
        print(f"MTL mean F1: {gap['mtl_mean_f1']:.4f} +/- {gap['mtl_std']:.4f}")
        print(f"UTL mean F1: {gap['utl_mean_f1']:.4f} +/- {gap['utl_std']:.4f}")
        print(f"GAP (MTL - UTL): {gap['gap_f1']:.4f}  ({gap['relative_drop_pct']:.1f}% relative drop)")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
