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

import sys

from evaluate import cross_check, entity_f1, generalization_gap, malformed_tag_rate
from models.token_clf import TokenClassifier
from tqdm.auto import tqdm

TQDM_KWARGS = {
    "file": sys.stdout,
    "ncols": 88,
    "mininterval": 0.2,
}


def json_serializable(obj):
    """Fallback serializer for json.dumps to handle numpy/torch scalars."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


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
def predict(
    classifier: TokenClassifier,
    loader: DataLoader,
    device: torch.device,
    desc: str | None = None,
):
    classifier.model.eval()
    predictions: list[list[int]] = []
    references: list[list[int]] = []

    it = tqdm(loader, desc=desc, leave=False, **TQDM_KWARGS) if desc else loader
    for encoded, batch in it:
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
    checkpoint_dir: Path | None = None,
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
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        classifier.model.train()
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:>2}/{config.epochs}",
            leave=True,
            unit="b",
            **TQDM_KWARGS,
        )
        for encoded, _ in pbar:
            inputs = {k: v.to(device) for k, v in encoded.inputs.items()}
            loss = classifier.model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg_loss = epoch_loss / max(1, len(train_loader))
        val_pred, val_ref = predict(classifier, val_loader, device)
        val_f1 = entity_f1(val_pred, val_ref, classifier.model.config.id2label).f1
        history.append({"epoch": epoch, "train_loss": round(avg_loss, 4), "val_f1": round(val_f1, 4)})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in classifier.model.state_dict().items()}
            if checkpoint_dir is not None:
                torch.save(best_state, checkpoint_dir / "best_model.pt")
                meta = {
                    "epoch": epoch,
                    "best_val_f1": round(best_val_f1, 4),
                    "train_loss": round(avg_loss, 4),
                }
                (checkpoint_dir / "best_meta.json").write_text(
                    json.dumps(meta, indent=2, default=json_serializable), encoding="utf-8"
                )

        pbar.set_postfix(loss=f"{avg_loss:.3f}", val_f1=f"{val_f1:.3f}", best=f"{best_val_f1:.3f}")
        pbar.close()

    if best_state is not None:
        classifier.model.load_state_dict(best_state)

    test_pred, test_ref = predict(classifier, test_loader, device, desc="Test eval")
    id2label = classifier.model.config.id2label
    scores = entity_f1(test_pred, test_ref, id2label)

    res = {
        "test": scores.as_dict(),
        "best_val_f1": round(best_val_f1, 4),
        "history": history,
        "malformed_tags": malformed_tag_rate(test_pred, id2label),
        "cross_check": cross_check(test_pred, test_ref, id2label),
    }

    if checkpoint_dir is not None:
        (checkpoint_dir / "result.json").write_text(
            json.dumps(res, indent=2, default=json_serializable), encoding="utf-8"
        )
        (checkpoint_dir / "history.json").write_text(
            json.dumps(history, indent=2, default=json_serializable), encoding="utf-8"
        )

    return res


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
            run_dir = output_dir / f"seed_{seed}"
            result_file = run_dir / "result.json"

            if result_file.exists():
                print(f"\n=== Seed {seed} (resuming from checkpoint: {result_file}) ===", flush=True)
                try:
                    cached = json.loads(result_file.read_text(encoding="utf-8"))
                    print(f"  Loaded existing result: Test F1 {cached['test']['f1']:.4f}", flush=True)
                    runs.append(cached)
                    continue
                except Exception as e:
                    print(f"  Warning: Failed reading {result_file} ({e}), re-running Seed {seed}...", flush=True)

            print(f"\n=== Seed {seed} ===", flush=True)
            set_seed(seed)
            classifier = TokenClassifier(config.backbone, len(label_list), id2label, config.max_length)
            result = train_one(classifier, train_ex, val_ex, test_ex, config, device, checkpoint_dir=run_dir)
            result["seed"] = seed
            print(f"Seed {seed} -> Test F1: {result['test']['f1']:.4f}  |  {result['cross_check']}", flush=True)
            runs.append(result)

            # Incremental checkpoint of metrics.json after each seed completes
            f1s = [r["test"]["f1"] for r in runs]
            summary = {
                "config": vars(config) | {"seeds": list(config.seeds)},
                "runs": runs,
                "mean_f1": round(sum(f1s) / len(f1s), 4),
                "std_f1": round((sum((x - sum(f1s) / len(f1s)) ** 2 for x in f1s) / max(1, len(f1s) - 1)) ** 0.5, 4) if len(f1s) > 1 else 0.0,
            }
            out = output_dir / "metrics.json"
            out.write_text(json.dumps(summary, indent=2, default=json_serializable), encoding="utf-8")

        f1s = [r["test"]["f1"] for r in runs]
        summary = {
            "config": vars(config) | {"seeds": list(config.seeds)},
            "runs": runs,
            "mean_f1": round(sum(f1s) / len(f1s), 4),
            "std_f1": round((sum((x - sum(f1s) / len(f1s)) ** 2 for x in f1s) / max(1, len(f1s) - 1)) ** 0.5, 4) if len(f1s) > 1 else 0.0,
        }
        out = output_dir / "metrics.json"
        out.write_text(json.dumps(summary, indent=2, default=json_serializable), encoding="utf-8")
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
                print(f"\n{proof}", flush=True)

                for seed in config.seeds:
                    tag = f"{regime}/fold{fold_idx}/seed{seed}"
                    run_dir = output_dir / f"{regime}_fold{fold_idx}_seed{seed}"
                    result_file = run_dir / "result.json"

                    if result_file.exists():
                        print(f"\n=== {tag} (resuming from checkpoint: {result_file}) ===", flush=True)
                        try:
                            cached = json.loads(result_file.read_text(encoding="utf-8"))
                            print(f"  Loaded existing result: Test F1 {cached['test']['f1']:.4f}", flush=True)
                            all_results[regime].append(cached)
                            continue
                        except Exception as e:
                            print(f"  Warning: Failed reading {result_file} ({e}), re-running {tag}...", flush=True)

                    print(f"\n=== {tag} ===", flush=True)
                    print(split.describe(), flush=True)
                    set_seed(seed)

                    classifier = TokenClassifier(
                        config.backbone, len(label_list), id2label, config.max_length
                    )
                    result = train_one(
                        classifier, split.train, split.val, split.test, config, device, checkpoint_dir=run_dir
                    )
                    result["seed"] = seed
                    result["fold"] = fold_idx
                    result["regime"] = regime
                    result["held_out"] = list(split.held_out_templates)
                    print(f"{tag} -> Test F1: {result['test']['f1']:.4f}  |  {result['cross_check']}", flush=True)
                    all_results[regime].append(result)

                    # Incremental checkpoint of metrics.json after each fold/seed completes
                    mtl_f1s = [r["test"]["f1"] for r in all_results["MTL"]]
                    utl_f1s = [r["test"]["f1"] for r in all_results["UTL"]]
                    gap = generalization_gap(mtl_f1s, utl_f1s) if (mtl_f1s and utl_f1s) else {}

                    summary = {
                        "config": vars(config) | {"seeds": list(config.seeds)},
                        "protocol": "matched",
                        "n_train": n,
                        "MTL": all_results["MTL"],
                        "UTL": all_results["UTL"],
                        "gap": gap,
                    }
                    out = output_dir / "metrics.json"
                    out.write_text(json.dumps(summary, indent=2, default=json_serializable), encoding="utf-8")

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
        out.write_text(json.dumps(summary, indent=2, default=json_serializable), encoding="utf-8")
        print(f"\n{'=' * 60}")
        print(f"MTL mean F1: {gap['mtl_mean_f1']:.4f} +/- {gap['mtl_std']:.4f}")
        print(f"UTL mean F1: {gap['utl_mean_f1']:.4f} +/- {gap['utl_std']:.4f}")
        print(f"GAP (MTL - UTL): {gap['gap_f1']:.4f}  ({gap['relative_drop_pct']:.1f}% relative drop)")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
