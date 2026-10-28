"""Smoke test runner for KeyValue-Extraction experiments and ablation studies.

Verifies that model instantiation, data collation, forward/backward passes,
and prediction pipelines execute error-free for each experiment configuration
before launching long-running GPU training jobs.
"""

from __future__ import annotations

import argparse
import gc
import tempfile
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.vrdu_loader import VrduExample, chunk_vrdu_examples
from models.dann import compute_dann_alpha
from train import FormDataset, TrainConfig, build_model, dump_test_entities, make_collator, predict

sys.path.insert(0, str(ROOT_DIR / "scripts"))
from analyze_entity_errors import analyze


def create_smoke_examples(is_cpu: bool = False) -> list[VrduExample]:
    """Create 4 mock VRDU documents with multi-page structure and different template IDs."""
    examples = []
    templates = ["Amendment", "Dissemination_Report", "Short-Form", "Amendment"]
    for i in range(4):
        n_words = (30 if i % 2 == 0 else 80) if is_cpu else (60 if i % 2 == 0 else 400)
        words = [f"word_{k}" for k in range(n_words)]
        boxes = [[(k * 15) % 800, (k * 10) % 900, ((k * 15) % 800) + 50, ((k * 10) % 900) + 20] for k in range(n_words)]
        ner_tags = [(k % 7) for k in range(n_words)]
        ex = VrduExample(
            guid=f"smoke_doc_{i}.pdf",
            words=words,
            boxes=boxes,
            ner_tags=ner_tags,
            template_id=templates[i],
        )
        examples.append(ex)
    return examples


def run_smoke_test(config_path: str | Path, device: torch.device) -> bool:
    """Run a 1-step forward+backward smoke test on a single configuration."""
    print(f"\n{'='*70}")
    print(f"[*] RUNNING SMOKE TEST: {config_path}")
    print(f"{'='*70}")

    config = TrainConfig.from_yaml(config_path)
    if device.type == "cpu":
        config.max_length = min(config.max_length, 48)
        config.window_size = min(getattr(config, "window_size", 32), 32)
        config.window_overlap = min(getattr(config, "window_overlap", 16), 16)
    print(f"    Backbone: {config.backbone} | Graph-BROS: {getattr(config, 'use_graph_bros', False)} | DANN: {getattr(config, 'adversarial_dann', False)} | Sliding Window: {getattr(config, 'sliding_window', False)}")

    label_list = [
        "O",
        "B-file_date",
        "I-file_date",
        "B-signer_name",
        "I-signer_name",
        "B-registration_num",
        "I-registration_num",
    ]
    id2label = {i: lbl for i, lbl in enumerate(label_list)}

    # 1. Build model
    model = build_model(config, len(label_list), id2label, num_templates=config.num_templates)
    model.to(device)
    model_to_optimize = model if hasattr(model, "parameters") else model.model
    model_to_optimize.to(device)
    model_to_optimize.train()
    total_p = sum(p.numel() for p in model_to_optimize.parameters())
    train_p = sum(p.numel() for p in model_to_optimize.parameters() if p.requires_grad)
    print(f"    [+] Model built and loaded to device ({train_p:,} / {total_p:,} trainable params, {100.0 * train_p / max(1, total_p):.2f}%).")

    # 2. Prepare examples and collation
    raw_examples = create_smoke_examples(is_cpu=(device.type == "cpu"))
    train_examples = raw_examples[:2]
    val_examples = raw_examples[2:]

    if getattr(config, "sliding_window", False):
        w_size = getattr(config, "window_size", 384)
        w_overlap = getattr(config, "window_overlap", 128)
        train_examples = chunk_vrdu_examples(train_examples, window_size=w_size, overlap=w_overlap)
        val_examples = chunk_vrdu_examples(val_examples, window_size=w_size, overlap=w_overlap)

    is_dann = getattr(config, "adversarial_dann", False)
    template_to_idx = None
    if is_dann:
        unique_templates = sorted(list({getattr(ex, "template_id", "unk") for ex in train_examples}))
        template_to_idx = {t: idx for idx, t in enumerate(unique_templates)}

    anc_mode = getattr(config, "anchor_mode", "content_box" if getattr(config, "relative_coords", False) else "none")
    train_collate = make_collator(
        model,
        is_joint=False,
        is_train=True,
        coordinate_jitter=getattr(config, "coordinate_jitter", False),
        relative_coords=getattr(config, "relative_coords", False),
        anchor_mode=anc_mode,
        is_dann=is_dann,
        template_to_idx=template_to_idx,
    )
    eval_collate = make_collator(
        model,
        is_joint=False,
        is_train=False,
        relative_coords=getattr(config, "relative_coords", False),
        anchor_mode=anc_mode,
        is_dann=is_dann,
        template_to_idx=template_to_idx,
    )

    train_loader = DataLoader(FormDataset(train_examples), batch_size=2, collate_fn=train_collate)
    val_loader = DataLoader(FormDataset(val_examples), batch_size=2, collate_fn=eval_collate)
    print(f"    [+] DataLoaders initialized ({len(train_examples)} train chunks, {len(val_examples)} val chunks).")

    # 3. Test 1-step Training (Forward + Backward + Optimizer step)
    trainable_params = [p for p in model_to_optimize.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    batch_data = next(iter(train_loader))

    if is_dann:
        encoded, _, template_targets = batch_data
        inputs = {k: v.to(device) for k, v in encoded.inputs.items()}
        alpha = compute_dann_alpha(0, 100, gamma=getattr(config, "dann_gamma", 10.0))
        out = model_to_optimize(inputs=inputs, template_targets=template_targets.to(device), alpha=alpha)
        loss = out.total_loss
        assert out.token_logits is not None, "Token logits missing from DANN output"
        assert out.domain_logits is not None, "Domain logits missing from DANN output"
    else:
        encoded, _ = batch_data
        inputs = {k: v.to(device) for k, v in encoded.inputs.items()}
        out = model.model(**inputs)
        loss = out.loss

    assert loss is not None and not torch.isnan(loss) and not torch.isinf(loss), f"Loss is invalid: {loss}"
    print(f"    [+] Forward pass verified! Loss = {loss.item():.4f}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("    [+] Backward pass and optimizer step verified!")

    # 4. Test Inference (predict + soft-voting aggregation)
    model_to_optimize.eval()
    preds, refs = predict(model, val_loader, device=device)
    assert len(preds) > 0, f"Expected predictions, got {len(preds)}"
    print(f"    [+] Inference & soft-voting verified ({len(preds)} document predictions generated).")

    # 5. Test the error-analysis path: per-document entity dump -> analyzer report
    with tempfile.TemporaryDirectory() as tmp:
        dump_path = Path(tmp) / "test_entities.jsonl"
        dump_test_entities(dump_path, val_examples, preds, refs, id2label)
        report = analyze([dump_path])
        assert report, "Entity error report is empty"
    print(f"    [+] Entity dump & error analysis verified ({len(report)} template(s) in report).")

    # Clean up memory
    del model, model_to_optimize, optimizer, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[PASS] {config_path} smoke test PASSED successfully!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run quick smoke tests across all experiment configs.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[
            "configs/vrdu_bros.yaml",
            "configs/vrdu_track1_bros.yaml",
            "configs/vrdu_dann_lilt.yaml",
            "configs/vrdu_dann_bros.yaml",
            "configs/vrdu_graph_bros.yaml",
        ],
        help="List of YAML config paths to test",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Smoke Test Suite on device: {device}")
    print(f"[*] Configurations to test: {args.configs}")

    passed = []
    failed = []
    for cfg_path in args.configs:
        path = Path(cfg_path)
        if not path.exists():
            print(f"[!] Warning: Config file not found: {path}")
            failed.append((str(path), "File not found"))
            continue
        try:
            ok = run_smoke_test(path, device)
            if ok:
                passed.append(str(path))
            else:
                failed.append((str(path), "Returned False"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append((str(path), str(e)))

    print(f"\n{'='*70}")
    print(f"[*] SMOKE TEST SUITE SUMMARY")
    print(f"{'='*70}")
    print(f"    Passed ({len(passed)}/{len(args.configs)}):")
    for p in passed:
        print(f"      [PASS] {p}")
    if failed:
        print(f"    Failed ({len(failed)}/{len(args.configs)}):")
        for f, err in failed:
            print(f"      [FAIL] {f} -> {err}")
        sys.exit(1)
    else:
        print("\n[+] ALL EXPERIMENT CONFIGURATIONS PASSED SMOKE TESTS CLEANLY!")


if __name__ == "__main__":
    main()
