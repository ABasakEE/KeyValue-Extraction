"""Smoke test for the windowed joint training + end-to-end evaluation (cell 3 of the Study 19 notebook).

Runs the unit checks, then a tiny real training run (16 train / 8 val documents, 1 epoch per curriculum phase, form windows
on) and the end-to-end evaluation of the saved checkpoint on 3 documents.  Raises on any failure.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(ROOT / "tests"), str(Path(__file__).resolve().parent)):
    sys.path.insert(0, p)

import test_e2e_link  # noqa: E402
from data.xfund_loader import ID2LABEL, LABEL_LIST, load_xfund_dataset  # noqa: E402
from train import TrainConfig, build_model, train_one  # noqa: E402
from xfund_e2e_eval import evaluate_seed  # noqa: E402


def main() -> None:
    test_e2e_link.main()

    cfg = TrainConfig.from_yaml(ROOT / "configs" / "xfund_joint_lilt_windows.yaml")
    assert cfg.form_window == 510 and cfg.task == "joint"
    cfg.phase1_epochs = cfg.phase2_epochs = cfg.phase3_epochs = 1
    cfg.max_phase1_epochs = cfg.max_phase2_epochs = cfg.max_phase3_epochs = 1
    cfg.epochs = 3
    cfg.batch_size = 1  # the smoke test checks the code path, and must fit a small CPU machine
    dim = (5 if cfg.use_geometry else 0) + (5 if cfg.use_containment else 0) + (2 if cfg.use_line_of_sight else 0) + (4 if cfg.use_typography else 0)
    cfg.spatial_in_features = dim  # what train.py main() derives; train_one is called directly here

    train, val, test = load_xfund_dataset(cfg.data_root, languages=list(cfg.languages))
    train, val, test = train[:16], val[:8], test[:3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp = ROOT / "results" / "_smoke_xfund_e2e"
    shutil.rmtree(tmp, ignore_errors=True)
    model = build_model(cfg, len(LABEL_LIST), ID2LABEL)
    res = train_one(model, train, val, test, cfg, device, checkpoint_dir=tmp / "seed_0")
    assert 0.0 <= res["test_linking"]["f1"] <= 1.0
    assert (tmp / "seed_0" / "best_model.pt").exists(), "checkpoint was not written"

    model.to(device).eval()
    out = evaluate_seed(model, test, {d.guid for d in test}, stride=128)
    for key in ("all|single|entity", "all|single|link", "all|windowed|entity", "all|windowed|link"):
        assert 0.0 <= out[key]["f1"] <= 1.0 and out[key]["gold"] > 0, (key, out[key])
    shutil.rmtree(tmp, ignore_errors=True)
    print("smoke_xfund_e2e: ok", {k: v["f1"] for k, v in out.items() if isinstance(v, dict) and k.startswith("all")})


if __name__ == "__main__":
    main()
