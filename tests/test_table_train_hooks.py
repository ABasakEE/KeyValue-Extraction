import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from train import TrainConfig, apply_init_checkpoint


def _linear(fill=None):
    m = torch.nn.Linear(4, 3)
    if fill is not None:
        with torch.no_grad():
            m.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) * fill)
            m.bias.copy_(torch.tensor([0.1, 0.2, 0.3]))
    return m


def test_init_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "seed_13").mkdir()
        src = _linear(0.5)
        torch.save(src.state_dict(), tmp / "seed_13" / "best_model.pt")
        cfg = TrainConfig()
        cfg.init_checkpoint = str(tmp / "seed_{seed}" / "best_model.pt")
        dst = _linear()
        path = apply_init_checkpoint(dst, cfg, 13)
        x = torch.ones(2, 4)
        assert torch.equal(src(x), dst(x))
        assert Path(path).as_posix().endswith("seed_13/best_model.pt"), path
    print("test_init_checkpoint_roundtrip passed")


def test_missing_init_checkpoint_raises():
    with tempfile.TemporaryDirectory() as t:
        cfg = TrainConfig()
        cfg.init_checkpoint = str(Path(t) / "seed_{seed}" / "best_model.pt")
        try:
            apply_init_checkpoint(_linear(), cfg, 21)
        except FileNotFoundError as e:
            assert "seed_21" in str(e), str(e)
        else:
            raise AssertionError("expected FileNotFoundError")
    print("test_missing_init_checkpoint_raises passed")


def test_unset_init_checkpoint_noop():
    m = _linear(0.5)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    assert apply_init_checkpoint(m, TrainConfig(), 13) is None
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k])
    print("test_unset_init_checkpoint_noop passed")


def test_crop_config_fields_default_off():
    cfg = TrainConfig()
    assert cfg.crop_aug is False
    assert cfg.init_checkpoint == ""
    assert (cfg.crop_max_per_form, cfg.crop_merge_dist, cfg.crop_seed) == (4, 50.0, 13)
    print("test_crop_config_fields_default_off passed")


if __name__ == "__main__":
    test_init_checkpoint_roundtrip()
    test_missing_init_checkpoint_raises()
    test_unset_init_checkpoint_noop()
    test_crop_config_fields_default_off()
    print("ALL TABLE TRAIN HOOK TESTS PASSED")
