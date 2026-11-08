"""CPU smoke test for Study 26 (table-link arms).

Pre-check: every arm config differs from xfund_joint_lilt_windows.yaml only in the allowed keys.
Then, on tiny data (1 epoch per phase, batch 1, 16 train / 8 val forms): a pretrain on 16 synthetic documents (10 would give exactly 10 steps and hit a OneCycleLR zero-division with warmup 0.1), A0/A2/A1/A3
mini-runs through train_single_seed (so init_checkpoint is really loaded from the pretrain checkpoint), evaluate_seed on 3 val
documents per arm (asserting every group key), and the synthetic-test evaluation on 3 documents.  Raises on any failure;
temp dirs are removed at the end.  Prints SMOKE OK.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(Path(__file__).resolve().parent)):
    sys.path.insert(0, p)

import train as train_mod  # noqa: E402
from data.funsd_loader import ID2LABEL, LABEL_LIST, load_funsd_split  # noqa: E402
from data.synthetic_generator import generate_synthetic_dataset  # noqa: E402
from train import TrainConfig, build_model, load_dataset, train_single_seed  # noqa: E402
from xfund_e2e_eval import evaluate_seed  # noqa: E402

CFG = ROOT / "configs"
BASE = CFG / "xfund_joint_lilt_windows.yaml"
INIT = "results/table_links/pretrain/seed_{seed}/best_model.pt"
ALLOWED = {  # arm -> keys that may differ from the windows config (output_dir is checked separately)
    "A0": {}, "A1": {"init_checkpoint": INIT}, "A2": {"crop_aug": True}, "A3": {"crop_aug": True, "init_checkpoint": INIT},
}
TRAIN_FAMILIES = ["invoice", "financial_statement", "procurement_order", "adbuy_contract", "fara_registration"]
TEST_FAMILIES = ["shipping_manifest", "medical_intake"]
GROUP_NAMES = ["below", "same_line", "above", "rank_0", "rank_1", "rank_2_4", "rank_5plus", "q_to_q"]
SEED = 13


def check_configs() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    for arm, extra in ALLOWED.items():
        raw = yaml.safe_load((CFG / f"xfund_table_{arm}.yaml").read_text(encoding="utf-8"))
        assert raw.pop("output_dir") == f"results/table_links/{arm}", arm
        diff = {k for k in set(raw) | set(base) if k != "output_dir" and raw.get(k, "<absent>") != base.get(k, "<absent>")}
        assert diff == set(extra), (arm, diff)
        assert all(raw[k] == v for k, v in extra.items()), arm
        TrainConfig.from_yaml(CFG / f"xfund_table_{arm}.yaml")
    pre = TrainConfig.from_yaml(CFG / "xfund_synth_pretrain.yaml")
    assert pre.dataset == "funsd" and pre.data_root == "data/synthetic_table" and pre.form_window == 510
    assert (pre.max_phase1_epochs, pre.max_phase2_epochs, pre.max_phase3_epochs, pre.epochs) == (2, 2, 3, 7)
    assert pre.output_dir == "results/table_links/pretrain"
    print("config checks ok: arms differ from the windows config only in", {a: sorted(e) for a, e in ALLOWED.items()}, flush=True)


def shrink(cfg: TrainConfig, output_dir: Path) -> TrainConfig:
    cfg.phase1_epochs = cfg.phase2_epochs = cfg.phase3_epochs = 1
    cfg.max_phase1_epochs = cfg.max_phase2_epochs = cfg.max_phase3_epochs = 1
    cfg.epochs = 3
    cfg.batch_size = 1
    cfg.spatial_in_features = (5 if cfg.use_geometry else 0) + (5 if cfg.use_containment else 0) + (2 if cfg.use_line_of_sight else 0) + (4 if cfg.use_typography else 0)
    cfg.output_dir = str(output_dir)
    cfg.save_weights = True
    return cfg


def load_weights(model, path: Path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    return model.eval()


def main() -> None:
    t0 = time.time()
    check_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_syn = Path(tempfile.mkdtemp(prefix="smoke_table_syn_"))
    tmp_out = ROOT / "results" / "_smoke_table_links"
    shutil.rmtree(tmp_out, ignore_errors=True)
    inits: list = []
    real_apply = train_mod.apply_init_checkpoint

    def spy(model, config, seed):
        r = real_apply(model, config, seed)
        inits.append(r)
        return r

    train_mod.apply_init_checkpoint = spy
    try:
        generate_synthetic_dataset(
            tmp_syn, n_train=16, n_test=4, seed=7, use_augraphy=False, train_families=TRAIN_FAMILIES, test_families=TEST_FAMILIES,
            stacked_ratio=0.3, tall_tables=True, save_images=False,
        )

        # 1. pretrain on synthetic
        pre = shrink(TrainConfig.from_yaml(CFG / "xfund_synth_pretrain.yaml"), tmp_out / "pretrain")
        pre.data_root = str(tmp_syn)
        tr, va, te, labels, id2label = load_dataset(pre)
        tr, va, te = tr[:16], va[:8], te[:3]
        r = train_single_seed(SEED, pre, tr, va, te, labels, id2label, device, tmp_out / "pretrain")
        pre_ckpt = tmp_out / "pretrain" / f"seed_{SEED}" / "best_model.pt"
        assert pre_ckpt.exists(), "pretrain checkpoint missing"
        assert inits == [None], inits
        print("pretrain ok, test linking f1", r["test_linking"]["f1"], flush=True)

        # 2-3. arms
        xcfg = TrainConfig.from_yaml(CFG / "xfund_table_A0.yaml")
        xtr, xva, _xte, xlabels, xid2label = load_dataset(xcfg)
        xtr, xva_eval = xtr[:16], xva[:8]
        xte = xva[:3]
        keys_found: dict[str, int] = {}
        for arm in ("A0", "A2", "A1", "A3"):
            cfg = shrink(TrainConfig.from_yaml(CFG / f"xfund_table_{arm}.yaml"), tmp_out / arm)
            if cfg.init_checkpoint:
                assert cfg.init_checkpoint == INIT
                cfg.init_checkpoint = str(tmp_out / "pretrain" / "seed_{seed}" / "best_model.pt")  # keep {seed} templating
            before = len(inits)
            res = train_single_seed(SEED, cfg, xtr, xva_eval, xte, xlabels, xid2label, device, tmp_out / arm)
            used = inits[before:]
            want_init = "init_checkpoint" in ALLOWED[arm]
            assert (used == [str(pre_ckpt)]) if want_init else (used == [None]), (arm, used)
            assert 0.0 <= res["test_linking"]["f1"] <= 1.0
            model = load_weights(build_model(cfg, len(xlabels), xid2label), tmp_out / arm / f"seed_{SEED}" / "best_model.pt").to(device)
            out = evaluate_seed(model, xva[:3], set(), stride=128)
            for key in ("all|windowed|entity", "all|windowed|link", *(f"all|windowed|group|{g}" for g in GROUP_NAMES)):
                assert key in out and 0.0 <= out[key]["f1"] <= 1.0, (arm, key)
            keys_found[arm] = sum(k.startswith("all|windowed|group|") for k in out)
            print(f"{arm} ok: init={used}, group cells={keys_found[arm]}", flush=True)
            del model

        # 4. synthetic-test evaluation (xfund_e2e_eval --data synthetic path)
        syn_test = load_funsd_split(tmp_syn, "test")[:3]
        assert syn_test, "no synthetic test docs"
        model = load_weights(build_model(pre, len(LABEL_LIST), ID2LABEL), pre_ckpt).to(device)
        out = evaluate_seed(model, syn_test, set(), stride=128)
        assert "all|windowed|link" in out and "sibling_free|windowed|link" not in out
        assert all(f"all|windowed|group|{g}" in out for g in GROUP_NAMES)
        print("synthetic eval ok", {k: v["f1"] for k, v in out.items() if isinstance(v, dict) and k.startswith("all|windowed")}, flush=True)
    finally:
        train_mod.apply_init_checkpoint = real_apply
        shutil.rmtree(tmp_syn, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)
    print(f"wall time {time.time() - t0:.0f}s")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
