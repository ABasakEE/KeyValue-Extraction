"""Smoke test on REAL VRDU data for study 16: one epoch of the Track-1 BROS recipe on a handful of documents from two
registration templates, tested on a third, then check that test_entities.jsonl and test_probs.jsonl are written and
that the probability dump respects the requested document list and matches each document's word count.

Usage: python scripts/smoke_vrdu_probs.py        (needs data/vrdu and a GPU or a lot of patience)
"""

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    import torch
    from data.vrdu_loader import load_vrdu
    from routing_eval import validate_probs
    from train import TrainConfig, train_single_seed

    examples, label_list, id2label = load_vrdu(ROOT / "data" / "vrdu", "registration-form")
    by_t = defaultdict(list)
    for e in sorted(examples, key=lambda e: e.guid):
        by_t[e.template_id].append(e)
    seen, held = ["Amendment", "Dissemination_Report"], "Short-Form"
    train = by_t[seen[0]][:6] + by_t[seen[1]][:6]
    val = by_t[seen[0]][6:8] + by_t[seen[1]][6:8]
    test = by_t[held][:4]
    keep = {test[0].guid, test[2].guid}                       # request only 2 of the 4 test documents

    tmp = Path(tempfile.mkdtemp())
    guid_file = tmp / "guids.json"
    guid_file.write_text(json.dumps(sorted(keep)), encoding="utf-8")
    config = TrainConfig.from_yaml(ROOT / "configs" / "vrdu_track1_bros_probs.yaml")
    config.epochs, config.batch_size, config.dump_probs_guids = 1, 2, str(guid_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_single_seed(13, config, train, val, test, label_list, id2label, device, tmp / "run", log_prefix="[vrdu smoke]")

    run_dir = tmp / "run" / "seed_13"
    assert (run_dir / "result.json").exists() and (run_dir / "test_entities.jsonl").exists(), list(run_dir.glob("*"))
    n = validate_probs(run_dir / "test_probs.jsonl", {e.guid: len(e.words) for e in test}, keep)
    assert n == 2, n
    entities = [json.loads(l) for l in (run_dir / "test_entities.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(entities) == 4, len(entities)                  # span dump covers every test document, probs only the requested ones
    print(f"[+] vrdu smoke passed: {n}/4 probability rows (requested 2), files {sorted(p.name for p in run_dir.glob('*'))}")


if __name__ == "__main__":
    main()
