"""Checks for src/data/xfund_windows.py and src/e2e_link.py.  Needs data/xfund and the cached LiLT weights.

    python tests/test_e2e_link.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT / "src"), str(ROOT)):
    sys.path.insert(0, p)

from data.xfund_loader import ID2LABEL, LABEL_LIST, load_xfund_split  # noqa: E402
from data.xfund_windows import sub_example, window_examples, word_windows  # noqa: E402
from e2e_link import gold_structure, link_entities, predict_document, predict_window  # noqa: E402
from models.joint_extractor import prepare_entity_batch  # noqa: E402
from train import TrainConfig, build_model  # noqa: E402


def main() -> None:
    cfg = TrainConfig.from_yaml(ROOT / "configs" / "xfund_joint_lilt.yaml")
    torch.manual_seed(0)
    model = build_model(cfg, len(LABEL_LIST), ID2LABEL).eval()
    docs = [d for d in load_xfund_split(ROOT / "data" / "xfund", "val", ["fr"]) if len(d.words) > 250][:3]
    assert docs, "need data/xfund/fr.val.json"

    for ex in docs:
        wins = word_windows(model.tokenizer, ex.words, 510, 128)
        assert wins[0][0] == 0 and wins[-1][1] == len(ex.words), "windows must cover the document"
        assert all(b2 > a2 and a2 > a1 and b2 >= b1 for (a1, b1), (a2, b2) in zip(wins, wins[1:])), "windows must advance"
        for wa, wb in wins:
            enc = model.tokenizer(ex.words[wa:wb], boxes=[[0, 0, 0, 0]] * (wb - wa), truncation=False)
            assert len(enc["input_ids"]) <= 512, (ex.guid, wa, wb, len(enc["input_ids"]))

    # the relation half equals JointFormExtractor.forward on the same window with the same entities
    ex = docs[0]
    win = window_examples([ex], model.tokenizer, 510, 255)[0]
    enc = model.encode([win.words], [win.boxes], [win.ner_tags])
    spans, _, types = prepare_entity_batch([win], enc.word_ids)
    inputs = {k: v for k, v in enc.inputs.items() if k != "labels"}
    with torch.no_grad():
        ref = model(inputs=inputs, entity_spans=spans, gold_links=None, entity_types=types)
        hid = model.encoder(**inputs, output_hidden_states=True).hidden_states[-1][0]
        logits, pairs = link_entities(model, hid, inputs["bbox"][0], spans[0], types[0])
    assert torch.allclose(logits, ref.relation_logits[0], atol=1e-5), "link_entities diverges from forward"
    assert pairs == ref.predicted_links[0]

    # inference helpers
    ents, pr = predict_window(model, sub_example(ex, 0, min(200, len(ex.words)), with_entities=False))
    assert all(0 <= a < b <= 200 for _, a, b in ents) and all(i != j and max(i, j) < len(ents) for i, j in pr)
    full = predict_document(model, ex, word_windows(model.tokenizer, ex.words, 510, 128))
    assert all(0 <= a < b <= len(ex.words) for _, a, b in full[0])
    assert all(e in {(a, b) for _, a, b in full[0]} for l in full[1] for e in l), "every link endpoint must be a kept entity"

    g_ents, g_links = gold_structure(ex)
    assert g_ents and g_links and len(g_links) <= len({(s, d) for s, d in ex.links if s != d})
    print("test_e2e_link: ok")


if __name__ == "__main__":
    main()
