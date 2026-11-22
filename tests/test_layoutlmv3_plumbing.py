"""LayoutLMv3 plumbing on real VRDU documents (needs data/vrdu/registration-form and the layoutlmv3 weights from the Hub; CPU is enough).

    python tests/test_layoutlmv3_plumbing.py

Checks: the page of a window, the image tensor (shape, range, caching, blank control), that the collator adds `pixel_values`, that one forward and backward
step with the real model gives a finite loss and gradients on the visual stream, and that the page image changes the logits while the blank image does not depend on the page.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from data.page_images import PageImages  # noqa: E402
from data.vrdu_loader import chunk_vrdu_examples, load_vrdu  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from train import FormDataset, TrainConfig, build_model, make_collator  # noqa: E402


def examples(n=3):
    exs, _, id2label = load_vrdu(ROOT / "data" / "vrdu", "registration-form")
    multi = [e for e in exs if e.n_pages > 1][:1]
    return (multi + [e for e in exs if e.n_pages == 1])[:n], id2label


def test_page_images():
    exs, _ = examples()
    pi = PageImages(ROOT / "data" / "vrdu" / "registration-form" / "main" / "pdfs")
    for ex in exs:
        t = pi(ex)
        assert t.shape == (3, 224, 224) and -1.0 <= t.min() and t.max() <= 1.0 and t.std() > 0.05, (ex.guid, t.shape, t.std())
    assert len(pi._cache) == len(exs) and torch.equal(pi(exs[0]), pi(exs[0]))
    blank = PageImages(pi.pdf_dir, mode="blank")
    assert blank(exs[0]).abs().max() == 0 and torch.equal(blank(exs[0]), blank(exs[1]))
    ex = exs[0]
    two = type(ex)(guid=ex.guid, words=ex.words[:5], boxes=ex.boxes[:5], ner_tags=ex.ner_tags[:5], template_id=ex.template_id, page_ids=[1, 0, 1, 0, 0], n_pages=2)
    assert PageImages.page_of(two) == 0 and PageImages.page_of(type(two)(guid="x", words=["a"], boxes=[[0, 0, 1, 1]], ner_tags=[0], template_id="t", page_ids=[])) == 0


def test_model_step():
    exs, id2label = examples()
    cfg = TrainConfig.from_yaml(ROOT / "configs" / "vrdu_layoutlmv3.yaml")
    cfg.max_length = 128
    model = build_model(cfg, len(id2label), id2label)
    assert model.image_loader.mode == "pages"
    batch = chunk_vrdu_examples(exs, window_size=60, overlap=20)[:3]
    loader = DataLoader(FormDataset(batch), batch_size=3, collate_fn=make_collator(model, is_train=True))
    encoded, _ = next(iter(loader))
    px = encoded.inputs["pixel_values"]
    assert px.shape == (3, 3, 224, 224) and "bbox" in encoded.inputs and encoded.inputs["bbox"].max() <= 1000
    model.train()
    out = model.model(**encoded.inputs)
    assert torch.isfinite(out.loss), out.loss
    out.loss.backward()
    patch_grad = [p.grad for n, p in model.model.named_parameters() if "patch_embed" in n and p.grad is not None]
    assert patch_grad and any(g.abs().sum() > 0 for g in patch_grad), "no gradient reached the visual stream"
    model.eval()
    with torch.no_grad():
        inp = {k: v for k, v in encoded.inputs.items() if k != "labels"}
        real = model.model(**inp).logits
        other = model.model(**{**inp, "pixel_values": torch.flip(px, dims=[0])}).logits   # another window's image
        blank = model.model(**{**inp, "pixel_values": torch.zeros_like(px)}).logits
    assert (real - other).abs().max() > 1e-6 or (real - blank).abs().max() > 1e-6, "the image has no effect on the logits"
    cfg_b = TrainConfig.from_yaml(ROOT / "configs" / "vrdu_layoutlmv3_blank.yaml")
    assert cfg_b.image_mode == "blank" and cfg_b.backbone == "layoutlmv3" and cfg_b.relative_coords is False and cfg_b.coordinate_jitter is False
    try:
        cfg.image_mode = "none"
        build_model(cfg, len(id2label), id2label)
        raise AssertionError("a backbone that needs images must refuse image_mode none")
    except ValueError:
        pass


if __name__ == "__main__":
    test_page_images()
    print("[+] page images ok")
    test_model_step()
    print("[+] model step ok")
