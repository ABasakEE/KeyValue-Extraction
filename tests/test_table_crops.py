import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.data.funsd_loader import FormExample
from src.data.xfund_windows import crop_example, crop_examples, link_components


def make_form(links=True):
    """Two key->value pairs far apart plus one header->2 cells table group; entity 6 is an unlinked singleton."""
    words, boxes, eids = [], [], []
    ents = {}

    def add(eid, label, ws, x, y):
        idx = []
        for k, w in enumerate(ws):
            idx.append(len(words))
            words.append(w)
            boxes.append([x + k * 40, y, x + k * 40 + 35, y + 20])
            eids.append(eid)
        ents[eid] = {"label": label, "token_indices": idx, "extra": eid}

    add(0, "question", ["Name"], 50, 100)
    add(1, "answer", ["Ann", "Lee"], 200, 100)
    add(2, "question", ["City"], 50, 500)
    add(3, "answer", ["Pune"], 200, 500)
    add(4, "header", ["Item"], 100, 820)
    add(5, "answer", ["Pen"], 100, 880)
    add(7, "answer", ["Ink"], 250, 880)
    add(6, "other", ["Footer"], 700, 960)
    ln = [(0, 1), (2, 3), (4, 5), (4, 7)] if links else []
    return FormExample(
        guid="toy", words=words, boxes=boxes, ner_tags=[0] * len(words), image_path="x.png", image_size=(100, 100),
        entity_ids=eids, links=ln, entities=ents,
    )


def test_components_follow_links():
    ex = make_form()
    assert len(link_components(ex)) == 3
    assert len(link_components(ex, merge_dist=1000)) == 1


def test_crop_keeps_links_whole():
    ex = make_form()
    crops = crop_examples([ex], random.Random(1), max_per_form=3)
    assert crops
    for c in crops:
        assert c.links
        for s, d in c.links:
            assert s in c.entities and d in c.entities
        for i, e in c.entities.items():
            assert [c.words[k] for k in e["token_indices"]] == [ex.words[k] for k in ex.entities[i]["token_indices"]]
            assert e["label"] == ex.entities[i]["label"] and e["extra"] == i


def test_crop_boxes_renormalised():
    ex = make_form()
    for seed in range(5):
        for c in crop_examples([ex], random.Random(seed), max_per_form=3):
            assert all(0 <= v <= 1000 for b in c.boxes for v in b)
            assert min(b[0] for b in c.boxes) < 300


def test_form_without_links_gives_no_crops():
    assert crop_examples([make_form(links=False)], random.Random(0)) == []


def test_partial_entity_dropped_with_its_links():
    ex = make_form()
    # entity 1 has words centred at x 217 and 257; this region holds only the first, so entity 1 and link (0,1) go
    c = crop_example(ex, (0, 0, 230, 1000), "p")
    assert c is not None and 0 in c.entities and 1 not in c.entities
    assert all(1 not in l for l in c.links) and (2, 3) in c.links
    assert crop_example(ex, (0, 90, 230, 130), "q") is None  # only link (0,1) was there and it is gone
    assert c.guid == "toy#cropp"


def test_degenerate_region_boxes_in_range():
    c = crop_example(make_form(), (500, 500, 500, 500), "z")
    assert c is None or all(0 <= v <= 1000 for b in c.boxes for v in b)


def test_deterministic_with_seed():
    ex = make_form()
    a = crop_examples([ex], random.Random(13))
    b = crop_examples([ex], random.Random(13))
    assert [c.guid for c in a] == [c.guid for c in b]
    assert [c.boxes for c in a] == [c.boxes for c in b]


def test_max_per_form():
    ex = make_form()
    for m in (1, 2, 4):
        assert len(crop_examples([ex], random.Random(3), max_per_form=m)) <= m


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} passed")
    print("ALL TABLE CROP TESTS PASSED")
