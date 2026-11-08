import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from faker import Faker
from PIL import Image

from src.data.funsd_loader import load_funsd_split
from src.data.synthetic_generator import (
    FAMILIES,
    FormDocument,
    FormField,
    FormSection,
    SyntheticFormRenderer,
    generate_invoice_document,
    generate_medical_intake_document,
    generate_adbuy_contract_document,
    generate_fara_registration_document,
    generate_synthetic_dataset,
)


def test_template_generators():
    fake = Faker(seed=13)
    doc_inv = generate_invoice_document(fake, 0)
    assert doc_inv.template_name == "invoice_0"
    assert len(doc_inv.fields) >= 6
    assert doc_inv.table is not None and len(doc_inv.table.rows) > 0

    doc_med = generate_medical_intake_document(fake, 1)
    assert doc_med.template_name == "intake_1"
    assert len(doc_med.fields) >= 8

    doc_ad = generate_adbuy_contract_document(fake, 2)
    assert doc_ad.template_name == "adbuy_2"
    assert len(doc_ad.fields) >= 8

    doc_fara = generate_fara_registration_document(fake, 3)
    assert doc_fara.template_name == "fara_3"
    assert len(doc_fara.fields) >= 8


def test_synthetic_renderer_and_funsd_schema():
    fake = Faker(seed=42)
    renderer = SyntheticFormRenderer(fake=fake, use_augraphy=False)
    doc = generate_invoice_document(fake, 0)

    img, ann = renderer.render(doc)
    assert isinstance(img, Image.Image)
    assert img.size == (1654, 2338)

    assert "form" in ann
    form_entities = ann["form"]
    assert len(form_entities) > 0

    has_question = False
    has_answer = False
    has_header = False
    has_linking = False

    for ent in form_entities:
        assert "id" in ent
        assert "text" in ent
        assert "box" in ent
        assert len(ent["box"]) == 4
        assert "label" in ent
        assert ent["label"] in {"question", "answer", "header", "other"}
        assert "words" in ent
        assert len(ent["words"]) > 0
        assert "linking" in ent

        if ent["label"] == "question":
            has_question = True
        if ent["label"] == "answer":
            has_answer = True
        if ent["label"] == "header":
            has_header = True
        if len(ent["linking"]) > 0:
            has_linking = True

    assert has_question, "Should contain question entities"
    assert has_answer, "Should contain answer entities"
    assert has_header, "Should contain header entities"
    assert has_linking, "Should contain directed key-value links"


def test_generate_synthetic_dataset_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        summary = generate_synthetic_dataset(
            output_dir=tmp_dir,
            n_train=2,
            n_test=2,
            seed=99,
            use_augraphy=False,
        )
        assert summary["training_data_generated"] == 2
        assert summary["testing_data_generated"] == 2

        train_img_dir = Path(tmp_dir) / "training_data" / "images"
        train_ann_dir = Path(tmp_dir) / "training_data" / "annotations"
        assert len(list(train_img_dir.glob("*.png"))) == 2
        assert len(list(train_ann_dir.glob("*.json"))) == 2


def _centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def test_sections_emit_question_to_question_links():
    fake = Faker(seed=1)
    renderer = SyntheticFormRenderer(fake=fake, use_augraphy=False)
    fields = [FormField(f"Field {i}", f"value {i}") for i in range(3)]
    doc = FormDocument("sec_0", "TEST FORM", [], sections=[FormSection("Applicant Details", fields)])
    _, ann = renderer.render(doc)
    ents = {e["id"]: e for e in ann["form"]}
    heads = [e for e in ents.values() if e["text"] == "Applicant Details"]
    assert len(heads) == 1 and heads[0]["label"] == "question"
    dsts = [d for s, d in heads[0]["linking"] if s == heads[0]["id"]]
    assert len(dsts) == 3
    for d in dsts:
        assert ents[d]["label"] == "question"
        assert ents[d]["text"].startswith("Field")


def test_stacked_ratio_one_puts_values_below_keys():
    fake = Faker(seed=2)
    renderer = SyntheticFormRenderer(fake=fake, use_augraphy=False, stacked_ratio=1.0)
    doc = FormDocument("h_0", "T", [FormField(f"K{i}", f"v{i}", "horizontal") for i in range(6)])
    _, ann = renderer.render(doc)
    ents = {e["id"]: e for e in ann["form"]}
    n = 0
    for e in ann["form"]:
        for s, d in e["linking"]:
            if ents[s]["label"] == "question" and ents[d]["label"] == "answer":
                assert ents[d]["box"][1] > ents[s]["box"][3]
                n += 1
    assert n >= 6


def test_family_split_is_disjoint():
    with tempfile.TemporaryDirectory() as tmp:
        generate_synthetic_dataset(
            tmp, n_train=10, n_test=4, use_augraphy=False,
            train_families=["invoice", "fara_registration"],
            test_families=["medical_intake"], save_images=False,
        )
        train = list((Path(tmp) / "training_data" / "annotations").glob("*.json"))
        test = list((Path(tmp) / "testing_data" / "annotations").glob("*.json"))
        assert len(train) == 10 and len(test) == 4
        assert all("invoice" in p.name or "fara" in p.name for p in train)
        assert all("medical" in p.name for p in test)
        assert not list((Path(tmp) / "training_data").glob("images/*.png"))


def test_tall_tables_have_many_rows():
    import random as _r
    from src.data.synthetic_generator import generate_invoice_document as gen
    fake = Faker(seed=3)
    _r.seed(3)
    doc = gen(fake, 0, n_rows=12)
    data_rows = [r for r in doc.table.rows if not r.is_subheader and not r.is_summary]
    assert len(data_rows) >= 8
    renderer = SyntheticFormRenderer(fake=fake, use_augraphy=False)
    _, ann = renderer.render(doc)
    ents = {e["id"]: e for e in ann["form"]}
    far = 0
    for e in ann["form"]:
        if e["label"] != "header":
            continue
        cx, cy = _centre(e["box"])
        order = sorted(
            (o["id"] for o in ann["form"] if o["id"] != e["id"]),
            key=lambda i: (_centre(ents[i]["box"])[0] - cx) ** 2 + (_centre(ents[i]["box"])[1] - cy) ** 2,
        )
        for s, d in e["linking"]:
            if s == e["id"] and order.index(d) >= 5:
                far += 1
    assert far >= 1


def test_locales_change_values_not_keys():
    with tempfile.TemporaryDirectory() as tmp:
        generate_synthetic_dataset(
            tmp, n_train=7, n_test=0, use_augraphy=False, locales=["es_ES"],
            save_images=False, tall_tables=True, stacked_ratio=0.4,
        )
        texts = set()
        for p in (Path(tmp) / "training_data" / "annotations").glob("*.json"):
            for e in json.loads(p.read_text(encoding="utf-8"))["form"]:
                if e["label"] == "question":
                    texts.add(e["text"])
        assert "Invoice Number:" in texts and "Patient Legal Name:" in texts


def test_locale_is_independent_of_family():
    locs = ["es_ES", "fr_FR", "de_DE", "it_IT", "pt_PT"]
    fams = ["invoice", "financial_statement", "procurement_order", "adbuy_contract", "fara_registration"]
    with tempfile.TemporaryDirectory() as tmp:
        generate_synthetic_dataset(
            tmp, n_train=50, n_test=0, use_augraphy=False, train_families=fams,
            locales=locs, save_images=False,
        )
        seen = {}
        for p in (Path(tmp) / "training_data" / "annotations").glob("*.json"):
            fam = next(f for f in fams if f in p.name)
            seen.setdefault(fam, set()).add(json.loads(p.read_text(encoding="utf-8"))["locale"])
        assert any(len(v) >= 2 for v in seen.values()), seen


def test_loader_works_without_images():
    with tempfile.TemporaryDirectory() as tmp:
        generate_synthetic_dataset(tmp, n_train=7, n_test=7, use_augraphy=False, save_images=False)
        for split in ("train", "test"):
            exs = load_funsd_split(tmp, split)
            assert len(exs) == 7
            assert all(e.links and e.image_size == (1654, 2338) for e in exs)


def test_families_registry_keys():
    assert set(FAMILIES) == {
        "invoice", "financial_statement", "procurement_order", "shipping_manifest",
        "medical_intake", "adbuy_contract", "fara_registration",
    }


if __name__ == "__main__":
    test_template_generators()
    print("test_template_generators passed")
    test_synthetic_renderer_and_funsd_schema()
    print("test_synthetic_renderer_and_funsd_schema passed")
    test_generate_synthetic_dataset_end_to_end()
    print("test_generate_synthetic_dataset_end_to_end passed")
    test_sections_emit_question_to_question_links()
    print("test_sections_emit_question_to_question_links passed")
    test_stacked_ratio_one_puts_values_below_keys()
    print("test_stacked_ratio_one_puts_values_below_keys passed")
    test_family_split_is_disjoint()
    print("test_family_split_is_disjoint passed")
    test_tall_tables_have_many_rows()
    print("test_tall_tables_have_many_rows passed")
    test_locales_change_values_not_keys()
    print("test_locales_change_values_not_keys passed")
    test_locale_is_independent_of_family()
    print("test_locale_is_independent_of_family passed")
    test_loader_works_without_images()
    print("test_loader_works_without_images passed")
    test_families_registry_keys()
    print("test_families_registry_keys passed")
    print("ALL SYNTHETIC GENERATOR TESTS PASSED")
