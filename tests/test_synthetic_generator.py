import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from faker import Faker
from PIL import Image

from src.data.synthetic_generator import (
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


if __name__ == "__main__":
    test_template_generators()
    print("test_template_generators passed")
    test_synthetic_renderer_and_funsd_schema()
    print("test_synthetic_renderer_and_funsd_schema passed")
    test_generate_synthetic_dataset_end_to_end()
    print("test_generate_synthetic_dataset_end_to_end passed")
    print("ALL SYNTHETIC GENERATOR TESTS PASSED")
