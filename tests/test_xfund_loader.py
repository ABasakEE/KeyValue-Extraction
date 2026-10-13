"""Unit tests for XFUND loader."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import json
import tempfile
import unittest

from src.data.xfund_loader import _parse_xfund_doc, load_xfund_dataset


class TestXfundLoader(unittest.TestCase):

    def test_parse_xfund_doc(self):
        doc = {
            "uid": "es_test_01",
            "img": {"fname": "es_test_01.jpg", "width": 1000, "height": 1000},
            "document": [
                {
                    "id": 0,
                    "text": "Factura No.:",
                    "box": [100, 100, 200, 150],
                    "label": "question",
                    "words": [
                        {"text": "Factura", "box": [100, 100, 150, 150]},
                        {"text": "No.:", "box": [155, 100, 200, 150]},
                    ],
                    "linking": [[0, 1]],
                },
                {
                    "id": 1,
                    "text": "F-2026",
                    "box": [220, 100, 300, 150],
                    "label": "answer",
                    "words": [
                        {"text": "F-2026", "box": [220, 100, 300, 150]},
                    ],
                    "linking": [[0, 1]],
                },
                {
                    "id": 2,
                    "text": "EMPRESA ABC",
                    "box": [100, 50, 400, 80],
                    "label": "header",
                    "words": [
                        {"text": "EMPRESA", "box": [100, 50, 250, 80]},
                        {"text": "ABC", "box": [260, 50, 400, 80]},
                    ],
                    "linking": [],
                },
            ],
        }

        ex = _parse_xfund_doc(doc, "es")
        self.assertIsNotNone(ex)
        self.assertEqual(ex.guid, "es_es_test_01")
        self.assertEqual(len(ex.words), 5)  # Factura, No.:, F-2026, EMPRESA, ABC
        self.assertEqual(len(ex.boxes), 5)
        self.assertEqual(len(ex.ner_tags), 5)
        self.assertEqual(ex.links, [(0, 1)])
        self.assertIn(0, ex.entities)
        self.assertIn(1, ex.entities)
        self.assertIn(2, ex.entities)
        self.assertEqual(ex.entities[0]["label"], "question")
        self.assertEqual(ex.entities[1]["label"], "answer")
        self.assertEqual(ex.entities[2]["label"], "header")

    def test_load_xfund_dataset_with_mock_files(self):
        sample_doc = {
            "documents": [
                {
                    "uid": "es_01",
                    "img": {"fname": "es_01.jpg", "width": 1000, "height": 1000},
                    "document": [
                        {
                            "id": 0,
                            "text": "Total:",
                            "box": [100, 100, 180, 140],
                            "label": "question",
                            "words": [{"text": "Total:", "box": [100, 100, 180, 140]}],
                            "linking": [[0, 1]],
                        },
                        {
                            "id": 1,
                            "text": "100 EUR",
                            "box": [200, 100, 280, 140],
                            "label": "answer",
                            "words": [{"text": "100", "box": [200, 100, 240, 140]}, {"text": "EUR", "box": [245, 100, 280, 140]}],
                            "linking": [[0, 1]],
                        },
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "es.train.json").write_text(json.dumps(sample_doc), encoding="utf-8")
            (tmp_path / "es.val.json").write_text(json.dumps(sample_doc), encoding="utf-8")

            train_ex, val_ex, test_ex = load_xfund_dataset(tmp_path, languages=["es"], val_ratio=0.5)
            self.assertEqual(len(train_ex) + len(val_ex), 1)
            self.assertEqual(len(test_ex), 1)


if __name__ == "__main__":
    unittest.main()
