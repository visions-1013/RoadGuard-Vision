import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.app import (
    MODEL_NAMES,
    ModelRegistryError,
    _comparison_rows,
    _load_yolo_weight,
    _registry_notice,
    load_project_model,
    validate_model_registry,
)


class AppTests(unittest.TestCase):
    def test_model_registry_rejects_missing_training_weight(self):
        with self.assertRaisesRegex(ModelRegistryError, "找不到训练权重"):
            validate_model_registry({"YOLO11n": {"weights": "weights/missing.pt"}})

    def test_model_registry_accepts_existing_training_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            weight_path = Path(tmp) / "best.pt"
            weight_path.write_bytes(b"course-project-weight")

            registry = validate_model_registry(
                {"YOLO11n": {"weights": str(weight_path), "recommended": True}}
            )

        self.assertEqual(registry["YOLO11n"]["weights"], str(weight_path.resolve()))
        self.assertTrue(registry["YOLO11n"]["recommended"])

    def test_comparison_rows_always_show_all_supported_models(self):
        rows = _comparison_rows(
            {
                "YOLO11n": {
                    "weights": "models/n11_best.pt",
                    "source": "models",
                    "status": "未评估",
                    "metrics": {},
                }
            }
        )

        self.assertEqual([row[0] for row in rows], list(MODEL_NAMES))
        self.assertEqual(rows[0][:3], ["YOLO11n", "未评估", "models/n11_best.pt"])
        self.assertEqual(rows[2][:3], ["YOLO26n", "未训练", ""])

    def test_empty_registry_notice_explains_where_to_place_models(self):
        notice = _registry_notice({})

        self.assertIn("没有可用训练权重", notice)
        self.assertIn("models", notice)

    def test_project_model_is_loaded_once_per_weight_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            weight_path = Path(tmp) / "best.pt"
            weight_path.write_bytes(b"course-project-weight")
            registry = {"YOLO11n": {"weights": str(weight_path)}}
            calls = []

            def fake_yolo(path):
                calls.append(path)
                return object()

            with mock.patch.dict(
                "sys.modules", {"ultralytics": SimpleNamespace(YOLO=fake_yolo)}
            ):
                first = load_project_model(registry, "YOLO11n")
                second = load_project_model(registry, "YOLO11n")

        self.assertIs(first, second)
        self.assertEqual(calls, [str(weight_path.resolve())])


if __name__ == "__main__":
    unittest.main()
