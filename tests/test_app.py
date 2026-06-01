import tempfile
import unittest
from pathlib import Path

from src.app import ModelRegistryError, validate_model_registry


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


if __name__ == "__main__":
    unittest.main()

