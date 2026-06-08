import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.cloud_training import (
    available_trained_weights,
    extract_rdd2022_archives,
    find_rdd2022_archives,
    prepare_yolo_dataset,
    sync_best_weight,
    train_selected_models,
    validate_cuda_environment,
)

from tests.test_data_utils import VALID_PNG, VALID_XML


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_name(index):
        return f"Fake GPU {index}"

    @staticmethod
    def device_count():
        return 1


class _FakeTorch:
    __version__ = "test"
    cuda = _FakeCuda()


class _FakeResult:
    def __init__(self, save_dir):
        self.save_dir = save_dir


class _FakeModel:
    calls = []

    def __init__(self, model_id):
        self.model_id = str(model_id)

    def train(self, **kwargs):
        self.calls.append({"model_id": self.model_id, **kwargs})
        if kwargs.get("resume"):
            save_dir = Path(self.model_id).parent.parent
        else:
            save_dir = Path(kwargs["project"]) / kwargs["name"]
        weight = save_dir / "weights" / "best.pt"
        weight.parent.mkdir(parents=True, exist_ok=True)
        weight.write_bytes(b"trained-weight")
        return _FakeResult(save_dir)


class CloudTrainingTests(unittest.TestCase):
    def setUp(self):
        _FakeModel.calls = []

    def test_extract_rdd2022_archives_returns_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "RDD2022_Japan.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("RDD2022/Japan/train/images/road.jpg", b"image")

            dataset_root = extract_rdd2022_archives([archive], root / "extracted")

            self.assertEqual(dataset_root, root / "extracted" / "RDD2022")
            self.assertTrue((dataset_root / "Japan" / "train" / "images" / "road.jpg").is_file())

    def test_extract_rdd2022_archives_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside.txt", "bad")

            with self.assertRaisesRegex(ValueError, "不安全"):
                extract_rdd2022_archives([archive], root / "extracted")

    def test_find_rdd2022_archives_requires_all_project_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = [
                "RDD2022_Japan.zip",
                "RDD2022_India.zip",
                "RDD2022_Czech.zip",
                "RDD2022_United_States.zip",
                "RDD2022_China_MotorBike.zip",
                "RDD2022_China_Drone.zip",
            ]
            for name in expected:
                (root / name).write_bytes(b"zip-placeholder")

            archives = find_rdd2022_archives(root)

            self.assertEqual([path.name for path in archives], expected)
            (root / "RDD2022_China_Drone.zip").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "China_Drone"):
                find_rdd2022_archives(root)

    def test_prepare_yolo_dataset_rebuild_removes_stale_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "source" / "Japan" / "images" / "road.png"
            xml = root / "source" / "Japan" / "annotations" / "road.xml"
            image.parent.mkdir(parents=True)
            xml.parent.mkdir(parents=True)
            image.write_bytes(VALID_PNG)
            xml.write_text(VALID_XML, encoding="utf-8")
            stale = root / "data" / "rdd2022_yolo" / "labels" / "train" / "stale.txt"
            stale.parent.mkdir(parents=True)
            stale.write_text("stale", encoding="utf-8")

            summary = prepare_yolo_dataset(root / "source", root / "data", rebuild=True)

            self.assertFalse(stale.exists())
            self.assertTrue(Path(summary["data_yaml"]).is_file())
            self.assertEqual(summary["converted_samples"], 1)

    def test_prepare_yolo_dataset_reuses_complete_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            yolo = data / "rdd2022_yolo"
            for split in ("train", "val", "test"):
                images = yolo / "images" / split
                images.mkdir(parents=True)
                (images / f"{split}.jpg").write_bytes(b"keep")
                labels = yolo / "labels" / split
                labels.mkdir(parents=True)
                (labels / f"{split}.txt").write_text("keep", encoding="utf-8")
            (data / "data.yaml").write_text("path: existing", encoding="utf-8")
            (data / "issues.csv").write_text(
                "subset,image_path,xml_path,issue_type,reason\nJapan,a.jpg,a.xml,invalid_box,bad\n",
                encoding="utf-8-sig",
            )

            summary = prepare_yolo_dataset(root / "missing-source", data, rebuild=False)

            self.assertTrue(summary["reused"])
            self.assertEqual(summary["issues"], 1)
            self.assertTrue((yolo / "labels" / "train" / "train.txt").is_file())

    def test_prepare_yolo_dataset_does_not_reuse_incomplete_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            yolo = data / "rdd2022_yolo"
            for split in ("train", "val", "test"):
                (yolo / "images" / split).mkdir(parents=True)
                labels = yolo / "labels" / split
                labels.mkdir(parents=True)
                (labels / f"{split}.txt").write_text("stale", encoding="utf-8")
            (data / "data.yaml").write_text("path: incomplete", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "RDD2022"):
                prepare_yolo_dataset(root / "missing-source", data, rebuild=False)

    def test_sync_and_available_weights_use_real_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "runs" / "YOLO11n" / "weights" / "best.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"weight")

            target = sync_best_weight("YOLO11n", source, root / "weights")
            registry = available_trained_weights(
                ["YOLO11n", "YOLO11s"], root / "weights"
            )

            self.assertEqual(target.read_bytes(), b"weight")
            self.assertEqual(list(registry), ["YOLO11n"])
            self.assertEqual(registry["YOLO11n"]["weights"], str(target.resolve()))

    def test_available_weights_prefers_deployment_model_and_reads_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deployment_weight = root / "models" / "n11_best.pt"
            deployment_weight.parent.mkdir()
            deployment_weight.write_bytes(b"deployment")
            training_weight = root / "weights" / "YOLO11n" / "best.pt"
            training_weight.parent.mkdir(parents=True)
            training_weight.write_bytes(b"training")
            metrics = root / "reports" / "model_comparison.csv"
            metrics.parent.mkdir()
            metrics.write_text(
                "model,status,precision,recall,map50,map50_95,weight_mb,inference_ms,recommended\n"
                "YOLO11n,已评估,0.91,0.82,0.88,0.71,5.2,14.5,True\n",
                encoding="utf-8-sig",
            )

            registry = available_trained_weights(
                ["YOLO11n", "YOLO26n"],
                root / "weights",
                models_root=root / "models",
                metrics_path=metrics,
            )

            self.assertEqual(registry["YOLO11n"]["weights"], str(deployment_weight.resolve()))
            self.assertEqual(registry["YOLO11n"]["source"], "models")
            self.assertEqual(registry["YOLO11n"]["status"], "已评估")
            self.assertEqual(registry["YOLO11n"]["metrics"]["map50_95"], 0.71)
            self.assertTrue(registry["YOLO11n"]["recommended"])
            self.assertNotIn("YOLO26n", registry)

    def test_available_weights_falls_back_to_training_weight_without_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_weight = root / "weights" / "YOLO26n" / "best.pt"
            training_weight.parent.mkdir(parents=True)
            training_weight.write_bytes(b"training")

            registry = available_trained_weights(
                ["YOLO26n"],
                root / "weights",
                models_root=root / "models",
                metrics_path=root / "reports" / "missing.csv",
            )

            self.assertEqual(registry["YOLO26n"]["weights"], str(training_weight.resolve()))
            self.assertEqual(registry["YOLO26n"]["source"], "weights")
            self.assertEqual(registry["YOLO26n"]["status"], "未评估")
            self.assertEqual(registry["YOLO26n"]["metrics"], {})

    def test_validate_cuda_environment_reports_gpu(self):
        info = validate_cuda_environment(torch_module=_FakeTorch())

        self.assertEqual(info["device"], 0)
        self.assertEqual(info["gpu_name"], "Fake GPU 0")
        self.assertEqual(info["gpu_count"], 1)

    def test_validate_cuda_environment_rejects_cpu_only_runtime(self):
        class CpuOnlyTorch:
            __version__ = "test"

            class cuda:
                @staticmethod
                def is_available():
                    return False

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            validate_cuda_environment(torch_module=CpuOnlyTorch())

    def test_train_selected_models_syncs_weights_and_writes_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_yaml = root / "data.yaml"
            data_yaml.write_text("names: {}", encoding="utf-8")

            records = train_selected_models(
                {"YOLO11n": "yolo11n.pt"},
                data_yaml,
                {"epochs": 1, "device": 0},
                root / "runs",
                model_factory=_FakeModel,
            )

            weight = root / "weights" / "YOLO11n" / "best.pt"
            saved_records = json.loads(
                (root / "runs" / "training_records.json").read_text(encoding="utf-8")
            )
            self.assertTrue(weight.is_file())
            self.assertEqual(records, saved_records)
            self.assertEqual(records[0]["best_weight"], str(weight.resolve()))
            self.assertEqual(_FakeModel.calls[0]["data"], str(data_yaml.resolve()))

    def test_train_selected_models_resumes_from_last_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_yaml = root / "data.yaml"
            data_yaml.write_text("names: {}", encoding="utf-8")
            last_weight = root / "runs" / "train" / "YOLO11n" / "weights" / "last.pt"
            last_weight.parent.mkdir(parents=True)
            last_weight.write_bytes(b"last")

            train_selected_models(
                {"YOLO11n": "yolo11n.pt"},
                data_yaml,
                {"epochs": 50, "device": 0},
                root / "runs",
                resume_paths={"YOLO11n": last_weight},
                model_factory=_FakeModel,
            )

            self.assertEqual(_FakeModel.calls[0], {"model_id": str(last_weight.resolve()), "resume": True})

    def test_train_selected_models_rejects_missing_local_pretrained_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_yaml = root / "data.yaml"
            data_yaml.write_text("names: {}", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "预训练权重"):
                train_selected_models(
                    {"YOLO11n": root / "pretrained" / "yolo11n.pt"},
                    data_yaml,
                    {"epochs": 1, "device": 0},
                    root / "runs",
                    model_factory=_FakeModel,
                )

            self.assertEqual(_FakeModel.calls, [])


if __name__ == "__main__":
    unittest.main()
