"""Cloud Notebook helpers for reproducible RDD2022 conversion and YOLO training."""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .constants import INCLUDED_SUBSETS
from .data_utils import (
    IMAGE_SUFFIXES,
    convert_dataset,
    export_issues_csv,
    generate_data_yaml,
    merge_converted_splits,
    resolve_subset_directories,
)

DEFAULT_MODEL_SPECS = {
    "YOLO11n": "yolo11n.pt",
    "YOLO11s": "yolo11s.pt",
    "YOLO26n": "yolo26n.pt",
    "YOLO26s": "yolo26s.pt",
}
DEFAULT_CLOUD_MODELS = ("YOLO11n", "YOLO26n")
MODEL_NAMES = tuple(DEFAULT_MODEL_SPECS)
DEPLOYMENT_MODEL_FILENAMES = {
    "YOLO11n": "n11_best.pt",
    "YOLO11s": "s11_best.pt",
    "YOLO26n": "n26_best.pt",
    "YOLO26s": "s26_best.pt",
}
METRIC_FIELDS = (
    "precision",
    "recall",
    "map50",
    "map50_95",
    "weight_mb",
    "inference_ms",
)
RDD2022_ARCHIVE_NAMES = (
    "RDD2022_Japan.zip",
    "RDD2022_India.zip",
    "RDD2022_Czech.zip",
    "RDD2022_United_States.zip",
    "RDD2022_China_MotorBike.zip",
    "RDD2022_China_Drone.zip",
)
DEFAULT_TRAIN_ARGS = {
    "imgsz": 640,
    "epochs": 50,
    "batch": 8,
    "seed": 42,
    "device": 0,
}


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the repository root from a Notebook or source directory."""
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "src").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("找不到项目根目录，请确认已上传并解压完整项目压缩包")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_extract_zip(archive_path: Path, extract_root: Path) -> None:
    root = extract_root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if not _is_within(destination, root):
                raise ValueError(f"压缩包包含不安全路径：{member.filename}")
        archive.extractall(root)


def _locate_rdd2022_root(extract_root: Path) -> Path:
    candidates = [extract_root]
    candidates.extend(path for path in extract_root.rglob("*") if path.is_dir())
    ranked = []
    for candidate in candidates:
        matched = sum((candidate / subset).is_dir() for subset in INCLUDED_SUBSETS)
        if matched:
            ranked.append((matched, len(candidate.parts), candidate))
    if not ranked:
        raise FileNotFoundError("解压后未找到 RDD2022 国家子集目录")
    return max(ranked, key=lambda item: (item[0], -item[1]))[2]


def extract_rdd2022_archives(
    archive_paths: Sequence[str | Path], extract_root: str | Path
) -> Path:
    """Safely extract official RDD2022 ZIP archives and return their dataset root."""
    root = Path(extract_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = [Path(path).expanduser().resolve() for path in archive_paths]
    if not paths:
        raise ValueError("请至少配置一个 RDD2022 ZIP 压缩包")
    for archive_path in paths:
        if not archive_path.is_file():
            raise FileNotFoundError(f"找不到数据压缩包：{archive_path}")
        if not zipfile.is_zipfile(archive_path):
            raise ValueError(f"不是有效 ZIP 压缩包：{archive_path}")
        _safe_extract_zip(archive_path, root)
    return _locate_rdd2022_root(root)


def find_rdd2022_archives(downloads_root: str | Path) -> list[Path]:
    """Return the six required official subset archives in a stable order."""
    root = Path(downloads_root).expanduser().resolve()
    archives = [root / name for name in RDD2022_ARCHIVE_NAMES]
    missing = [path.name for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少 RDD2022 压缩包：" + ", ".join(missing))
    return archives


def validate_cuda_environment(torch_module=None) -> dict:
    """Return CUDA runtime details or fail before an accidental CPU training run."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as error:
            raise RuntimeError("缺少 PyTorch，无法检查 CUDA 训练环境") from error
    if not torch_module.cuda.is_available():
        raise RuntimeError("未检测到可用 NVIDIA CUDA GPU，已停止训练以避免误用 CPU")
    return {
        "torch_version": str(torch_module.__version__),
        "device": 0,
        "gpu_name": str(torch_module.cuda.get_device_name(0)),
        "gpu_count": int(torch_module.cuda.device_count()),
    }


def _has_supported_subset(source_root: Path) -> bool:
    for subset in INCLUDED_SUBSETS:
        images_dir, annotations_dir = resolve_subset_directories(source_root / subset)
        if images_dir.is_dir() and annotations_dir.is_dir():
            return True
    return False


def _remove_generated_directory(path: Path, data_root: Path) -> None:
    resolved = path.resolve()
    root = data_root.resolve()
    if resolved.parent != root:
        raise ValueError(f"拒绝清理 data_root 之外的目录：{resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _existing_yolo_summary(source: Path, data: Path) -> dict | None:
    yolo_root = data / "rdd2022_yolo"
    split_counts = {}
    for split in ("train", "val", "test"):
        images_dir = yolo_root / "images" / split
        labels_dir = yolo_root / "labels" / split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            return None
        image_count = sum(
            1
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        label_count = sum(1 for path in labels_dir.rglob("*.txt") if path.is_file())
        if image_count == 0 or image_count != label_count:
            return None
        split_counts[split] = label_count
    data_yaml = data / "data.yaml"
    if not data_yaml.is_file():
        return None
    issues_csv = data / "issues.csv"
    issue_count = 0
    if issues_csv.is_file():
        with issues_csv.open(encoding="utf-8-sig", newline="") as handle:
            issue_count = sum(1 for _ in csv.DictReader(handle))
    return {
        "source_root": str(source),
        "intermediate_root": str((data / "rdd2022_intermediate").resolve()),
        "yolo_root": str(yolo_root.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "issues_csv": str(issues_csv.resolve()),
        "converted_samples": sum(split_counts.values()),
        "issues": issue_count,
        "split_counts": split_counts,
        "reused": True,
    }


def prepare_yolo_dataset(
    source_root: str | Path, data_root: str | Path, rebuild: bool
) -> dict:
    """Convert RDD2022 XML annotations, split samples, and generate data.yaml."""
    source = Path(source_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    if not rebuild:
        existing = _existing_yolo_summary(source, data)
        if existing is not None:
            return existing
    if not source.is_dir() or not _has_supported_subset(source):
        raise FileNotFoundError(f"未找到可转换的 RDD2022 训练目录：{source}")
    intermediate = data / "rdd2022_intermediate"
    yolo_root = data / "rdd2022_yolo"
    if rebuild:
        _remove_generated_directory(intermediate, data)
        _remove_generated_directory(yolo_root, data)
    data.mkdir(parents=True, exist_ok=True)
    summary = convert_dataset(source, intermediate)
    if not summary["converted_samples"]:
        raise RuntimeError("没有可转换的 RDD2022 样本，请检查图片和 XML 标注")
    issues_csv = export_issues_csv(summary["issues"], data / "issues.csv")
    split_counts = merge_converted_splits(intermediate, yolo_root, seed=42)
    data_yaml = generate_data_yaml(data / "data.yaml", yolo_root)
    return {
        "source_root": str(source),
        "intermediate_root": str(intermediate.resolve()),
        "yolo_root": str(yolo_root.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "issues_csv": str(issues_csv.resolve()),
        "converted_samples": summary["converted_samples"],
        "issues": len(summary["issues"]),
        "split_counts": split_counts,
        "reused": False,
    }


def sync_best_weight(
    model_name: str, best_weight: str | Path, weights_root: str | Path
) -> Path:
    """Copy one trained best.pt into the stable weights registry location."""
    source = Path(best_weight).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到训练产生的最佳权重：{source}")
    target = Path(weights_root).expanduser().resolve() / model_name / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _read_model_metrics(metrics_path: str | Path | None) -> dict[str, dict]:
    if metrics_path is None:
        return {}
    path = Path(metrics_path).expanduser().resolve()
    if not path.is_file():
        return {}
    records = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model_name = row.get("model", "").strip()
            if not model_name:
                continue
            metrics = {}
            for field in METRIC_FIELDS:
                value = row.get(field, "").strip()
                if value:
                    metrics[field] = float(value)
            records[model_name] = {
                "metrics": metrics,
                "recommended": row.get("recommended", "").strip().lower()
                in {"1", "true", "yes", "是"},
            }
    return records


def available_trained_weights(
    model_names: Sequence[str],
    weights_root: str | Path,
    *,
    models_root: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> dict:
    """Build a registry containing only real local project weights."""
    root = Path(weights_root).expanduser().resolve()
    deployment_root = (
        Path(models_root).expanduser().resolve() if models_root is not None else None
    )
    report = _read_model_metrics(metrics_path)
    registry = {}
    for model_name in model_names:
        deployment_name = DEPLOYMENT_MODEL_FILENAMES.get(model_name)
        deployment_weight = (
            deployment_root / deployment_name
            if deployment_root is not None and deployment_name is not None
            else None
        )
        training_weight = root / model_name / "best.pt"
        if deployment_weight is not None and deployment_weight.is_file():
            weight = deployment_weight
            source = "models"
        elif training_weight.is_file():
            weight = training_weight
            source = "weights"
        else:
            continue
        metrics_record = report.get(model_name, {})
        metrics = metrics_record.get("metrics", {})
        registry[model_name] = {
            "weights": str(weight.resolve()),
            "source": source,
            "status": "已评估" if metrics else "未评估",
            "metrics": metrics,
            "recommended": bool(metrics_record.get("recommended")),
        }
    return registry


def _validate_pretrained_reference(pretrained_id: str | Path) -> str:
    reference = Path(pretrained_id).expanduser()
    is_explicit_path = isinstance(pretrained_id, Path) or reference.is_absolute() or reference.parent != Path(".")
    if is_explicit_path:
        resolved = reference.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"找不到本地预训练权重：{resolved}")
        return str(resolved)
    return str(pretrained_id)


def train_selected_models(
    model_specs: Mapping[str, str | Path],
    data_yaml: str | Path,
    train_args: Mapping,
    output_root: str | Path,
    resume_paths: Mapping[str, str | Path] | None = None,
    *,
    model_factory: Callable | None = None,
    weights_root: str | Path | None = None,
) -> list[dict]:
    """Train or resume selected models, synchronize best weights, and write records."""
    yaml_path = Path(data_yaml).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"找不到 data.yaml：{yaml_path}")
    runs_root = Path(output_root).expanduser().resolve()
    stable_weights_root = (
        Path(weights_root).expanduser().resolve()
        if weights_root is not None
        else runs_root.parent / "weights"
    )
    runs_root.mkdir(parents=True, exist_ok=True)
    if model_factory is None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("缺少 ultralytics，无法训练模型") from error
        model_factory = YOLO

    records = []
    resume_lookup = resume_paths or {}
    for model_name, pretrained_id in model_specs.items():
        if model_name in resume_lookup:
            resume_weight = Path(resume_lookup[model_name]).expanduser().resolve()
            if not resume_weight.is_file():
                raise FileNotFoundError(f"找不到断点权重：{resume_weight}")
            result = model_factory(str(resume_weight)).train(resume=True)
            source_id = str(resume_weight)
            resumed = True
        else:
            pretrained_reference = _validate_pretrained_reference(pretrained_id)
            result = model_factory(pretrained_reference).train(
                data=str(yaml_path),
                project=str(runs_root / "train"),
                name=model_name,
                **dict(train_args),
            )
            source_id = pretrained_reference
            resumed = False
        best_weight = Path(result.save_dir) / "weights" / "best.pt"
        stable_weight = sync_best_weight(model_name, best_weight, stable_weights_root)
        records.append(
            {
                "model": model_name,
                "source": source_id,
                "resumed": resumed,
                "best_weight": str(stable_weight.resolve()),
                **dict(train_args),
            }
        )
    records_path = runs_root / "training_records.json"
    records_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records
