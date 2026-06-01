"""RDD2022 review and Pascal VOC to YOLO conversion helpers."""

from __future__ import annotations

import csv
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .constants import CLASS_CODE_TO_ID, DEFAULT_SEED, INCLUDED_SUBSETS

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class IssueRecord:
    subset: str
    image_path: str
    xml_path: str
    issue_type: str
    reason: str


def is_image_decodable(image_path: str | Path) -> bool:
    """Return whether an image can be decoded, using Pillow when available."""
    path = Path(image_path)
    if not path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except ImportError:
        data = path.read_bytes()
        return (
            data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9")
        ) or data.startswith(b"\x89PNG\r\n\x1a\n")
    except (OSError, ValueError):
        return False


def read_voc_annotation(xml_path: str | Path) -> dict:
    """Read image size and target-class boxes from one Pascal VOC XML file."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError("XML 缺少 size 节点")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError("XML 图片尺寸非法")

    boxes = []
    for item in root.findall("object"):
        class_code = item.findtext("name", "").strip()
        if class_code not in CLASS_CODE_TO_ID:
            continue
        node = item.find("bndbox")
        if node is None:
            boxes.append({"class_code": class_code, "box": None})
            continue
        try:
            box = tuple(
                float(node.findtext(name, "nan"))
                for name in ("xmin", "ymin", "xmax", "ymax")
            )
        except ValueError:
            box = None
        boxes.append({"class_code": class_code, "box": box})
    return {"width": width, "height": height, "boxes": boxes}


def validate_box(
    box: Sequence[float] | None, image_size: tuple[int, int]
) -> tuple[bool, str]:
    """Validate one VOC box against `(width, height)`."""
    if box is None or len(box) != 4:
        return False, "框坐标格式错误"
    width, height = image_size
    x1, y1, x2, y2 = box
    if not all(value == value for value in box):
        return False, "框坐标格式错误"
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False, "框坐标超出图片范围"
    if x2 <= x1 or y2 <= y1:
        return False, "框宽高必须为正数"
    return True, ""


def voc_box_to_yolo(
    box: Sequence[float], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Convert VOC `(x1, y1, x2, y2)` to normalized YOLO coordinates."""
    valid, reason = validate_box(box, image_size)
    if not valid:
        raise ValueError(reason)
    width, height = image_size
    x1, y1, x2, y2 = box
    return (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def export_issues_csv(issues: Iterable[IssueRecord], csv_path: str | Path) -> Path:
    """Export conversion issues as a UTF-8 BOM CSV for spreadsheet use."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subset", "image_path", "xml_path", "issue_type", "reason"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(issue) for issue in issues)
    return path


def _record_issue(
    issues: list[IssueRecord],
    subset: str,
    image_path: Path,
    xml_path: Path,
    issue_type: str,
    reason: str,
) -> None:
    issues.append(
        IssueRecord(subset, str(image_path), str(xml_path), issue_type, reason)
    )


def convert_dataset(
    source_root: str | Path,
    output_root: str | Path,
    included_subsets: Sequence[str] = INCLUDED_SUBSETS,
) -> dict:
    """Convert supported subset directories into an intermediate YOLO dataset."""
    source = Path(source_root)
    output = Path(output_root)
    issues: list[IssueRecord] = []
    converted_samples = 0

    allowed_subsets = set(included_subsets)
    for subset_dir in sorted(
        path for path in source.iterdir() if path.is_dir() and path.name in allowed_subsets
    ):
        subset = subset_dir.name
        images_dir = subset_dir / "images"
        annotations_dir = subset_dir / "annotations"
        if not images_dir.exists():
            continue
        image_paths = sorted(
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for image_path in image_paths:
            relative_path = image_path.relative_to(images_dir)
            xml_path = (annotations_dir / relative_path).with_suffix(".xml")
            if not xml_path.is_file():
                _record_issue(
                    issues, subset, image_path, xml_path, "missing_xml", "缺少 XML 标注"
                )
                continue
            if not is_image_decodable(image_path):
                _record_issue(
                    issues, subset, image_path, xml_path, "corrupt_image", "图片无法解码"
                )
                continue
            try:
                annotation = read_voc_annotation(xml_path)
            except (ET.ParseError, OSError, ValueError) as error:
                _record_issue(
                    issues, subset, image_path, xml_path, "invalid_xml", str(error)
                )
                continue

            yolo_rows = []
            for item in annotation["boxes"]:
                box = item["box"]
                valid, reason = validate_box(
                    box, (annotation["width"], annotation["height"])
                )
                if not valid:
                    _record_issue(
                        issues, subset, image_path, xml_path, "invalid_box", reason
                    )
                    continue
                values = voc_box_to_yolo(
                    box, (annotation["width"], annotation["height"])
                )
                yolo_rows.append(
                    f"{CLASS_CODE_TO_ID[item['class_code']]} "
                    + " ".join(f"{value:.6f}" for value in values)
                )

            if not yolo_rows:
                continue
            target_image = output / "images" / subset / relative_path
            target_label = (output / "labels" / subset / relative_path).with_suffix(
                ".txt"
            )
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_image)
            target_label.write_text("\n".join(yolo_rows) + "\n", encoding="utf-8")
            converted_samples += 1

    return {"converted_samples": converted_samples, "issues": issues}


def split_subset(
    samples: Sequence[str | Path], seed: int = DEFAULT_SEED
) -> dict[str, list[str | Path]]:
    """Split one subset reproducibly into 80%, 10%, and 10% partitions."""
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_end = int(count * 0.8)
    val_end = train_end + int(count * 0.1)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def merge_converted_splits(
    converted_root: str | Path,
    dataset_root: str | Path,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    """Split each converted subset independently, then merge into YOLO folders."""
    converted = Path(converted_root)
    dataset = Path(dataset_root)
    counts = {"train": 0, "val": 0, "test": 0}
    labels_root = converted / "labels"
    if not labels_root.exists():
        return counts

    for subset_dir in sorted(path for path in labels_root.iterdir() if path.is_dir()):
        labels = sorted(subset_dir.rglob("*.txt"))
        for split, split_labels in split_subset(labels, seed).items():
            for label_path in split_labels:
                relative = label_path.relative_to(labels_root)
                image_dir = converted / "images" / relative.parent
                candidates = [
                    path
                    for suffix in IMAGE_SUFFIXES
                    if (path := (image_dir / relative.stem).with_suffix(suffix)).is_file()
                ]
                if not candidates:
                    continue
                target_label = dataset / "labels" / split / relative
                target_image = dataset / "images" / split / relative.with_suffix(
                    candidates[0].suffix
                )
                target_label.parent.mkdir(parents=True, exist_ok=True)
                target_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_path, target_label)
                shutil.copy2(candidates[0], target_image)
                counts[split] += 1
    return counts


def generate_data_yaml(yaml_path: str | Path, dataset_root: str | Path) -> Path:
    """Write an Ultralytics dataset configuration with fixed class ordering."""
    path = Path(yaml_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = Path(dataset_root).resolve().as_posix()
    lines = [
        f"path: {dataset}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(
        f"  {class_id}: {code}"
        for code, class_id in sorted(CLASS_CODE_TO_ID.items(), key=lambda item: item[1])
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
