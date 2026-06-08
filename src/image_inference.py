"""Single-image inference and annotation."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .constants import CLASS_ID_TO_CODE
from .priority import score_defect, summarize_risk

CHINESE_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)
ASCII_PRIORITY_LEVELS = {"低": "low", "中": "medium", "高": "high"}


def _to_list(value):
    return value.cpu().tolist() if hasattr(value, "cpu") else list(value)


def normalize_ultralytics_result(result) -> list[dict]:
    """Normalize the first Ultralytics result into project detection records."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _to_list(boxes.xyxy)
    confidences = _to_list(boxes.conf)
    class_ids = _to_list(boxes.cls)
    detections = []
    for box, confidence, class_id in zip(xyxy, confidences, class_ids):
        code = CLASS_ID_TO_CODE.get(int(class_id))
        if code is None:
            continue
        normalized_box = [float(value) for value in box]
        detections.append(
            {
                "class_code": code,
                "confidence": float(confidence),
                "box": normalized_box,
                "x1": normalized_box[0],
                "y1": normalized_box[1],
                "x2": normalized_box[2],
                "y2": normalized_box[3],
            }
        )
    return detections


def _image_shape(image) -> tuple[int, int]:
    if hasattr(image, "shape"):
        return tuple(image.shape[:2])
    if hasattr(image, "height") and hasattr(image, "width"):
        return image.height, image.width
    raise ValueError("无法读取图片尺寸")


def _annotation_label(defect: dict, include_chinese: bool) -> str:
    level = defect["priority_level"]
    if include_chinese:
        return (
            f"{defect['class_code']} {defect['class_name']} "
            f"{defect['confidence']:.2f} {level}"
        )
    return (
        f"{defect['class_code']} {defect['confidence']:.2f} "
        f"{ASCII_PRIORITY_LEVELS.get(level, level)}"
    )


def _load_chinese_font(size: int = 18):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for font_path in CHINESE_FONT_CANDIDATES:
        path = Path(font_path)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return None


def _draw_pillow_annotations(image, detections, font):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    try:
        annotated = Image.fromarray(image) if hasattr(image, "shape") else image.copy()
        draw = ImageDraw.Draw(annotated)
    except Exception:
        return None
    include_chinese = font is not None
    text_font = font or ImageFont.load_default()
    for defect in detections:
        x1, y1, x2, y2 = (int(value) for value in defect["box"])
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
        draw.text(
            (x1, max(0, y1 - 22)),
            _annotation_label(defect, include_chinese),
            fill=(255, 0, 0),
            font=text_font,
        )
    return annotated


def draw_annotations(image, detections):
    """Draw Chinese labels when a suitable font exists, otherwise use ASCII."""
    annotated = image.copy() if hasattr(image, "copy") else image
    pillow_annotated = _draw_pillow_annotations(
        annotated, detections, _load_chinese_font()
    )
    if pillow_annotated is not None:
        return pillow_annotated
    try:
        import cv2
    except ImportError:
        return annotated
    if not hasattr(annotated, "shape") or annotated.__class__.__module__ != "numpy":
        return annotated
    for defect in detections:
        x1, y1, x2, y2 = (int(value) for value in defect["box"])
        label = _annotation_label(defect, include_chinese=False)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def analyze_image(image, model, conf_threshold) -> dict:
    """Run image inference, score detections, and return GUI-ready data."""
    started = perf_counter()
    results = model.predict(source=image, conf=conf_threshold, verbose=False)
    detections = normalize_ultralytics_result(results[0]) if results else []
    image_shape = _image_shape(image)
    details = [
        score_defect(detection, image_shape, detections) for detection in detections
    ]
    elapsed_ms = (perf_counter() - started) * 1000
    message = "未发现缺陷" if not details else f"检测完成，共发现 {len(details)} 个缺陷"
    return {
        "annotated_image": draw_annotations(image, details),
        "details": details,
        "risk_summary": summarize_risk(details),
        "inference_ms": elapsed_ms,
        "message": message,
    }
