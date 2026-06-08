"""Video patrol inference, ByteTrack deduplication, and exports."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .constants import CLASS_CODE_TO_NAME, CLASS_ID_TO_CODE
from .priority import score_defect, summarize_risk

CSV_FIELDS = [
    "track_id",
    "class_code",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "area_ratio",
    "priority_score",
    "priority_level",
    "best_frame_index",
]


def summarize_unique_tracks(detections: list[dict]) -> list[dict]:
    """Keep the highest-confidence record for each ByteTrack identifier."""
    records = {}
    for detection in detections:
        track_id = int(detection["track_id"])
        previous = records.get(track_id)
        if previous is None or detection["confidence"] > previous["confidence"]:
            records[track_id] = dict(detection)
    return [records[track_id] for track_id in sorted(records)]


def score_unique_tracks(tracks: list[dict], image_shape) -> list[dict]:
    """Score representative tracks using only neighbors from the same frame."""
    details = []
    for track in tracks:
        frame_index = track["best_frame_index"]
        nearby_tracks = [
            candidate
            for candidate in tracks
            if candidate["best_frame_index"] == frame_index
        ]
        details.append(score_defect(track, image_shape, nearby_tracks))
    return details


def _to_list(value):
    return value.cpu().tolist() if hasattr(value, "cpu") else list(value)


def normalize_tracking_result(result, frame_index: int) -> list[dict]:
    """Normalize one Ultralytics tracking frame, ignoring untracked boxes."""
    boxes = getattr(result, "boxes", None)
    ids = getattr(boxes, "id", None) if boxes is not None else None
    if boxes is None or ids is None:
        return []
    detections = []
    for box, confidence, class_id, track_id in zip(
        _to_list(boxes.xyxy),
        _to_list(boxes.conf),
        _to_list(boxes.cls),
        _to_list(ids),
    ):
        code = CLASS_ID_TO_CODE.get(int(class_id))
        if code is None:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        detections.append(
            {
                "track_id": int(track_id),
                "class_code": code,
                "class_name": CLASS_CODE_TO_NAME[code],
                "confidence": float(confidence),
                "box": [x1, y1, x2, y2],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "best_frame_index": frame_index,
            }
        )
    return detections


def export_video_csv(records: list[dict], output_dir: str | Path) -> Path:
    """Export unique tracked defects without overwriting previous patrol results."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = directory / f"patrol_details_{timestamp}.csv"
    try:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    except OSError:
        csv_path.unlink(missing_ok=True)
        raise
    return csv_path


def _remove_incomplete_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _empty_result(message: str) -> dict:
    return {
        "annotated_video": None,
        "csv_path": None,
        "details": [],
        "risk_summary": summarize_risk([]),
        "message": message,
    }


def analyze_video(video_path, model, conf_threshold, output_dir) -> dict:
    """Run ByteTrack video patrol and export an annotated video plus CSV."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = Path(video_path)
    if not path.is_file():
        return _empty_result("视频文件不存在，无法解码")
    try:
        import cv2
    except ImportError:
        return _empty_result("缺少 OpenCV 依赖，无法解码视频")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return _empty_result("视频无法解码，请检查文件格式")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        return _empty_result("视频无法解码，请检查文件内容")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    annotated_path = output / f"patrol_annotated_{timestamp}.mp4"
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        _remove_incomplete_file(annotated_path)
        return _empty_result("无法创建标注视频文件")

    frame_detections = []
    try:
        results = model.track(
            source=str(path),
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            verbose=False,
        )
        for frame_index, result in enumerate(results):
            frame_detections.extend(normalize_tracking_result(result, frame_index))
            plotted = result.plot()
            writer.write(plotted)
    except Exception as error:
        writer.release()
        _remove_incomplete_file(annotated_path)
        return _empty_result(f"视频分析失败：{error}")
    writer.release()

    unique = summarize_unique_tracks(frame_detections)
    details = score_unique_tracks(unique, (height, width))
    try:
        csv_path = export_video_csv(details, output)
    except OSError as error:
        _remove_incomplete_file(annotated_path)
        return _empty_result(f"CSV 导出失败：{error}")
    message = (
        "未发现缺陷，整体风险为低"
        if not details
        else f"巡检完成，ByteTrack 近似去重后共发现 {len(details)} 个缺陷"
    )
    return {
        "annotated_video": str(annotated_path),
        "csv_path": str(csv_path),
        "details": details,
        "risk_summary": summarize_risk(details),
        "message": message,
    }
