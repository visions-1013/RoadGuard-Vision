"""Explainable maintenance-priority scoring."""

from __future__ import annotations

from math import hypot
from typing import Iterable, Mapping, Sequence

from .constants import (
    AREA_RATIO_MEDIUM,
    AREA_RATIO_SMALL,
    BASE_SCORES,
    CLASS_CODE_TO_NAME,
    DENSITY_BONUS,
    DENSITY_DISTANCE_RATIO,
    PRIORITY_HIGH_MIN,
    PRIORITY_MEDIUM_MIN,
)


def _box_from(detection: Mapping) -> Sequence[float]:
    if "box" in detection:
        return detection["box"]
    return [
        detection["x1"],
        detection["y1"],
        detection["x2"],
        detection["y2"],
    ]


def _center(box: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def priority_level(score: int) -> str:
    if score >= PRIORITY_HIGH_MIN:
        return "高"
    if score >= PRIORITY_MEDIUM_MIN:
        return "中"
    return "低"


def score_defect(detection, image_shape, nearby_detections) -> dict:
    """Score one defect without adding model confidence to severity."""
    height, width = image_shape[:2]
    box = _box_from(detection)
    x1, y1, x2, y2 = box
    area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / (width * height)
    if area_ratio < AREA_RATIO_SMALL:
        area_score = 5
    elif area_ratio <= AREA_RATIO_MEDIUM:
        area_score = 15
    else:
        area_score = 30

    center_x, center_y = _center(box)
    max_distance = hypot(width, height) * DENSITY_DISTANCE_RATIO
    density_score = 0
    for nearby in nearby_detections:
        if nearby is detection:
            continue
        nearby_x, nearby_y = _center(_box_from(nearby))
        if hypot(center_x - nearby_x, center_y - nearby_y) <= max_distance:
            density_score = DENSITY_BONUS
            break

    class_code = detection["class_code"]
    score = BASE_SCORES[class_code] + area_score + density_score
    result = dict(detection)
    result.update(
        {
            "class_name": CLASS_CODE_TO_NAME[class_code],
            "area_ratio": area_ratio,
            "base_score": BASE_SCORES[class_code],
            "area_score": area_score,
            "density_score": density_score,
            "priority_score": score,
            "priority_level": priority_level(score),
        }
    )
    return result


def summarize_risk(scored_defects: Iterable[Mapping]) -> dict:
    """Summarize risk using the highest defect level."""
    counts = {"低": 0, "中": 0, "高": 0}
    for defect in scored_defects:
        counts[defect["priority_level"]] += 1
    overall = "高" if counts["高"] else "中" if counts["中"] else "低"
    return {"total": sum(counts.values()), "level_counts": counts, "overall_risk": overall}

