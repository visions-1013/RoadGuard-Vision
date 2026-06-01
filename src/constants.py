"""Shared class mapping and scoring constants."""

CLASS_CODE_TO_ID = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
CLASS_ID_TO_CODE = {class_id: code for code, class_id in CLASS_CODE_TO_ID.items()}
CLASS_CODE_TO_NAME = {
    "D00": "纵向裂缝",
    "D10": "横向裂缝",
    "D20": "网状裂缝",
    "D40": "坑洞",
}

BASE_SCORES = {"D00": 20, "D10": 25, "D20": 40, "D40": 50}
AREA_RATIO_SMALL = 0.01
AREA_RATIO_MEDIUM = 0.05
DENSITY_DISTANCE_RATIO = 0.15
DENSITY_BONUS = 10
PRIORITY_MEDIUM_MIN = 40
PRIORITY_HIGH_MIN = 70
DEFAULT_SEED = 42
INCLUDED_SUBSETS = (
    "Japan",
    "India",
    "Czech",
    "United_States",
    "China_MotorBike",
    "China_Drone",
)
EXCLUDED_SUBSETS = ("Norway",)
