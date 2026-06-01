import unittest

from src.priority import score_defect, summarize_risk


def detection(class_code, box):
    return {"class_code": class_code, "confidence": 0.9, "box": box}


class PriorityTests(unittest.TestCase):
    def test_base_scores_match_each_class(self):
        expected = {"D00": 25, "D10": 30, "D20": 45, "D40": 55}
        for class_code, score in expected.items():
            with self.subTest(class_code=class_code):
                result = score_defect(
                    detection(class_code, [0, 0, 5, 5]),
                    (100, 100),
                    [],
                )
                self.assertEqual(result["priority_score"], score)

    def test_area_score_uses_documented_boundaries(self):
        cases = [
            ([0, 0, 9, 10], 5),
            ([0, 0, 10, 10], 15),
            ([0, 0, 50, 10], 15),
            ([0, 0, 51, 10], 30),
        ]
        for box, expected in cases:
            with self.subTest(box=box):
                result = score_defect(detection("D00", box), (100, 100), [])
                self.assertEqual(result["area_score"], expected)

    def test_nearby_bonus_is_added_only_once(self):
        current = detection("D00", [0, 0, 10, 10])
        nearby = [
            detection("D10", [1, 1, 11, 11]),
            detection("D20", [2, 2, 12, 12]),
        ]
        result = score_defect(current, (100, 100), nearby)
        self.assertEqual(result["density_score"], 10)
        self.assertEqual(result["priority_score"], 45)

    def test_levels_cover_low_medium_and_high(self):
        self.assertEqual(
            score_defect(detection("D00", [0, 0, 5, 5]), (100, 100), [])[
                "priority_level"
            ],
            "低",
        )
        self.assertEqual(
            score_defect(detection("D20", [0, 0, 10, 10]), (100, 100), [])[
                "priority_level"
            ],
            "中",
        )
        self.assertEqual(
            score_defect(detection("D40", [0, 0, 60, 10]), (100, 100), [])[
                "priority_level"
            ],
            "高",
        )

    def test_empty_summary_has_low_overall_risk(self):
        self.assertEqual(
            summarize_risk([]),
            {
                "total": 0,
                "level_counts": {"低": 0, "中": 0, "高": 0},
                "overall_risk": "低",
            },
        )

    def test_summary_uses_highest_risk(self):
        summary = summarize_risk(
            [{"priority_level": "低"}, {"priority_level": "高"}, {"priority_level": "中"}]
        )
        self.assertEqual(summary["overall_risk"], "高")
        self.assertEqual(summary["level_counts"], {"低": 1, "中": 1, "高": 1})


if __name__ == "__main__":
    unittest.main()

