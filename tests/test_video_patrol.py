import tempfile
import unittest
from pathlib import Path

from src.video_patrol import CSV_FIELDS, analyze_video, summarize_unique_tracks


class VideoPatrolTests(unittest.TestCase):
    def test_unique_tracks_keep_highest_confidence_record_and_frame(self):
        detections = [
            {"track_id": 7, "confidence": 0.4, "best_frame_index": 1},
            {"track_id": 7, "confidence": 0.9, "best_frame_index": 5},
            {"track_id": 8, "confidence": 0.6, "best_frame_index": 3},
        ]
        unique = summarize_unique_tracks(detections)

        self.assertEqual(len(unique), 2)
        self.assertEqual(unique[0]["confidence"], 0.9)
        self.assertEqual(unique[0]["best_frame_index"], 5)

    def test_csv_fields_match_required_contract(self):
        self.assertEqual(
            CSV_FIELDS,
            [
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
            ],
        )

    def test_invalid_video_returns_chinese_error_and_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "exports"
            result = analyze_video(Path(tmp) / "missing.mp4", object(), 0.25, output_dir)

            self.assertTrue(output_dir.exists())
            self.assertIn("无法解码", result["message"])
            self.assertEqual(result["details"], [])
            self.assertEqual(result["risk_summary"]["overall_risk"], "低")


if __name__ == "__main__":
    unittest.main()

