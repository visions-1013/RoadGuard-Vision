import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.video_patrol import (
    CSV_FIELDS,
    analyze_video,
    export_video_csv,
    score_unique_tracks,
    summarize_unique_tracks,
)


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

    def test_unique_tracks_do_not_add_density_bonus_across_frames(self):
        tracks = [
            {
                "track_id": 1,
                "class_code": "D00",
                "confidence": 0.9,
                "box": [0, 0, 10, 10],
                "best_frame_index": 1,
            },
            {
                "track_id": 2,
                "class_code": "D00",
                "confidence": 0.8,
                "box": [5, 5, 15, 15],
                "best_frame_index": 2,
            },
        ]

        details = score_unique_tracks(tracks, (100, 100))

        self.assertEqual([item["density_score"] for item in details], [0, 0])

    def test_unique_tracks_add_density_bonus_within_same_frame(self):
        tracks = [
            {
                "track_id": 1,
                "class_code": "D00",
                "confidence": 0.9,
                "box": [0, 0, 10, 10],
                "best_frame_index": 1,
            },
            {
                "track_id": 2,
                "class_code": "D00",
                "confidence": 0.8,
                "box": [5, 5, 15, 15],
                "best_frame_index": 1,
            },
        ]

        details = score_unique_tracks(tracks, (100, 100))

        self.assertEqual([item["density_score"] for item in details], [10, 10])

    def test_csv_export_removes_partial_file_after_write_error(self):
        class BrokenWriter:
            def writeheader(self):
                return None

            def writerows(self, records):
                raise OSError("disk full")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "exports"
            with mock.patch("src.video_patrol.csv.DictWriter", return_value=BrokenWriter()):
                with self.assertRaisesRegex(OSError, "disk full"):
                    export_video_csv([], output_dir)

            self.assertEqual(list(output_dir.glob("*.csv")), [])


if __name__ == "__main__":
    unittest.main()
