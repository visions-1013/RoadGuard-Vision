import base64
import csv
import tempfile
import unittest
from pathlib import Path

from src.data_utils import (
    IssueRecord,
    convert_dataset,
    generate_data_yaml,
    split_subset,
    voc_box_to_yolo,
)


VALID_XML = """\
<annotation>
  <size><width>100</width><height>200</height></size>
  <object>
    <name>D00</name>
    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>100</ymax></bndbox>
  </object>
</annotation>
"""
VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class DataUtilsTests(unittest.TestCase):
    def test_voc_box_to_yolo_normalizes_coordinates(self):
        self.assertEqual(
            voc_box_to_yolo((10, 20, 50, 100), (100, 200)),
            (0.3, 0.3, 0.4, 0.4),
        )

    def test_convert_dataset_records_missing_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "source" / "Japan" / "images" / "road.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"not-a-real-image")

            summary = convert_dataset(root / "source", root / "output")

            self.assertEqual(summary["converted_samples"], 0)
            self.assertIn("missing_xml", [issue.issue_type for issue in summary["issues"]])

    def test_convert_dataset_records_corrupt_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "source" / "Japan" / "images" / "road.jpg"
            xml = root / "source" / "Japan" / "annotations" / "road.xml"
            image.parent.mkdir(parents=True)
            xml.parent.mkdir(parents=True)
            image.write_bytes(b"not-a-real-image")
            xml.write_text(VALID_XML, encoding="utf-8")

            summary = convert_dataset(root / "source", root / "output")

            self.assertEqual(summary["converted_samples"], 0)
            self.assertIn("corrupt_image", [issue.issue_type for issue in summary["issues"]])

    def test_convert_dataset_skips_invalid_box_without_writing_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "source" / "Japan" / "images" / "road.png"
            xml = root / "source" / "Japan" / "annotations" / "road.xml"
            image.parent.mkdir(parents=True)
            xml.parent.mkdir(parents=True)
            image.write_bytes(VALID_PNG)
            xml.write_text(
                VALID_XML.replace("<xmax>50</xmax>", "<xmax>150</xmax>"),
                encoding="utf-8",
            )

            summary = convert_dataset(root / "source", root / "output")

            self.assertEqual(summary["converted_samples"], 0)
            self.assertIn("invalid_box", [issue.issue_type for issue in summary["issues"]])
            self.assertEqual(list((root / "output").rglob("*.txt")), [])

    def test_convert_dataset_excludes_norway_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "source" / "Norway" / "images" / "road.png"
            xml = root / "source" / "Norway" / "annotations" / "road.xml"
            image.parent.mkdir(parents=True)
            xml.parent.mkdir(parents=True)
            image.write_bytes(VALID_PNG)
            xml.write_text(VALID_XML, encoding="utf-8")

            summary = convert_dataset(root / "source", root / "output")

            self.assertEqual(summary["converted_samples"], 0)
            self.assertEqual(summary["issues"], [])

    def test_split_subset_is_reproducible_and_uses_expected_ratios(self):
        samples = [f"sample-{index:02d}" for index in range(20)]
        first = split_subset(samples, seed=42)
        second = split_subset(samples, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 16)
        self.assertEqual(len(first["val"]), 2)
        self.assertEqual(len(first["test"]), 2)

    def test_generate_data_yaml_keeps_class_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "data.yaml"
            generate_data_yaml(yaml_path, Path(tmp) / "dataset")
            content = yaml_path.read_text(encoding="utf-8")

            self.assertIn("0: D00", content)
            self.assertIn("1: D10", content)
            self.assertIn("2: D20", content)
            self.assertIn("3: D40", content)

    def test_issue_record_exports_expected_columns(self):
        issue = IssueRecord("Japan", "a.jpg", "a.xml", "invalid_box", "bad")
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "issues.csv"
            from src.data_utils import export_issues_csv

            export_issues_csv([issue], csv_path)
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["issue_type"], "invalid_box")


if __name__ == "__main__":
    unittest.main()
