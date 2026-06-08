import json
import unittest
from pathlib import Path


class CloudNotebookTests(unittest.TestCase):
    def test_cloud_notebook_is_configured_for_direct_run_all(self):
        notebook = json.loads(
            Path("notebooks/00_cloud_training.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn("AUTO_INSTALL_DEPENDENCIES = True", source)
        self.assertIn("find_rdd2022_archives", source)
        self.assertIn("REBUILD_DATASET = False", source)
        self.assertIn("RUN_TRAINING = True", source)
        self.assertIn("SKIP_TRAINED_MODELS = True", source)


if __name__ == "__main__":
    unittest.main()
