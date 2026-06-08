import json
import unittest
from pathlib import Path


class GuiNotebookTests(unittest.TestCase):
    def test_gui_notebook_auto_discovers_models_without_requiring_weights(self):
        notebook = json.loads(
            Path("notebooks/06_gradio_gui.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn("models_root=PROJECT_ROOT / 'models'", source)
        self.assertIn(
            "metrics_path=PROJECT_ROOT / 'reports' / 'model_comparison.csv'", source
        )
        self.assertNotIn("if not model_registry:", source)
        self.assertNotIn("找不到任何本项目真实训练权重", source)


if __name__ == "__main__":
    unittest.main()
