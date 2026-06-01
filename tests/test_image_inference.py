import unittest

from src.image_inference import analyze_image


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeImage:
    shape = (100, 100, 3)

    def copy(self):
        return self


class FakeModel:
    def __init__(self, result):
        self.result = result

    def predict(self, source, conf, verbose):
        return [self.result]


class ImageInferenceTests(unittest.TestCase):
    def test_empty_detection_returns_low_risk_and_message(self):
        model = FakeModel(FakeResult(FakeBoxes([], [], [])))
        result = analyze_image(FakeImage(), model, 0.25)

        self.assertEqual(result["details"], [])
        self.assertEqual(result["risk_summary"]["overall_risk"], "低")
        self.assertEqual(result["message"], "未发现缺陷")

    def test_confidence_is_reported_but_not_added_to_score(self):
        model = FakeModel(FakeResult(FakeBoxes([[0, 0, 10, 10]], [0.99], [0])))
        result = analyze_image(FakeImage(), model, 0.25)

        self.assertEqual(result["details"][0]["confidence"], 0.99)
        self.assertEqual(result["details"][0]["priority_score"], 35)


if __name__ == "__main__":
    unittest.main()

