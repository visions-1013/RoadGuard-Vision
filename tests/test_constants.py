import unittest

from src.constants import CLASS_CODE_TO_ID, CLASS_CODE_TO_NAME, CLASS_ID_TO_CODE


class ConstantsTests(unittest.TestCase):
    def test_class_order_is_fixed_and_unique(self):
        self.assertEqual(
            CLASS_CODE_TO_ID,
            {"D00": 0, "D10": 1, "D20": 2, "D40": 3},
        )
        self.assertEqual(
            CLASS_ID_TO_CODE,
            {0: "D00", 1: "D10", 2: "D20", 3: "D40"},
        )
        self.assertEqual(len(CLASS_CODE_TO_NAME), 4)
        self.assertEqual(len(set(CLASS_CODE_TO_NAME.values())), 4)


if __name__ == "__main__":
    unittest.main()

