import unittest
from train import get_classes
from train import get_anchors


class TestGoogleSearchFunctions(unittest.TestCase):

    def setUp(self):
        self.seed=1234
        self.classes_path="./data/data_cards/cards.names"
        self.yolo_anchors="./data/data_cards/yolo_anchors.txt"

    def test_get_classes(self):
        self.assertEqual(len(get_classes(self.classes_path)),52)

    def test_get_anchors(self):
        self.assertEqual(get_anchors(self.yolo_anchors).shape, (9, 2))


if __name__ == '__main__':
    unittest.main()

