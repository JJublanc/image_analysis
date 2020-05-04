import unittest
from src.model_compile import get_classes, get_anchors
from src.utils import get_initial_stage_and_epoch


class TestGoogleSearchFunctions(unittest.TestCase):

    def setUp(self):
        self.seed=1234
        self.classes_path="./data/cards/cards.names"
        self.yolo_anchors="./data/cards/yolo_anchors.txt"
        self.weights_file_name = "yolov3_weight-train_stage-2-epoch-50_.h5"

    def test_get_classes(self):
        self.assertEqual(len(get_classes(self.classes_path)),52)

    def test_get_anchors(self):
        self.assertEqual(get_anchors(self.yolo_anchors).shape, (9, 2))

    def test_get_initial_stage_and_epoch(self):
        self.assertEqual(get_initial_stage_and_epoch(self.weights_file_name), (2, 50))
        self.assertEqual(get_initial_stage_and_epoch("yolo.h5"), (0, 0))


if __name__ == '__main__':
    unittest.main()

