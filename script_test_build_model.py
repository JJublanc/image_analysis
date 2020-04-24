from train import *
anchors = get_anchors("./data/data_cards/yolo_anchors.txt")
create_model((416, 416),
             anchors,
             num_classes=52,
             load_pretrained=True,
             weights_path="./load_and_convert_weights/weights/yolov3_weights.h5")

