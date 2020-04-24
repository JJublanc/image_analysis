from train import *
anchors = get_anchors("./data/data_cards/yolo_anchors.txt")
create_model((416, 416), anchors, num_classes = 52, load_pretrained=False)