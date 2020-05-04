import numpy as np
from tensorflow.keras.optimizers import Adam
import sys
from src.model_compile import get_classes, get_anchors, get_initial_stage_and_epoch, \
                              create_model, data_generator_wrapper

from src.yolov3_class import YOLO
import PIL
import cv2
import os

def main():

    ###########################################
    # Instaciate the object of the YOLO class #
    ###########################################

    model_path = './load_and_convert_weights/weights/yolov3_cards_weights_train_stage-1-epoch-50_.h5'

    #model_path = "./load_and_convert_weights/weights/yolov3_coco_weights_small_stage-1-epoch-1_.h5"
    anchors_path = 'data/cards/yolo_anchors.txt',
    classes_path = 'data/coco/coco_names.txt',

    # path
    folder = "/Users/johanjublanc/DataScienceProjects/video_analysis/data/random_images"
    file_name = "car"
    file_path = os.path.join(folder, file_name + ".jpg")
    file_yolo_path = os.path.join(folder, file_name + "_yolo.jpg")
    yolo_transformer = YOLO(model_path=model_path,
                            anchors_path='data/cards/yolo_anchors.txt',
                            # classes_path='data/cards/cards_names.txt',
                            classes_path='data/coco/coco_names.txt',
                            score=0.3,
                            iou=0.45,
                            model_image_size=(416, 416),
                            gpu_num=1)
    image = PIL.Image.open(file_path)
    image_boxed = yolo_transformer.detect_image(image)
    image_boxed.save(file_yolo_path)
    yolo_transformer.close_session()


if __name__ == "__main__":
    main()

