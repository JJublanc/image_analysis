import tensorflow as tf
import sys
sys.path.append("../")
from src.utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import numpy as np
from yolov3 import YOLOv3Net
import os

print(os.listdir())
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

# tf.config.experimental.set_memory_growth(physical_devices[0], True)
model_size = (416, 416, 3)
num_classes = 80
class_name = './data/cards/cards.names'
max_output_size = 40
max_output_size_per_class= 20
iou_threshold = 0.5
confidence_threshold = 0.5
cfgfile = './load_and_convert_weights/cfg/yolov3.cfg'
weightfile = './load_and_convert_weights/weights/yolov3_cards_weights_train_stage-1-epoch-50_.h5'
image_name = "card_random"
img_path = "./data/random_images"


def main(img_path, image_name):
    model = YOLOv3Net(cfgfile,model_size,num_classes)
    model.load_weights(weightfile, by_name=True, skip_mismatch=True)
    class_names = load_class_names(class_name)
    image = cv2.imread(os.path.join(img_path, "{}.jpg".format(image_name)))
    image = np.array(image)
    image = tf.expand_dims(image, 0)
    resized_frame = resize_image(image, (model_size[0],model_size[1]))
    pred = model.predict(resized_frame)
    boxes, scores, classes, nums = output_boxes( \
        pred, model_size,
        max_output_size=max_output_size,
        max_output_size_per_class=max_output_size_per_class,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold)
    image = np.squeeze(image)
    img = draw_outputs(image, boxes, scores, classes, nums, class_names)
    print(pred.shape)
    # win_name = 'Image detection'
    # cv2.imshow(win_name, img)
    # time.sleep(20)
    # cv2.destroyAllWindows()

    #If you want to save the result, uncommnent the line below:
    cv2.imwrite(os.path.join(img_path, "{}_yolo.jpg".format(image_name)), img)


main(img_path, image_name)





