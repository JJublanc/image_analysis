import numpy as np
from src.utils import load_class_names
import os
import pandas as pd
import zipfile

def unzip_folder():
    directory_to_extract_to = "data/cards"
    path_to_zip_file = "data/cards/playing-cards-dataset.zip"
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    os.rename("data/cards/train_zipped", "data/cards/train")
    os.rename("data/cards/test_zipped", "data/cards/test")
    os.remove("data/cards/playing-cards-dataset.zip")

def unzip_data(data_kind="test"):

    directory_to_extract_to = "data/cards/{}".format(data_kind)
    path_to_zip_file = "data/cards/{}_zipped.zip".format(data_kind)
    try:
        os.mkdir(directory_to_extract_to)
    except:
        print("diretory already exists")
        pass

    if len(os.listdir("data/cards/{}".format(data_kind))) != 0:
        print("{} data are already unzipped".format(data_kind))
    else:
        directory_to_extract_to = "data/cards/{}".format(data_kind)
        path_to_zip_file = "data/cards/{}_zipped.zip".format(data_kind)

        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

def get_card_num(card):
    cards = load_class_names("data/cards/cards.names")
    cards_serie = pd.Series(cards)
    return cards_serie[cards_serie==card].index[0]


def resize_boxes_limits(x_min, y_min, x_max, y_max, origin, destination):
    return x_min*destination[0]//origin[0],\
           y_min*destination[1]//origin[1],\
           x_max*destination[0]//origin[0],\
           y_max*destination[1]//origin[1]


def resize_and_add_class_num(df):
    df[["x_min_resized",
                 "y_min_resized",
                 "x_max_resized",
                 "y_max_resized"]] = df.apply(lambda x : resize_boxes_limits(x["xmin"],
                                                                             x["ymin"],
                                                                             x["xmax"],
                                                                             x["ymax"],
                                                                             [x["width"],
                                                                              x["height"]],
                                                                            [460, 460]),
                                              axis=1)\
                                                           .apply(pd.Series)

    df["class_num"] = df["class"].map(get_card_num)
    return df


def save_csv_resized_class_num(data_kind):
    df = pd.read_csv("./data/cards/{}_cards_label.csv".format(data_kind), nrows=5)
    if len(df.columns) < 13:
        df = pd.read_csv("./data/cards/{}_cards_label.csv".format(data_kind))
        df = resize_and_add_class_num(df)
        df.to_csv("./data/cards/{}_cards_label.csv".format(data_kind))


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format
    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value
    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true