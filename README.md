First we'll start with a quick description of what we used as code and data, and then, you'll find 
what you need to do to reproduce our results


# Data
We used cards data set available [here](https://www.kaggle.com/hugopaigneau/playing-cards-dataset#train.record
). It is composed of a train set and test set, and the boxes are already labeled. Yet, a part of the 
code helps to refactor the boxes to match the YOLOv3 expected format

Therefore we have to set YOLO to 52 output classes as we have 52 cards in a game

All the images are representing a hand of cards (around 6 cards). Therefore, YOLO will 
have to find (around) 6 boxes. Those hands are put over backgrounds 
that will help to generalize.

# Steps to reproduce the project
## 0- Requirements

Make sure requirements are fullfiled doing `pip install -r requirements.txt`

## 1- Download model weights and transform it to H5 format
### 1-1 Download weights
The weight are available [here](https://pjreddie.com/media/files/yolov3.weights) 
(**carefull: click will launch download !!**). The weights are taking around 250Mo

Put the weight here: `load_and_convert_weights/weights` 

*_Documentation on Yolov3 model is accessible [here](https://machinelearningspace.com/yolov3-tensorflow-2-part-2/)_*

### 1-2 Convert model to h5

In this project, we'll use model saved under h5 format. To convert, launch `convert_weights.py`. 
This will save the weights under `load_and_convert_weights/weights/yolov3_weights.h5`.


## 2- Download and prepare data
[Data](https://www.kaggle.com/hugopaigneau/playing-cards-dataset) are to be loaded in the following folder: `data/cards`. Make sure you download both csv label files and zipped file

(Reminder: you can download using kaggle api: `kaggle datasets dowload -d hugopaigneau/playing-ca`)

Then, by launching in terminal `python script_unzip_and_process_images.py`, files will be unzipped in subfolders `train` and `test`.Those two folders contain jpg and xml files.

Also the original zip file is deleted during the operation

Also, the script will create `annotation_test.txt` and `annotation_train.txt`, that is formatting the boxes
correctly.

Finally, in `test_cards_label.csv` & `train_cards_label.csv`, new columns are build, in order to resize the images
to format (416, 416). Boxes are also resized to fit with the new images format.


## 3- Training !

`python train.py train'


# Référence

* https://kaggle.com/hugopaigneau/playing-cards-dataset
* https://machinelearningspace.com/yolov3-tensorflow-2-part-2/
* https://github.com/qqwweee/keras-yolo3/tree/e6598d13c703029b2686bc2eb8d5c09badf42992
* https://pjreddie.com/media/files/yolov3.weights
