from process_images import unzip_folder, save_csv_resized_class_num
from utils import create_annotation_txt
import os
import pandas as pd
from shutil import copyfile

# kaggle datasets download -d hugopaigneau/playing-cards-dataset

#########################
# unzip the main folder #
#########################

if "playing-cards-dataset.zip" in os.listdir('data/data_cards'):
    unzip_folder()
    print("folder unzipped - OK")
else:
    print("no file to unzip")

############################
# Create annotations files #
############################
for data_kind in ["train", "test"]:
    if "annotation_{}.txt".format(data_kind) not in os.listdir("data/data_cards/"):
        print("os.listdir()==", os.listdir("data/data_cards/"))
        ##################################################################
        # original csv are process to match format expected by the model #
        ##################################################################
        save_csv_resized_class_num(data_kind)

        ###########################################
        # A txt file is created from the csv file #
        ###########################################
        labels = pd.read_csv("data/data_cards/{}_cards_label.csv".format(data_kind),
                             index_col=False)
        text_series = create_annotation_txt(labels, "./data/data_cards/{}/".format(data_kind))
        text_series.to_csv("data/data_cards/annotation_{}.txt".format(data_kind),
                           index=False,
                           header=False)
        print("csv {} saved - OK".format(data_kind))

        ################################################
        # Remove unwanted characters from the txt file #
        ################################################
        with open("./data/data_cards/annotation_{}.txt".format(data_kind), "r") as file: # Read in the file
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace('"', '')

            # Write the file out again
        with open("./data/data_cards/annotation_{}.txt".format(data_kind),"w") as file:
            file.write(filedata)
        print("name modified for {} - OK".format(data_kind))
    else:
        print("{} annotations already exist".format(data_kind))

#####################################################
# Create a small annotation file to make some tests #
#####################################################
if "annotation_small.txt" in os.listdir("data/data_cards/"):
    os.remove("./data/data_cards/annotation_small.txt")

with open("./data/data_cards/annotation_train.txt", "r") as f:
    lines = f.readlines()
    for line in lines[:72]:
        with open("data/data_cards/annotation_small.txt", "a") as f1:
            f1.writelines(line)
            copyfile(line.split(" ")[0], line.split(" ")[0].replace("train", "small"))

