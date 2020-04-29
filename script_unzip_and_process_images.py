from process_images import unzip_data, save_csv_resized_class_num
from utils import create_annotation_txt
import sys
import pandas as pd
from process_images import unzip_folder

# kaggle datasets download -d hugopaigneau/playing-cards-dataset

unzip_folder()
print("folder unzipped - OK")

# Annotations
for data_kind in ["train", "test"]:
    save_csv_resized_class_num(data_kind)
    
    labels = pd.read_csv("data/data_cards/{}_cards_label.csv".format(data_kind),
                         index_col=False)
    text_series = create_annotation_txt(labels, "./data/data_cards/{}/".format(data_kind))
    text_series.to_csv("data/data_cards/annotation_{}.txt".format(data_kind),
                       index=False,
                       header=False)
    print("csv {} saved - OK".format(data_kind))

    with open("data/data_cards/annotation_{}.txt".format(data_kind),"r") as file: # Read in the file
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('"', '')

    # Write the file out again
    with open("data/data_cards/annotation_{}.txt".format(data_kind),"w") as file: # Read in the file
        file.write(filedata)
    print("name modified for {} - OK".format(data_kind))

