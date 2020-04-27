from process_images import unzip_data, save_csv_resized_class_num
from utils import create_annotation_txt
import sys
import pandas as pd

data_kind = sys.argv[1]

unzip_data(data_kind=data_kind)
save_csv_resized_class_num(data_kind)

# Annotations
labels = pd.read_csv("data/data_cards/{}_cards_label.csv".format(data_kind),
                     index_col=False)
text_series = create_annotation_txt(labels, "data/data_cards/{}".format(data_kind))
text_series.to_csv("data/data_cards/annotation_{}.txt".format(data_kind),
                   index=False,
                   header=False)


