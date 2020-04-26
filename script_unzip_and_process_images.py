from process_images import unzip_data, save_csv_resized_class_num, create_anotation_txt
import sys

data_kind = sys.argv[1]


unzip_data(data_kind=data_kind)
save_csv_resized_class_num(data_kind)
create_anotation_txt(path="data/data_cards/", data_kind)



