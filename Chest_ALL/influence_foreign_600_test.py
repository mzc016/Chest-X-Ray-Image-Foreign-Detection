import os
import pandas as pd

data_dir = './trainAndTest/test/'
img_list = os.listdir(data_dir)
prediction_localization = pd.read_csv('./trainAndTest/localization_all_600_1105.csv', na_filter=False)  # image_name prediction
label_localization = pd.read_csv('./threelabel/auditor_info_new.csv', na_filter=False)  # image_name annot
# 生成预测值csv的元组，完整的csv文件，不带index
prediction_tuples = prediction_localization.itertuples(index=False)
label_tuple = label_localization.itertuples(index=False)

# label只是获取sop到series的映射
sop_to_series = {}
for row in label_tuple:
    sop_to_series[row.sop] = row.SeriesInstanceUID
