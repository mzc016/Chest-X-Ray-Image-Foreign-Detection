import torch
import torchvision
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

OBJECT_SEP = ';'
ANNOTATION_SEP = ' '

data_dir = './trainAndTest/'

device = torch.device('cuda:0')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_classes = 2  # object (foreground); background
csvname = 'recon_15epoch_3_31_outtest'
print(csvname)


def draw_annotation(im, anno_str, thresh1, thresh2, fill=(255, 63, 63, 40), resize=2):  # 预测结果
    if anno_str == '':
        return False
    draw = ImageDraw.Draw(im, mode="RGBA")
    flag = 0
    for anno in anno_str.split(OBJECT_SEP):
        anno = list(map(float, anno.split(ANNOTATION_SEP)))
        for index in range(1, len(anno)):
            anno[index] = anno[index]*2
        if thresh1 <= anno[0] <= thresh2:
            flag = 1
            draw.rectangle(anno[1:], fill=fill)
            font = ImageFont.truetype("方正楷体简体.ttf", 20, encoding="unic")  # 设置字体
            draw.text((anno[1], anno[4]), str(anno[0]), 'fuchsia', font)
    if flag == 0:
        return False
    else:
        return True


def draw_annotation2(im, points_str, fill=(63, 63, 255, 55)):  # 标签
    draw = ImageDraw.Draw(im, mode="RGBA")
    for anno in points_str:
        anno = anno.replace('[', '')
        anno = anno.replace(']', '')
        anno = anno.replace(' ', '')
        points = anno.split(',')
        anno = list(map(int, points))
        draw.rectangle(anno, fill=fill)


# prediction_localization = pd.read_csv('../cla_loc_csv/' + csvname + '.csv', na_filter=False)  # 含异物和不含异物都有
prediction_localization = pd.read_csv('../cla_loc_csv/' + csvname + '.csv', na_filter=False)
label_localization = pd.read_csv('./trainAndTest/recon_out_test_label.csv', na_filter=False, encoding='gb18030')  # 只有异物数据

img_list = os.listdir(data_dir + 'out_test_recon_jpg')
length = len(img_list)

print(f'{length} pictures in {data_dir}test/')


# 生成预测值csv的元组
prediction_tuples = prediction_localization.itertuples(index=False)
label_tuples = label_localization.itertuples(index=False)



count = 0
for i in tqdm(prediction_tuples):  # 预测是对所有的图像结果进行遍历，包含了有异物和没异物的
    im = Image.open(data_dir + 'out_test_recon_jpg/' + i.image_name).convert("RGB")
    w = im.width
    h = im.height
    im = im.resize((w*2, h*2), Image.ANTIALIAS)
    thresh1 = 0.5
    thresh2 = 1
    target_path = data_dir + 'result/recon_out_pre/' + str(thresh1) + '_' + str(thresh2) + '_resize/'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    if draw_annotation(im, i.prediction, thresh1, thresh2):
        count += 1
    else:
        print(i.image_name)
    pic_path = target_path + getattr(i, 'image_name')
    im.save(pic_path)

print(count)

#
# label_tuples_merge = dict(zip(img_list, [None for _ in range(length)]))
#
# for si, poi,role in zip(label_localization.SOPInstanceUID, label_localization.points, label_localization.role):
#     # 下方文件处理时，需要此处的第一列带有后缀名
#     si = si + '.jpg'
#     poi = poi[13:-2]
#     temp = poi.replace('[', '')
#     temp = temp.replace(']', '')
#     temp = temp.replace(' ', '')
#     temp = temp.split(',')
#     add = []
#     for i in range(len(temp)//4):
#         add.append(','.join(temp[4*i:4*(i+1)]))
#     if not label_tuples_merge[si]:
#         label_tuples_merge[si] = add
#     else:
#         if role == 'auditor':
#             label_tuples_merge[si] = add
#
# count = 0
# for k in tqdm(label_tuples_merge.keys()):  # k是纯净的文件名
#     # k就是图片名称，也是key
#     im = Image.open(data_dir + "out_test_recon_jpg/" + k).convert("RGB")
#     if label_tuples_merge[k]:
#         count += 1
#     draw_annotation2(im, label_tuples_merge[k])
#     pic_path = data_dir + 'result/recon_out_test_label/' + k
#     im.save(pic_path)
# print(count)
# #
