import random
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
#  读取所有2911训练集，读取所有选框，换算成1000边长内的，然后进行聚类，获取5种长宽比，然后按我心情进行设置


def new_cal_iou(anchor_w, anchor_h, clus_w, clus_h):
    min_w = min(anchor_w, clus_w)
    min_h = min(anchor_h, clus_h)
    intersection = min_w * min_h
    iou = intersection / (anchor_w * anchor_h + clus_w * clus_h - intersection)
    return iou

lbael_localization = pd.read_csv('./1022labels.csv', na_filter=False)  # 包含所有异物
label_tuples = lbael_localization.itertuples(index=False)

train_img_list = os.listdir('./trainAndTest/train/')
wait_for_cluster_anchor = []
# reshape_std = 800
# for row in tqdm(label_tuples):
#     if row.sop+'.jpg' in train_img_list:  # 当前行的异物对应的sop就是训练数据，说明这是待聚类的异物框
#         img = Image.open('./trainAndTest/train/' + row.sop + '.jpg').convert("RGB")
#         width, height = img.size[0], img.size[1]
#         anno = row.points.replace('[', '')
#         anno = anno.replace(']', '')
#         anno = anno.replace(' ', '')
#         points = anno.split(',')
#         anno = list(map(int, points))
#         anno[0] = anno[0] / width * reshape_std
#         anno[1] = anno[1] / height * reshape_std
#         anno[2] = anno[2] / width * reshape_std
#         anno[3] = anno[3] / height * reshape_std
#         wait_for_cluster_anchor.append([abs(anno[2]-anno[0]), abs(anno[3]-anno[1])])
#
#
# print(len(wait_for_cluster_anchor))
# f = open('anchor_wh.txt', 'w+')
# for i in range(len(wait_for_cluster_anchor)):
#     for j in range(2):
#         strNum = str(wait_for_cluster_anchor[i][j])
#         f.write(strNum)
#         f.write(' ')
#     f.write('\n')
# f.close()

file = open('anchor_wh.txt', 'r')
arr1 = file.readlines()

for item in arr1:
    wait_for_cluster_anchor.append(list(map(float, item.split(' ')[:2])))

num_class = 24
k_anchor_wh = random.sample(wait_for_cluster_anchor, num_class)

for T in range(10000):
    groups = [[] for _ in range(num_class)] # 五个组，依次以上述抽取的5个长宽为中心
    for clus in wait_for_cluster_anchor:  # 遍历所有候选宽高
        ious = []
        for anchor in k_anchor_wh:  # anchor就是聚类中心，
            ious.append(new_cal_iou(anchor[0], anchor[1], clus[0], clus[1]))
        #  聚类的原理就是越相近的越聚在一起，所以iou越大说明越相近，这里取最小的ious就出现了错误
        # min_idx = ious.index(min(ious))
        # groups[min_idx].append(clus)
        # print('找到和几个聚类中心距离最短的，也就是iou最大的，最接近的')
        max_idx = ious.index(max(ious))  # 找到最相近的anchor，然后直接找到其下标
        groups[max_idx].append(clus)
    old_k_anchor_wh = []
    for i in range(num_class):
        old_k_anchor_wh.append(k_anchor_wh[i])
    for i, group in enumerate(groups):  # 单个anchor对应的组内
        if len(group) == 0:
            continue
        sum_w = 0
        sum_h = 0
        for item in group:
            sum_w += item[0]
            sum_h += item[1]
        k_anchor_wh[i] = [sum_w/len(group), sum_h/len(group)]
    k_anchor_wh.sort(key=lambda x: (x[0]))
    print(T, k_anchor_wh)
    if k_anchor_wh == old_k_anchor_wh:
        break



