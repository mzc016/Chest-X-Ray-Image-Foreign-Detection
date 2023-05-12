import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

foreignThreshold = 0
img_list = os.listdir('./trainAndTest/test')

label_list = dict.fromkeys(img_list, 0)
pre_list = dict.fromkeys(img_list, 0)
csvname = 'model_c_iou_rewrite2_without_800_clahe_15epoch.csv'
print(csvname)
prediction_pd = pd.read_csv('../cla_loc_csv/' + csvname, na_filter=False)
label_pd = pd.read_csv('./trainAndTest/1022labels.csv', na_filter=False)

prediction_tuples = prediction_pd.itertuples(index=False)
label_tuples = label_pd.itertuples(index=False)
#  这个破元组读一次好像就不能读了，我给你换成字典行了吧，草
label_tuples_dict = {}
for row in label_tuples:
    label_tuples_dict[row.sop + '.jpg'] = row.points
prediction_tuples_dict = {}
for row in prediction_tuples:
    prediction_tuples_dict[row.image_name] = row.prediction
tpr = []
fpr = []
maxacc = 0
maxacc_cut = 0
optthreshold = 0
maxcut = 0
for i in range(0, 1001):
    foreignThreshold = i/1000
    for j, img_name in enumerate(img_list):
        label_list[img_name] = 0
        pre_list[img_name] = 0

    for key in label_tuples_dict.keys():
        if key in label_list.keys():
            label_list[key] = 1  # label文件中，查到一个异物在某个图像中，那么该图像就是含异物图像

    for key in prediction_tuples_dict.keys():
        if prediction_tuples_dict[key]:
            for pre in prediction_tuples_dict[key].split(';'):
                pre = list(map(float, pre.split(' ')))
                if pre[0] >= foreignThreshold:  # 查看所有检测框的置信度，有任何大于阈值的检测框说明该图像被预测含有异物
                    pre_list[key] = 1
                    break
    tp = 0  # 完全正确
    fp = 0  # 无异物判断为有异物
    tn = 0  # 无异物判断为无异物
    fn = 0  # 有异物判断为无异物
    for img_name in img_list:
        label = label_list[img_name]
        pre = pre_list[img_name]
        if label == 1:  # 人工判断有异物
            if pre == 1:
                tp += 1
            else:
                fn += 1
        else:  # 人工判断为无异物
            if pre == 1:
                fp += 1
            else:
                tn += 1
    if foreignThreshold==0.691:
        print("tp,fn,fp,tn")
        print(tp,fn,fp,tn, foreignThreshold)
        print((tp+tn, tp+tn+fn+fp, (tp+tn)/(tp+tn+fp+fn)))
    if (tp+tn)/(tp+tn+fp+fn) > maxacc:
        maxacc = (tp+tn)/(tp+tn+fp+fn)
        maxacc_cut = foreignThreshold

    tpr.append(tp/(tp+fn))
    fpr.append(fp/(fp+tn))
    if maxcut < tp/(tp+fn) - fp/(fp+tn):
        maxcut = tp/(tp+fn) - fp/(fp+tn)
        optthreshold = foreignThreshold
# tpr = np.array(tpr)
# fpr = np.array(fpr)
print(tpr)
print(fpr)
print('标准做法，最大化tpr和fpr的差值： ', maxcut, optthreshold)
print('非标准化做法，直接在过程中计算acc，取最大的acc以及对应的阈值： ', maxacc, maxacc_cut)
plt.plot(fpr, tpr, color='b', label='roc')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

auc = 0

for i in range(1,1001):
    auc += (fpr[i-1]-fpr[i])*tpr[i]
print(auc)
auc += tpr[0]*(1-fpr[0])
print(auc)