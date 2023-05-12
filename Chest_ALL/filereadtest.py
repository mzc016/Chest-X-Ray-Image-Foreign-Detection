import numpy as np
import os
import pandas as pd
data_dir = './'



train_list = os.listdir(data_dir + 'trainAndTest/train')
print('train have ' + str(len(train_list)))
test_list = os.listdir(data_dir + 'trainAndTest/test')
print('test have ' + str(len(test_list)))
val_list = os.listdir(data_dir + 'trainAndTest/val')
print('val have ' + str(len(val_list)))
print('allwith have ' + str(len(os.listdir('./all_with'))))
print('allwithout have ' + str(len(os.listdir('./all_without'))))

# with_list = os.listdir(data_dir + 'all_with')
# print('with have ' + str(len(with_list)))
# without_list = os.listdir(data_dir + 'all_without')
# print('without have ' + str(len(without_list)))


# label_localization = pd.read_csv(data_dir + 'threelabel/auditor_info_new.csv', na_filter=False)  # 第三批数据的label测试
label_localization = pd.read_csv(data_dir + 'trainAndTest/1022labels.csv', na_filter=False)  # 第三批数据的label测试
label_tuples = label_localization.itertuples(index=False)
dict_label = {}
lines = 0
for row in label_tuples:
    dict_label[row.sop + '.jpg'] = row.points
    lines += 1
print('foreign object：' + str(lines))
count = 0
all = 0
for item in test_list:
    if item in dict_label.keys():
        count += 1
    all += 1

print(str(count)+'/'+str(all))
count = 0
all = 0
for item in train_list:
    if item in dict_label.keys():
        count += 1
    all += 1

print(str(count)+'/'+str(all))

count = 0
all = 0
for item in val_list:
    if item in dict_label.keys():
        count += 1
    all += 1

print(str(count)+'/'+str(all))

#  这里出现了五五开的有异物和无异物图像列表，存在少量无异物图像被判定为含有异物，只能说明他们原始是无异物标签的
#  第一批数据有少量无异物图像含有异物但是没有对应标签，而很可能之后的图像重名后存在含有义务的图像，最终导致数据偏差，我不想改了
#  ——————————————————————————
#  更新了一下数据，重新随机选择，这次将无异物图像的抽取范围也选择在最后一批数据中，同时还去除了第三批数据中，在前几批数据产生的标注情况

#  这里处理一下数据的类别，按照图像统计性别、年龄的分布

f = open('dictall.txt', 'r+')
dic_all = eval(f.read())
print(len(dic_all))
T = ['train', 'val', 'test']
for choose in T:
    ageyou = 0
    agewu = 0
    arr = []
    age1 = 0
    age0 = 0
    for item in os.listdir('./trainAndTest/' + choose):  # 遍历当前的文件夹内的文件
        itemkey = os.path.splitext(item)[0]
        if itemkey in dic_all:  # 去除文件名后缀，保持和dic_all的key格式一致
            if dic_all[itemkey][0]:
                ageyou += 1
                if int(dic_all[itemkey][0]) != 0:
                    age1 += 1
                    arr.append(int(dic_all[itemkey][0]))
                else:
                    age0 += 1

            if dic_all[itemkey][0]=='':
                agewu += 1
    print(choose, '--------------------------------')
    print('均值:', np.mean(arr))
    # print('方差:', np.var(arr))
    print('标准差', np.std(arr))
    print('有年龄', ageyou)
    print('无年龄', agewu)
    print('年龄记录为0', age0)
    print('年龄记录正常', age1)




















