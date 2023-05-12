import os
from shutil import copy
import random
from tqdm import tqdm

#  该文件用于随机分配数据集，分为训练、测试、验证三个集合
#  其中为数据质量进行了特殊处理，因此无法使用集成的数据折叠方法
all_with = os.listdir('./all_with')  # 全部的含异物图像
all_without = os.listdir('./all_without')  # 全部的无异物图像
well_with = os.listdir('../Chest_20211008/with3375')  # 20211008的数据标注比较优质
well_without = os.listdir('../Chest_20211008/without2235')
CONTORL_PARA = 0.8  # 抽取比例
TEST_NUM = 600

#  下面先构建测试集,分为无异物部分和含异物部分，测试中保持1：1
well_with_1000 = random.sample(well_with, TEST_NUM)  # 优质异物数据集中选取1000张作为最后的测试集
for i, item in enumerate(well_with_1000):
    well_with_1000[i] = item.split('_')[1]  # 去除下划线的傻逼操作，烦死我了

all_without_1000 = random.sample(all_without, TEST_NUM)  # 从整体数据中挑1000张作为测试集的无异物图像
test_with = well_with_1000
test_without = all_without_1000
print('test have', len(test_with), len(test_without))

# 其实一般数据集，这些left的数据都是训练集，而我们需要继续划分为训练集和验证集，这里先进行划分
all_with_left = list(set(all_with) - set(well_with_1000))
all_without_left = list(set(all_without) - set(all_without_1000))
print('with test', len(all_with), len(well_with_1000), len(all_with_left))  # 测试一下重复性
print('without test', len(all_without), len(all_without_1000), len(all_without_left))

#  下面构建训练集和验证集，训练集只需要含异物图像，因此从剩余部分随机挑选，剩下的含异物图像和同等数量的无异物图像构成
train_with = random.sample(all_with_left, int(len(all_with_left)*CONTORL_PARA))
val_with = list(set(all_with_left) - set(train_with))
print('train and val', len(train_with), len(val_with))
val_without = random.sample(all_without_left, len(val_with))
print('val have', len(val_with), len(val_without))

# # 开始复制数据，先进行含异物图像的分配，主要为600测试，2911训练，728验证
with_file = os.listdir('./all_with')
for filename in tqdm(with_file):  # 全部训练数据的文件名
    from_path = './all_with/' + filename
    if filename in test_with:
        to_path = './trainAndTest/test/' + filename
    elif filename in train_with:
        to_path = './trainAndTest/train/' + filename
    elif filename in val_with:
        to_path = './trainAndTest/val/' + filename
    copy(from_path, to_path)

#  然后进行无异物图像的分配，600测试，0训练，728验证
without_file = os.listdir('./all_without')
for filename in tqdm(without_file):
    from_path = './all_without/' + filename
    if filename in test_without:
        to_path = './trainAndTest/test/' + filename
        copy(from_path, to_path)
    elif filename in val_without:
        to_path = './trainAndTest/val/' + filename
        copy(from_path, to_path)


#  这里的基本处理方法就是合并三批数据的无异物图像和含异物图像
#  然后出现的问题主要在于第一批数据中存在漏标，而第二批图像中出现了对应的标注，导致无异物图像中存在含异物图像
