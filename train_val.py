import torch
import numpy as np
import os
import pandas as pd
from tools import utils
from torch.utils.data import Dataset

from tools.engine import train_one_epoch
# import models.model_se_stn_nonl as modelss
import net.use_for_me_model_se_stn_nonl as modelme
import net.use_for_me_model_mobilev2 as mobilev2
import torchvision.transforms as transforms
from dataSet import ForeignObjectDataset
from tqdm import tqdm
from test import mzc_test
from Chest_ALL.cal_map_hospital import CAL_MAP
import wandb
np.random.seed(0)
torch.manual_seed(0)
# from sklearn.metrics import roc_auc_score, roc_curve, auc

OBJECT_SEP = ';'
ANNOTATION_SEP = ' '
#  默认800*800， ciou， noclahe, frozenBN
savename = 'model_noeca_33pooling11_train4_nomixup_stn_anchor140_20epoch'
data_dir = './Chest_ALL/trainAndTest/'
device = torch.device('cuda:0')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



# 参数配置  下面还有个模型配置也很关键
reshape_std = 800  # 在dataset初始化test数据集和最后产生最后的预测框都需要用到这个reshape参数
num_classes = 2  # 一种前景，一种背景
with_clahe = 0  # 0不适用clahe ，1 适用clahe

# Build Dataset
# 7111的图片,其中5534带标签
labels_tr = pd.read_csv(data_dir + '1022labels.csv', na_filter=False)  # 无jpg后缀
img_list = []
for item in labels_tr.sop:
    img_list.append(item + '.jpg')
dict_tr = dict(zip(img_list, [[] for _ in range(len(img_list))]))  # 看起来上面的img_list重复了，实际上key不会重复
for si, poi in zip(labels_tr.sop, labels_tr.points):
    # 下方文件处理时，需要此处的第一列带有后缀名
    si = si + '.jpg'
    dict_tr[si].append(poi)
print("含异物图像数量：", len(dict_tr))

dataset_train = ForeignObjectDataset(datafolder=data_dir + 'train/',
                                     datatype='train',
                                     labels_dict=dict_tr,
                                     reshape_std=reshape_std,
                                     mixup=0)
dataset_val = ForeignObjectDataset(datafolder=data_dir + 'val/',
                                   datatype='dev',
                                   labels_dict=dict_tr,
                                   reshape_std=reshape_std,
                                   mixup=0)

data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=8, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)


data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)


# model_ft = modelss.get_detection_model(num_classes)
# model_ft = modelme.get_detection_local_model(num_classes,
#                                              pretrain_path="./faster_rcnn/backbone/convnext_tiny_1k_224_ema.pth")
# model_ft = modelme.get_detection_local_model(num_classes)
# model_ft.load_state_dict(torch.load("model_0306.pt"))
# print('使用了修改了rpn的本地实现的faster')

# model_ft = modelme.get_detection_model(num_classes, 0)
model_ft = modelme.get_detection_local_model1(num_classes, 0, 'frozenBN')  # resnet50  没有rewrite
# model_ft = modelme.get_detection_local_model1(num_classes, 1, 1)  # resnet 50
# model_ft = modelme.get_detection_local_model2(num_classes, 1, 0)  # convnext
# model_ft = modelme.get_detection_local_model3(num_classes)  # 3  STN
# model_ft = modelme.get_detection_local_model4(num_classes)  # 4 SE
# model_ft = modelme.get_detection_local_model5(num_classes)  # 5 Nonlocal
# model_ft = modelme.get_detection_local_model6(num_classes)  # 6 nonlocal + stn (经过测试，stn覆盖了se的功能)
# model_ft = modelme.get_detection_local_model7(num_classes)  # 7 nonlocal + se
# model_ft = mobilev2.create_model(num_classes)
model_ft.to(device)
params = [p for p in model_ft.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,  # 每过step_size次，更新学习率
                                               gamma=0.1)  # 学习率下降因子

#  为了模型！！
# print('载入奇怪的模型2')
# model_ft.load_state_dict(torch.load('./pth_save/model_noeca_33pooling11_train4_nomixup_20epoch19.pt'), strict=False)

num_epochs = 20
if num_epochs==0:
    print('只进行测试')
else:
    print('训练模式num_epochs', num_epochs)

# 这里是进行多次15epoch的迭代测试
# model_ft.load_state_dict(torch.load('model_c_iou_without_800_clahe_45epoch.pt'), strict=False)
# print('载入了15个epoch预训练，因此是', num_epochs+45)



auc_max = 0
if not os.path.exists('./pth_save'):
    os.mkdir('./pth_save')

for epoch in range(num_epochs):

    train_one_epoch(model_ft, optimizer, data_loader, device, epoch, print_freq=20)

    lr_scheduler.step()
    # if epoch == num_epochs/2-1:
    #     torch.save(model_ft.state_dict(), savename + '_half.pt')
    torch.save(model_ft.state_dict(), './pth_save/' + savename + str(epoch) + '.pt')

    a = mzc_test(savename='', model=model_ft, choose='frozenBN')  #没有model载入才需要选择choose
    a.start()
    matrix, aa = CAL_MAP(THRESHOLD=0.5, CONFIDENCE=1, csvname='temp.csv', data_dir='./Chest_ALL/trainAndTest/',
                         csv_dir='./pth_save/', label_csv='./Chest_ALL/threelabel/auditor_info_new.csv').cal()
    print(aa)
    continue
    # model_ft.eval()
    #
    # val_pred = []
    # val_label = []
    # for batch_i, (image, label, width, height) in enumerate(data_loader_val):
    #     image = list(img.to(device) for img in image)
    #
    #     # 先进行acc和auc的计算
    #     val_label.append(label[-1])  # 一个batch中一张图像，所以无所谓0下标和-1下标
    #     outputs = model_ft(image)
    #     if len(outputs[-1]['boxes']) == 0:
    #         val_pred.append(0)
    #     else:
    #         val_pred.append(torch.max(outputs[-1]['scores']).tolist())  # 若是含有多个box那就选一个置信度最高的代表该图片
    # # 经过遍历，得到了所有图像的含异物情况，通过阈值进行二值化
    # val_pred_label = []
    # for i in range(len(val_pred)):
    #     if val_pred[i] >= 0.5:
    #         val_pred_label.append(1)
    #     else:
    #         val_pred_label.append(0)
    #
    # number = 0
    # for i in range(len(val_pred_label)):
    #     if val_pred_label[i] == val_label[i]:
    #         number += 1
    # acc = number / len(val_pred_label)
    #
    # auc = roc_auc_score(val_label, val_pred)
    # print('Epoch: ', epoch, '| val acc: %.4f' % acc, '| val auc: %.4f' % auc)
    #
    # if auc > auc_max:
    #     auc_max = auc
    #     print('Best Epoch: ', epoch, '| val acc: %.4f' % acc, '| Best val auc: %.4f' % auc_max)



print('-----------test--------------')
dataset_test = ForeignObjectDataset(datafolder=data_dir + 'test/',
                                    datatype='dev',
                                    labels_dict=dict_tr,
                                    reshape_std=reshape_std,
                                    mixup=0)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

# model

model = model_ft
model.to(device)
# model.load_state_dict(torch.load(savename + '.pt'), strict=False)
# torch.save(model.state_dict(), 'model_biao_1230_gai.pth', _use_new_zipfile_serialization=False)
# sys.exit()
model.eval()

locs = []

for image, label, width, height in tqdm(data_loader_test):  #

    image = list(img.to(device) for img in image)
    outputs = model(image)


    center_points = []
    center_points_preds = []  # scores

    if len(outputs[-1]['boxes']) == 0:
        center_points.append([])
        center_points_preds.append('')
        locs.append('')
    else:
        new_output_index = torch.where((outputs[-1]['scores'] > 0))
        new_boxes = outputs[-1]['boxes'][new_output_index]
        new_scores = outputs[-1]['scores'][new_output_index]
        # new_labels = outputs[-1]['labels'][new_output_index]

        for i in range(len(new_boxes)):
            new_box = new_boxes[i].tolist()
            # center_x = (new_box[0] + new_box[2]) / 2
            # center_y = (new_box[1] + new_box[3]) / 2
            center_x1 = new_box[0]
            center_y1 = new_box[1]
            center_x2 = new_box[2]
            center_y2 = new_box[3]
            center_points.append([center_x1 / reshape_std * width[-1], center_y1 / reshape_std * height[-1],
                                  center_x2 / reshape_std * width[-1], center_y2 / reshape_std * height[-1]])
        center_points_preds += new_scores.tolist()
        # center_points_labels += new_labels.tolist()

        line = ''
        for i in range(len(new_boxes)):
            if i == len(new_boxes) - 1:
                line += str(center_points_preds[i]) \
                        + ' ' + str(center_points[i][0]) + ' ' + str(center_points[i][1]) \
                        + ' ' + str(center_points[i][2]) + ' ' + str(center_points[i][3])
            else:
                line += str(center_points_preds[i]) \
                        + ' ' + str(center_points[i][0]) + ' ' + str(center_points[i][1]) \
                        + ' ' + str(center_points[i][2]) + ' ' + str(center_points[i][3]) + ';'
        locs.append(line)

loc_res = pd.DataFrame({'image_name': dataset_test.image_files_list_test, 'prediction': locs})
loc_res.to_csv('./cla_loc_csv/' + savename + '.csv', columns=['image_name', 'prediction'], sep=',', index=None)
print('localization.csv generated.')
