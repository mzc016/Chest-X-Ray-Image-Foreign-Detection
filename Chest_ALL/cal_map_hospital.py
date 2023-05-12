
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pycocotools


def cal_iou(a, b):
    #  x1 y1 x2 y2
    #  0  1  2  3
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    left_max = max(a[0], b[0])
    top_max = max(a[1], b[1])
    bottom_min = min(a[3], b[3])
    right_min = min(a[2], b[2])
    inter = max(0, (right_min - left_max)) * max(0, (bottom_min - top_max))
    return inter / (a_area + b_area - inter)


class CAL_MAP(object):
    def __init__(self, THRESHOLD = 0.5, CONFIDENCE = 1, csvname = '', data_dir = './trainAndTest/',
                 csv_dir = '../cla_loc_csv/', label_csv = './threelabel/auditor_info_new.csv', test_dir = 'test/'):
        self.THRESHOLD = THRESHOLD
        self.OBJECT_SEP = ';'
        self.ANNOTATION_SEP = ' '
        self.THRESHOLD = THRESHOLD
        self.CONFIDENCE = CONFIDENCE
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        self.test_dir = test_dir
        if csvname == '':
            print('give me csvname')
            return
        # elif csvname == 'temp.csv':
        #     self.csvname = '../pth_save/' + csvname
        else:
            self.csvname = self.csv_dir + csvname
        self.label_csv = label_csv

    def cal(self):
        img_list = os.listdir(self.data_dir + self.test_dir)

        empty_list = [[] for _ in range(len(img_list))]
        img_label_dict = dict(zip(img_list, empty_list))

        prediction_localization = pd.read_csv(self.csvname, na_filter=False)  # image_name prediction
        label_localization = pd.read_csv(self.label_csv, na_filter=False, encoding='gb18030')  # image_name annot

        # 生成预测值csv的元组，完整的csv文件，不带index
        prediction_tuples = prediction_localization.itertuples(index=False)
        label_tuples = label_localization.itertuples(index=False)

        # for row in label_tuples:  #  ./threelabel/auditor_info_new.csv 中的csv要这样处理
        #     points_str = row.points
        #     points_str = points_str.replace(']', '')
        #     points_str = points_str.replace('[', '')
        #     points_str = points_str.replace(' ', '')
        #     points = points_str.split(',')
        #     anno = list(map(int, points))
        #
        #     if row.sop + '.jpg' in img_label_dict.keys():
        #         img_label_dict[row.sop + '.jpg'].append(anno)

        # 这里是定位片的label的处理方法，他们导出的label就是一张图片直接对应了所有的异物检测框
        for row in label_tuples:
            si = row.SOPInstanceUID
            poi = row.points
            role = row.role
            # 18<=age<38
            # 38<age<=58
            # 58<age<=78
            # 78<age
            age = int(row.age)
            sex = row.sex
            if sex=='M':
                continue
            # 下方文件处理时，需要此处的第一列带有后缀名
            si = si + '.jpg'
            poi = poi[13:-2]
            temp = poi.replace('[', '')
            temp = temp.replace(']', '')
            temp = temp.replace(' ', '')
            temp = temp.split(',')
            add = []
            for i in range(len(temp) // 4):
                add.append(list(map(int, temp[4 * i:4 * (i + 1)])))
            if not img_label_dict[si]:
                img_label_dict[si] = add
            else:
                if role == 'auditor':
                    img_label_dict[si] = add

        num_gt = 0  # 用于计算precision和recall的两个值
        num_det = 0
        pre_matrix = []  # 两列 预测值和是否为TP
        pr_matrix = []  # 两列 recall precision
        for row in tqdm(prediction_tuples):
            pre_boxes = []
            if row.prediction:  # 把四个坐标放入,并累计检测框数量
                for pre in row.prediction.split(self.OBJECT_SEP):
                    try:
                        pre = list(map(float, pre.split(self.ANNOTATION_SEP)))
                    except ValueError:
                        print(row.image_name)
                    pre_boxes.append(pre)
                    num_det += 1
            gt_boxes = img_label_dict[row.image_name]
            num_gt += len(gt_boxes)

            # 预存储，一部分fp框直接取下标为1的位置为0即可，表示没找到合适的gt框，这些可以直接作为fp传入最终结果
            # 而按照pr图像的定义，一个gt框应该只对应一个tp结果，这也和nms算法的概念一致
            # 但是图像中存在多个异物聚集的整体检测框等，为了最大程度防止最终tp数量超过gt数量，必须直接限制单个gt对应单个tp
            pppre_matrix = []
            for pre in pre_boxes:  # 上面获得了单个图像内含有预测值的预测框，和只有坐标的gt框
                flag = -1
                iou = 0.0
                for i, gt in enumerate(gt_boxes):
                    temiou = cal_iou(gt, pre[1:])
                    if temiou > iou:
                        flag = i
                        iou = temiou
                if iou >= self.THRESHOLD:
                    pppre_matrix.append([pre[0], 1, flag, iou])  # 置信度、tp、对应gt的编号、iou的值
                else:
                    if pre[0] < self.CONFIDENCE:  # 反常情况，当置信度特别高时却没找到gtbox，判定是label的问题
                        pre_matrix.append([pre[0], 0, flag, iou])
                    else:
                        pre_matrix.append(
                            [pre[0], 1, flag, iou, row.image_name])  # 因为就不存在gt对应，因此直接算作tp，理论上来讲，num_gt也要加一
                        num_gt += 1
            len_gt_boxes = len(gt_boxes)
            # for i in range(len_gt_boxes):  # 相同的flag 只保留置信度最高的，从而能使面积最大
            #     pre_confidence = 0
            #     for pre_tp_flag_iou in pppre_matrix:  # 该循环找到i号gt下最大的IOU值
            #         if pre_tp_flag_iou[2] == i and pre_tp_flag_iou[0] > pre_confidence:
            #             pre_confidence = pre_tp_flag_iou[0]
            #     for pre in pppre_matrix:
            #         if pre[2] == i:
            #             if pre[0] != pre_confidence:
            #                 pre[1] = 0
            #                 pre_matrix.append(pre)
            #             else:
            #                 pre_matrix.append(pre)
            for i in range(len_gt_boxes):  # 相同的flag 只保留iou最大的，其余直接将
                # tp项置0
                iou_max = 0
                for pre_tp_flag_iou in pppre_matrix:
                    if pre_tp_flag_iou[2] == i and pre_tp_flag_iou[3] > iou_max:
                        iou_max = pre_tp_flag_iou[3]
                for j, pre in enumerate(pppre_matrix):
                    if pre[2] == i and pre[3] == iou_max:
                        pre_matrix.append(pre)
                        pppre_matrix[j][2] = -1
                        break
                for pre in pppre_matrix:
                    if pre[2] == i:
                        pre[1] = 0
                        pre.append("只能有一个巫妖王")
                        pre_matrix.append(pre)

            # for i in range(len_gt_boxes):  # 最原始的，最不合理的做法，未将重复检测算作fp
            #     iou_max = [0, 0, 1, 0]
            #     iou_save = iou_max
            #     for pre_tp_flag_iou in pppre_matrix:
            #         if pre_tp_flag_iou[2] == i:
            #             if pre_tp_flag_iou[3] > iou_max[3]:
            #                 iou_max = pre_tp_flag_iou
            #     if iou_max != iou_save:
            #         pre_matrix.append(iou_max)

        # 按置信度排序
        sort_pre_matrix = sorted(pre_matrix, key=lambda x: x[0], reverse=True)
        # for pre in sort_pre_matrix:
        #     print(pre)

        acc_tp = 0  # 累计的tp数量和fp数量
        acc_fp = 0
        for item in sort_pre_matrix:
            recall = 0.0
            precision = 0.0
            if item[1] == 1:
                acc_tp += 1
            else:
                acc_fp += 1
            # if acc_tp > num_gt:
            #     print('阈值过小，tp数量超过gt数量')
            #     os._exit(0)
            precision = acc_tp / (acc_tp + acc_fp)
            recall = acc_tp / num_gt
            pr_matrix.append([recall, precision])

        print(acc_tp, '/', num_gt)
        print('num_det', num_det)

        # plt.figure()
        # plt.plot(np.array(pr_matrix)[:, 0], np.array(pr_matrix)[:, 1])
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])

        print('阈值', self.THRESHOLD)
        # 直接估计面积
        auc_wrong = 0
        for index in range(len(pr_matrix)):
            if index != len(pr_matrix) - 1:
                auc_wrong += (pr_matrix[index + 1][0] - pr_matrix[index][0]) * pr_matrix[index][1]
        print(auc_wrong, '插值前')

        # 开始插值，计算插值后面积
        inter_pr_matrix = []
        auc_area = 0
        for index in range(11):
            precision_max = 0
            for item in pr_matrix:
                if item[0] < index / 10:
                    continue
                else:
                    if item[1] > precision_max:
                        precision_max = item[1]
            if index != 10:
                auc_area += 0.1 * precision_max
            inter_pr_matrix.append([index / 10, precision_max])

        # print(inter_pr_matrix)

        # plt.figure()
        # plt.plot(np.array(inter_pr_matrix)[:, 0], np.array(inter_pr_matrix)[:, 1])
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.title(csvname + str(self.THRESHOLD))
        # plt.show()

        print(auc_area, '插值后')
        print(self.csvname)
        return inter_pr_matrix, auc_area

if __name__ == '__main__':
    csvname = 'recon_15epoch_3_31_outtest.csv'
    print(csvname)
    map50_95 = 0
    plt.figure()
    plt.title('0.5...0.05...0.95mAP')
    # colors = ['green', 'red', 'gold', 'yellow', 'blue', 'lime', 'chocolate', 'plum', 'steelblue', 'slategray']
    for thre in range(9, -1, -1):
        print('----------------')
        threeee = round(0.5+thre*0.05, 2)
        matrix, aa = CAL_MAP(THRESHOLD=threeee, CONFIDENCE=1, csvname = csvname,
                             label_csv='./trainAndTest/recon_out_test_label.csv', test_dir='out_test_recon_jpg').cal()
        plt.plot(np.array(matrix)[:, 0], np.array(matrix)[:, 1], label = str(threeee))
        map50_95 += aa
    plt.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()
    print('0.50-0.95的平均map', map50_95/10)





