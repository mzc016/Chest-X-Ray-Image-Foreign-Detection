import torch
import net.use_for_me_model_se_stn_nonl as modelme
from tools import utils
from dataSet import ForeignObjectDataset
import pandas as pd

import numpy as np
import os
from tqdm import tqdm

from Chest_ALL.cal_map_hospital import CAL_MAP
from collections import defaultdict
# from jpg.test_num import getSopDict

# prepare for some parameter


class mzc_test:
    def __init__(self, savename='', with_clahe=0, reshape_std=800, num_classes=2, model = None, choose = 'frozenBN',
                 data_dir = './Chest_ALL/trainAndTest/test'):
        np.random.seed(0)
        torch.manual_seed(0)
        self.device = torch.device('cuda')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.num_classes = num_classes  # object (foreground); background
        # prepare for csv
        dict_tr = {}
        # dict_tr1 = getSopDict('425.csv').getIt()
        # dict_tr2 = getSopDict('837lr.csv').getIt()
        # dict_tr3 = getSopDict('580lr.csv').getIt()
        #
        # dict_tr.update(dict_tr1)
        # dict_tr.update(dict_tr2)
        # dict_tr.update(dict_tr3)
        self.savename = savename
        if self.savename == '':
            print('空savename-----')
        else:
            print('有savename，还需载入')
        self.reshape_std = reshape_std
        self.dataset_test = ForeignObjectDataset(datafolder=data_dir,
                                                 datatype='dev',
                                                 labels_dict=dict_tr,
                                                 reshape_std=reshape_std)
        self.data_loader_test = torch.utils.data.DataLoader(
            self.dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)
        if model is None:
            print('根据savename参数载入成功')
            self.model = modelme.get_detection_local_model1(self.num_classes, 0, choose)
            self.model.to(self.device)
            self.model.load_state_dict(torch.load('./pth_save/' + self.savename + '.pt'), strict=False)
            self.model.eval()
        else:
            print('无需savename，直接载入model')
            self.model = model
            self.model.eval()

    def start(self):



        locs = []
        # if os.path.exists('./pth_save/temp.csv'):
        #     return
        for image, label, width, height in tqdm(self.data_loader_test):  #

            image = list(img.to(self.device) for img in image)
            outputs = self.model(image)

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
                    center_points.append([center_x1 / self.reshape_std * width[-1], center_y1 / self.reshape_std * height[-1],
                                          center_x2 / self.reshape_std * width[-1], center_y2 / self.reshape_std * height[-1]])
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

        loc_res = pd.DataFrame({'image_name': self.dataset_test.image_files_list_test, 'prediction': locs})
        loc_res.to_csv('./pth_save/' + 'temp.csv', columns=['image_name', 'prediction'], sep=',',
                       index=None)
        print('localization.csv generated.')


if __name__ == '__main__':
    save_name1 = 'model_noeca_33pooling11_train4_nomixup_stn_anchor140_20epoch7'
    # save_name1 = 'model_eca_33pooling_rewrite_30epoch'
    res = []
    n = 1
    for i in range(n):
        if n == 1:
            temp_name = save_name1
        else:
            temp_name = save_name1 + str(i)
        a = mzc_test(savename=temp_name, choose='frozenBN')
        a.start()
        # #  注意这里的csv_dir = pth_save 其实是存储模型文件的地方，但是和上方的temp.csv对应，暂存而已
        # matrix, aa = CAL_MAP(THRESHOLD=0.5, CONFIDENCE=1, csvname='temp.csv', data_dir='./Chest_ALL/trainAndTest/',
        #                      csv_dir='./pth_save/', label_csv='./Chest_ALL/threelabel/auditor_info_new.csv').cal()
        # res.append([i, aa])
    print(res)
