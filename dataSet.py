import os
import torch
from PIL import Image
from torchvision import transforms
import cv2 as cv


class ForeignObjectDataset(object):

    def __init__(self, datafolder, datatype='train', transform=None, labels_dict={}, reshape_std=600, mixup = 0):
        self.RESIZE_STD = reshape_std
        self.datafolder = datafolder  # 路径
        self.datatype = datatype
        self.labels_dict = labels_dict
        self.image_files_list_train = [s for s in sorted(os.listdir(datafolder)) if s in labels_dict.keys()]
        # 原baseline根据对不同csv文件处理来保持该方法类的函数一致，但是因为我们用的是唯一的字典，所以需要区分train和test的不同图片列表
        self.image_files_list_test = [s for s in sorted(os.listdir(datafolder))]

        # 在原始数据集翻倍之前，记录长度，方便超过这个长的的idx下直接进行图像的翻转
        self.transposeLength = len(self.image_files_list_train)
        print('实际训练图像数量（在测试过程中不使用）', self.transposeLength)
        print('测试图像数量（在训练过程中表示载入图像数量）', len(self.image_files_list_test))
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.RESIZE_STD, self.RESIZE_STD)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            if mixup:
                self.transform1 = transforms.Compose([
                    transforms.Resize((self.RESIZE_STD, self.RESIZE_STD))
                ])
                self.transform2 = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        # self.annotations = [labels_dict[i] for i in self.image_files_list]
        print('resize标准为:', self.RESIZE_STD)
        # self.with_clahe = with_clahe
        # if with_clahe:
        #     print("使用clahe")
        # else:
        #     print("不适用clahe")
        self.mixup = mixup
        if mixup:
            print('使用mixup方法')
        else:
            print('不适用mixup方法')

    def get_box(self, idx):  # 根据图像返回boxes

        img_name = self.image_files_list_train[idx]
        img_path = os.path.join(self.datafolder, img_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]
        boxes = []
        if img_name in self.labels_dict.keys():
            annotation = self.labels_dict[img_name]
            for anno in annotation:
                temp = anno.replace('[', '')
                temp = temp.replace(']', '')
                temp = temp.replace(' ', '')
                temp = temp.split(',')  # 单个gt框的四个坐标值
                # 进行归一化，到600*600
                xmin = float(temp[0]) / width * self.RESIZE_STD
                ymin = float(temp[1]) / height * self.RESIZE_STD
                xmax = float(temp[2]) / width * self.RESIZE_STD
                ymax = float(temp[3]) / height * self.RESIZE_STD
                if xmin > xmax:
                    temp = xmin
                    xmin = xmax
                    xmax = temp
                if ymin > ymax:
                    temp = ymin
                    ymin = ymax
                    ymax = temp

                boxes.append([xmin, ymin, xmax, ymax])
        return img, boxes

    def __getitem__(self, idx):
        # load images
        if self.datatype == 'train':

            boxes = []
            if self.mixup and idx >= self.transposeLength:  # 只有mixup的时候才会大于等于数据量，此时这个部分的数据是mixup的部分
                img1, boxes1 = self.get_box(idx-self.transposeLength)
                img2, boxes2 = self.get_box(idx-self.transposeLength//2)
                boxes.extend(boxes1)
                boxes.extend(boxes2)
            else:
                img3, boxes3 = self.get_box(idx)
                boxes.extend(boxes3)

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class

            labels = torch.ones((len(boxes),), dtype=torch.int64)  # 每个gt框的标签自然是1
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            image_id = torch.tensor([idx])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd  # 难检测
            img = None
            if self.transform is not None:
                if self.mixup and idx >= self.transposeLength:
                    img = Image.blend(self.transform1(img1), self.transform1(img2), 0.5)
                    img = self.transform2(img)
                else:
                    img = self.transform(img3)
            return img, target

        if self.datatype == 'dev':  # 还需要修改！！！也可以不改（因为不使用这里提供的标签类型
            img_name = self.image_files_list_test[idx]
            img_path = os.path.join(self.datafolder, img_name)
            img = Image.open(img_path).convert("RGB")
            width, height = img.size[0], img.size[1]
            if img_name in self.labels_dict.keys():
                label = 1
            else:
                label = 0

            if self.transform is not None:
                img = self.transform(img)

            return img, label, width, height

    def __len__(self):
        if self.datatype == 'dev':
            return len(self.image_files_list_test)
        elif self.datatype == 'train':
            if self.mixup:  # mixup时，前面正常训练，后面使用融合的数据
                return len(self.image_files_list_train)//2 * 3
            else:
                return len(self.image_files_list_train)

