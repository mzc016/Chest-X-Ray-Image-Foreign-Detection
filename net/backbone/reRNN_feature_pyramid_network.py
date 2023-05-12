from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict


class eca_block(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 一维卷积，这里的前两个1是输入输出通道，这里就是变成了类似于fc的效果，但是又不一样，全连接和1*1卷积类似，可以完成对整体的注意，也就是
        # 没有足够的局部注意力，而1维卷积是在最后的维度上进行的，也就是[batch_size, channel, text_len]的最后一维，文本长度
        # 这里channel直接化简为了1，也就是非常基础的一维卷积了，也就是n个一维词向量，你每三个词进行一次卷积求和，然后得到新的那个一维词向量
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size() # b代表b个样本，c为通道数，h为高度，w为宽度

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  # [b, c, 1, 1]

        # Two different branches of ECA module
        # torch.squeeze()这个函数主要对数据的维度进行压缩,torch.unsqueeze()这个函数 主要是对数据维度进行扩充
        # 这里主要就是把1*1的图像压缩成了1维向量，然后将通道维度进行了后移，从而对通道进行了一维卷积，然后转回来
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)


        # Multi-scale information fusion多尺度信息融合
        y = self.sigmoid(y)
        # 原网络中克罗内克积，也叫张量积，为两个任意大小矩阵间的运算
        return x * y.expand_as(x)

class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Arguments:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    """

    def __init__(self, in_channels_list, out_channels, extra_blocks=None, use_eca=0):
        super(FeaturePyramidNetwork, self).__init__()
        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.inner_blocks = nn.ModuleList()
        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks
        print('-----------fpn修改情况-------------')
        if self.extra_blocks:
            self.conv_extra = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
            self.relu_extra = nn.ReLU(inplace=True)
            print('使用了额外的更大感受野的层,这里采用了步长为2的卷积的方式')
            # print('使用了maxpooling')
            # pass

        self.use_eca = use_eca
        if use_eca:
            self.eca = eca_block()
            print('使用了eca')
        else:
            print('未使用eca')
        print('---------------------------------')
        self.usecount = 0

    def get_result_from_inner_blocks(self, x, idx):
        # type: (Tensor, int) -> Tensor
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x, idx):
        # type: (Tensor, int) -> Tensor
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x):
        # type: (Dict[str, Tensor]) -> Dict[str, Tensor]
        """
        Computes the FPN for a set of feature maps.
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())  # [0,1,2,3]
        x = list(x.values())  # [x1, x2, x3, x4] 具体为tensor

        # 将resnet layer4的channel调整到指定的out_channels
        # 因为最低分辨率要变大并且和上一层对应的inner结果相加，所以先计算最后一层的
        # last_inner = self.inner_blocks[-1](x[-1])  是下面这个代码的简单翻译
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        # result中保存着每个预测特征层
        results = []

        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        # results.append(self.layer_blocks[-1](last_inner))
        self.usecount += 1


        results.append(self.get_result_from_layer_blocks(last_inner, -1))  # 这里加一个

        for idx in range(len(x) - 2, -1, -1):  # 这里就是x=4 ,所以就是2,1,0 正好是layer3 2 1 对应的fpn输入
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]  # 四个维度bchw，只用长宽
            # 低一级的分辨率进行临近插值，然后和当前的inner_lateral相加
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            if self.use_eca:
                results.insert(0, self.eca(
                    self.get_result_from_layer_blocks(last_inner, idx)))  # ！！！！！！这里是额外增加的eca，和3*3卷积接续
            else:
                results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))  # 这里是头插，所以依然是从高分别率到低的顺序

        if self.usecount % 3 != 0:
            for i in range(4):
                results[i] = torch.cat([results[i], x[i]], dim=1)
        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None and self.usecount % 3 == 0:  # 同一次训练中，反复载入fpn会一直累加计数器，而每三次要训练一下额外模块
            # results, names = self.extra_blocks(results, x, names)  # 原始的maxpooling方法
            #新加的卷积降分辨率方式~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            names.append('pool')
            results.append(self.relu_extra(self.conv_extra(results[-1])))
        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out


class FakeFPN11(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super(FakeFPN11, self).__init__()
        # self.inner_block_module1 = nn.Conv2d(in_channels_list[0], out_channels, 1)  # 256层，分辨率高
        # self.inner_block_module2 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        # self.inner_block_module3 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        # self.inner_block_module4 = nn.Conv2d(in_channels_list[3], out_channels, 1)
        self.inner_block_module1 = nn.Conv2d(out_channels + in_channels_list[0], in_channels_list[0], 3, 1, 1)  # 256层，分辨率高
        self.inner_block_module2 = nn.Conv2d(out_channels + in_channels_list[1], in_channels_list[1], 3, 1, 1)
        self.inner_block_module3 = nn.Conv2d(out_channels + in_channels_list[2], in_channels_list[2], 3, 1, 1)
        self.inner_block_module4 = nn.Conv2d(out_channels + in_channels_list[3], in_channels_list[3], 3, 1, 1)
        print('---通道反向增加模块33卷积---')

    def forward(self, x):
        # type: (Dict[str, Tensor]) -> Dict[str, Tensor]
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())  # [0,1,2,3]
        x = list(x.values())  # [x1, x2, x3, x4] 具体为tensor

        inner_4 = self.inner_block_module4(x[3])  # 最后一层通道数最多，对应最后一个卷积
        inner_3 = self.inner_block_module3(x[2])
        inner_2 = self.inner_block_module2(x[1])
        inner_1 = self.inner_block_module1(x[0])
        results = [inner_1, inner_2, inner_3, inner_4]

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(torch.nn.Module):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(self, x, y, names):  # 这里的x很可能是已经统一从不同阶段resnet 卷积得到的256通道特征图金字塔了，而堆叠顺序是
        # type: (List[Tensor], List[Tensor], List[str]) -> Tuple[List[Tensor], List[str]]
        names.append("pool")  #  输入图像为800*800，特征金字塔中均为4的batchsize，256的通道数，然后是200*200 100 100 50 50  25 25
        x.append(F.max_pool2d(x[-1], 1, 2, 0))  # input, kernel_size, stride, padding  # 对最小图像产生的特征图进行继续maxpool2d，产生13*13的特征图
        return x, names
