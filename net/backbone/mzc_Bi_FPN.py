from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


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

class BiFeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels, is_first, extra_blocks=None):  # 默认是第一个
        super(BiFeaturePyramidNetwork, self).__init__()
        # [256,512,1024,2048]  全部变成 [256,256,256,256]
        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.conv_c5_in = nn.Conv2d(in_channels_list[-1], 256, 1)  # 降维
        self.conv_c4_in = nn.Conv2d(in_channels_list[-2], 256, 1)  # 降维
        self.conv_c3_in = nn.Conv2d(in_channels_list[-3], 256, 1)  # 降维
        self.conv_c2_in = nn.Conv2d(in_channels_list[-4], 256, 1)  # 降维

        # 产生不同的cx_in,如果是第一个bifpn层，就是进行同样的降维
        self.conv_c5_in2 = nn.Conv2d(in_channels_list[-1], 256, 1)  # 降维
        self.conv_c4_in2 = nn.Conv2d(in_channels_list[-2], 256, 1)  # 降维
        self.conv_c3_in2 = nn.Conv2d(in_channels_list[-3], 256, 1)  # 降维

        # 产生不同的cx_in，如果是第二个以及后续的bifpn层，因为已经有一组不同降维的输入了，所以其他输入就改用1*1卷积重新产生
        self.conv_c5_in3 = nn.Conv2d(256, 256, 1)  # 产生不同的分支
        self.conv_c4_in3 = nn.Conv2d(256, 256, 1)  # 产生不同的分支
        self.conv_c3_in3 = nn.Conv2d(256, 256, 1)  # 产生不同的分支



        self.conv_4 = nn.Conv2d(256, 256, 3, padding=1)  # 在不同尺度融合之后进行一次卷积，之后才是maxpooling和上面一层重新加起来
        self.conv_3 = nn.Conv2d(256, 256, 3, padding=1)  # 同上
        self.conv_2_3 = nn.Conv2d(256, 256, 3, padding=1)  # 从三个地方获取了信息，得到了2层的结果，接着我要进行卷积，卷积结果指向3层
        self.conv_3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_4_5 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_5_out = nn.Conv2d(256, 256, 3, padding=1)

        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=(2, 2))
        self.upsample_5_4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_4_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_3_2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)  # 到p4的合并
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # 在p2到p5接入results之前,再进行一次eca,因为在二阶段进行eca,不如在特征金字塔中,同时进行
        self.eca_p2 = eca_block()
        self.eca_p3 = eca_block()
        self.eca_p4 = eca_block()
        self.eca_p5 = eca_block()

        self.epsilon = 1e-4
        self.swish = Swish()
        self.relu = nn.ReLU()
        self.is_first = is_first
        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks

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
        names = list(x.keys())  # 0 1 2 3
        x = list(x.values())   # layer1结果 l2结果 l3结果  l4结果
        results = []
        #  x[0] c2 、 x1 c3 、x2 c4 、 x3 c5
        if self.is_first:
            c5in = self.conv_c5_in(x[3])
            c4in = self.conv_c4_in(x[2])
            c3in = self.conv_c3_in(x[1])
            c2in = self.conv_c2_in(x[0])

            c5in_2 = self.conv_c5_in2(x[3])
            c4in_2 = self.conv_c4_in2(x[2])
            c3in_2 = self.conv_c3_in2(x[1])
        else:  # 不是第一次，通道数就是对齐的，所以直接拿来用就行了，每一层的长宽是固定的
            c5in = x[3]
            c4in = x[2]
            c3in = x[1]
            c2in = x[0]

            c5in_2 = self.conv_c5_in3(x[3])
            c4in_2 = self.conv_c4_in3(x[2])
            c3in_2 = self.conv_c3_in3(x[1])

        p4_w1 = self.relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_mid = self.conv_4(self.swish(weight[0] * c4in + weight[1] * self.upsample_5_4(c5in)))

        p3_w1 = self.relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_mid = self.conv_3(self.swish(weight[0] * c3in + weight[1] * self.upsample_4_3(p4_mid)))

        p2_w1 = self.relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p2_out = self.conv_2_3(self.swish(weight[0] * c2in + weight[1] * self.upsample_3_2(p3_mid)))
        results.append(self.eca_p2(p2_out))  # 插入结果集，按照分辨率从高到低     从这里开始向上走了

        p3_w2 = self.relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p3_out = self.conv_3_4(self.swish(weight[0] * c3in_2 + weight[1] * p3_mid + self.maxpooling(p2_out)))
        results.append(self.eca_p3(p3_out))

        p4_w2 = self.relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv_4_5(self.swish(weight[0] * c4in_2 + weight[1] * p4_mid + self.maxpooling(p3_out)))
        results.append(self.eca_p4(p4_out))

        p5_w1 = self.relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_out = self.conv_5_out(self.swish(weight[0] * c5in_2 + weight[1] * self.maxpooling(p4_out)))
        results.append(self.eca_p5(p5_out))

        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

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
