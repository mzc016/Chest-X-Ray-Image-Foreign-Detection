import torch


# from faster_rcnn.backbone import resnet50_fpn_backbone, convnext_fpn_backbone, resnet50_refpn_backbone
# from faster_rcnn.backbone import resnet50_bifpn_backbone

from faster_rcnn.backbone import resnet50_fpn_backbone, convnext_fpn_backbone, resnet50_refpn_backbone
from faster_rcnn.backbone import resnet50_bifpn_backbone

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# from faster_rcnn.network_files import FasterRCNN, FastRCNNPredictor
from faster_rcnn.network_files import FasterRCNN, FastRCNNPredictor

device = torch.device('cuda:0')


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        print(x.shape)
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


# add by mzc--------------↓
class LocalizationNetwork(nn.Module):
    """
    空间变换网络
    1.读入输入图片，并利用其卷积网络提取特征
    2.使用特征计算基准点，基准点的个数由参数fiducial指定，参数channel指定输入图像的通道数
    3.计算基准点的方法是使用两个全连接层将卷积网络输出的特征进行降维，从而得到基准点集合
    """

    def __init__(self, fiducial, channel):
        """
        初始化方法

        :param fiducial: 基准点的数量
        :param channel: 输入图像通道数
        输入tensor为2048*256*7*7
        """
        super(LocalizationNetwork, self).__init__()
        self.fiducial = fiducial # 指定基准点个数
        self.channel = channel   # 指定输入图像的通道数
        # 提取特征使用的卷积网络
        self.ConvNet = nn.Sequential(
            nn.Conv2d(self.channel, 512, 3, 1, padding=1, bias=False),  # 256 -> 512
            nn.BatchNorm2d(512), nn.ReLU(True),  # [N, 512, 7, 7]
            nn.AdaptiveAvgPool2d(1))  # [N, 512, 1, 1]
        # 计算基准点使用的两个全连接层
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.fiducial * 2)
        # 将全连接层2的参数初始化为0
        self.localization_fc2.weight.data.fill_(0)
        """
        全连接层2的偏移量bias需要进行初始化，以便符合RARE Paper中所介绍的三种初始化形式，三种初始化方式详见
        https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Robust_Scene_Text_CVPR_2016_paper.pdf，Fig. 6 (a)
        下初始化方法为三种当中的第一种
        """
        ctrl_pts_x = np.linspace(-1.0, 1.0, fiducial // 2)  # 1*3
        ctrl_pts_y_top = np.linspace(0.0, -1.0, fiducial // 2)  # 1*3
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, fiducial // 2)  # 1*3
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)  # 第二维度插入新维度 1*2*3
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)  # 1*2*3
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)  # 2*1*2*3
        # 修改全连接层2的偏移量
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, x):
        """
        前向传播方法

        :param x: 输入图像，规模[batch_size, C, H, W]
        :return: 输出基准点集合C，用于图像校正，规模[batch_size, fiducial, 2]
        """
        # 获取batch_size
        batch_size = x.size(0)
        # 提取特征
        features = self.ConvNet(x).view(batch_size, -1)
        # 使用特征计算基准点集合C
        features = self.localization_fc1(features)
        C = self.localization_fc2(features).view(batch_size, self.fiducial, 2)
        return C

class GridGenerator(nn.Module):
    """网格生成网络

    Grid Generator of RARE, which produces P_prime by multipling T with P."""

    def __init__(self, fiducial, output_size):
        """
        初始化方法
        :param fiducial: 基准点与基本基准点的个数
        :param output_size: 校正后图像的规模

        基本基准点是被校正后的图片的基准点集合
        """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        # 基准点与基本基准点的个数
        self.fiducial = fiducial
        # 校正后图像的规模
        self.output_size = output_size # 假设为[w, h]
        # 论文公式当中的C'，C'是基本基准点，也就是被校正后的图片的基准点集合
        self.C_primer = self._build_C_primer(self.fiducial)
        # 论文公式当中的P'，P'是校正后的图片的像素坐标集合，规模为[h·w, 2]，集合中有n个元素，每个元素对应校正图片的一个像素的坐标
        self.P_primer = self._build_P_primer(self.output_size)
        # 如果使用多GPU，则需要寄存器缓存register buffer
        self.register_buffer("inv_delta_C_primer",
                             torch.tensor(self._build_inv_delta_C_primer(self.fiducial, self.C_primer)).float())
        self.register_buffer("P_primer_hat",
                             torch.tensor(
                                 self._build_P_primer_hat(self.fiducial, self.C_primer, self.P_primer)).float())

    def _build_C_primer(self, fiducial):
        """
        构建基本基准点集合C'，即被校正后的图片的基准点，应该是一个矩形的fiducial个点集合

        :param fiducial: 基本基准点的个数，跟基准点个数相同
        该方法生成C'的方法与前面的空间变换网络相同
        """
        ctrl_pts_x = np.linspace(-1.0, 1.0, fiducial // 2)
        ctrl_pts_y_top = -1 * np.ones(fiducial // 2)
        ctrl_pts_y_bottom = np.ones(fiducial // 2)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C_primer = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C_primer

    def _build_P_primer(self, output_size):
        """
        构建校正图像像素坐标集合P'，构建的方法为按照像素靠近中心的程度形成等差数列作为像素横纵坐标值

        :param output_size: 模型输出的规模
        :return : 校正图像的像素坐标集合
        """
        w, h = output_size
        # 等差数列output_grid_x
        output_grid_x = (np.arange(-w, w, 2) + 1.0) / w
        # 等差数列output_grid_y
        output_grid_y = (np.arange(-h, h, 2) + 1.0) / h
        """
        使用np.meshgrid将output_grid_x中每个元素与output_grid_y中每个元素组合形成一个坐标
        注意，若output_grid_x的规模为[w], output_grid_y为[h]，则生成的元素矩阵规模为[h, w, 2]
        """
        P_primer = np.stack(np.meshgrid(output_grid_x, output_grid_y), axis=2)
        # 在返回时将P'进行降维，将P'从[h, w, 2]降为[h·w, 2]
        return P_primer.reshape([-1, 2])  # [HW, 2]

    def _build_inv_delta_C_primer(self, fiducial, C_primer):
        """
        计算deltaC'的逆，该矩阵为常量矩阵，在确定了fiducial与C'之后deltaC'也同时被确定

        :param fiducial: 基准点与基本基准点的个数
        :param C_primer: 基本基准点集合C'
        :return: deltaC'的逆
        """
        # 计算C'梯度公式中的R，R中的元素rij等于dij的平方再乘dij的平方的自然对数，dij是C'中第i个元素与C'中第j个元素的欧式距离，R矩阵是个对称矩阵
        R = np.zeros((fiducial, fiducial), dtype=float)
        # 对称矩阵可以简化for循环
        for i in range(0, fiducial):
            for j in range(i, fiducial):
                R[i, j] = R[j, i] = np.linalg.norm(C_primer[i] - C_primer[j])
        np.fill_diagonal(R, 1)  # 填充对称矩阵对角线元素，都为1
        R = (R ** 2) * np.log(R ** 2)  # 或者R = 2 * (R ** 2) * np.log(R)

        # 使用不同矩阵进行拼接，组成deltaC'
        delta_C_primer = np.concatenate([
            np.concatenate([np.ones((fiducial, 1)), C_primer, R], axis=1),       # 规模[fiducial, 1+2+fiducial]，deltaC'计算公式的第一行
            np.concatenate([np.zeros((1, 3)), np.ones((1, fiducial))], axis=1),  # 规模[1, 3+fiducial]，deltaC'计算公式的第二行
            np.concatenate([np.zeros((2, 3)), np.transpose(C_primer)], axis=1)   # 规模[2, 3+fiducial]，deltaC'计算公式的第三行
        ], axis=0)                                                               # 规模[fiducial+3, fiducial+3]

        # 调用np.linalg.inv求deltaC'的逆
        inv_delta_C_primer = np.linalg.inv(delta_C_primer)
        return inv_delta_C_primer

    def _build_P_primer_hat(self, fiducial, C_primer, P_primer):
        """
        求^P'，即论文公式当中由校正后图片像素坐标经过变换矩阵T后反推得到的原图像素坐标P集合公式当中的P'帽，P = T^P'

        :param fiducial: 基准点与基本基准点的个数
        :param C_primer: 基本基准点集合C'，规模[fiducial, 2]
        :param P_primer: 校正图像的像素坐标集合，规模[h·w, 2]
        :return: ^P'，规模[h·w, fiducial+3]
        """
        n = P_primer.shape[0]  # P_primer的规模为[h·w, 2]，即n=h·w
        # PAPER: d_{i,k} is the euclidean distance between p'_i and c'_k
        P_primer_tile = np.tile(np.expand_dims(P_primer, axis=1), (1, fiducial, 1))  # 规模变化 [h·w, 2] -> [h·w, 1, 2] -> [h·w, fiducial, 2]
        C_primer = np.expand_dims(C_primer, axis=0)                                  # 规模变化 [fiducial, 2] -> [1, fiducial, 2]
        # 此处相减是对于P_primer_tile的每一行都减去C_primer，因为这两个矩阵规模不一样
        dist = P_primer_tile - C_primer
        # 计算求^P'公式中的dik，dik为P'中第i个点与C'中第k个点的欧氏距离
        r_norm = np.linalg.norm(dist, ord=2, axis=2, keepdims=False)                 # 规模 [h·w, fiducial]
        # r'ik = d^2ik·lnd^2ik
        r = 2 * np.multiply(np.square(r_norm), np.log(r_norm + self.eps))
        # ^P'i = [1, x'i, y'i, r'i1,......, r'ik]的转置，k=fiducial
        P_primer_hat = np.concatenate([np.ones((n, 1)), P_primer, r], axis=1)        # 规模 经过垂直拼接[h·w, 1]，[h·w, 2]，[h·w, fiducial]形成[h·w, fiducial+3]
        return P_primer_hat

    def _build_batch_P(self, batch_C):
        """
        求本batch每一张图片的原图像素坐标集合P

        :param batch_C: 本batch原图的基准点集合C
        :return: 本batch的原图像素坐标集合P，规模[batch_size, h, w, 2]
        """
        # 获取batch_size
        batch_size = batch_C.size(0)
        # 将本batch的基准点集合进行扩展，使其规模从[batch_size, fiducial, x] -> [batch_size, fiducial+3, 2]
        batch_C_padding = torch.cat((batch_C, torch.zeros(batch_size, 3, 2).float().to(device)), dim=1)

        # 按照论文求解T的公式求T，规模变化[fiducial+3, fiducial+3] × [batch_size, fiducial+3, 2] -> [batch_size, fiducial+3, 2]
        batch_T = torch.matmul(self.inv_delta_C_primer, batch_C_padding)
        # 按照论文公式求原图像素坐标的公式求解本batch的原图像素坐标集合P，P = T^P'
        # [h·w, fiducial+3] × [batch_size, fiducial+3, 2] -> [batch_size, h·w, 2]
        batch_P = torch.matmul(self.P_primer_hat, batch_T)
        # 将P从[batch_size, h·w, 2]转换到[batch_size, h, w, 2]
        return batch_P.reshape([batch_size, self.output_size[1], self.output_size[0], 2])

    def forward(self, batch_C):
        return self._build_batch_P(batch_C)


class TPSSpatialTransformerNetwork(nn.Module):
    """Rectification Network of RARE, namely TPS based STN"""

    def __init__(self, fiducial, input_size, output_size, channel):
        """Based on RARE TPS

        :param fiducial: number of fiducial points
        :param input_size: (w, h) of the input image
        :param output_size: (w, h) of the rectified image
        :param channel: input image channel
        """
        super(TPSSpatialTransformerNetwork, self).__init__()
        self.fiducial = fiducial
        self.input_size = input_size
        self.output_size = output_size
        self.channel = channel
        self.LNet = LocalizationNetwork(self.fiducial, self.channel)
        self.GNet = GridGenerator(self.fiducial, self.output_size)

    def forward(self, x):
        """
        :param x: batch input image [batch_size, c, w, h]
        :return: rectified image [batch_size, c, h, w]
        """
        # 求原图的基准点集合C
        C = self.LNet(x)  # [batch_size, fiducial, 2]
        # 求原图对应校正图像素的像素坐标集合P
        P = self.GNet(C) # [batch_size, h, w, 2]
        # 按照P对x进行采样，对于越界的位置在网格中采用边界的pixel value进行填充
        rectified = nn.functional.grid_sample(x, P, padding_mode='border', align_corners=True)  #规模[batch_size, c, h, w]
        # print(np.shape(rectified))
        return rectified


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c//2, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c//2, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c//2, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

#
class STNBlock(torch.nn.Module):
    def __init__(self, inchannel):
        super(STNBlock, self).__init__()
        self.localization = torch.nn.Sequential(  # 2048, 256, 7, 7 to  2048, 10, 3, 3
            torch.nn.Conv2d(inchannel, 64, kernel_size=3, padding=0),
            # !!!!!!!!!!!!!!!!!!!!!!!!!这里直接对接了roipooling ，所以要通道对齐
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 10, kernel_size=3, padding=0),
            # torch.nn.MaxPool2d(2, stride=2),
            torch.nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = torch.nn.Sequential(
            torch.nn.Linear(10 * 3 * 3, 32),
            torch.nn.ReLU(True),
            torch.nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        print(x.shape)
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = torch.nn.functional.affine_grid(theta, x.size())
        x = torch.nn.functional.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        return x


class SEblock(torch.nn.Module):
    def __init__(self, inchannel):
        super(SEblock, self).__init__()
        self.se = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Conv2d(inchannel, inchannel // 4, kernel_size=1),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(),
            torch.nn.Conv2d(inchannel // 4, inchannel, kernel_size=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x1 = x
        x = self.se(x)
        x = x1.mul(x)
        return x


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

class ReWriteRoiHeadsBoxHead_norewrite(torch.nn.Module):  # 重写一个RoiHeads层
    def __init__(self):
        super(ReWriteRoiHeadsBoxHead_norewrite, self).__init__()
        self.a = TwoMLPHead(12544, 1024)

    def forward(self, x):
        x = self.a(x)
        return x

class ReWriteRoiHeadsBoxHead(torch.nn.Module):  # 重写一个RoiHeads层
    def __init__(self):
        super(ReWriteRoiHeadsBoxHead, self).__init__()
        self.NonLocal = NonLocalBlock(256)
        self.stn = TPSSpatialTransformerNetwork(6, (7, 7), (7, 7), 256)
        self.SEblock = SEblock(256)
        self.a = TwoMLPHead(12544, 1024)

    def forward(self, x):
        x = self.stn(x)
        x = self.NonLocal(x)
        x = self.SEblock(x)
        x = self.a(x)
        return x


class ReWriteRoiHeadsBoxHead_noNonLocal(torch.nn.Module):  # 重写一个RoiHeads层
    def __init__(self):
        super(ReWriteRoiHeadsBoxHead_noNonLocal, self).__init__()
        self.stn = TPSSpatialTransformerNetwork(6, (7, 7), (7, 7), 256)
        self.SEblock = SEblock(256)
        self.a = TwoMLPHead(256*49, 1024)   # 192*49或256*49

    def forward(self, x):
        x = self.stn(x)
        x = self.SEblock(x)
        x = self.a(x)
        return x

class ReWriteRoiHeadsBoxHead_STN(torch.nn.Module):  # 重写一个RoiHeads层
    def __init__(self):
        super(ReWriteRoiHeadsBoxHead_STN, self).__init__()
        # self.stn = STNBlock(256)
        # print('普通的stn')
        self.stn = TPSSpatialTransformerNetwork(6, (7, 7), (7, 7), 256)
        print('tppspatial stn')
        self.a = TwoMLPHead(256*49, 1024)   # 192*49或256*49

    def forward(self, x):
        x = self.stn(x)
        x = self.a(x)
        return x

class ReWriteRoiHeadsBoxHead_SE(torch.nn.Module):  # 重写一个RoiHeads层
    def __init__(self):
        super(ReWriteRoiHeadsBoxHead_SE, self).__init__()
        # self.SEblock = SEblock(256)
        self.se = eca_block()
        print('使用了eca模块')
        self.a = TwoMLPHead(256*49, 1024)   # 192*49或256*49

    def forward(self, x):
        x = self.se(x)
        x = self.a(x)
        return x

class ReWriteRoiHeadsBoxHead_Nonlocal(torch.nn.Module):  # 重写一个RoiHeads层
    def __init__(self):
        super(ReWriteRoiHeadsBoxHead_Nonlocal, self).__init__()
        self.NonLocalBlock = NonLocalBlock(256)
        self.a = TwoMLPHead(256*49, 1024)   # 192*49或256*49

    def forward(self, x):
        x = self.NonLocalBlock(x)
        x = self.a(x)
        return x

class ReWriteRoiHeadsBoxHead_nonlocal_stn(torch.nn.Module):  # 重写一个RoiHeads层
    def __init__(self):
        super(ReWriteRoiHeadsBoxHead_nonlocal_stn, self).__init__()
        self.NonLocalBlock = NonLocalBlock(256)
        self.stn = STNBlock(256)
        self.a = TwoMLPHead(256*49, 1024)   # 192*49或256*49

    def forward(self, x):
        x = self.NonLocalBlock(x)
        x = self.stn(x)
        x = self.a(x)
        return x

class ReWriteRoiHeadsBoxHead_nonlocal_se(torch.nn.Module):  # 重写一个RoiHeads层
    def __init__(self):
        super(ReWriteRoiHeadsBoxHead_nonlocal_se, self).__init__()
        self.NonLocalBlock = NonLocalBlock(256)
        self.SEBlock = SEblock(256)
        self.a = TwoMLPHead(256*49, 1024)   # 192*49或256*49

    def forward(self, x):
        x = self.NonLocalBlock(x)
        x = self.SEBlock(x)
        x = self.a(x)
        return x



# add by mzc--------------↑



def get_detection_local_model3(num_classes):
    # ----
    backbone = resnet50_fpn_backbone(trainable_layers=3, use_eca=1)  # 使用冻结的bn层
    print('基础resnet+fpn 使用了frozen bn')
    model = FasterRCNN(backbone=backbone, num_classes=91)
    weights_dict = torch.load("./faster_rcnn/backbone/fasterrcnn_resnet50_fpn_coco.pth")
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # ----

    model.roi_heads.box_head = ReWriteRoiHeadsBoxHead_STN()
    print(' use STN')
    return model

def get_detection_local_model4(num_classes):
    # ----
    backbone = resnet50_fpn_backbone(trainable_layers=3, use_eca=1)  # 使用冻结的bn层
    print('基础resnet+fpn 使用了frozen bn')
    model = FasterRCNN(backbone=backbone, num_classes=91)
    weights_dict = torch.load("./faster_rcnn/backbone/fasterrcnn_resnet50_fpn_coco.pth")
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # ----

    model.roi_heads.box_head = ReWriteRoiHeadsBoxHead_SE()
    print('use SE')
    return model

def get_detection_local_model5(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d, trainable_layers=3)
    model = FasterRCNN(backbone=backbone, num_classes=91)
    weights_dict = torch.load("./faster_rcnn/backbone/fasterrcnn_resnet50_fpn_coco.pth")
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.box_head = ReWriteRoiHeadsBoxHead_Nonlocal()
    print('use Nonlocal')
    return model

def get_detection_local_model6(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d, trainable_layers=3)
    model = FasterRCNN(backbone=backbone, num_classes=91)
    weights_dict = torch.load("./faster_rcnn/backbone/fasterrcnn_resnet50_fpn_coco.pth")
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.box_head = ReWriteRoiHeadsBoxHead_nonlocal_stn()
    print('use Nonlocal + STN')
    return model

def get_detection_local_model7(num_classes):
    backbone = resnet50_fpn_backbone(trainable_layers=3)
    print('use frozen bn')
    model = FasterRCNN(backbone=backbone, num_classes=91)
    weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth")
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.box_head = ReWriteRoiHeadsBoxHead_nonlocal_se()
    print('use Nonlocal + SE')
    return model



def get_detection_model(num_classes, use_rewrite):  # 修改的第一版，se模块附加
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # 双分支预测头，直接采用了fasterrcnn的预测头
    if use_rewrite:
        model.roi_heads.box_head = ReWriteRoiHeadsBoxHead()
        print('原版+nonlocal+stn+se')
    else:
        print('使用最纯净原版')
    return model





def get_detection_local_model2(num_classes, use_rewrite, use_nonlocal):  # 改了 convnext！！！！！！！！！！！！！
    backbone = convnext_fpn_backbone(pretrain_path='', trainable_layers=3)
    model = FasterRCNN(backbone=backbone, num_classes=91)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if use_rewrite:
        if use_nonlocal:
            model.roi_heads.box_head = ReWriteRoiHeadsBoxHead()
            print('改backbone+改损失+nonlocal+stn+se')
        else:
            model.roi_heads.box_head = ReWriteRoiHeadsBoxHead_noNonLocal()
            print('改backbone+改损失+stn+se')
    else:
        print('改backbone+改损失')
    return model

def get_detection_local_model1(num_classes, use_stn, choose):
    backbone = None
    if choose == 'BN':
        backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d, trainable_layers=3)
        print('使用普通bn')
    elif choose == 'frozenBN':
        backbone = resnet50_fpn_backbone(trainable_layers=4, use_eca=0)  # 使用冻结的bn层
        print('基础resnet+fpn 使用了frozen bn, trainable=4')
    elif choose == 'bifpn':
        backbone = resnet50_bifpn_backbone(trainable_layers=3, nums_bi=1)
        print('使用了frozenBN,而且加了一次的bifp1')
    elif choose == 'refpn':
        backbone = resnet50_refpn_backbone(trainable_layers=3, use_eca=1)
        print('尝试进行refpn，仿rnn的fpn')
    else:
        print('发生什么事了')

    model = FasterRCNN(backbone=backbone, num_classes=91)
    weights_dict = torch.load("./faster_rcnn/backbone/fasterrcnn_resnet50_fpn_coco.pth")
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if use_stn:
        model.roi_heads.box_predictor = ReWriteRoiHeadsBoxHead_STN()
        print('stn+改损失')
    else:
        # model.roi_heads.box_predictor = ReWriteRoiHeadsBoxHead_norewrite()
        print('原版+改损失')
    # if use_rewrite:
    #     if use_nonlocal:
    #         model.roi_heads.box_head = ReWriteRoiHeadsBoxHead()
    #         print('原版+改损失函数+nonlocal+stn+se')
    #     else:
    #         model.roi_heads.box_head = ReWriteRoiHeadsBoxHead_noNonLocal()
    #         print('原版+改随时函数+stn+se')
    # else:
    #     print('使用原版+改损失函数')
    return model


if __name__=='__main__':
    get_detection_local_model2(2, 1, 1)