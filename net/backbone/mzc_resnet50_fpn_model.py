import os
from collections import OrderedDict  # 有序字典其实并不是真的有序，而是可以保留插入顺序

import torch
import torch.nn as nn
from torch.jit.annotations import List, Dict
from torchvision.ops.misc import FrozenBatchNorm2d

from .mzc_Bi_FPN import BiFeaturePyramidNetwork, LastLevelMaxPool   # 注意这里修改了路径


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = norm_layer(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = norm_layer(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = norm_layer(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)  # 这里判断shortcut是否需要变更通道数

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.include_top = include_top  # 是否包括顶端的分类头
        self.in_channel = 64
        # 3是图像的RGB通道，self的inchannel是当前层的输出通道数
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 这个条件肯定有问题，因为可以看出，每一个stage的开头都需要进行downsample，可能是针对不同channel设计才写的这个条件
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion))  # 第一个

        layers = []
        #  确认了downsample之后，创建stage中的一个block，这里除了downsample参数外，stride参数也是只有一个stage的第一个block需要的
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, norm_layer=norm_layer))
        self.in_channel = channel * block.expansion  # 然后给这个类变量赋值当前stage的输出通道数

        for _ in range(1, block_num):  # 然后不需要dowmsample和stride默认为1的其余block进行堆叠
            layers.append(block(self.in_channel, channel, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def overwrite_eps(model, eps):  #
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


class IntermediateLayerGetter(nn.ModuleDict):  # 中间层提取
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    __annotations__ = {
        "return_layers": Dict[str, str],  # 规定了字典格式，好像也还是普通字典，就是键值都是str类型
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        # return_layers 就是 {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        # 遍历模型子模块按顺序存入有序字典
        # 只保存layer4及其之前的结构，舍去之后不用的结构
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                # 这个layers字典里按顺序全部存进去了，如果看到returnlayers的内容，就相当于遍历到了，进行一个删除
                # 其实就是遍历到layer4之后，该找的都找到了，就后面的不要了
                del return_layers[name]
            if not return_layers:
                break
        # 这里把layers有序字典传入了super，不太懂，不过感觉上来讲，我觉得这个类，也就是self，就是这个字典了
        # 根据后面看起来，这玩意已经是个网络结构了，也就是只包含layer4以及前面所有resnet层的网络结构
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers  # 找到需要的模型前半段之后，恢复return_layers的内容

    def forward(self, x):
        out = OrderedDict()
        # 依次遍历模型的所有子模块，并进行正向传播，
        # 收集layer1, layer2, layer3, layer4的输出
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]  # 传入的return_layers字典中，键为res内的名称，值为后面fpn的取名
                out[out_name] = x
        return out  # 如此一来，out就搜集了4个stage的输出结果了


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        extra_blocks: ExtraFPNBlock
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extra_blocks=None, nums_bi=0):
        #  这里传入了backbon和几个参数，其中in_channels_list其实就是每个layer的输出通道数量，用来和fpn对齐的
        #  而这里的out_channels规定了后面fpn每个层的大小
        super(BackboneWithFPN, self).__init__()
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        # 这个body中提取中间层，这里的返回值应当只是几个stage的输出特征图
        self.nums_bi = nums_bi
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        if self.nums_bi == 1:
            self.BiFPN1 = BiFeaturePyramidNetwork(  # 非最后一层，不用extra_blocks, 作为第一层，is_first=True，用于调整通道
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks,
                is_first=True
            )

        elif self.nums_bi == 5:
            self.BiFPN1 = BiFeaturePyramidNetwork(  # 非最后一层，不用extra_blocks, 作为第一层，is_first=True，用于调整通道
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                is_first=True
            )
            self.BiFPN2 = BiFeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                is_first=False
            )
            self.BiFPN3 = BiFeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                is_first=False
            )
            self.BiFPN4 = BiFeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                is_first=False
            )
            self.BiFPN5 = BiFeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                is_first=False,
                extra_blocks=extra_blocks
            )
        elif nums_bi == 2:
            self.BiFPN1 = BiFeaturePyramidNetwork(  # 非最后一层，不用extra_blocks, 作为第一层，is_first=True，用于调整通道
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                is_first=True
            )
            self.BiFPN2 = BiFeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks,
                is_first=False
            )
        else:
            print('出大问题！！')


        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        # 这里就是module调用的特殊之处，网络层不需要作为初始化时候的参数，只在forward时候进行，此处就是从body获取特征层，然后fpn自然就拿到了
        if self.nums_bi == 1:
            x = self.BiFPN1(x)
        elif self.nums_bi == 2:
            x = self.BiFPN1(x)
            x = self.BiFPN2(x)
        elif self.nums_bi == 5:
            x = self.BiFPN1(x)
            x = self.BiFPN2(x)
            x = self.BiFPN3(x)
            x = self.BiFPN4(x)
            x = self.BiFPN5(x)
        return x


def resnet50_bifpn_backbone(pretrain_path="",
                            norm_layer=FrozenBatchNorm2d,  # FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新
                            trainable_layers=3,
                            returned_layers=None,
                            extra_blocks=None,
                            nums_bi=0):
    """
    搭建resnet50_fpn——backbone
    Args:
        pretrain_path: resnet50的预训练权重，如果不使用就默认为空
        norm_layer: 官方默认的是FrozenBatchNorm2d，即不会更新参数的bn层(因为如果batch_size设置的很小会导致效果更差，还不如不用bn层)
                    如果自己的GPU显存很大可以设置很大的batch_size，那么自己可以传入正常的BatchNorm2d层
                    (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        trainable_layers: 指定训练哪些层结构
        returned_layers: 指定哪些层的输出需要返回
        extra_blocks: 在输出的特征层基础上额外添加的层结构

    Returns:

    """
    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3],
                             include_top=False,
                             norm_layer=norm_layer)

    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)

    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "{} is not exist.".format(pretrain_path)
        # 载入预训练权重
        print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5  # 这里考虑前面的特征提取已经很有效了，不需要训练
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # 如果要训练所有层结构的话，不要忘了conv1后还有一个bn1
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    # freeze layers
    # 比如我们选择了layer4 layer3 laryer2,那么下面这个循环中，就会把layer1和conv1的参数冻结，因为这是基础的细节获取，没必要再训练了
    for name, parameter in resnet_backbone.named_parameters():
        # 只训练不在layers_to_train列表中的层结构
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    # 返回的特征层个数肯定大于0小于5
    assert min(returned_layers) > 0 and max(returned_layers) < 5

    # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    # in_channel 为layer4的输出特征矩阵channel = 2048
    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
    # 记录resnet50提供给fpn的每个特征层channel
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]  # 256 512 1024 2048
    # 通过fpn后得到的每个特征层的channel
    out_channels = 256
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels,
                           extra_blocks=extra_blocks, nums_bi=nums_bi)
