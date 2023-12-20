import torch
import torch.nn as nn
from torchvision import models


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResUNet(nn.Module):

    def __init__(self, in_channel, out_channel, block, num_block):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.dconv_last = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, out_channel, 1)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = self.conv1(x)  # shape(b,64,32, 32)
        temp = self.maxpool(conv1)  # shape(b,64, 16,16)
        conv2 = self.conv2_x(temp) # shape(b,64,16,16)
        conv3 = self.conv3_x(conv2) # shape(b, 128, 8,8)
        conv4 = self.conv4_x(conv3) # shape(b, 256, 4, 4)
        bottle = self.conv5_x(conv4) # shape(b, 512, 2, 2)
        # output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)
        # output = self.fc(output)
        x = self.upsample(bottle)
        # print(x.shape)
        # print(conv4.shape)
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        # print(x.shape)
        # print(conv3.shape)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        # print(x.shape)
        # print(conv2.shape)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up1(x)
        x = self.upsample(x)
        # print(x.shape)
        # print(conv1.shape)
        x = torch.cat([x, conv1], dim=1)
        out = self.dconv_last(x)

        return out

    def load_pretrained_weights(self):
        # 这个函数的功能是加载预训练的resnet34模型的权重到自定义的模型中
        # 输入：无
        # 输出：无
        # 异常：如果加载失败，会抛出异常

        model_dict = self.state_dict()  # 获取自定义模型的状态字典
        resnet34_weights = models.resnet34(True).state_dict()  # 获取预训练的resnet34模型的状态字典，并下载权重文件
        count_res = 0  # 初始化resnet34模型的键的计数器
        count_my = 0  # 初始化自定义模型的键的计数器

        reskeys = list(resnet34_weights.keys())  # 获取resnet34模型的状态字典的键的列表
        mykeys = list(model_dict.keys())  # 获取自定义模型的状态字典的键的列表
        # print(self) # 打印自定义模型的结构（可选）
        # print(models.resnet34()) # 打印resnet34模型的结构（可选）
        # print(reskeys) # 打印resnet34模型的状态字典的键（可选）
        # print(mykeys) # 打印自定义模型的状态字典的键（可选）

        corresp_map = []  # 初始化一个空列表，用于存储对应的键的映射
        while (True):  # 后缀相同的放入list
            reskey = reskeys[count_res]  # 获取resnet34模型的状态字典的第count_res个键
            mykey = mykeys[count_my]  # 获取自定义模型的状态字典的第count_my个键

            if "fc" in reskey:  # 如果resnet34模型的键中包含"fc"，说明已经遍历完了所有的卷积层，可以跳出循环
                break

            while reskey.split(".")[-1] not in mykey:  # 如果resnet34模型的键的后缀不在自定义模型的键中，说明还没有找到对应的键，需要继续向后遍历
                count_my += 1  # 自定义模型的键的计数器加一
                mykey = mykeys[count_my]  # 获取自定义模型的状态字典的下一个键

            corresp_map.append([reskey, mykey])  # 找到对应的键后，将它们作为一个列表添加到映射列表中
            count_res += 1  # resnet34模型的键的计数器加一
            count_my += 1  # 自定义模型的键的计数器加一

        for k_res, k_my in corresp_map:  # 遍历映射列表中的每一对键
            if k_res == "conv1.weight":  # 如果resnet34模型的键是"conv1.weight"，则需要对它进行特殊处理
                model_dict[k_my] = resnet34_weights[k_res][:, :1, :, :]
            else:
                model_dict[k_my] = resnet34_weights[k_res]  # 将resnet34模型的状态字典中对应的值赋给自定义模型的状态字典中对应的键

        try:  # 尝试加载自定义模型的状态字典
            self.load_state_dict(model_dict)  # 加载自定义模型的状态字典
            print("Loaded resnet34 weights in mynet !")  # 打印成功的信息
        except:  # 如果加载失败
            print("Error resnet34 weights in mynet !")  # 打印错误的信息
            raise  # 抛出异常


def getResUNet(in_channel, out_channel, pretrain=True):
    model = ResUNet(in_channel, out_channel, BasicBlock, [3, 4, 6, 3])
    if pretrain:
        model.load_pretrained_weights()
    return model


# if __name__ == '__main__':
#     net = getResUNet(1, 1, True)
#     print(net)
#     x = torch.rand((1, 1, 512, 512))
#     print(net.forward(x).shape)
