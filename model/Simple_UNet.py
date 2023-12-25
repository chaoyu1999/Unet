import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义UNet模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 编码器部分
        self.inc = nn.Sequential(
            single_conv(1, 64),  # 单层卷积
            single_conv(64, 64)  # 单层卷积
        )

        self.down1 = nn.AvgPool2d(2)  # 下采样
        self.conv1 = nn.Sequential(
            single_conv(64, 128),  # 单层卷积
            single_conv(128, 128),  # 单层卷积
            single_conv(128, 128)  # 单层卷积
        )

        self.down2 = nn.AvgPool2d(2)  # 下采样
        self.conv2 = nn.Sequential(
            single_conv(128, 256),  # 单层卷积
            single_conv(256, 256),  # 单层卷积
            single_conv(256, 256),  # 单层卷积
            single_conv(256, 256),  # 单层卷积
            single_conv(256, 256),  # 单层卷积
            single_conv(256, 256)  # 单层卷积
        )

        # 解码器部分
        self.up1 = up(256)  # 上采样
        self.conv3 = nn.Sequential(
            single_conv(128, 128),  # 单层卷积
            single_conv(128, 128),  # 单层卷积
            single_conv(128, 128)  # 单层卷积
        )

        self.up2 = up(128)  # 上采样
        self.conv4 = nn.Sequential(
            single_conv(64, 64),  # 单层卷积
            single_conv(64, 64)  # 单层卷积
        )

        self.outc = outconv(64, 1)  # 输出层

    # 前向传播
    def forward(self, x):
        inx = self.inc(x)  # 编码器部分

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)  # 解码器部分
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)  # 解码器部分
        conv4 = self.conv4(up2)

        out = self.outc(conv4)  # 输出
        return out


# 定义单层卷积模块
class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # 3x3卷积
            nn.GELU()  # GELU激活函数
        )

    def forward(self, x):
        return self.conv(x)


# 定义上采样模块
class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)  # 转置卷积

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 输入为CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x


# 定义输出层
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)  # 1x1卷积

    def forward(self, x):
        x = self.conv(x)
        return x


# 定义简化版UNet模型
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.unet = UNet()

    # 前向传播
    def forward(self, x):
        out = self.unet(x) + x
        return out
