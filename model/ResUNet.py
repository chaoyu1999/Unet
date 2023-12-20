import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        return out


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, ):
        super(up_conv, self).__init__()
        self.ps = nn.PixelShuffle(2)
        self.up = nn.Sequential(
            nn.Conv2d(ch_in // 4, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.ps(x)
        x = self.up(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResUNet, self).__init__()
        downsample = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

        # down sampling
        self.w1 = nn.Conv2d(in_ch, 64, kernel_size=1, padding=0, stride=1)
        self.conv1 = BasicBlock(in_ch, 64)
        self.pool1 = downsample

        self.conv2 = BasicBlock(64, 128)  # 不变
        # to increase the dimensions
        self.w2 = nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=1)
        self.pool2 = downsample

        self.conv3 = BasicBlock(128, 256)
        # to increase the dimensions
        self.w3 = nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=1)
        self.pool3 = downsample

        self.conv4 = BasicBlock(256, 512)
        # to increase the dimensions
        self.w4 = nn.Conv2d(256, 512, kernel_size=1, padding=0, stride=1)
        self.pool4 = downsample

        self.conv5 = BasicBlock(512, 1024)
        # to increase the dimensions
        self.w5 = nn.Conv2d(512, 1024, kernel_size=1, padding=0, stride=1)

        # up sampling
        self.up6 = up_conv(1024, 512)
        self.conv6 = BasicBlock(1024, 512)

        self.up7 = up_conv(512, 256)
        self.conv7 = BasicBlock(512, 256)

        self.up8 = up_conv(256, 128)
        self.conv8 = BasicBlock(256, 128)

        self.up9 = up_conv(128, 64)
        self.conv9 = BasicBlock(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # 下采样部分
        down0_res = self.w1(x)
        down0 = self.conv1(x) + down0_res
        down1 = self.pool1(self.relu(down0))

        down1_res = self.w2(down1)
        down1 = self.conv2(down1) + down1_res
        down2 = self.pool2(self.relu(down1))

        down2_res = self.w3(down2)
        down2 = self.conv3(down2) + down2_res
        down3 = self.pool3(self.relu(down2))

        down3_res = self.w4(down3)
        down3 = self.conv4(down3) + down3_res
        down4 = self.pool4(self.relu(down3))

        down4_res = self.w5(down4)

        down5 = self.conv5(down4) + down4_res

        # 上采样部分
        up_6 = self.up6(self.relu(down5))
        merge6 = torch.cat([up_6, down3], dim=1)
        c6 = self.conv6(self.relu(merge6))

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, down2], dim=1)
        c7 = self.conv7(self.relu(merge7))

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, down1], dim=1)
        c8 = self.conv8(self.relu(merge8))

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, down0], dim=1)
        c9 = self.conv9(self.relu(merge9))

        c10 = self.conv10(self.relu(c9))

        return self.relu(c10)


def getResUNet(in_channel, out_channel):
    model = ResUNet(in_channel, out_channel)
    return model


if __name__ == '__main__':
    net = getResUNet(1, 1)
    print(net)
    x = torch.rand((1, 1, 512, 512))
    print(net.forward(x).shape)
