from model.DRUNet_parts import *
from model.CBAM import *
import torch.nn.functional as F


class DRUNet_CBAM_Multiscale(nn.Module):
    def __init__(self):
        super(DRUNet_CBAM_Multiscale, self).__init__()
        self.input = InputCov(1, 64)
        self.cbam = CBAM(64)
        self.down1 = Down(64, 128)

        self.left1 = LeftGlobalResBlock(128, 128)
        self.down2 = Down(128, 256)
        self.left2 = LeftGlobalResBlock(256, 256)
        self.down3 = Down(256, 512)
        self.left3 = LeftGlobalResBlock(512, 512)
        self.up1 = Up()
        self.right1 = RightGlobalResBlock(256, 256)
        self.out_1 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, padding=0, stride=1),
            nn.ReLU()
        )

        self.up2 = Up()
        self.right2 = RightGlobalResBlock(128, 128)
        self.out_2 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1, padding=0, stride=1),
            nn.ReLU()
        )

        self.up3 = Up()
        self.right3 = RightGlobalResBlock(64, 64)
        # self.output = OutputCov(64, 1)
        self.ca = ChannelAttentionModule(256 + 128 + 64)
        self.output = OutputCov(256 + 128 + 64, 1)


    def forward(self, init_img):
        input_map_ = self.input(init_img)
        input_map = input_map_ + self.cbam(input_map_)
        down1_map = self.down1(input_map)
        left1_map = self.left1(down1_map)
        down2_map = self.down2(left1_map)
        left2_map = self.left2(down2_map)
        down3_map = self.down3(left2_map)
        left3_map = self.left3(down3_map)

        up1_map = self.up1(left3_map, left2_map)
        right1_map = self.right1(up1_map)
        r1 = F.interpolate(
            right1_map, scale_factor=4, mode="bilinear", align_corners=False
        )

        # out_1 = self.out_1(right1_map)
        up2_map = self.up2(right1_map, left1_map)
        right2_map = self.right2(up2_map)
        # out_2 = self.out_2(right2_map)
        r2 = F.interpolate(
            right2_map, scale_factor=2, mode="bilinear", align_corners=False
        )
        up3_map = self.up3(right2_map, input_map_)
        right3_map = self.right3(up3_map)
        # output = self.output(right3_map)

        cat_out = torch.cat([right3_map, r1, r2], dim=1)
        output = cat_out + self.ca(cat_out) * cat_out
        output = self.output(output)
        return output


class DRUNet_CBAM(nn.Module):
    def __init__(self):
        super(DRUNet_CBAM, self).__init__()
        self.input = InputCov(1, 64)
        self.cbam = CBAM(64)
        self.down1 = Down(64, 128)

        self.left1 = LeftGlobalResBlock(128, 128)
        self.down2 = Down(128, 256)
        self.left2 = LeftGlobalResBlock(256, 256)
        self.down3 = Down(256, 512)
        self.left3 = LeftGlobalResBlock(512, 512)
        self.up1 = Up()
        self.right1 = RightGlobalResBlock(256, 256)
        # self.out_1 = nn.Sequential(
        #     nn.Conv2d(256, 1, kernel_size=1, padding=0, stride=1),
        #     nn.ReLU()
        # )

        self.up2 = Up()
        self.right2 = RightGlobalResBlock(128, 128)
        # self.out_2 = nn.Sequential(
        #     nn.Conv2d(128, 1, kernel_size=1, padding=0, stride=1),
        #     nn.ReLU()
        # )

        self.up3 = Up()
        self.right3 = RightGlobalResBlock(64, 64)
        self.output = OutputCov(64, 1)
        # self.ca = ChannelAttentionModule(256 + 128 + 64)
        # self.output = OutputCov(256 + 128 + 64, 1)


    def forward(self, init_img):
        input_map_ = self.input(init_img)
        input_map = input_map_ + self.cbam(input_map_)
        down1_map = self.down1(input_map)
        left1_map = self.left1(down1_map)
        down2_map = self.down2(left1_map)
        left2_map = self.left2(down2_map)
        down3_map = self.down3(left2_map)
        left3_map = self.left3(down3_map)

        up1_map = self.up1(left3_map, left2_map)
        right1_map = self.right1(up1_map)
        # r1 = F.interpolate(
        #     right1_map, scale_factor=4, mode="bilinear", align_corners=False
        # )

        # out_1 = self.out_1(right1_map)
        up2_map = self.up2(right1_map, left1_map)
        right2_map = self.right2(up2_map)
        # out_2 = self.out_2(right2_map)
        # r2 = F.interpolate(
        #     right2_map, scale_factor=2, mode="bilinear", align_corners=False
        # )
        up3_map = self.up3(right2_map, input_map_)
        right3_map = self.right3(up3_map)
        # output = self.output(right3_map)

        # cat_out = torch.cat([right3_map, r1, r2], dim=1)
        # output = cat_out + self.ca(cat_out) * cat_out
        output = self.output(right3_map)
        return 1, 2 , output



class DRUNet(nn.Module):
    def __init__(self):
        super(DRUNet, self).__init__()
        self.input = InputCov(1, 64)
        self.down1 = Down(64, 128)
        self.left1 = LeftGlobalResBlock(128, 128)
        self.down2 = Down(128, 256)
        self.left2 = LeftGlobalResBlock(256, 256)
        self.down3 = Down(256, 512)
        self.left3 = LeftGlobalResBlock(512, 512)
        self.up1 = Up()
        self.right1 = RightGlobalResBlock(256, 256)
        self.up2 = Up()
        self.right2 = RightGlobalResBlock(128, 128)
        self.up3 = Up()
        self.right3 = RightGlobalResBlock(64, 64)
        self.output = OutputCov(64, 1)

    def forward(self, init_img, _=None, __=None):
        input_map = self.input(init_img)
        down1_map = self.down1(input_map)
        left1_map = self.left1(down1_map)
        down2_map = self.down2(left1_map)
        left2_map = self.left2(down2_map)
        down3_map = self.down3(left2_map)
        left3_map = self.left3(down3_map)
        up1_map = self.up1(left3_map, left2_map)
        right1_map = self.right1(up1_map)
        up2_map = self.up2(right1_map, left1_map)
        right2_map = self.right2(up2_map)
        up3_map = self.up3(right2_map, input_map)
        right3_map = self.right3(up3_map)
        output = self.output(right3_map)
        return _, __, output