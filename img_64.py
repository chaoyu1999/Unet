import numpy as np
import os
import cv2
from PIL import Image


def get_patch(
    full_input_img, full_target_img, full_model_img, patch_n=10, patch_size=64
):
    # assert为断言函数
    # 只有当 输入图像大小 == 输出图像大小，程序才会继续执行
    assert full_input_img.shape == full_target_img.shape

    patch_input_imgs = []  # 声明处理后的输入图像为patch_input_imgs列表
    patch_target_imgs = []  # 声明处理后的标签图像为patch_target_imgs列表
    patch_model_imgs = []
    h, w = full_input_img.shape  # h , w 为图像的高和宽（即大小）
    new_h, new_w = patch_size, patch_size  # new_h, new_w 为补丁高、宽

    # 随机生成多少个补丁。该函数返回np类型的输入图像补丁及目标图像补丁
    for _ in range(patch_n):
        # np.random.randint(a,b) - 表示返回一个[a,b]之间的整型数
        # 设置补丁图像的边界情况。设置开始的点，patch图像直接加上补丁图像高、宽即可。
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # 声明补丁图像 patch_input_img / patch_target_img
        patch_input_img = full_input_img[top : top + new_h, left : left + new_w]
        patch_target_img = full_target_img[top : top + new_h, left : left + new_w]
        patch_model_img = full_model_img[top : top + new_h, left : left + new_w]

        # patch_input_imgs是一个列表。直接将处理后的补丁图像加进列表即可。
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
        patch_model_imgs.append(patch_model_img)
        # 最后返回np.array类型的补丁图像。
    return (
        np.array(patch_input_imgs),
        np.array(patch_target_imgs),
        np.array(patch_model_imgs),
    )


def main():
    img_dir = "F:\\Unet-master\\save\\fig\\input"
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        # 读取图像
        join = os.path.join(img_dir, img_name)
        img_input = np.array(Image.open(join))
        img_target = np.array(Image.open(join.replace("input", "target")))
        img_model = np.array(Image.open(join.replace("input", "output")))
        # 获取补丁图像
        patch_input, patch_target, patch_model = get_patch(
            img_input, img_target, img_model
        )
        # 保存补丁图像
        for i in range(len(patch_input)):
            cv2.imwrite(
                f"F:\\Unet-master\\save\\patch\\input\\{img_name.replace('.tif','')}_{i}.tif",
                patch_input[i],
            )
            cv2.imwrite(
                f"F:\\Unet-master\\save\\patch\\target\\{img_name.replace('.tif','')}_{i}.tif",
                patch_target[i],
            )
            cv2.imwrite(
                f"F:\\Unet-master\\save\\patch\\output\\{img_name.replace('.tif','')}_{i}.tif",
                patch_model[i],
            )
        pass


# execute
main()
