import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# 拼接主观图像-------------------------------------------------------------------
# 读取文件夹中的文件列表
proj_path = "E:/cy/Unet"  # 项目路径
db = "/pictures/DB_0/"  # DB_0是实际数据集，DB_1是测试数据集
imgs_path = proj_path + db
folders = os.listdir(imgs_path)[:-1]
# 补全文件路径
folders = [imgs_path + folder for folder in folders]
images = [os.listdir(folder) for folder in folders]
for i in range(len(images)):
    folder = folders[i]
    images[i] = [folder + "/" + image for image in images[i]]

images = np.concatenate(images)  # 合并所有的字符串列表为一个大的字符串列表
images = np.array(images).reshape(-1, images.shape[0] // len(folders))  # 转换为NumPy数组
# images = np.array(images).reshape(-1, 53)  # 转换为NumPy数组

for i in range(images.shape[1]):
    # 读取图像
    images_array = np.array(
        [np.asarray(Image.open(image)) for image in images[:, i]]
    )  # (6, 512, 512)
    # 将图片拼成一行
    image = np.concatenate(images_array, axis=1)  # (512, 3072)
    # 将图片转换为0~255
    image = (image - image.min()) / (image.max() - image.min()) * 255
    # 转换为uint8
    image = image.astype(np.uint8)

    # 保存图片
    cv2.imwrite(imgs_path + "cat/{}".format(images[0, i].split('/')[-1]), image)
