import os
import cv2


def process_images(directory):
    # 获取目录下的所有文件
    files = os.listdir(directory)

    for file in files:
        # 判断文件是否为图片文件
        if file.endswith(".png"):
            # 构建图片文件的完整路径
            file_path = os.path.join(directory, file)

            # 读取图片
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图像方式读取

            # 将小于50的像素值置为5
            image[image < 50] = 5
            # 保存修改后的图片
            cv2.imwrite(file_path, image)

            print(f"Processed image: {file}")


# 调用函数，传入目录路径
process_images("pictures/DB_1/1_unet")
