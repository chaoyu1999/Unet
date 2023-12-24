import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse

from sklearn.metrics import mean_absolute_error as compare_mae

# 计算图像的指标，即平均相似度、峰值信噪比、均方误差 等相关指标

# 输入图像文件夹路径
save_path = 'E:/cy/Unet/save_NAF_LOSS'
cbcts_path = '%s/fig/input/' % save_path
# 目标图像文件夹路径
rpcts_path = '%s/fig/target/' % save_path
# 模型输出图像文件夹路径
models_path = '%s/fig/output/' % save_path

# 获取输入图像文件夹中的文件列表，即输入数据，带有环形伪影的数据集
cbcts_file = os.listdir(cbcts_path)

# 定义原始图像和模型输出图像指标列表
cbct_psnr = []
cbct_ssim = []
cbct_mae = []
cbct_rmse = []

model_psnr = []
model_ssim = []
model_mae = []
model_rmse = []

# 遍历每个输入图像文件
for cbct_file in cbcts_file:
    # 构建输入图像路径
    cbct_path = cbcts_path + '/' + cbct_file
    # 读取输入图像并转换为灰度图
    cbct_img = np.asarray(Image.open(cbct_path).convert('F'))
    # 将像素值限制在0到1之间
    cbct_img = np.maximum(cbct_img, 0)
    cbct_img = np.minimum(cbct_img, 1)

    # 构建目标图像路径
    rpct_path = rpcts_path + '/' + 'target_' + cbct_file.split('.')[0][6:] + '.tif'
    # 读取目标图像并转换为灰度图
    rpct_img = np.asarray(Image.open(rpct_path).convert('F'))
    # 将像素值限制在0到1之间
    rpct_img = np.maximum(rpct_img, 0)
    rpct_img = np.minimum(rpct_img, 1)

    # 构建模型输出图像路径
    model_path = models_path + '/' + 'output_' + cbct_file.split('.')[0][6:] + '.tif'
    # 读取模型输出图像并转换为灰度图
    model_img = np.asarray(Image.open(model_path).convert('F'))
    # 将像素值限制在0到1之间
    model_img = np.maximum(model_img, 0)
    model_img = np.minimum(model_img, 1)

    # 计算原始图像和模型输出图像的峰值信噪比和结构相似度
    cbct_psnr.append(compare_psnr(image_true=rpct_img, image_test=cbct_img))
    cbct_ssim.append(compare_ssim(im1=rpct_img, im2=cbct_img, data_range=1))
    model_psnr.append(compare_psnr(image_true=rpct_img, image_test=model_img))
    model_ssim.append(compare_ssim(im1=rpct_img, im2=model_img, data_range=1))

    # 将图像像素值转换为HU值
    cbct_img = cbct_img * 4095 - 1000
    rpct_img = rpct_img * 4095 - 1000
    model_img = model_img * 4095 - 1000

    # 计算原始图像和模型输出图像的平均绝对误差和均方根误差
    cbct_mae.append(compare_mae(y_true=rpct_img, y_pred=cbct_img))
    model_mae.append(compare_mae(y_true=rpct_img, y_pred=model_img))
    cbct_rmse.append(np.sqrt(compare_mse(image0=rpct_img, image1=cbct_img)))
    model_rmse.append(np.sqrt(compare_mse(image0=rpct_img, image1=model_img)))

# 打印原始图像和模型输出图像的指标结果
print('-------------------------------------------------------------------------')
print('cbct: mae {:.4f}±{:.4f}  rmse {:.4f}±{:.4f} psnr {:.4f}±{:.4f}  ssim {:.4f}±{:.4f}'.
      format(np.mean(cbct_mae), np.std(cbct_mae), np.mean(cbct_rmse), np.std(cbct_rmse), np.mean(cbct_psnr),
             np.std(cbct_psnr), np.mean(cbct_ssim), np.std(cbct_ssim)))
print('model: mae {:.4f}±{:.4f}  rmse {:.4f}±{:.4f} psnr {:.4f}±{:.4f}  ssim {:.4f}±{:.4f}'.
      format(np.mean(model_mae), np.std(model_mae), np.mean(model_rmse), np.std(model_rmse), np.mean(model_psnr),
             np.std(model_psnr), np.mean(model_ssim), np.std(model_ssim)))
