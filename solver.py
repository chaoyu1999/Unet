import os
import time
import numpy as np
import matplotlib
import cv2

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from measure import compute_measure  # 计算指标函数
# from prep import printProgressBar  # 从另一文件中引入打印进度条函数
from networks import U_Net  # 从网络模块中引入RED-CNN模型
from model.DRUNet_model import DRUNet_CBAM, DRUNet, DRUNet_CBAM_Multiscale
from model.ResUNet import getResUNet
from model.NAF_Simple_UNet import NAFUNet, MS_SSIM_L1_LOSS
from model.Simple_UNet import SimpleUNet


# 该文件是处理步骤的集合(包括保存加载模型、学习率衰减、归一化、保存图像、训练和测试过程)

# 定义一个类Solver类，继承object类，其为所有类的基类。
class Solver(object):
    def __init__(self, args, data_loader):  # 调用该类时，需传参args和data_loader
        # 初始化函数，self表示未创建的类实例对象
        # args表示系统命令行参数

        self.mode = args.mode  # 表示训练 or 测试
        self.load_mode = args.load_mode  # 表示数据加载方式 - >10 or <=10
        self.data_loader = data_loader  # dataloader的定义/获取方式

        # 设置设备声明，可能是cpu或CUDA，根据电脑配置进行设置
        if args.device:  # 如果命令行设备参数设置，则选择命令行参数对应的设备类型
            self.device = torch.device(args.device)
        else:  # 选择设备 - （ GPU  or  CPU ）
            ####
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 设置实例对象的参数值
        # 一些归一化和图像色彩数值之间的配置
        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        # 设置训练模型和结果图像保存的位置
        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        # 设置一共训练多少轮（全部数据的总训练轮数
        self.num_epochs = args.num_epochs

        # 设置打印频数
        self.print_iters = args.print_iters

        # 设置衰减频数
        self.decay_iters = args.decay_iters

        # 设置保存频数
        self.save_iters = args.save_iters

        # 设置具体测试选择的轮数
        self.test_iters = args.test_iters

        # 设置是否要保存图像
        self.result_fig = args.result_fig

        # 设置切片图像的大小
        self.patch_size = args.patch_size

        # 确定网络结构 - 残差卷积自编码网络模型
        ####
        self.U_Net = NAFUNet()

        # 判断有几个CUDA处理器，选择其中一个
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print("Use {} GPUs".format(torch.cuda.device_count()))

            # self.REDCNN = nn.DataParallel(self.REDCNN)
            self.U_Net = nn.DataParallel(self.U_Net)
            # self.R2U_Net = nn.DataParallel(self.R2U_Net)
            # self.AttU_Net = nn.DataParallel(self.AttU_Net)
            # self.R2AttU_Net = nn.DataParallel(self.R2AttU_Net)
            # self.unet_sqq = nn.DataParallel(self.unet_sqq)
        ####
        # self.REDCNN.to(self.device) # 模型也用CUDA进行训练
        self.U_Net.to(self.device)

        self.lr = args.lr  # 设置学习率
        # self.criterion = nn.MSELoss()  # 设置损失函数，为均方误差
        '''这里改了loss函数为MS_SSIM_L1_LOSS'''
        self.criterion = MS_SSIM_L1_LOSS(data_range=1.0)
        ####
        # self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr) # 设置优化器
        self.optimizer = optim.Adam(self.U_Net.parameters(), self.lr)
        # self.optimizer = optim.Adam(self.R2U_Net.parameters(), self.lr)
        # self.optimizer = optim.Adam(self.AttU_Net.parameters(), self.lr)
        # self.optimizer = optim.Adam(self.R2AttU_Net.parameters(), self.lr)
        # self.optimizer = optim.Adam(self.unet_sqq.parameters(), self.lr)

    def save_model(self, iter_):  # 保存这个训练模型，iter_表示轮数 第几个
        # f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        # torch.save(self.REDCNN.state_dict(), f)

        ####
        f = os.path.join(self.save_path, "U_Net_{}iter.ckpt".format(iter_))
        torch.save(self.U_Net.state_dict(), f)

        # f = os.path.join(self.save_path, 'R2U_Net{}iter.ckpt'.format(iter_))
        # torch.save(self.R2U_Net.state_dict(), f)

        # f = os.path.join(self.save_path, 'AttU_Net{}iter.ckpt'.format(iter_))
        # torch.save(self.AttU_Net.state_dict(), f)

        # f = os.path.join(self.save_path, 'R2AttU_Net{}iter.ckpt'.format(iter_))
        # torch.save(self.R2U_Net.state_dict(), f)

        # f = os.path.join(self.save_path, 'unet_sqq{}iter.ckpt'.format(iter_))
        # torch.save(self.unet_sqq.state_dict(), f)

    # 加载模型
    def load_model(self, iter_):
        device = torch.device(self.device)
        ####
        # f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        f = os.path.join(self.save_path, "U_Net_{}iter.ckpt".format(iter_))
        # f = os.path.join(self.save_path, 'R2U_Net{}iter.ckpt'.format(iter_))
        # f = os.path.join('model_pretrained', 'T2T_vit_{}iter.ckpt'.format(iter_))
        # f = os.path.join(self.save_path, 'AttU_Net_{}iter.ckpt'.format(iter_))
        # f = os.path.join(self.save_path, 'R2AttU_Net_{}iter.ckpt'.format(iter_))
        # f = os.path.join(self.save_path, 'unet_sqq_{}iter.ckpt'.format(iter_))
        ####
        # self.REDCNN.load_state_dict(torch.load(f, map_location=device))
        self.U_Net.load_state_dict(torch.load(f, map_location=device))
        # self.R2U_Net.load_state_dict(torch.load(f, map_location=device))
        # self.AttU_Net.load_state_dict(torch.load(f, map_location=device))
        # self.UR2AttU_Net.load_state_dict(torch.load(f, map_location=device))
        # self.unet_sqq.load_state_dict(torch.load(f, map_location=device))

        # f = os.path.join(self.save_path, 'REDCNN_{}iter.ckpt'.format(iter_))
        # if self.multi_gpu:
        #     state_d = OrderedDict()
        #     for k, v in torch.load(f):
        #         n = k[7:]
        #         state_d[n] = v
        #     self.REDCNN.load_state_dict(state_d)
        # else:
        #     self.REDCNN.load_state_dict(torch.load(f))

    # 学习率衰减
    def lr_decay(self):
        lr = self.lr * 0.5
        # 更新学习率
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    # 对图像进行归一化处理
    def denormalize_(self, image):
        # image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    # 当使用标量数据且没有显式的*norm*、*vmin*和*vmax*时，定义色彩图覆盖的数据范围
    # 当给出*norm*时，使用*vmin*/*vmax*是错误的；当使用RGB(A)数据时，参数*vmin*/*vmax*被忽略。
    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    # 保存图像
    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray)
        # ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title("input", fontsize=30)  # 半剂量
        ax[0].set_xlabel(
            "PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(
                original_result[0], original_result[1], original_result[2]
            ),
            fontsize=20,
        )
        ax[1].imshow(pred, cmap=plt.cm.gray)
        # ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title("output", fontsize=30)
        ax[1].set_xlabel(
            "PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(
                pred_result[0], pred_result[1], pred_result[2]
            ),
            fontsize=20,
        )
        ax[2].imshow(y, cmap=plt.cm.gray)
        # ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title("target", fontsize=30)  # 全剂量

        # 保存图像
        f.savefig(os.path.join(self.save_path, "fig", "result_{}.png".format(fig_name)))
        # 关闭创建的一个图像对象
        plt.close()

    # 开启训练模式
    def train(self):
        train_losses = []  # 记录训练过程中的损失值
        total_iters = 0  # 记录迭代次数
        start_time = time.time()  # 定义时间函数
        loss_all = []
        loss_list = []  # 存储loss，计算平均loss
        best_loss = float("inf")
        # 遍历训练轮数，训练一轮就将所有的数据训练一遍
        for epoch in range(1, self.num_epochs):
            torch.cuda.empty_cache()
            ####
            # self.REDCNN.train(True) # 开启训练模式，REDCNN自带train()函数
            self.U_Net.train(True)
            # self.R2U_Net.train(True)
            # self.AttU_Net.train(True)
            # self.R2AttU_Net.train(True)
            # self.unet_sqq.train(True)

            # dataloader是一个列表，其中的每个元素是一个列表，是batchsize个(1批次)图像及标签
            # enumerate()表示将列表进行元组打包，返回索引值及数据 [ 图像 + 标签 ]
            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1  # 每训练一个batchsize，增加一次迭代数

                # add 1 channel
                # 进行维度扩充，在指定位置dim=0处加上维数为1的维度，并转换为cuda等形式
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                # print(x.shape)

                # 如果设置了patch的大小，则用patch进行训练
                if self.patch_size:  # patch training
                    # 改变图像数据的大小，依次是（批次、通道、高、宽）
                    x = x.view(-1, 1, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)
                else:
                    x = x.view(-1, 1, 512, 512)
                    y = y.view(-1, 1, 512, 512)

                # y1 = F.interpolate(
                #     y, size=(32, 32), mode="bilinear", align_corners=False
                # )
                #
                # y2 = F.interpolate(
                #     y, size=(64, 64), mode="bilinear", align_corners=False
                # )
                ####
                # pred = self.REDCNN(x)  # 将数据放到模型中进行训练
                pred = self.U_Net(x)
                # pred = self.R2U_Net(x)
                # pred = self.AttU_Net(x)
                # pred = self.R2AttU_Net(x)
                # pred = self.unet_sqq(x)

                # loss_1 = self.criterion(o1, y1)
                # loss_2 = self.criterion(o2, y2)
                loss_3 = self.criterion(pred, y)
                loss = loss_3  # + loss_1 + loss_2
                # 确定损失函数 - 预测值与实际值之间的差
                ####
                # self.REDCNN.zero_grad()  # 模型梯度清零，为反向传播优化参数作准备
                self.U_Net.zero_grad()

                self.optimizer.zero_grad()  # 梯度清零

                loss.backward()  # 反向传播

                self.optimizer.step()  # 梯度优化

                train_losses.append(loss_3.item())  # 将此次过程中的损失值添到损失列表中
                loss_all.append(loss_3.item())
                loss_list.append(loss_3.item())

                # print，当满足某条件时 进行打印
                if total_iters % self.print_iters == 0:
                    print(
                        "STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(
                            total_iters,
                            epoch,
                            self.num_epochs,
                            iter_ + 1,
                            len(self.data_loader),
                            np.mean(loss_list),
                            time.time() - start_time,
                        )
                    )
                    loss_list = []

                # learning rate decay，当满足某条件时 进行学习率的衰减
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()

                # save model，当满足某条件时 进行模型的保存
                if total_iters % self.save_iters == 0:
                    pass  # 暂时不保存模型
                    # self.save_model(total_iters)
                    # os.path.join()表示连接函数，作用是连接字符串
                    # np.save(
                    #     os.path.join(
                    #         self.save_path, "loss_{}_iter.npy".format(total_iters)
                    #     ),
                    #     np.array(train_losses),
                    # )

                # 保存最好的模型
                if total_iters > 10000 and np.mean(loss_list) < best_loss:
                    best_loss = loss.item()
                    self.save_model("best_" + str(total_iters))

        # 保存最后一次的训练结果
        self.save_model(total_iters)

        np.save(
            os.path.join(self.save_path, "loss_{}_iter.npy".format(total_iters)),
            np.array(train_losses),
        )
        print("total_iters:", total_iters)

        # 绘制loss折线，用红色线条绘制
        ## save loss figure
        # plt.plot(np.array(loss_all), "r")  ## print out the loss curve
        # plt.show()
        # 将所作的图保存在save文件夹中的loss.png中
        # plt.savefig('save/loss.png') # 保存文件名

    # 定义测试函数
    def test(self):
        ####
        # del self.REDCNN
        del self.U_Net

        ####
        # load，加载模型
        # self.REDCNN = RED_CNN().to(self.device)
        self.U_Net = DRUNet_CBAM_Multiscale().to(self.device)
        # self.U_Net = R2U_Net().to(self.device)
        # self.U_Net = AttU_Ne().to(self.device)
        # self.U_Net = R2AttU_Net().to(self.device)
        # self.U_Net = unet_sqq().to(self.device)

        path_root = r"E:\cy\结果汇总\NAFUNet\save_DRUNet_CBAM_Multiscale"
        path_input = "%s/fig/input/" % path_root
        path_target = "%s/fig/target/" % path_root
        path_output = "%s/fig/output/" % path_root
        # 检查并创建路径
        self.create_path_if_not_exists(path_root)
        self.create_path_if_not_exists(path_input)
        self.create_path_if_not_exists(path_target)
        self.create_path_if_not_exists(path_output)
        # 加载模型，第test_iters次训练的模型
        # self.load_model(self.test_iters)
        self.U_Net.load_state_dict(torch.load(path_root + "/U_Net_best.ckpt", map_location=self.device))
        # compute PSNR, SSIM, RMSE - 计算评价指标
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        # 不要更新梯度的模式开始 - 即测试阶段
        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                shape_ = x.shape[-1]

                shape_a = x.shape[-2]

                # 维度变化，增加一个维度
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                ####
                # pred = self.REDCNN(x)  # 放入模型，进行预测
                # _, __, pred = self.U_Net(x)
                pred = self.U_Net(x)
                # pred = self.R2U_Net(x)
                # pred = self.AttU_Ne(x)
                # pred = self.R2AttU_Net(x)
                # pred = self.unet_sqqunet_sqq(x)

                # denormalize, truncate
                # x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                # y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                # pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                # 笛卡尔坐标系图像
                # x = self.trunc(x.view(shape_, shape_).cpu().detach())
                # y = self.trunc(y.view(shape_, shape_).cpu().detach())
                # pred = self.trunc(pred.view(shape_, shape_).cpu().detach())

                # 极坐标系图像
                x = self.trunc(x.view(shape_a, shape_).cpu().detach())
                y = self.trunc(y.view(shape_a, shape_).cpu().detach())
                pred = self.trunc(pred.view(shape_a, shape_).cpu().detach())

                x, y, pred = x.numpy(), y.numpy(), pred.numpy()
                # 保存图像
                # x为带有环形伪影的图像
                Image.fromarray(x).save(path_input + "input_" + str(i) + ".tif")
                # y为目标图像 即没有环形伪影的图像
                Image.fromarray(y).save(path_target + "target_" + str(i) + ".tif")
                # pred为预测值，即通过模型得到的测试结果
                Image.fromarray(pred).save(path_output + "output_" + str(i) + ".tif")

                # data_range = self.trunc_max - self.trunc_min # 数值范围 - 色彩范围
                #
                # 计算图像质量评价指标
                # original_result, pred_result = compute_measure(x, y, pred, data_range)
                # ori_psnr_avg += original_result[0]
                # ori_ssim_avg += original_result[1]
                # ori_rmse_avg += original_result[2]
                # pred_psnr_avg += pred_result[0]
                # pred_ssim_avg += pred_result[1]
                # pred_rmse_avg += pred_result[2]

                # # save result figure
                # # 一开始设置可以保存图像了，即result_fig = true
                # if self.result_fig: # 如果该值不为空，则保存该图像
                #     self.save_fig(x, y, pred, i, original_result, pred_result)
                #
                # # 打印进度条
                # printProgressBar(i, len(self.data_loader),
                #                  prefix="Compute measurements ..",
                #                  suffix='Complete', length=25)

            print("\n")
            # 打印平均信噪比、平均结构相似性的值
            print(
                "Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}".format(
                    ori_psnr_avg / len(self.data_loader),
                    ori_ssim_avg / len(self.data_loader),
                    ori_rmse_avg / len(self.data_loader),
                )
            )
            print("\n")
            print(
                "Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}".format(
                    pred_psnr_avg / len(self.data_loader),
                    pred_ssim_avg / len(self.data_loader),
                    pred_rmse_avg / len(self.data_loader),
                )
            )

    # 测试文件中的图像
    def test_img(self):
        del self.U_Net
        self.U_Net = NAFUNet().to(self.device)
        self.load_model(self.test_iters)
        data_path = "C:/Users/25351/Downloads/new_data/input"  # 测试图像文件夹的路径
        save_path = os.path.join("C:/Users/25351/Downloads/new_data/", "drunet")
        with torch.no_grad():
            for f in os.listdir(data_path):
                if f.endswith(".tif"):
                    img = cv2.imread(os.path.join(data_path, f), cv2.IMREAD_UNCHANGED)
                    img = np.asarray(img)
                    # 下采样到512x512
                    resized_image = cv2.resize(img, (512, 512))

                    # 归一化操作
                    img_source = cv2.normalize(
                        resized_image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F
                    )
                    # 将img_source转换为形状为(1, 1, H, W)的浮点型张量
                    img = (
                        torch.from_numpy(img_source)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .float()
                        .to(self.device)
                    )
                    _, __, pred = self.U_Net(img)
                    pred = self.trunc(
                        self.denormalize_(pred.view(512, 512).cpu().detach())
                    )
                    pred = pred.numpy()
                    # 归一化操作
                    pred = cv2.normalize(pred, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

                    # 还原回0-255范围再拼接，防止图像显示出问题（背景白斑）
                    img_source = cv2.normalize(
                        resized_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                    )
                    pred = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                    # 拼接图像并保存
                    # img = np.concatenate((img_source, pred), axis=1)
                    cv2.imwrite(os.path.join(save_path, f), pred)

    # 创建一个函数来检查路径是否存在，如果不存在则创建它
    def create_path_if_not_exists(self, p):
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"路径已创建：{p}")
        else:
            print(f"路径已存在：{p}")
