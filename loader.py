import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_aug import data_augmentation


# from torchvision import transforms

# 自定义Dataset类。注意自定义Dataset需要继承基类Dataset!!!!

# 自定义Dataloader !!! 


class ct_dataset(Dataset):

    # 参数含义值
    # self表示未被创建的实例对象；
    # mode表示运行模式，是训练还是测试
    # load_mode表示内存设置情况，是内存<10的情况，还是内存>10的情况
    # saved_path 表示dicom图像文件转换为np文件后，np文件保存的位置
    # test_patient 表示测试数据的文件夹
    # patch_n 表示随机选取的补丁的个数
    # patch_size 表示补丁图像的大小
    # transform 表示图像增强的方式

    def __init__(self, mode, load_mode, saved_path, test_patient, patch_n=None, patch_size=None, transform=None):

        # 该句为断言函数
        # 表示，若mode为train/test，则正常运行。否则报错“mode is 'train' or 'test'”
        assert mode in ['train', 'test', 'test_img'], "mode not is 'train' or 'test' 'test_img'"

        # load_mode表示内存情况，若RAM大于10GB，则使load_mode=1，会更快。
        assert load_mode in [0, 1], "load_mode is 0 or 1"

        # .npy文件是一种将numpy数组保存到磁盘的格式。训练后的结果常用.npy文件进行保存。
        # glob()方法返回所有匹配的文件路径列表

        # glob()该句：获取saved_path文件夹(保存np图像的文件夹)下所有以_input.npy结尾的文件的路径
        # 并sort()表示按照文件名进行排序(如文件名是带有数字后缀的)，不是随机排序的

        # 读取由dicom->numpy的图像数据文件，直接读取
        # input_path是一个list，其中每个元素为一个后缀为_input.npy的文件路径
        input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))

        # 该句：得到saved_path文件夹(保存np图像的文件夹)下所有以_target.npy结尾的文件的路径
        target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))

        self.load_mode = load_mode  # 内存大小的声明及利用，RAM>10，使用load_mode=1更快

        self.patch_n = patch_n  # 随机选中的补丁的个数

        self.patch_size = patch_size  # 补丁的大小

        # self.transform = transform  # 图像增强方法

        # self.transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ToPILImage()
        # ])

        self.transform = transform

        # 若为训练模式，则加载数据集方式：
        if mode == 'train':
            # 如果是训练模式，则在saved_path文件夹中保存的np图像文件中找属于训练集文件的图像及标签文件
            # 注意：input_path是一个列表
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]

            # 如果选择内存小于10的选项：则按批次进行加载，不全局加载。
            if load_mode == 0:  # batch data load
                self.input_ = input_  # 此时的input_是输入图像文件的路径，通过读取路径获取图像信息
                self.target_ = target_  # 此时的target_是目标图像文件的路径

            # 如果选择内存大于10的选项，则直接加载np图像(即加载文件名)，input是np图像文件
            else:  # all data load () ， 直接加载np图像文件，而非图像文件路径
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]

        # 若为测试模式，则加载数据集方式：
        else:  # mode =='test'
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]

            # 如果内存<10
            if load_mode == 0:
                # 注意：input_ 及 target_ 为输入图像列表和标签图像列表
                self.input_ = input_
                self.target_ = target_

            # 如果内存>10G，则以numpy数组形式加载
            else:
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]

    # 求长度
    def __len__(self):
        # 求这个目标图像列表的长度
        return len(self.target_)

    # 获取单个元素的信息 - 原始图像及对应标签图像
    def __getitem__(self, idx):

        # 通过索引，获取单张图像 / 标签的路径
        input_img, target_img = self.input_[idx], self.target_[idx]

        # 判断内存模式，若<10，则加载该路径，获取np类型的图像数据。
        # 判断内存模式，若>10，np图像文件已被转换。
        if self.load_mode == 0:
            input_img, target_img = np.load(input_img), np.load(target_img)

        # 如果图像增强属性不为空的话，将输入、目标图像进行图像增强处理。
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        # 如果patch_size不为空的话，则获取输入补丁图像、输出补丁图像。
        if self.patch_size:
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)

            # 返回补丁信息（包含输入补丁图像信息、补丁标签图像信息）
            return (input_patches, target_patches)
        else:
            # 如果未定义补丁信息，则直接返回输入图像及标签图像信息
            return (input_img, target_img)


# 获取图像补丁。形参为输入图像列表、标签图像列表、补丁个数、补丁大小。
# 该函数返回np数组类型的输入补丁图像列表、标签补丁图像列表
def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    # assert为断言函数
    # 只有当 输入图像大小 == 输出图像大小，程序才会继续执行
    assert full_input_img.shape == full_target_img.shape

    patch_input_imgs = []  # 声明处理后的输入图像为patch_input_imgs列表
    patch_target_imgs = []  # 声明处理后的标签图像为patch_target_imgs列表

    h, w = full_input_img.shape  # h , w 为图像的高和宽（即大小）
    new_h, new_w = patch_size, patch_size  # new_h, new_w 为补丁高、宽

    # 随机生成多少个补丁。该函数返回np类型的输入图像补丁及目标图像补丁
    for _ in range(patch_n):
        # np.random.randint(a,b) - 表示返回一个[a,b]之间的整型数
        # 设置补丁图像的边界情况。设置开始的点，patch图像直接加上补丁图像高、宽即可。
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # 声明补丁图像 patch_input_img / patch_target_img
        patch_input_img = full_input_img[top:top + new_h, left:left + new_w]
        patch_target_img = full_target_img[top:top + new_h, left:left + new_w]

        # patch_input_imgs是一个列表。直接将处理后的补丁图像加进列表即可。
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)

        """
        # # ## data augment 2/2
        tmp = np.random.randint(1, 8)
        patch_input_img = data_augmentation(patch_input_img, tmp)
        patch_target_img = data_augmentation(patch_target_img, tmp)

        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
        """

        # 最后返回np.array类型的补丁图像。
    return np.array(patch_input_imgs), np.array(patch_target_imgs)


# 该函数返回dataloader，可放入模型中进行训练了 - 即数据满足模型训练的要求了。
def get_loader(mode='train', load_mode=0,
               saved_path=None, test_patient='L506',
               patch_n=None, patch_size=None,
               transform=None, batch_size=32, num_workers=6):
    dataset_ = ct_dataset(mode, load_mode, saved_path, test_patient, patch_n, patch_size, transform)

    # 设置批次大小，是否打乱数据等属性。
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if mode == 'test':
        data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 返回这个dataloader
    return data_loader
