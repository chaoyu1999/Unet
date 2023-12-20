import os

# import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  ## 1/2 ,multi GPU

import argparse
from torch.backends import cudnn

from loader import get_loader
from solver import Solver


def main(args):
    # 该命令行可以优化运行效率。
    cudnn.benchmark = True

    # 判断项目中是否存在命令行声明的文件。若没有，则创建，并打印提示信息。
    # save_path保存的是生成的训练文件和图像的信息
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)  # 创建新文件夹
        print('Create path : {}'.format(args.save_path))

    # 命令行中写的result_fig是否存在？若存在，则确定图像的保存地址文件夹
    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        # 若不存在，可理解为不存在命令行中声明的结果图像文件夹
        if not os.path.exists(fig_path):
            # 则进行创建该详细目标文件夹
            os.makedirs(fig_path)
            # 最后，打印创建成功等提示信息
            print('Create path : {}'.format(fig_path))

    # 定义dataloader数据加载器。dataset -> dataloader 
    # 其中，get_loader()的参数可以去该函数文件中寻找
    data_loader = get_loader(mode=args.mode,  # 判断是训练模式 or 测试模式
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,  # 设置保存的地址
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode == 'train' else None),
                             patch_size=(args.patch_size if args.mode == 'train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode == 'train' else 1),
                             num_workers=args.num_workers)

    # 命令行语句 & 数据集，进行打包
    solver = Solver(args, data_loader)

    # 判断模式（ 训练模式 / 测试模式 ）
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    elif args.mode == 'test_img':
        solver.test_img()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建解析对象

    # 添加参数 - 项目中所能用到的参数 几乎都在这了！
    # 设置网络模式 - 训练 or 测试
    parser.add_argument('--mode', type=str, default='train')

    # 设置运行模式（0/1），内存<10 设为0，内存>10 设为1。
    parser.add_argument('--load_mode', type=int, default=0)

    # 设置图像数据的路径 - 文件夹名为AAPM-Mayo-CT-Challenge，当前文件的上一层
    # parser.add_argument('--data_path', type=str, default='./AAPM-Mayo-CT-Challenge/')

    # 添加图像转换的保存路径[由dicom -> numpy] - 文件夹名字为npy_img。 
    # parser.add_argument('--saved_path', type=str, default='/home/Desktop/RED-CNN-Unet/
    # npy_img/')

    # 读取数据，先进行归一化的数据
    # parser.add_argument('--saved_path', type=str, default='/home/Desktop/new_norm_data/')

    project_path = 'E:/cy/Unet'
    parser.add_argument('--saved_path', type=str, default=('%s/a_new_npy/' % project_path))

    # 结果保存路径（运行文件 + 生成图像数据）
    parser.add_argument('--save_path', type=str, default=("%s/save_ResUnet/" % project_path))

    # 测试文件夹(test_patient)的名字 设为‘L506’
    # 将转换的np CT图像文件，通过条件遍历(是否以input/target结尾)，一同保存到列表里面
    # 读取这些文件，作为dataloader。
    # 在此过程中，如何区分训练集和测试集？ 判断是否存在测试集文件夹的名称test_patient。
    parser.add_argument('--test_patient', type=str, default='L506')  # 测试文件夹的名称

    # 是否保存图像文件信息
    parser.add_argument('--result_fig', type=bool, default=True)

    # 设置标准化的范围最大最小值
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)

    # 设置未进行标准化的像素色彩范围
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    # 设置图像增强，可根据要求进行设置
    parser.add_argument('--transform', type=bool, default=False)

    # 若用补丁训练，则设置补丁数量
    # if patch training, batch size is (--patch_n * --batch_size)
    parser.add_argument('--patch_n', type=int, default=4)  # 10/0

    # 设置补丁大小
    parser.add_argument('--patch_size', type=int, default=64)  # 128/0

    # 设置批次大小
    parser.add_argument('--batch_size', type=int, default=8)

    # 设置总数据迭代的轮数
    parser.add_argument('--num_epochs', type=int, default=200)

    # 设置打印次数
    parser.add_argument('--print_iters', type=int, default=20)

    # 设置学习率衰减的次数  原来=3000
    parser.add_argument('--decay_iters', type=int, default=10000)

    # 设置  原来=1000
    parser.add_argument('--save_iters', type=int, default=1000)

    # 设置测试时，选用的第几轮数  原来=1000
    parser.add_argument('--test_iters', type=int, default=49750)

    # 设置学习率
    parser.add_argument('--lr', type=float, default=1e-4)

    # 设置设备参数
    parser.add_argument('--device', type=str)  # 可以进行default = [2,3]

    # 设置进程数  原来=7
    parser.add_argument('--num_workers', type=int, default=7)

    # 设置是否为多gpu状态，若是多个，则可以选择使用哪个GPU
    parser.add_argument('--multi_gpu', type=bool, default=False)

    # 解析参数 - 可以取到里面的值
    args = parser.parse_args()

    # 调用运行主函数main()
    main(args)
