# -*- coding: utf-8 -*-
# @Time    : *
# @Author  : *
# @Email   : *
# @File    : utils.py
# @Software: PyCharm


import os

import torch
import torch.nn as nn
from torchvision.utils import make_grid


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)


def record_image(writer, image_list, cur_iter, nrow):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=nrow), cur_iter)


def load_model(model, pretrained, statistic=None):
    # model.load_state_dict(torch.load(pretrained))
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if statistic is not None:
        statistic['mu'] = weights['statistic']['mu']
        statistic['var'] = weights['statistic']['var']


def save_checkpoint(model, epoch, iteration, statistic, prefix=""):
    state = {'model': model, 'statistic': statistic}
    model_out_path = "model/" + prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


class EMA:
    '''ExponentialMovingAverage'''
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()


class MMD_loss(nn.Module):
    '''
    Maximum Mean Discrepancy, MMD
    '''
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


# if __name__ == '__main__':
#     import argparse
#
#     def parse_arg():
#         '''
#         建立一个函数来管理超参数
#         :return:
#         '''
#         parser = argparse.ArgumentParser('参数管理')
#
#         # parser.add_argument('--dataroot', default="/data/zhangdan/dataset/celebAHQ/celeba-256/", type=str, help='数据文件路径')
#         parser.add_argument('--dataroot', default="F:/Work/Dataset/celebAHQ/celeba-256/", type=str, help='数据文件路径')
#         parser.add_argument('--trainfiles', default="celeba_hq_attr.list", type=str, help='label的保存路径')
#         parser.add_argument('--outfile', default='results', type=str, help='输出图像保存地址')
#         parser.add_argument('--noise_dim', type=int, default=512, help='噪声潜变量维度')
#         parser.add_argument('--batch_size', type=int, default=32, help='batch size')
#         parser.add_argument('--test_batch_size', type=int, default=16, help='batch size')
#         parser.add_argument('--nEpochs', type=int, default=500, help='epochs')
#         parser.add_argument('--sample_step', type=int, default=1000, help='保存图像的迭代次数')
#         parser.add_argument('--save_step', type=int, default=1, help='保存模型的epoch')
#
#         parser.add_argument('--channels', default="32, 64, 128, 256, 512, 512", type=str, help='模型通道数')
#         parser.add_argument('--trainsize', type=int, default=29000, help='训练样本大小')
#         parser.add_argument('--input_height', type=int, default=256, help='读取图像的尺寸')
#         parser.add_argument('--input_width', type=int, default=None, help='读取图像的尺寸')
#         parser.add_argument('--output_height', type=int, default=256, help='图像大小')
#         parser.add_argument('--output_width', type=int, default=None, help='图像大小')
#         parser.add_argument('--crop_height', type=int, default=None, help='图像剪切尺寸')
#         parser.add_argument('--crop_width', type=int, default=None, help='图像剪切尺寸')
#
#         parser.add_argument('--g_lr', type=float, default=0.0002, help='生成器的learning rate')
#         parser.add_argument('--d_lr', type=float, default=0.0002, help='判别器的learning rate')
#         parser.add_argument('--lr_decay', type=float, default=1., help='decay')
#         parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
#         parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
#         parser.add_argument('--lambda_gp', type=float, default=10.0, help='WGANGP的调节参数')
#
#         parser.add_argument('--m_plus', type=float, default=120.0, help='公式中的m')
#         parser.add_argument('--weight_neg', type=float, default=0.5, help='alpha * loss_adv， 公式中的alpha')
#         parser.add_argument('--weight_rec', type=float, default=0.05, help='beta * ae_loss， 公式中的beta，重构误差权重')
#         parser.add_argument('--weight_kl', type=float, default=1., help='gamma * kl(q||p)_loss，散度项权重')
#
#         # # parser.add_argument('--mean', type=float, default=[0.485, 0.456, 0.406], help='图像归一化的均值')
#         # # parser.add_argument('--std', type=float, default=[0.229, 0.224, 0.225], help='图像归一化的方差')
#         # parser.add_argument('--mean', type=float, default=[0.5, 0.5, 0.5], help='图像归一化的均值')
#         # parser.add_argument('--std', type=float, default=[0.5, 0.5, 0.5], help='图像归一化的方差')
#
#         parser.add_argument('--use_tensorboard', type=bool, default=False)
#         parser.add_argument("--pretrained", default="", type=str, help="预训练模型的路径和文件名")
#         parser.add_argument("--start_epoch", default=0, type=int, help="默认为0")
#         parser.add_argument("--num_vae", type=int, default=0, help="是否用VAE预训练")
#
#         return parser.parse_known_args()[0]
#
#
#     from datasets import *
#     from networks import *
#     import matplotlib.pyplot as plt
#
#
#     config = parse_arg()
#     SAMPLE_SIZE = 500
#     buckets = 50
#     mmd_loss = MMD_loss()
#
#     # --------------build models -------------------------
#     str_to_list = lambda x: [int(xi) for xi in x.split(',')]
#     model = IntroVAE(img_dim=3, noise_dim=config.noise_dim,
#                      channels=str_to_list(config.channels), image_size=config.output_height).cuda()
#     fix_noise = torch.zeros(config.batch_size, config.noise_dim).normal_(0, 1).cuda()
#     fake_image = model.sample(fix_noise)
#
#     # -----------------load dataset--------------------------
#     image_list = [x for x in os.listdir(config.dataroot) if is_image_file(x)]
#     train_list = image_list[:config.trainsize]
#     assert len(train_list) > 0
#
#     train_set = MyDataset(train_list, config.dataroot, input_height=None, crop_height=None,
#                           output_height=config.output_height, is_mirror=True)
#     train_data_loader = MyDataLoader(train_set, config.batch_size)
#
#     data = []
#     for i, t in enumerate(train_data_loader.get_iter()):
#         data.append(t)
#         if i == 1:
#             break
#     train_data_1, train_data_2 = data[0].cuda(), data[1].cuda()
#
#     print(mmd_loss(train_data_1.view(config.batch_size, -1), train_data_2.view(config.batch_size, -1)))
#     print(mmd_loss(train_data_1.view(config.batch_size, -1), fake_image.view(config.batch_size, -1)))
#     print(mmd_loss(train_data_2.view(config.batch_size, -1), fake_image.view(config.batch_size, -1)))
