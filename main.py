# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 13:23
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : main.py
# @Software: PyCharm


import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import torch.nn.functional as F

from datasets import *
from utils import *
from networks import *

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_arg():
    parser = argparse.ArgumentParser('参数管理')

    parser.add_argument('--dataroot', default="", type=str, help='data path')
    parser.add_argument('--outfile', default='results', type=str, help='output path')
    parser.add_argument('--noise_dim', type=int, default=512, help='latent space')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=5, help='test batch size')
    parser.add_argument('--nEpochs', type=int, default=1000, help='epochs')
    parser.add_argument('--sample_step', type=int, default=1000, help='iteration for saving image')
    parser.add_argument('--save_step', type=int, default=2, help='epoch for saving mdoel')

    parser.add_argument('--channels', default="16, 32, 64, 128, 256, 512, 512, 512", type=str, help='channels for each layer')
    parser.add_argument('--trainsize', type=int, default=29000, help='training sample')
    parser.add_argument('--input_height', type=int, default=1024, help='input image size')
    parser.add_argument('--input_width', type=int, default=None, help='input image size')
    parser.add_argument('--output_height', type=int, default=1024, help='output image size')
    parser.add_argument('--output_width', type=int, default=None, help='output image size')
    parser.add_argument('--crop_height', type=int, default=None, help='crop image size')
    parser.add_argument('--crop_width', type=int, default=None, help='crop image size')

    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for generator')
    parser.add_argument('--e_lr', type=float, default=0.0002, help='learning rate for encoder')
    parser.add_argument('--lr_decay', type=float, default=1., help='decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2')

    parser.add_argument('--m_plus', type=float, default=160.0, help='m in formula')
    parser.add_argument('--weight_neg', type=float, default=0.5, help='alpha * loss_adv， alpha in formula')
    parser.add_argument('--weight_rec', type=float, default=0.0025, help='lambda * ae_loss， lambda in formula')
    parser.add_argument('--weight_kl', type=float, default=1., help='gamma * kl(q||p)_loss，gamma in formula')

    # # parser.add_argument('--mean', type=float, default=[0.485, 0.456, 0.406], help='normalized mean')
    # # parser.add_argument('--std', type=float, default=[0.229, 0.224, 0.225], help='normalized var')

    parser.add_argument('--use_tensorboard', type=bool, default=False)
    parser.add_argument("--pretrained", default="", type=str, help="path for pre-training model")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--num_vae", type=int, default=10, help="epochs for VAE pre-training")
    parser.add_argument("--num_gan", type=int, default=10, help="epochs for GAN pre-training, without ExponentialMovingAverage")

    return parser.parse_known_args()[0]


def main():
    config = parse_arg()
    disp_str = ''
    for attr in sorted(dir(config), key=lambda x: len(x)):
        if not attr.startswith('_'):
            disp_str += ' {} : {}\n'.format(attr, getattr(config, attr))
    print(disp_str)

    try:
        os.makedirs(config.outfile)
        print('mkdir:', config.outfile)
    except OSError:
        pass

    seed = np.random.randint(0, 10000)
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    # --------------build models -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    str_to_list = lambda x: [int(xi) for xi in x.split(',')]
    model = IntroVAE(cdim=3, hdim=config.noise_dim,
                     channels=str_to_list(config.channels), image_size=config.output_height).to(device)
    if config.pretrained:
        load_model(model, config.pretrained)
    print(model)

    g_optimizer = torch.optim.Adam(model.generator.parameters(), config.g_lr, betas=(config.beta1, config.beta2))
    e_optimizer = torch.optim.Adam(model.encoder.parameters(), config.e_lr, betas=(config.beta1, config.beta2))
    ema = EMA(0.999)

    # -----------------load dataset--------------------------
    image_list = [x for x in os.listdir(config.dataroot) if is_image_file(x)]
    train_list = image_list[:config.trainsize]
    test_list = image_list[config.trainsize:]
    assert len(train_list) > 0
    assert len(test_list) >= 0

    train_set = MyDataset(train_list, config.dataroot, input_height=None, crop_height=None,
                          output_height=config.output_height, is_mirror=True)
    test_set = MyDataset(test_list, config.dataroot, input_height=None, crop_height=None,
                         output_height=config.output_height, is_mirror=False)
    train_data_loader = MyDataLoader(train_set, config.batch_size)
    test_data_loader = MyDataLoader(test_set, config.test_batch_size, shuffle=False)

    fix_noise = torch.zeros(config.test_batch_size, config.noise_dim).normal_(0, 1).to(device)

    start_time = time.time()
    cur_iter = 0

    def train_vae(epoch, iteration, real_images, cur_iter, statistic):
        info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration,
                                                                                  len(train_data_loader),
                                                                                  time.time() - start_time)

        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossG_rec_kl]'

        # =========== Update E ================
        real_mu, real_logvar, z, rec_images = model(real_images)

        # update statistic at each iteration
        mu = torch.mean(real_mu, dim=0).detach()
        var = torch.mean(real_logvar.exp(), dim=0).detach()
        new_mu = (statistic['mu'] * iteration + mu) / (iteration + 1)
        new_var = statistic['var'] * iteration / (iteration + 1) + (statistic['mu'] - new_mu) ** 2 * iteration / (
                    iteration + 1) + var / (iteration + 1) + (mu - new_mu) ** 2 / (iteration + 1)
        statistic['mu'] = new_mu
        statistic['var'] = new_var

        loss_rec = model.reconstruction_loss(rec_images, real_images, True)
        loss_kl = model.kl_loss(real_mu, real_logvar).mean()

        loss = loss_rec + loss_kl

        g_optimizer.zero_grad()
        e_optimizer.zero_grad()
        loss.backward()
        e_optimizer.step()
        g_optimizer.step()

        info += 'Rec: {:.4f}, KL: {:.4f}, '.format(loss_rec.item(), loss_kl.item())
        print(info)

        if cur_iter % config.sample_step == 0:
            model.eval()
            real_images = test_data_loader.next()
            real_images = real_images.to(device)

            _, _, _, rec_images = model(real_images)

            eps = [torch.cuda.FloatTensor(statistic['mu'].size()).normal_() for _ in range(config.test_batch_size)]
            eps = torch.stack(eps)
            noise = eps + statistic['mu'] / 10
            fake_images = model.sample(noise)
            fix_fake_images = model.sample(fix_noise)

            save_image(torch.cat([real_images, rec_images, fake_images, fix_fake_images], dim=0).data.cpu(),
                       '{}/image_{}.jpg'.format(config.outfile, cur_iter), nrow=config.test_batch_size // 2)
            model.train()

    def train(epoch, iteration, real_images, cur_iter, statistic):
        info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration,
                                                                                  len(train_data_loader),
                                                                                  time.time() - start_time)

        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossG_rec_kl]'

        # =========== Update E ================
        real_mu, real_logvar, z, rec_images = model(real_images)
        rec_mu, rec_logvar = model.encode(rec_images.detach())

        # update statistic at each iteration
        mu = torch.mean(real_mu, dim=0).detach()
        var = torch.mean(real_logvar.exp(), dim=0).detach()
        new_mu = (statistic['mu'] * iteration + mu) / (iteration + 1)
        new_var = statistic['var'] * iteration / (iteration + 1) + (statistic['mu'] - new_mu) ** 2 * iteration / (iteration + 1) + var / (iteration + 1) + (mu - new_mu) ** 2 / (iteration + 1)
        statistic['mu'] = new_mu
        statistic['var'] = new_var

        loss_rec = model.reconstruction_loss(rec_images, real_images, True)
        lossE_real_kl = model.kl_loss(real_mu, real_logvar).mean()
        _lossE_rec_kl = model.kl_loss(rec_mu, rec_logvar)
        lossE_rec_kl = F.relu(config.m_plus - _lossE_rec_kl).mean()

        loss_margin = lossE_real_kl + lossE_rec_kl * config.weight_neg

        lossE = loss_rec * config.weight_rec + \
                loss_margin * config.weight_kl
        e_optimizer.zero_grad()
        lossE.backward()
        e_optimizer.step()

        # ========= Update G ==================
        real_mu, real_logvar, z, rec_images = model(real_images)
        rec_mu, rec_logvar = model.encode(rec_images)

        loss_rec = model.reconstruction_loss(rec_images, real_images, True)
        lossG_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean() * config.weight_neg

        lossG = loss_rec * config.weight_rec + \
                lossG_rec_kl * config.weight_kl

        g_optimizer.zero_grad()
        lossG.backward()
        g_optimizer.step()

        if epoch >= config.num_vae + config.num_gan:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema.update(name, param.data)

        info += 'Rec: {:.4f}, '.format(loss_rec.item())
        info += 'Kl_E: real {:.4f}, rec {:.4f}, '.format(lossE_real_kl.item(), _lossE_rec_kl.mean().item())
        info += 'Kl_G: rec {:.4f}, '.format(lossG_rec_kl.item())
        print(info)

        if cur_iter % config.sample_step == 0:
            model.eval()
            real_images = test_data_loader.next()
            real_images = real_images.to(device)

            _, _, _, rec_images = model(real_images)

            eps = [torch.cuda.FloatTensor(statistic['mu'].size()).normal_() for _ in range(config.test_batch_size)]
            eps = torch.stack(eps)
            noise = eps + statistic['mu'] / 10
            fake_images = model.sample(noise)
            fix_fake_images = model.sample(fix_noise)

            save_image(torch.cat([real_images, rec_images, fake_images, fix_fake_images], dim=0).data.cpu(),
                       '{}/image_{}.jpg'.format(config.outfile, cur_iter), nrow=config.test_batch_size // 2)
            model.train()

    for epoch in range(config.start_epoch, config.nEpochs + 1):
        # reset statistic to N(0, I)
        statistic = {'mu': torch.zeros(size=(config.noise_dim,)).to(device),
                     'var': torch.ones(size=(config.noise_dim,)).to(device)}

        model.train()
        for iteration, real_images in enumerate(train_data_loader.get_iter()):
            real_images = real_images.to(device)
            # --------------train------------
            if epoch < config.num_vae:
                train_vae(epoch, iteration, real_images, cur_iter, statistic)
            elif epoch == config.num_vae + config.num_gan:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        ema.register(name, param.data)
                train(epoch, iteration, real_images, cur_iter, statistic)
            else:
                train(epoch, iteration, real_images, cur_iter, statistic)

            cur_iter += 1

        # save models
        if epoch % config.save_step == config.save_step - 1:
            save_checkpoint(model, epoch, 0, statistic, '')


if __name__ == "__main__":
    main()
