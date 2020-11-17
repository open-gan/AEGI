# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 13:24
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : datasets.py
# @Software: PyCharm


import numpy as np
import os
import lmdb
import cv2

from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def load_image(file_path, input_height=128, input_width=None, output_height=128, output_width=None,
              crop_height=None, crop_width=None, is_random_crop=True, is_mirror=True, is_gray=False):
    '''
    读取图像，是否做增强
    '''
    if input_width is None:
        input_width = input_height
    if output_width is None:
        output_width = output_height
    if crop_width is None:
        crop_width = crop_height

    img = Image.open(file_path)
    if is_gray is False and img.mode != 'RGB':
        img = img.convert('RGB')
    if is_gray and img.mode != 'L':
        img = img.convert('L')

    # 随机进行水平翻转
    if is_mirror and np.random.randint(0, 1) == 0:
        img = ImageOps.mirror(img)

    if input_height is not None:
        img = img.resize((input_width, input_height), Image.BICUBIC)

    # 去掉左，上，右，下四个边上的行/列数
    if crop_height is not None:
        [w, h] = img.size
        if is_random_crop:
            #print([w,cropSize])
            cx1 = np.random.randint(0, w - crop_width)
            cx2 = w - crop_width - cx1
            cy1 = np.random.randint(0, h - crop_height)
            cy2 = h - crop_height - cy1
        else:
            cx2 = cx1 = int(round((w-crop_width)/2.))
            cy2 = cy1 = int(round((h-crop_height)/2.))
        img = ImageOps.crop(img, (cx1, cy1, cx2, cy2))

    img = img.resize((output_width, output_height), Image.BICUBIC)
    return img


class MyDataset(Dataset):
    '''
    preprocess dataset
    '''
    def __init__(self, image_list, root_path,
                 input_height=128, input_width=None, output_height=128, output_width=None,
                 crop_height=None, crop_width=None, is_random_crop=False, is_mirror=True, is_gray=False):
        super(MyDataset, self).__init__()

        self.root_path = root_path
        self.image_filenames = image_list
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.is_gray = is_gray

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.crop_height = crop_height
        self.crop_width = crop_width

        self.input_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = load_image(os.path.join(self.root_path, self.image_filenames[index]),
                         self.input_height, self.input_width, self.output_height, self.output_width,
                         self.crop_height, self.crop_width, self.is_random_crop, self.is_mirror, self.is_gray)
        img = self.input_transform(img)
        return img

    def __len__(self):
        return len(self.image_filenames)


class MyDataLoader(object):
    '''
    dataloader with next and iter
    '''
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.unlimit_gen = self.generator(True)

    def generator(self, inf=False):
        while True:
            data_loader = DataLoader(dataset=self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=self.shuffle,
                                     num_workers=12,
                                     pin_memory=True,
                                     drop_last=self.drop_last)
            for images in data_loader:
                yield images
            if not inf:
                break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return len(self.dataset)//self.batch_size


class MyDatasetLSUN(Dataset):
    '''
    preprocess dataset
    '''
    def __init__(self, root_path, input_size=128, output_size=128,
                 is_random_crop=False, is_mirror=True, is_gray=False):
        super(MyDatasetLSUN, self).__init__()

        self.root_path = root_path
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.is_gray = is_gray

        self.env = lmdb.open(root_path, readonly=True)  # 打开数据文件
        self.txn = self.env.begin()  # 生成处理句柄
        self.cur = self.txn.cursor()  # 生成迭代器指针

        self.keys = []
        with open(root_path + '/keys.txt', 'rb') as q:
            for key in q.readlines():
                self.keys.append(key.strip())

        self.input_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([output_size, output_size]),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        key = self.keys[index]

        image_bin = self.txn.get(key)
        image_buf = np.frombuffer(image_bin, dtype=np.uint8)
        img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])

        img = self.input_transform(img)
        return img

    def __len__(self):
        '''trainsize 3033042，testsize 300'''
        return len(self.keys)


class MyDataLoaderLSUN(object):
    '''
    dataloader with next and iter
    '''
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.unlimit_gen = self.generator(True)

    def generator(self, inf=False):
        while True:
            data_loader = DataLoader(dataset=self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=self.shuffle,
                                     num_workers=0,  # 0意味着所有的数据都会被load进主进程
                                     pin_memory=True,
                                     drop_last=self.drop_last)
            for images in data_loader:
                yield images
            if not inf:
                break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return len(self.dataset)//self.batch_size
