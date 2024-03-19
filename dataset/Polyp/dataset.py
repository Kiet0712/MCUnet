import os
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import random
import torch
class PolypDataset(Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        super().__init__()

        self.trainsize = trainsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        image = TF.adjust_brightness(image,random.uniform(-0.1,0.1))
        image = TF.adjust_contrast(image,random.uniform(0.9,1.1))
        image = TF.adjust_saturation(image,random.uniform(0.9,1.1))
        image = TF.adjust_hue(image,random.uniform(-0.02,0.02))
        gt = self.gt_transform(gt)
        if random.uniform(0,1)>0.5:
            image = TF.hflip(image)
            gt = TF.hflip(gt)
        if random.uniform(0,1)>0.5:
            image = TF.vflip(image)
            gt = TF.vflip(gt)
        return (image, gt)

    def filter_files(self):
        assert (len(self.images) == len(self.gts))
        images = []
        gts = []
        for (img_path, gt_path) in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if (img.size == gt.size):
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert (img.size == gt.size)
        (w, h) = img.size
        if ((h < self.trainsize) or (w < self.trainsize)):
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return (img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST))
        else:
            return (img, gt)

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, trainsize):
    dataset = PolypDataset(image_root, gt_root, trainsize)
    return dataset


class test_dataset(Dataset):

    def __init__(self, image_root, gt_root, testsize):
        super().__init__()
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        image = self.img_transform(image)
        gt = self.gt_transform(self.binary_loader(self.gts[index]))
        name = self.images[index].split('/')[(- 1)]
        if name.endswith('.jpg'):
            name = (name.split('.jpg')[0] + '.png')
        return (image, gt)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size