import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import random
import numbers
import math
class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))
class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))
class LesionDataset(Dataset):
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
        if random.uniform(0,1)>0.5:
            image,gt = RandomSizedCrop(self.trainsize)(image,gt)
        image = self.img_transform(image)
        image = TF.adjust_brightness(image,random.uniform(-0.1,0.1)+1)
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
    dataset = LesionDataset(image_root, gt_root, trainsize)
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