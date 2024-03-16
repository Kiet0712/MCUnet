import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
from PIL import Image, ImageOps


class Polyp(data.Dataset):
    def __init__(self,root=None,transform=None):
        super().__init__()
        self.transform= transform
        self.root = root
        self.file_name = os.listdir(os.path.join(self.root,'images'))
    def __len__(self):
        return len(self.file_name)
    def __getitem__(self, idx):
        file_name = self.file_name[idx]
        image_path = os.path.join(self.root,'images',file_name)
        mask_path = os.path.join(self.root,'masks',file_name)
        
