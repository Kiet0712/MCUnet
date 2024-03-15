import os
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import nibabel as nib
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import json
def pad_or_crop_image(image, seg=None, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    if seg is not None:
        seg = np.pad(seg, padlist)
        return image, seg
    return image


def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right


def get_crop_slice(target_size, dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)


def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image


def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function used for the challenge (seems to work the best).

    Remove outliers voxels first, then min-max scale.

    Warnings
    --------
    This will not do it channel wise!!
    """

    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = normalize(image)
    return image


def zscore_normalise(img: np.ndarray) -> np.ndarray:
    slices = (img != 0)
    img[slices] = (img[slices] - np.mean(img[slices])) / np.std(img[slices])
    return img


def remove_unwanted_background(image, threshold=1e-5):
    """Use to crop zero_value pixel from MRI image.
    """
    dim = len(image.shape)
    non_zero_idx = np.nonzero(image > threshold)
    min_idx = [np.min(idx) for idx in non_zero_idx]
    # +1 because slicing is like range: not inclusive!!
    max_idx = [np.max(idx) + 1 for idx in non_zero_idx]
    bbox = tuple(slice(_min, _max) for _min, _max in zip(min_idx, max_idx))
    return image[bbox]


def random_crop2d(*images, min_perc=0.5, max_perc=1.):
    """Crop randomly but identically all images given.

    Could be used to pass both mask and image at the same time. Anything else will
    throw.

    Warnings
    --------
    Only works for channel first images. (No channel image will not work).
    """
    if len(set(tuple(image.shape) for image in images)) > 1:
        raise ValueError("Image shapes do not match")
    shape = images[0].shape
    new_sizes = [int(dim * random.uniform(min_perc, max_perc)) for dim in shape]
    min_idx = [random.randint(0, ax_size - size) for ax_size, size in zip(shape, new_sizes)]
    max_idx = [min_id + size for min_id, size in zip(min_idx, new_sizes)]
    bbox = list(slice(min_, max(max_, 1)) for min_, max_ in zip(min_idx, max_idx))
    # DO not crop channel axis...
    bbox[0] = slice(0, shape[0])
    # prevent warning
    bbox = tuple(bbox)
    cropped_images = [image[bbox] for image in images]
    if len(cropped_images) == 1:
        return cropped_images[0]
    else:
        return cropped_images


def random_crop3d(*images, min_perc=0.5, max_perc=1.):
    """Crop randomly but identically all images given.

    Could be used to pass both mask and image at the same time. Anything else will
    throw.

    Warnings
    --------
    Only works for channel first images. (No channel image will not work).
    """
    return random_crop2d(min_perc, max_perc, *images)
def nib_load(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError
    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data
def csv_to_list(csv_file_path,mode):
    extracted_info = []
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row if present
        for row in reader:
            if row[1] == mode:
                extracted_info.append(row[0])
    return extracted_info
def datafold_readSWINUNETR(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d['label'][13:28])
        else:
            tr.append(d['label'][13:28])

    return tr, val
def train_transformation(dataset):
    for key in dataset:
        if key!='label':
            dataset[key] = irm_min_max_preprocess(dataset[key])
    patient_img = np.stack([dataset[key] for key in dataset if key not in ['label']]).astype('float32')
    et = dataset['label']==4
    tc = np.logical_or(et,dataset['label']==1)
    wt = np.logical_or(tc,dataset['label']==2)
    patient_label = np.stack([et,tc,wt]).astype('float32')
    z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_img, axis=0) != 0)
    zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
    zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
    patient_img = patient_img[:, zmin:zmax, ymin:ymax, xmin:xmax]
    patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
    patient_img, patient_label = pad_or_crop_image(patient_img, patient_label, target_size=(128, 128, 128))
    return {
        'img':torch.from_numpy(patient_img),
        'label':torch.from_numpy(patient_label)
    }
def pad_last_dims(data, pad_value=0):
   shape = data.shape
   if any(dim % 2 for dim in shape[-3:]):
     pad_width = [(0, 0) for _ in range(len(shape) - 3)] + [(0, 1 if dim % 2 else 0) for dim in shape[-3:]]
     data = np.pad(data, pad_width, mode='constant', constant_values=pad_value)
   return data
def val_transformation(dataset):
    for key in dataset:
        if key!='label':
            dataset[key] = irm_min_max_preprocess(dataset[key])
    patient_img = np.stack([dataset[key] for key in dataset if key not in ['label']]).astype('float32')
    et = dataset['label']==4
    tc = np.logical_or(et,dataset['label']==1)
    wt = np.logical_or(tc,dataset['label']==2)
    patient_label = np.stack([et,tc,wt]).astype('float32')
    z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_img, axis=0) != 0)
    zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
    zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
    patient_img = patient_img[:, zmin:zmax, ymin:ymax, xmin:xmax]
    
    return {
        'img':torch.from_numpy(patient_img),
        'label':torch.from_numpy(patient_label),
        'crop_indices': ((zmin,zmax),(ymin,ymax),(xmin,xmax))
    }
class BRATS2021(Dataset):
    def __init__(self, file_list,mode,root_dir='/kaggle/working/brain_images'):
        self.root_dir = root_dir
        self.transform = None
        if mode == 'train':
            self.transform = train_transformation
        else:
            self.transform = val_transformation
        self.file_list = file_list
        self.mode = mode
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.file_list[idx]) + '/' + self.file_list[idx]
        name = self.file_list[idx] 
        seg_path = path + '_seg.nii.gz'
        flair_path = path + '_flair.nii.gz'
        t1ce_path = path + '_t1ce.nii.gz'
        t1_path = path + '_t1.nii.gz'
        t2_path = path + '_t2.nii.gz'
        seg = np.array(nib_load(seg_path),dtype='uint8')
        flair = np.array(nib_load(flair_path),dtype='float32')
        t1ce = np.array(nib_load(t1ce_path),dtype='float32')
        t1 = np.array(nib_load(t1_path),dtype='float32')
        t2 = np.array(nib_load(t2_path),dtype='float32')
        item = self.transform({'flair':flair, 't1':t1, 't1ce':t1ce, 't2':t2, 'label':seg})
        item['idx']=idx
        item['name']=name
        return item

dataset_zoo = {
    'BRATS2021':BRATS2021
}
def make_dataloader(cfg,mode,name):
    dataset = dataset_zoo[cfg.DATASET.NAME]
    data = dataset(
        file_list=name,
        
        mode=mode,
        root_dir=cfg.DATASET.ROOT_DIR
    )
    return DataLoader(
        dataset=data,
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=cfg.DATASET.PIN_MEMORY
    )