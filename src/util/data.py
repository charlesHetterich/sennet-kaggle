import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from typing import Optional
import tifffile as tiff
import numpy as np
import cv2

def process_mask(slice):
    mask = slice.astype(bool)
    return mask

def process_scan(_scan: np.ndarray):
    scan = _scan.astype(np.float32)
    smin, smax = np.min(scan), np.max(scan)
    scan = (255 * (scan - smin) / (smax - smin)).astype(np.uint8)
    scan = 255 - scan
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    return clahe.apply(scan)

class SenNet(Dataset):
    def _load_slice(self, pth, preprocess_fn):
        return torch.tensor(
            preprocess_fn(tiff.imread(pth))
        )

    def _load_scan(self, scan_pth, preprocess_fn):
        src_pth = self.folder + scan_pth 
        cache_fn = self.folder + '/cache' + scan_pth + '.pt'
        if os.path.exists(cache_fn):
            print(f'Loading {scan_pth} from cache')
            return torch.load(cache_fn)
        
        print(f'Loading & processing {scan_pth} from .tif source...')
        scan = torch.stack([
            self._load_slice(os.path.join(src_pth, f), preprocess_fn)
            for f in os.listdir(src_pth)
            if f.endswith('.tif')
        ]).unsqueeze(0)

        print(f'Saving {scan_pth} to cache...')
        os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
        torch.save(scan, cache_fn)
        return scan

    def __init__(
            self,
            patch_size: tuple[int, int, int] = (16, 16, 16),
            guarantee_vessel: float = 0.9,
            data_dir = "/root/data",
            samples = ["/train/kidney_1_dense"],
            data: Optional[tuple[list[torch.Tensor], list[torch.Tensor]]] = None,
            perm_slice_axis: bool = True
        ):
        self.patch_size = torch.tensor(patch_size)
        self.guarantee_vessel = guarantee_vessel
        self.folder = data_dir
        self.train_samples = samples
        self.perm_slice_axis = perm_slice_axis

        if data is not None:
            self.scans, self.labels = data
        else:
            self.scans, self.labels = [
                self._load_scan(sample + '/images', process_scan)
                for sample in self.train_samples
            ], [
                self._load_scan(sample + '/labels', process_mask)
                for sample in self.train_samples
            ]
        self.num_patches = sum([
            (scan.shape[1] * scan.shape[2] * scan.shape[3]) // (patch_size[0] * patch_size[1] * patch_size[2])
            for scan in self.scans
        ])


    def __len__(self):
        return self.num_patches

    def __getitem__(self, _):
        scan_idx = np.random.randint(len(self.scans))
        found_vessel = False if np.random.rand() < self.guarantee_vessel else True

        perm = torch.arange(0, len(self.patch_size))
        if self.perm_slice_axis:
            perm = torch.randperm(len(self.patch_size))

        patch_size = self.patch_size[perm]
        x, y, z = None, None, None
        while not found_vessel or x is None:
            x = np.random.randint(0, self.scans[scan_idx].shape[1] - patch_size[0])
            y = np.random.randint(0, self.scans[scan_idx].shape[2] - patch_size[1])
            z = np.random.randint(0, self.scans[scan_idx].shape[3] - patch_size[2])
            label = self.labels[scan_idx][
                :,
                x:x+patch_size[0],
                y:y+patch_size[1],
                z:z+patch_size[2],
            ]
            if not found_vessel:
                found_vessel = label.sum() > 0

        return self.scans[scan_idx][
            :,
            x:x+patch_size[0],
            y:y+patch_size[1],
            z:z+patch_size[2],
        ].permute(0, *torch.argsort(perm)+1), \
        label.permute(0, *torch.argsort(perm)+1), \
        torch.tensor([x, y, z])

class SweepCube(Dataset):
    def __init__(
            self, 
            data: torch.Tensor, 
            patch_size: tuple[int, int, int],
            stride: Optional[tuple[int, int, int]] = None
        ):
        assert data.ndim == 4, "Data must be 4-dimensional"

        self.data = data
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size

        self.patches = (
            (self.data.shape[1] - self.patch_size[0]) // self.stride[0] + 1,
            (self.data.shape[2] - self.patch_size[1]) // self.stride[1] + 1,
            (self.data.shape[3] - self.patch_size[2]) // self.stride[2] + 1,
        )

    def __len__(self):
        return (self.patches[0] * self.patches[1] * self.patches[2])

    def __getitem__(self, i):
        x, y, z = np.unravel_index(i, self.patches)
        x, y, z = x * self.stride[0], y * self.stride[1], z * self.stride[2]

        return self.data[
            :,
            x:x+self.patch_size[0],
            y:y+self.patch_size[1],
            z:z+self.patch_size[2],
        ], torch.tensor([x, y, z])

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def kidney_from_cache(kidney_id: str, aspect: str, cache_dir: str = '/root/data/cache'):
    assert aspect in ['images', 'labels'], f'aspect must be in ["images", "labels"]'
    return torch.load(f'{cache_dir}/train/{kidney_id}/{aspect}.pt')

KIDNEY_1_SHAPE = (1303, 912)
KIDNEY_2_SHAPE = (1041, 1511)
KIDNEY_3_SHAPE = (1706, 1510)
def label_from_rle(csv_pth: str, shape: tuple[int, int], kidney_id: str) -> torch.Tensor:
    df = pd.read_csv(csv_pth)
    if kidney_id is not None:
        idx = df['id'].str.startswith(kidney_id)
        df = df[idx]

    return torch.stack(
        [torch.tensor(rle_decode(row.rle, shape))
         for row in df.itertuples()]
    ).unsqueeze(0).bool()

def kidney_1_fixed() -> torch.Tensor:
    this_dir = os.path.dirname(os.path.realpath(__file__))
    return label_from_rle(f"{this_dir}/_data/fixed_kidney_1_dense_true_labels.csv", KIDNEY_1_SHAPE, 'kidney_1_dense')