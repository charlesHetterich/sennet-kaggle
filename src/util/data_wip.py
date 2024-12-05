import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from typing import Optional
import tifffile as tiff
import numpy as np
import cv2

def preprocess_slice(slice):
    image = slice.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val)
    image = image * 255
    image = image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    # image = np.tile(image[..., None], [1, 1, 3])
    # image = np.transpose(image, (2, 0, 1))
    # image = torch.tensor(image)
    return image

def preprocess_mask(slice):
    mask = slice.astype(bool)
    # mask = mask.astype('float32')
    # mask/=255.0
    # mask = torch.tensor(mask)
    return mask

def identity(x):
    return x.astype('float')

def prescan(_scan: np.ndarray):
    scan = _scan.astype(np.float32)
    smin, smax = np.min(scan), np.max(scan)
    scan = (255 * (scan - smin) / (smax - smin)).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    return clahe.apply(scan)

class SenNet(Dataset):
    folder = "/root/data"
    train_samples = [
        "/train/kidney_1_dense",
        # "/train/kidney_1_voi",
    ]

    def _load_slice(self, pth, preprocess_fn):
        return torch.tensor(
            preprocess_fn(tiff.imread(pth))
        )

    def _load_scan(self, scan_pth, preprocess_fn):
        scan = torch.stack([
            self._load_slice(os.path.join(scan_pth, f), preprocess_fn)
            for f in os.listdir(scan_pth)
            if f.endswith('.tif')
        ])
        # return scan
        return F.pad(
            scan,
            (self.chunk_size // 2, self.chunk_size // 2, \
             self.chunk_size // 2, self.chunk_size // 2, \
             self.chunk_size // 2, self.chunk_size // 2), 
        ).unsqueeze(0)

    def __init__(self, chunk_size: int = 96):
        self.chunk_size = chunk_size

        self.scans = []
        self.labels = []
        self.num_chunks = 0
        for sample in self.train_samples:
            sample_pth = self.folder +  sample + '/images'
            label_pth = self.folder +  sample + '/labels'
            self.scans.append(self._load_scan(sample_pth, prescan))
            self.labels.append(self._load_scan(label_pth, preprocess_mask))
            self.num_chunks += self.scans[-1].numel() // (chunk_size ** 3)

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        scan_idx = np.random.randint(len(self.scans))
        x = np.random.randint(0, self.scans[scan_idx].shape[1] - self.chunk_size)
        y = np.random.randint(0, self.scans[scan_idx].shape[2] - self.chunk_size)
        z = np.random.randint(0, self.scans[scan_idx].shape[3] - self.chunk_size)
        print(x, y, z)
        return self.scans[scan_idx][
            :,
            x:x+self.chunk_size,
            y:y+self.chunk_size,
            z:z+self.chunk_size,
        ], self.labels[scan_idx][
            :,
            x:x+self.chunk_size,
            y:y+self.chunk_size,
            z:z+self.chunk_size,
        ]


