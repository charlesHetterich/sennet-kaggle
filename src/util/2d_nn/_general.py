import os
from os import path
import torch
from torch.utils.data import Dataset
import torchvision
import pickle
from typing import Optional
import pandas as pd
import numpy as np
import tifffile as tiff
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split

class ImageGrabber:
    """
    # To Call:
    base_path = 'C:/Users/ryans/Downloads/blood-vessel-segmentation/train/'
    datasets = ['kidney_1_dense/', 'kidney_1_voi/', 'kidney_2/', 'kidney_3_dense/', 'kidney_3_sparse/']
    image_loader = ImageGrabber(base_path, datasets)
    image_loader.load()
    image_files = image_loader.image_files
    label_files = image_loader.label_files"""

    def __init__(self, base_path, datasets):
        self.base_path = base_path
        self.datasets = datasets
        self.image_files = []
        self.label_files = []

    def load(self):
        """Load the image and label files from the datasets."""
        for dataset in self.datasets:
            images_path = os.path.join(self.base_path, dataset, 'images/')
            labels_path = os.path.join(self.base_path, dataset, 'labels/')
            try:
                self.image_files.extend([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.tif')])
                self.label_files.extend([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.tif')])
            except FileNotFoundError:
                print(f"Skipping {dataset} because it does not have an images subfolder.")

        self.image_files = sorted(self.image_files)
        self.label_files = sorted(self.label_files)


def preprocess_image(path):
    image = tiff.imread(path).astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val)
    image = image * 255
    image = image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.resize(image, (512, 512))
    image = image[..., None]
    #image = np.tile(image, [1, 1, 3])
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image)
    return image


def preprocess_mask(path):
    mask = tiff.imread(path).astype(np.float32)
    mask = mask.astype('float32')
    mask = cv2.resize(mask, (512, 512))
    mask/=255.0
    mask = torch.tensor(mask)
    return mask


class CustomDataset(Dataset):
    def __init__(self, image_files, mask_files, input_size=(512, 512), augmentation_transforms=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.input_size = input_size
        #self.augmentation_transforms = augmentation_transforms
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = preprocess_image(image_path)
        mask = preprocess_mask(mask_path)
     
        return image, mask




'''class image_preprocess:
    def __init__(self, image_list):
        # image_list is a list of image file names
        self.image_list = image_list
    
    def normalize(self, image):
        # normalize an image using the given function
        image = tiff.imread(image).astype(np.float32)
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val)
        image = image * 255
        image = image.astype(np.uint8)
        return image
    
    def clahe(self, image, clip_limit=40.0, tile_size=(8, 8)):
        # apply contrast limited adaptive histogram equalization to an image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        image = clahe.apply(image)
        image = np.tile(image[..., None], [1, 1, 3])
        image = np.transpose(image, (2, 0, 1)) 
        return image'''


'''### TODO: Figure out what form (vector/matrix) this requires ###
    #def zca_whitening(image):
        # Flatten the image into a vector
        image = image.reshape(-1)
        # Standardize the image
        scaler = StandardScaler()
        image = scaler.fit_transform(image)
        # Compute the covariance matrix
        cov = np.cov(image, rowvar=False)
        # Compute the eigenvalues and eigenvectors
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        # Create the transformation matrix
        zca_matrix = eig_vecs @ np.diag(1.0 / np.sqrt(eig_vals + 1e-5)) @ eig_vecs.T
        # Apply the transformation to the image
        image = zca_matrix @ image
        # Reshape the image back to its original shape
        image = image.reshape(512, 512) # Change this to match your image size
        return image

    def apply(self):
        # apply the normalize, clahe, and preprocess_mask functions to each image and label file in the lists
        # and return two separate lists of processed images and labels
        image_list = [] 
        for image in self.image_list:
            image = self.normalize(image)
            image = self.clahe(image)
            image = torch.tensor(image)
            image_list.append(image)
        return image_list


class mask_preprocess:
    def __init__(self, label_files):
        # image_list is a list of image file names
        self.label_files = label_files

    def preprocess_mask(self, image):
        image = tiff.imread(image).astype(np.float32)
        image = image.astype('float32')
        image/=255.0
        img_ten = torch.tensor(image)
        return img_ten
    
    def apply(self):
        label_list = []
        for label in self.label_files:
            label = self.preprocess_mask(label)
            label_list.append(label)
        return label_list


class augmenter:
    def __init__(self, image_list, label_list):
        # image_list and label_list are lists of image and mask file names
        self.image_list = image_list
        self.label_list = label_list
    
    def augment_image(self, image, mask):
        # augment an image and a mask using the given function
        image_np = image.permute(1,2,0).numpy()
        mask_np = mask.numpy()

        transform = A.Compose([
            A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            A.RandomCrop(height=256, width=256,always_apply=True),
            A.RandomBrightness(p=1),
            A.OneOf(
                [
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),
        ])

        augmented = transform(image=image_np, mask=mask_np)
        augmented_image, augmented_mask = augmented['image'], augmented['mask']

        augmented_image = torch.tensor(augmented_image, dtype=torch.float32).permute(2,0,1)
        augmented_mask = torch.tensor(augmented_mask, dtype=torch.float32)

        return augmented_image, augmented_mask
    
    def apply(self):
        augmented_image_list = [] # list for augmented images
        augmented_label_list = [] # list for augmented masks
        for image, label in zip(self.image_list, self.label_list):
            augmented_image, augmented_label = self.augment_image(image, label)
            augmented_image_list.append(augmented_image)
            augmented_label_list.append(augmented_label)
        return augmented_image_list, augmented_label_list'''
