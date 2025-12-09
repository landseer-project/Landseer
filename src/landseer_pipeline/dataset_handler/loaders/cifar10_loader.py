
import numpy as np
import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms

def load_dataset(output_dir: str, download_dir: str):
    
    output_dir = os.path.abspath(output_dir)
    download_dir = os.path.abspath(download_dir)
    
    # Define comprehensive training transforms (same as DP training)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # Random crop with padding
        transforms.RandomHorizontalFlip(),         # Random horizontal flip
        transforms.RandomErasing(p=0.2),           # Random erasing (cutout)
        transforms.ToTensor(),                     # Convert to tensor
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # Normalization
    ])
    
    # Define test transforms (no augmentation, only normalization)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    
    # 1. Load datasets with transforms
    dataset = torchvision.datasets.CIFAR10(
        root=download_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
        #transform=transform_train  # Use augmented transforms for training
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=download_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
        #transform=transform_test
    )

    X = np.stack([np.array(img) for img, _ in dataset]) 
    
    Y = np.array([lbl for _, lbl in dataset])

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    np.save(output_dir + '/data.npy', X)  #saved shape is (50000, 3, 32, 32)
    np.save(output_dir + '/labels.npy', Y) #saved shape is (50000,)

    test_images = np.stack([np.array(img) for img, _ in test_dataset])
    test_labels = np.array([lbl for _, lbl in test_dataset])

    np.save(output_dir + '/test_data.npy', test_images) #saved shape is (10000, 3, 32, 32)
    np.save(output_dir + '/test_labels.npy', test_labels) #saved shape is (10000,)