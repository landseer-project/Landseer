import numpy as np
import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dynamically download + Feature Squeeze CIFAR-10")
    parser.add_argument("--output", default="/output", help="Output .npy file path for images")
    parser.add_argument("--bit-depth", type=int, default=4, help="Bit depth for squeezing")
    parser.add_argument("--download-dir", default="/data", help="Directory to download CIFAR-10")
    args = parser.parse_args()

    # 1. Dynamically download CIFAR-10
    dataset = torchvision.datasets.CIFAR10(
        root=args.download_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    X = np.stack([np.array(img) for img, _ in dataset])
    
    Y = np.array([lbl for _, lbl in dataset])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output + '/data.npy', X)
    np.save(args.output + '/labels.npy', Y)

    test_dataset = torchvision.datasets.CIFAR10(
        root=args.download_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    test_images = np.stack([np.array(img) for img, _ in test_dataset])
    test_labels = np.array([lbl for _, lbl in test_dataset])

    np.save(args.output + '/test_data.npy', test_images)
    np.save(args.output + '/test_labels.npy', test_labels)
