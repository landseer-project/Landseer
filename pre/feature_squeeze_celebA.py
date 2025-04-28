import numpy as np
import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

def feature_squeezing(X, bit_depth=4):
    max_val = 2 ** bit_depth - 1
    return np.round(X * max_val) / max_val

def load_celeba(data_dir, attribute_name='Smiling', target_size=(32, 32)):
    """
    Load CelebA dataset with specified attribute as binary labels
    Args:
        data_dir: Path to CelebA directory containing img_align_celeba and list_attr_celeba.csv
        attribute_name: Name of attribute to use as label ('Smiling')
        target_size: Size to resize images to (32, 32)
    Returns:
        images: numpy array of images [N, 3, 32, 32]
        labels: numpy array of binary labels [N]
    """
    # Paths
    img_dir = os.path.join(data_dir, 'img_align_celeba')
    attr_path = os.path.join(data_dir, 'list_attr_celeba.csv')
    
    # Read attributes file
    df = pd.read_csv(attr_path)
    
    # Verify attribute exists
    if attribute_name not in df.columns:
        raise ValueError(f"Attribute '{attribute_name}' not found. Available attributes: {list(df.columns[1:])}")
    
    # Process images and labels
    images = []
    labels = []
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()  # Converts to [0,1] range and (C, H, W) format
    ])
    
    for idx, row in df.iterrows():
        img_name = row['image_id']
        attr_value = row[attribute_name]
        
        # Load and transform image
        img_path = os.path.join(img_dir, img_name)
        try:
            img = Image.open(img_path)
            img_tensor = transform(img)  # [3, 32, 32]
            images.append(img_tensor.numpy())
            
            # Convert attribute to binary (original is -1/1, we want 0/1)
            labels.append(1 if attr_value == 1 else 0)
        except FileNotFoundError:
            print(f"Warning: Image {img_name} not found, skipping")
            continue
    
    return np.stack(images), np.array(labels)  # [N, 3, 32, 32], [N]

def split_dataset(images, labels, train_ratio=0.8, random_seed=42):
    """
    Split dataset into training and test sets
    """
    num_samples = len(images)
    num_train = int(num_samples * train_ratio)
    
    np.random.seed(random_seed)
    indices = np.random.permutation(num_samples)
    train_idx, test_idx = indices[:num_train], indices[num_train:]
    
    return images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process CelebA dataset with feature squeezing")
    parser.add_argument("--data-dir", default="./data/celeba",
                      help="Path to CelebA directory with img_align_celeba and list_attr_celeba.csv")
    parser.add_argument("--attribute", default="Smiling",
                      help="Attribute to use as label (default: Smiling)")
    parser.add_argument("--output", default="/output",
                      help="Output directory for .npy files")
    parser.add_argument("--bit-depth", type=int, default=4,
                      help="Bit depth for feature squeezing (default: 4)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                      help="Train/test split ratio (default: 0.8)")
    
    args = parser.parse_args()

    # 1. Load dataset
    print(f"Loading CelebA with attribute '{args.attribute}'...")
    X, Y = load_celeba(args.data_dir, args.attribute, (32, 32)) 
    
    # 2. Split dataset
    print("Splitting dataset...")
    X_train, Y_train, X_test, Y_test = split_dataset(X, Y, args.train_ratio)
    
    # 3. Apply feature squeezing
    print("Applying feature squeezing...")
    X_train_squeezed = feature_squeezing(X_train, args.bit_depth)
    X_test_squeezed = feature_squeezing(X_test, args.bit_depth)
    
    # 4. Save as .npy files
    print("Saving files...")
    os.makedirs(args.output, exist_ok=True)
    
    # Save as .npy files
    np.save(args.output + '/data.npy', X_train_squeezed)
    print(f"saved train data to {args.output}/data.npy")
    np.save(args.output + '/labels.npy', Y_train)
    print(f"saved train labels to {args.output}/labels.npy")
    np.save(args.output + '/test_data.npy', X_test_squeezed)
    np.save(args.output + '/test_labels.npy', Y_test)

    
    print(f"Results saved to {args.output}:")
    print(f"- train_data.npy: {X_train_squeezed.shape}")
    print(f"- train_labels.npy: {Y_train.shape}")
    print(f"- test_data.npy: {X_test_squeezed.shape}")
    print(f"- test_labels.npy: {Y_test.shape}")
    print("Done!")


# import numpy as np
# import argparse
# import os
# import torch
# import torchvision
# import torchvision.transforms as transforms

# def feature_squeezing(X, bit_depth=4):
#     max_val = 2 ** bit_depth - 1
#     return np.round(X * max_val) / max_val

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Dynamically download + Feature Squeeze CelebA")
#     parser.add_argument("--output", default="/app/output", help="Output directory for .npy files")
#     parser.add_argument("--bit-depth", type=int, default=4, help="Bit depth for squeezing")
#     parser.add_argument("--download-dir", default="/app/data", help="Directory to download CelebA")
#     parser.add_argument("--image-size", type=int, default=32, help="Size to resize images to")
#     parser.add_argument("--attribute", default="Smiling", help="Which attribute to use as labels")
#     args = parser.parse_args()

#     # Define transforms with resizing and normalization
#     transform = transforms.Compose([
#         transforms.Resize(args.image_size),
#         transforms.CenterCrop(args.image_size),
#         transforms.ToTensor(),
#     ])

#     # 1. Download CelebA (automatically splits into train/test)
#     train_dataset = torchvision.datasets.CelebA(
#         root=args.download_dir,
#         split='train',
#         target_type='attr',
#         transform=transform,
#         download=True
#     )

#     # val_dataset = torchvision.datasets.CelebA(
#     #     root=args.download_dir,
#     #     split='valid',
#     #     target_type='attr',
#     #     transform=transform,
#     #     download=True
#     # )

#     test_dataset = torchvision.datasets.CelebA(
#         root=args.download_dir,
#         split='test',
#         target_type='attr',
#         transform=transform,
#         download=True
#     )

#     # Function to process a dataset split
#     def process_dataset(dataset, attribute):
#         # Get index of the requested attribute
#         attr_names = dataset.attr_names
#         attr_idx = attr_names.index(attribute)
        
#         # Stack images and extract selected attribute
#         X = np.stack([np.array(img) for img, _ in dataset])
#         Y = np.array([attrs[attr_idx] for _, attrs in dataset])
#         return X, Y

#     # 2. Process each split
#     X_train, Y_train = process_dataset(train_dataset, args.attribute)
#    # X_val, Y_val = process_dataset(val_dataset, args.attribute)
#     X_test, Y_test = process_dataset(test_dataset, args.attribute)

#     # 3. Apply feature squeezing
#     X_train_squeezed = feature_squeezing(X_train, args.bit_depth)
#    # X_val_squeezed = feature_squeezing(X_val, args.bit_depth)
#     X_test_squeezed = feature_squeezing(X_test, args.bit_depth)

#     # 4. Save all splits
#     os.makedirs(args.output, exist_ok=True)
    
#     np.save(os.path.join(args.output, '/data.npy'), X_train_squeezed)
#     np.save(os.path.join(args.output, '/labels.npy'), Y_train)
    
#     # np.save(os.path.join(args.output, 'data.npy'), X_val_squeezed)
#     # np.save(os.path.join(args.output, 'labels.npy'), Y_val)
    
#     np.save(os.path.join(args.output, '/test_data.npy'), X_test_squeezed)
#     np.save(os.path.join(args.output, '/test_labels.npy'), Y_test)

#     print(f"Saved processed CelebA data to {args.output}")
#     print(f"Attribute used: {args.attribute}")
#     print(f"Train shape: {X_train_squeezed.shape}, Test shape: {X_test_squeezed.shape}")