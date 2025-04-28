import numpy as np
import pandas as pd
import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

def load_celeba_manually(data_dir, image_size=32, test_size=0.2, random_state=42):
    """
    Load CelebA dataset manually with train/test split
    Args:
        data_dir: Directory containing CelebA files
        image_size: Size to resize images to (default: 32)
        test_size: Fraction of data for test set (default: 0.2)
        random_state: Random seed for reproducibility
    Returns:
        X_train, Y_train, X_test, Y_test
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    # Paths to manual download files
    img_dir = os.path.join(data_dir, 'img_align_celeba')
    attr_path = os.path.join(data_dir, 'list_attr_celeba.csv')

    # Load attributes
    try:
        # Load CSV version (comma-separated)
        df = pd.read_csv(attr_path)  # Assumes column 'Smiling' exists
    except Exception as e:
        raise ValueError(f"Failed to load attributes: {str(e)}")

    # Get filenames and attributes
    filenames = df['image_id'].astype(str).values  # Prevents int64 path errors
    attributes = df.values

    # Find index of 'Smiling' attribute (convert from -1/1 to 0/1)
    attr_names = list(df.columns)
    try:
        smile_idx = attr_names.index('Smiling')
    except ValueError:
        raise ValueError("'Smiling' attribute not found in dataset")

    # Convert labels from -1/1 to 0/1 (where 1 = smiling)
    labels = (attributes[:, smile_idx] == 1).astype(np.int32)

    # Split into train and test
    train_files, test_files, train_labels, test_labels = train_test_split(
        filenames, labels, test_size=test_size, random_state=random_state
    )

    # Function to load images with error handling
    def load_images(file_list, label_list, transform):
        images = []
        valid_labels = []
        
        for filename, label in zip(file_list, label_list):
            img_path = os.path.join(img_dir, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    if transform:
                        img = transform(img)
                    images.append(img.numpy())
                    valid_labels.append(label)
            except Exception as e:
                print(f"Skipping {filename}: {str(e)}")
                continue
        
        return np.stack(images), np.array(valid_labels)

    # Load images
    X_train, Y_train = load_images(train_files, train_labels, transform)
    X_test, Y_test = load_images(test_files, test_labels, transform)

    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process manually downloaded CelebA dataset with 'Smiling' as target")
    parser.add_argument("--output", default="/output", help="Output directory for .npy files")
    parser.add_argument("--data_dir", default="/app/data/celeba", help="Directory containing manually downloaded CelebA files")
    parser.add_argument("--image_size", type=int, default=32, help="Size to resize images to")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data to use as test set")
    args = parser.parse_args()

    # 1. Load and process CelebA data
    print("Loading and processing manually downloaded CelebA dataset...")
    try:
        X_train, Y_train, X_test, Y_test = load_celeba_manually(
            data_dir=args.data_dir,
            image_size=args.image_size,
            test_size=args.test_size
        )
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        exit(1)

    # 2. Save all splits
    os.makedirs(args.output, exist_ok=True)
    
    # Save training data
    np.save(args.output + '/data.npy', X_train)
    print(f"train data saved to {args.output + '/data.npy'}")
    np.save(args.output + '/labels.npy', Y_train)
    print(f"train labels saved to {args.output + '/labels.npy'}")
    
    # Save test data
    np.save(args.output + '/test_data.npy', X_test)
    np.save(args.output + '/test_labels.npy', Y_test)


    print(f"Successfully processed and saved CelebA data to {args.output}")
    print(f"Dataset statistics:")
    print(f"  Training set: {len(X_train)} images, {Y_train.sum()} smiling ({Y_train.mean():.2%})")
    print(f"  Test set:     {len(X_test)} images, {Y_test.sum()} smiling ({Y_test.mean():.2%})")