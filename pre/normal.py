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
    parser.add_argument("--data-dir", default="/app/data/celeba", help="Directory containing manually downloaded CelebA files")
    parser.add_argument("--image-size", type=int, default=32, help="Size to resize images to")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use as test set")
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



# import numpy as np
# import pandas as pd
# import argparse
# import os
# import torch
# from torchvision import transforms
# from PIL import Image
# import pandas as pdFor 
# from sklearn.model_selection import train_test_split

# def load_celeba_manually(data_dir, image_size=64, test_size=0.2, random_state=42):
#     """
#     Load CelebA dataset manually with train/test split
#     """
#     # Define transforms
#     transform = transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.CenterCrop(image_size),
#         transforms.ToTensor(),
#     ])

#     # Paths to manual download files
#     img_dir = os.path.join(data_dir, 'img_align_celeba')
#     attr_path = os.path.join(data_dir, 'list_attr_celeba.csv')

#     # Load attributes
#     df = pd.read_csv(attr_path)
#     filenames = df['image_id'].values
#     attributes = df.drop('image_id', axis=1).values

#     # Split into train and test
#     train_files, test_files, train_attrs, test_attrs = train_test_split(
#         filenames, attributes, test_size=test_size, random_state=random_state
#     )

#     # Function to load images
#     def load_images(file_list, attr_list, transform):
#         images = []
#         labels = []
#         attr_idx = list(df.columns[1:]).index('Smiling')  # Index of 'Smile' attribute
        
#         for i, filename in enumerate(file_list):
#             img_path = os.path.join(img_dir, filename)
#             img = Image.open(img_path)
#             if transform:
#                 img = transform(img)
#             images.append(np.array(img))
#             labels.append(attr_list[i][attr_idx])
        
#         return np.stack(images), np.array(labels)

#     # Load images
#     X_train, Y_train = load_images(train_files, train_attrs, transform)
#     X_test, Y_test = load_images(test_files, test_attrs, transform)

#     return X_train, Y_train, X_test, Y_test

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Process manually downloaded CelebA dataset with 'Attractive' as target")
#     parser.add_argument("--output", default="./output", help="Output directory for .npy files")
#     parser.add_argument("--data-dir", default="./data/celeba", help="Directory containing manually downloaded CelebA files")
#     parser.add_argument("--csv-dir", default="./data/celeba", help="Directory containing manually downloaded csv files")
#     parser.add_argument("--image-size", type=int, default=32, help="Size to resize images to")
#     parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use as test set")
#     args = parser.parse_args()

#     # 1. Load and process CelebA data
#     print("Loading and processing manually downloaded CelebA dataset...")
#     X_train, Y_train, X_test, Y_test = load_celeba_manually(
#         data_dir=args.data_dir,
#         image_size=args.image_size,
#         test_size=args.test_size
#     )

#     # 2. Save all splits
#     os.makedirs(args.output, exist_ok=True)
    
#     # Save training data
#     np.save(os.path.join(args.output, 'data.npy'), X_train)
#     np.save(os.path.join(args.output, 'labels.npy'), Y_train)
    
#     # Save test data
#     np.save(os.path.join(args.output, 'test_data.npy'), X_test)
#     np.save(os.path.join(args.output, 'test_labels.npy'), Y_test)

#     print(f"Successfully processed and saved CelebA data to {args.output}")
#     print(f"Dataset statistics:")
#     print(f"  Training set: {X_train.shape[0]} images, {Y_train.sum()} attractive ({Y_train.sum()/len(Y_train):.2%})")
#     print(f"  Test set:     {X_test.shape[0]} images, {Y_test.sum()} attractive ({Y_test.sum()/len(Y_test):.2%})")



# # import numpy as np
# # import argparse
# # import os
# # import torch
# # import torchvision
# # import torchvision.transforms as transforms

# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser(description="Download and process CelebA dataset with 'Attractive' as target")
# #     parser.add_argument("--output", default="./output", help="Output directory for .npy files")
# #     parser.add_argument("--bit-depth", type=int, default=4, help="Bit depth for squeezing (unused in this version)")
# #     parser.add_argument("--download-dir", default="./data", help="Directory to download CelebA")
# #     parser.add_argument("--image-size", type=int, default=64, help="Size to resize images to")
# #     args = parser.parse_args()

# #     # Define transforms with resizing and conversion to tensor
# #     transform = transforms.Compose([
# #         transforms.Resize(args.image_size),
# #         transforms.CenterCrop(args.image_size),
# #         transforms.ToTensor(),
# #     ])

# #     # Function to process a dataset split
# #     def process_dataset(dataset, attribute="Attractive"):
# #         # Get index of the requested attribute
# #         attr_names = dataset.attr_names
# #         attr_idx = attr_names.index(attribute)
        
# #         # Stack images and extract selected attribute
# #         X = np.stack([np.array(img) for img, _ in dataset])
# #         Y = np.array([attrs[attr_idx] for _, attrs in dataset])
# #         return X, Y

# #     # 1. Download and process CelebA splits
# #     print("Downloading and processing CelebA dataset...")
    
# #     # Training set
# #     train_dataset = torchvision.datasets.CelebA(
# #         root=args.download_dir,
# #         split='train',
# #         target_type='attr',
# #         transform=transform,
# #         download=True
# #     )
# #     X_train, Y_train = process_dataset(train_dataset)
    
# #     # Validation set
# #     val_dataset = torchvision.datasets.CelebA(
# #         root=args.download_dir,
# #         split='valid',
# #         target_type='attr',
# #         transform=transform,
# #         download=True
# #     )
# #     X_val, Y_val = process_dataset(val_dataset)
    
# #     # Test set
# #     test_dataset = torchvision.datasets.CelebA(
# #         root=args.download_dir,
# #         split='test',
# #         target_type='attr',
# #         transform=transform,
# #         download=True
# #     )
# #     X_test, Y_test = process_dataset(test_dataset)

# #     # 2. Save all splits
# #     os.makedirs(args.output, exist_ok=True)
    
# #     # Save training data
# #     np.save(os.path.join(args.output, 'data.npy'), X_train)
# #     np.save(os.path.join(args.output, 'labels.npy'), Y_train)
    
# #     # Save validation data
# #     # np.save(os.path.join(args.output, 'val_data.npy'), X_val)
# #     # np.save(os.path.join(args.output, 'val_labels.npy'), Y_val)
    
# #     # Save test data
# #     np.save(os.path.join(args.output, 'test_data.npy'), X_test)
# #     np.save(os.path.join(args.output, 'test_labels.npy'), Y_test)

# #     print(f"Successfully processed and saved CelebA data to {args.output}")
# #     print(f"Dataset statistics:")
# #     print(f"  Training set:   {X_train.shape[0]} images, {Y_train.sum()} attractive ({Y_train.sum()/len(Y_train):.2%})")
# #     print(f"  Validation set: {X_val.shape[0]} images, {Y_val.sum()} attractive ({Y_val.sum()/len(Y_val):.2%})")
# #     print(f"  Test set:       {X_test.shape[0]} images, {Y_test.sum()} attractive ({Y_test.sum()/len(Y_test):.2%})")

# # import numpy as np
# # import argparse
# # import os
# # import torch
# # import torchvision
# # import torchvision.transforms as transforms

# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser(description="Dynamically download + Feature Squeeze CIFAR-10")
# #     parser.add_argument("--output", default="/output", help="Output .npy file path for images")
# #     parser.add_argument("--bit-depth", type=int, default=4, help="Bit depth for squeezing")
# #     parser.add_argument("--download-dir", default="/data", help="Directory to download CIFAR-10")
# #     args = parser.parse_args()

# #     # 1. Dynamically download CIFAR-10
# #     dataset = torchvision.datasets.CIFAR10(
# #         root=args.download_dir,
# #         train=True,
# #         download=True,
# #         transform=transforms.ToTensor()
# #     )

# #     X = np.stack([np.array(img) for img, _ in dataset])
    
# #     Y = np.array([lbl for _, lbl in dataset])

# #     os.makedirs(os.path.dirname(args.output), exist_ok=True)
# #     np.save(args.output + '/data.npy', X)
# #     np.save(args.output + '/labels.npy', Y)

# #     test_dataset = torchvision.datasets.CIFAR10(
# #         root=args.download_dir,
# #         train=False,
# #         download=True,
# #         transform=transforms.ToTensor()
# #     )

# #     test_images = np.stack([np.array(img) for img, _ in test_dataset])
# #     test_labels = np.array([lbl for _, lbl in test_dataset])

# #     np.save(args.output + '/test_data.npy', test_images)
# #     np.save(args.output + '/test_labels.npy', test_labels)
