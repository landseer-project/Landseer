import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
import torchvision.transforms as transforms

def load_dataset(output_dir, download_dir, attribute='Smiling', test_size=0.2, random_state=42):
    """
    Downloads CelebA dataset from Kaggle, loads and splits it based on a selected attribute,
    and saves it as .npy files.
    
    Args:
        output_dir (str): Where to store .npy outputs.
        attribute (str): Attribute column name to use as label (default: 'Smiling').
        test_size (float): Proportion of dataset to be used as test set.
        random_state (int): Random seed for reproducibility.
    """

    # --- Step 1: Download and unzip dataset ---
    dataset_name = "jessicali9530/celeba-dataset"
    download_path = download_dir
    os.makedirs(download_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()  # Uses ~/.kaggle/kaggle.json

    print("Downloading CelebA dataset...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print("Download completed!")

    # --- Step 2: Load attributes and validate ---
    attr_path = os.path.join(download_path, 'list_attr_celeba.csv')
    if not os.path.exists(attr_path):
        raise FileNotFoundError(f"Expected attributes file at {attr_path}")
    
    df = pd.read_csv(attr_path)
    if 'image_id' not in df.columns:
        df.insert(0, 'image_id', df.index.astype(str).str.zfill(6) + '.jpg')

    if attribute not in df.columns:
        raise ValueError(f"Attribute '{attribute}' not found in dataset.")

    img_dir = os.path.join(download_path, 'img_align_celeba', 'img_align_celeba')
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Expected image directory at {img_dir}")

    # Prepare labels
    filenames = df['image_id'].values
    labels = (df[attribute].values == 1).astype(np.int32)

    # --- Step 3: Split ---
    train_files, test_files, train_labels, test_labels = train_test_split(
        filenames, labels, test_size=test_size, random_state=random_state
    )

    # --- Step 4: Load and transform images ---
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    def load_images(file_list, label_list):
        images = []
        valid_labels = []
        for filename, label in zip(file_list, label_list):
            img_path = os.path.join(img_dir, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img = transform(img)
                    images.append(img.numpy())
                    valid_labels.append(label)
            except Exception as e:
                print(f"Skipping {filename}: {e}")
        return np.stack(images), np.array(valid_labels)

    print("Loading training images...")
    X_train, Y_train = load_images(train_files, train_labels)

    print("Loading test images...")
    X_test, Y_test = load_images(test_files, test_labels)

    # --- Step 5: Save ---
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'data.npy'), X_train)
    np.save(os.path.join(output_dir, 'labels.npy'), Y_train)
    np.save(os.path.join(output_dir, 'test_data.npy'), X_test)
    np.save(os.path.join(output_dir, 'test_labels.npy'), Y_test)
    np.save(os.path.join(output_dir, 'filenames.npy'), train_files)
    np.save(os.path.join(output_dir, 'test_filenames.npy'), test_files)

    # print(f"Saved {len(X_train)} train and {len(X_test)} test samples to '{output_dir}'.")

# Example usage
# load_dataset(output_dir='npy_split', download_dir='celeba_dataset')
