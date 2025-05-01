import scipy.io
import numpy as np
import cv2
import os
from PIL import Image

def mat_to_images(mat_file_path, output_folder, key=None):
    """
    Converts a .mat file to images and saves them.

    :param mat_file_path: Path to the .mat file.
    :param output_folder: Folder to save images.
    :param key: Key to extract image data (if needed).
    """
    # Load the .mat file
    mat = scipy.io.loadmat(mat_file_path)
    
    # If key is not provided, automatically find an array with image-like data
    if key is None:
        possible_keys = [k for k in mat.keys() if isinstance(mat[k], np.ndarray)]
        key = possible_keys[0] if possible_keys else None
    
    if key is None:
        raise ValueError("No valid array found in the .mat file")

    # Extract image data
    data = mat[key]

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    if len(data.shape) == 3:
        for i in range(data.shape[2]):
            img = data[:, :, i]  # Extract one slice
            img = normalize_and_convert(img)
            img_path = os.path.join(output_folder, f"image_{i}.png")
            img.save(img_path)
    
    elif len(data.shape) == 4:
        for i in range(data.shape[3]):
            img = data[:, :, :, i]
            img = normalize_and_convert(img)
            img_path = os.path.join(output_folder, f"image_{i}.png")
            img.save(img_path)
    
    else:
        raise ValueError("Unsupported data format in .mat file")

    print(f"Images saved in {output_folder}")

def normalize_and_convert(image_array):
    """
    Normalizes an image array and converts it to a PIL image.
    """
    image_array = image_array - np.min(image_array)  # Normalize
    image_array = (image_array / np.max(image_array) * 255).astype(np.uint8)
    
    if len(image_array.shape) == 2:  # Grayscale
        return Image.fromarray(image_array, mode="L")
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB
        return Image.fromarray(image_array, mode="RGB")
    else:
        raise ValueError("Unexpected image shape!")

mat_file_path = "data.mat"  
output_folder = "output_images"
mat_to_images(mat_file_path, output_folder)
