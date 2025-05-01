import json
import scipy.io
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y

def get_dataset_path(config_file):
    """Extracts the test dataset path from config.json."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    test_dataset_path = config["dataset"]["test_path"]
    if not os.path.exists(test_dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {test_dataset_path}")
    return test_dataset_path

def extract_labels_from_text(file_path):
    """Extracts the binary filter array (0-valid, 1-invalid) from a given text file."""
    with open(file_path, 'r') as file:
        text = file.read()    
    match = re.search(r'Test Data labels\s+\[(.*?)\]', text, re.DOTALL)
    if not match:
        raise ValueError("Could not find the test data labels array in the text file.")
    labels = np.array([int(num) for num in match.group(1).split()])
    return labels

def load_mat_dataset(mat_file):
    """Loads the dataset from a MATLAB .mat file and extracts X, y."""
    mat_data = scipy.io.loadmat(mat_file)    
    X = mat_data["X"]
    y = mat_data["y"].ravel()  
    
    X, y = check_X_y(X, y)
    
    return X, y

def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    return X_train, X_test, y_train, y_test

def filter_dataset(X_test, labels):
    """Filters the test dataset based on the labels (0 = keep, 1 = discard)."""
    return X_test[labels == 0] 

def save_filtered_dataset(filtered_data, output_file):
    """Saves the filtered dataset to a .mat file, creating necessary directories."""
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    scipy.io.savemat(output_file, {"filtered_X_test": filtered_data})
    print(f"Filtered dataset saved as {output_file}")

if __name__ == "__main__":
    config_file = "../config.json"  
    text_file = "/home/dhr33ti/MLProject/XGBOD_forked/xgbod_output.txt"  

    mat_file = get_dataset_path(config_file)
    labels = extract_labels_from_text(text_file)
    X, y = load_mat_dataset(mat_file)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    if len(labels) != X_test.shape[0]:
        raise ValueError(f"Mismatch: Labels ({len(labels)}) vs Test Data Rows ({X_test.shape[0]})")

    filtered_X_test = filter_dataset(X_test, labels)

    with open(config_file, 'r') as file:
        config = json.load(file)
    output_file = config["pipeline"]["pre_processing"]["output"]["path"] + "filtered_dataset.mat"

    save_filtered_dataset(filtered_X_test, output_file)
