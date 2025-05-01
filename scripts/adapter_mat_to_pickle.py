#!/usr/bin/env python
"""
Convert CIFAR-10 data files to the pickle format expected by the CIFAR-10 adversarial challenge.

This script takes a directory containing CIFAR-10 data files (could be in any format readable by scipy or numpy)
and converts them to the pickle format expected by the challenge repository.

Expected files in the input directory:
- data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5 (training batches)
- test_batch (test data)
- batches.meta (metadata with label names)

The script will skip readme.html or any other files.
"""

import os
import pickle
import numpy as np
import scipy.io as sio
import argparse
import glob
import re

def detect_file_format(filepath):
    """Detect the format of the input file based on extension"""
    if filepath.endswith('.mat'):
        return 'mat'
    elif filepath.endswith('.npy'):
        return 'npy'
    else:
        # Try to determine if it's already a pickle file
        try:
            with open(filepath, 'rb') as f:
                pickle.load(f)
            return 'pickle'
        except:
            raise ValueError(f"Unsupported file format for {filepath}")

def load_file(filepath):
    """Load data from file based on its detected format"""
    format_type = detect_file_format(filepath)
    
    if format_type == 'mat':
        return sio.loadmat(filepath)
    elif format_type == 'npy':
        return np.load(filepath, allow_pickle=True).item()
    elif format_type == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)

def convert_file_to_pickle(input_file, output_file, is_batch=True):
    """
    Convert a single CIFAR-10 file to pickle format
    
    Args:
        input_file (str): Path to the input file
        output_file (str): Path to save the pickle file
        is_batch (bool): True if this is a data batch file, False if it's the meta file
    """
    print(f"Converting {input_file} to {output_file}")
    
    # Skip readme.html
    if 'readme.html' in input_file:
        print(f"Skipping {input_file}")
        return
    
    # Load the data
    data = load_file(input_file)
    
    # Process based on file type
    if is_batch:
        # This is a data batch file
        # Make sure we have data and labels in the expected format
        if isinstance(data, dict):
            # If data is already a dictionary, ensure keys are bytes
            processed_data = {}
            
            # Look for data and labels keys
            data_key = None
            labels_key = None
            
            for key in data.keys():
                key_str = key if isinstance(key, str) else key.decode('utf-8') if isinstance(key, bytes) else str(key)
                if 'data' in key_str.lower():
                    data_key = key
                elif 'label' in key_str.lower():
                    labels_key = key
            
            if data_key is None or labels_key is None:
                raise ValueError(f"Could not find data and labels in {input_file}")
            
            # Ensure numpy array for data
            if not isinstance(data[data_key], np.ndarray):
                processed_data[b'data'] = np.array(data[data_key], dtype=np.uint8)
            else:
                processed_data[b'data'] = data[data_key].astype(np.uint8)
            
            # Ensure list for labels
            if isinstance(data[labels_key], np.ndarray):
                processed_data[b'labels'] = data[labels_key].flatten().tolist()
            else:
                processed_data[b'labels'] = list(data[labels_key])
        else:
            raise ValueError(f"Unexpected data format in {input_file}")
        
        # Save as pickle
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
    else:
        # This is the meta file
        if isinstance(data, dict):
            processed_data = {}
            
            # Look for label_names key
            label_names_key = None
            for key in data.keys():
                key_str = key if isinstance(key, str) else key.decode('utf-8') if isinstance(key, bytes) else str(key)
                if 'label' in key_str.lower() and 'name' in key_str.lower():
                    label_names_key = key
            
            if label_names_key is None:
                # Use default CIFAR-10 class names
                label_names = [
                    b'airplane', b'automobile', b'bird', b'cat', b'deer',
                    b'dog', b'frog', b'horse', b'ship', b'truck'
                ]
            else:
                # Extract label names
                label_names = data[label_names_key]
                if isinstance(label_names, np.ndarray):
                    if label_names.dtype.type is np.str_:
                        label_names = [name.encode('utf-8') for name in label_names.flatten()]
                    elif label_names.dtype.type is np.bytes_:
                        label_names = [name for name in label_names.flatten()]
                    else:
                        label_names = [str(name).encode('utf-8') for name in label_names.flatten()]
                elif isinstance(label_names, list):
                    label_names = [name.encode('utf-8') if isinstance(name, str) else name for name in label_names]
            
            processed_data[b'label_names'] = label_names
            
            # Save as pickle
            with open(output_file, 'wb') as f:
                pickle.dump(processed_data, f)
        else:
            raise ValueError(f"Unexpected data format in {input_file}")

def convert_cifar10_files(input_dir, output_dir):
    """
    Convert all CIFAR-10 files in the input directory to pickle format
    
    Args:
        input_dir (str): Directory containing CIFAR-10 files
        output_dir (str): Directory to save the pickle files
    """
    print(f"Converting files from {input_dir} to {output_dir}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Process data batch files
    batch_pattern = re.compile(r'data_batch_[1-5]')
    for i in range(1, 6):
        # Look for files with name data_batch_i (with or without extension)
        batch_files = glob.glob(os.path.join(input_dir, f'data_batch_{i}*'))
        
        # If no exact match, try to find files that contain the pattern
        if not batch_files:
            all_files = os.listdir(input_dir)
            for file in all_files:
                if f'data_batch_{i}' in file or (batch_pattern.search(file) and str(i) in file):
                    batch_files.append(os.path.join(input_dir, file))
        
        if not batch_files:
            raise ValueError(f"Could not find data_batch_{i} in {input_dir}")
        
        # Use the first matching file
        input_file = batch_files[0]
        output_file = os.path.join(output_dir, f'data_batch_{i}')
        convert_file_to_pickle(input_file, output_file, is_batch=True)
    
    # Process test batch file
    test_files = glob.glob(os.path.join(input_dir, 'test_batch*'))
    if not test_files:
        all_files = os.listdir(input_dir)
        for file in all_files:
            if 'test_batch' in file or 'test_data' in file:
                test_files.append(os.path.join(input_dir, file))
    
    if not test_files:
        raise ValueError(f"Could not find test_batch in {input_dir}")
    
    input_file = test_files[0]
    output_file = os.path.join(output_dir, 'test_batch')
    convert_file_to_pickle(input_file, output_file, is_batch=True)
    
    # Process meta file
    meta_files = glob.glob(os.path.join(input_dir, 'batches.meta*'))
    if not meta_files:
        all_files = os.listdir(input_dir)
        for file in all_files:
            if 'meta' in file or 'labels' in file:
                meta_files.append(os.path.join(input_dir, file))
    
    if meta_files:
        input_file = meta_files[0]
        output_file = os.path.join(output_dir, 'batches.meta')
        convert_file_to_pickle(input_file, output_file, is_batch=False)
    else:
        # Create default meta file
        print("No meta file found, creating default")
        label_names = [
            b'airplane', b'automobile', b'bird', b'cat', b'deer',
            b'dog', b'frog', b'horse', b'ship', b'truck'
        ]
        meta_data = {b'label_names': label_names}
        output_file = os.path.join(output_dir, 'batches.meta')
        with open(output_file, 'wb') as f:
            pickle.dump(meta_data, f)
    
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CIFAR-10 files to pickle format')
    parser.add_argument('--input', help='Directory containing CIFAR-10 data files')
    parser.add_argument('--output', default='cifar10_data', help='Directory to save pickle files (default: cifar10_data)')
    
    args = parser.parse_args()
    convert_cifar10_files(args.input_dir, args.output_dir)