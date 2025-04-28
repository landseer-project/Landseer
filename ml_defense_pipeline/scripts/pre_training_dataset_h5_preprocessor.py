#!/usr/bin/env python3
"""
Pre-training preprocessor script for dataset in h5 format
"""
import os
import h5py
import numpy as np

input_path = os.environ["INPUT_IR"]
output_dir = os.path.dirname(input_path)
tool_input_format = os.environ.get("TOOL_INPUT_FORMAT", "h5")

print(f"Pre-training preprocessor: Converting dataset from IR format to {tool_input_format}")
print(f"INPUT_IR = {input_path}")
print(f"Is it a file? {os.path.isfile(input_path)}")
print(f"Is it a directory? {os.path.isdir(input_path)}")
print("Directory listing of /data:")
print(os.listdir("/data"))


try:
    with h5py.File(input_path, "r") as f:
        X_train = f["X_train"][:]
        y_train = f["y_train"][:]
        X_test = f["X_test"][:]
        y_test = f["y_test"][:]
    
    print(f"Loaded dataset from {input_path}")
    print(f"  Training data shape: {X_train.shape}")
    print(f"  Training labels shape: {y_train.shape}")
    print(f"  Test data shape: {X_test.shape}")
    print(f"  Test labels shape: {y_test.shape}")
    
    # already h5, no conversion needed
    if tool_input_format == "h5":
        print("No conversion needed - tool accepts h5 format")
    
    elif tool_input_format == "pickle":
        import pickle
        
        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }
        
        # Save as pickle
        output_pickle = os.path.join(output_dir, "converted_dataset.pickle")
        with open(output_pickle, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Dataset converted to pickle format: {output_pickle}")
    
    elif tool_input_format == "numpy":
        # Save as numpy arrays
        output_dir_numpy = os.path.join(output_dir, "numpy_arrays")
        os.makedirs(output_dir_numpy, exist_ok=True)
        
        np.save(os.path.join(output_dir_numpy, "X_train.npy"), X_train)
        np.save(os.path.join(output_dir_numpy, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir_numpy, "X_test.npy"), X_test)
        np.save(os.path.join(output_dir_numpy, "y_test.npy"), y_test)
        
        print(f"Dataset converted to numpy arrays in: {output_dir_numpy}")
    
    else:
        print(f"Warning: Conversion to {tool_input_format} not implemented")
    
    print("Pre-processing completed successfully")

except Exception as e:
    print(f"Error in pre-processing: {e}")
    raise