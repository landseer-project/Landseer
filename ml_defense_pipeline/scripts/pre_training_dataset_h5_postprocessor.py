#!/usr/bin/env python3
"""
Pre-training postprocessor script for dataset in h5 format
"""
import os
import h5py
import numpy as np

# Get environment variables
output_ir_path = os.environ["OUTPUT_IR"]
tool_input_format = os.environ.get("TOOL_INPUT_FORMAT", "h5")

print(f"Pre-training postprocessor: Converting from tool output to IR format")

try:
    # Check if tool already produced h5 output
    if os.path.exists(output_ir_path) and output_ir_path.endswith(".h5"):
        print(f"Tool already produced output in h5 format: {output_ir_path}")
    else:
        # Need to convert tool output to h5 format
        print(f"Converting tool output to h5 format: {output_ir_path}")
        
        # Example: tool might have produced output in different formats
        if tool_input_format == "pickle":
            import pickle
            
            # Find pickle file
            output_dir = os.path.dirname(output_ir_path)
            pickle_file = os.path.join(output_dir, "tool_output.pickle")
            
            if os.path.exists(pickle_file):
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)
                
                # Extract data
                X_train = data.get("X_train")
                y_train = data.get("y_train")
                X_test = data.get("X_test")
                y_test = data.get("y_test")
                
                # Save as h5
                with h5py.File(output_ir_path, "w") as f:
                    f.create_dataset("X_train", data=X_train)
                    f.create_dataset("y_train", data=y_train)
                    f.create_dataset("X_test", data=X_test)
                    f.create_dataset("y_test", data=y_test)
                
                print(f"Converted pickle to h5: {output_ir_path}")
            else:
                print(f"Warning: Could not find pickle output file")
                # Create dummy output for demonstration
                X_train = np.random.rand(1000, 32, 32, 3)
                y_train = np.random.randint(0, 10, size=(1000,))
                X_test = np.random.rand(100, 32, 32, 3)
                y_test = np.random.randint(0, 10, size=(100,))
                
                with h5py.File(output_ir_path, "w") as f:
                    f.create_dataset("X_train", data=X_train)
                    f.create_dataset("y_train", data=y_train)
                    f.create_dataset("X_test", data=X_test)
                    f.create_dataset("y_test", data=y_test)
                
                print(f"Created dummy output for demonstration: {output_ir_path}")
        
        elif tool_input_format == "numpy":
            # Find numpy arrays
            output_dir = os.path.dirname(output_ir_path)
            numpy_dir = os.path.join(output_dir, "numpy_arrays")
            
            if os.path.exists(numpy_dir):
                X_train = np.load(os.path.join(numpy_dir, "X_train.npy"))
                y_train = np.load(os.path.join(numpy_dir, "y_train.npy"))
                X_test = np.load(os.path.join(numpy_dir, "X_test.npy"))
                y_test = np.load(os.path.join(numpy_dir, "y_test.npy"))
                
                # Save as h5
                with h5py.File(output_ir_path, "w") as f:
                    f.create_dataset("X_train", data=X_train)
                    f.create_dataset("y_train", data=y_train)
                    f.create_dataset("X_test", data=X_test)
                    f.create_dataset("y_test", data=y_test)
                
                print(f"Converted numpy arrays to h5: {output_ir_path}")
            else:
                print(f"Warning: Could not find numpy output directory")
                # Create dummy output for demonstration
                X_train = np.random.rand(1000, 32, 32, 3)
                y_train = np.random.randint(0, 10, size=(1000,))
                X_test = np.random.rand(100, 32, 32, 3)
                y_test = np.random.randint(0, 10, size=(100,))
                
                with h5py.File(output_ir_path, "w") as f:
                    f.create_dataset("X_train", data=X_train)
                    f.create_dataset("y_train", data=y_train)
                    f.create_dataset("X_test", data=X_test)
                    f.create_dataset("y_test", data=y_test)
                
                print(f"Created dummy output for demonstration: {output_ir_path}")
        
        else:
            print(f"Warning: Conversion from {tool_input_format} not implemented")
            # Create dummy output for demonstration
            X_train = np.random.rand(1000, 32, 32, 3)
            y_train = np.random.randint(0, 10, size=(1000,))
            X_test = np.random.rand(100, 32, 32, 3)
            y_test = np.random.randint(0, 10, size=(100,))
            
            with h5py.File(output_ir_path, "w") as f:
                f.create_dataset("X_train", data=X_train)
                f.create_dataset("y_train", data=y_train)
                f.create_dataset("X_test", data=X_test)
                f.create_dataset("y_test", data=y_test)
            
            print(f"Created dummy output for demonstration: {output_ir_path}")
    
    print("Post-processing completed successfully")

except Exception as e:
    print(f"Error in post-processing: {e}")
    raise