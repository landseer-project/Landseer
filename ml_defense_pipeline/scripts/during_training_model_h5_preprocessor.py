#!/usr/bin/env python3
"""
During-training preprocessor script for model in h5 format
"""
import os
import h5py
import numpy as np

# Get environment variables
input_path = os.environ["INPUT_IR"]
output_dir = os.path.dirname(input_path)
tool_input_format = os.environ.get("TOOL_INPUT_FORMAT", "h5")

print(f"During-training preprocessor: Converting model from IR format to {tool_input_format}")

try:
    # If tool accepts h5 format, no conversion needed
    if tool_input_format == "h5":
        print("No conversion needed - tool accepts h5 format")
    
    # Otherwise, need to convert model format
    else:
        print(f"Converting model from h5 to {tool_input_format}")
        
        if tool_input_format == "tensorflow":
            # Try to load as TensorFlow model
            try:
                import tensorflow as tf
                
                model = tf.keras.models.load_model(input_path)
                
                # Save in TensorFlow SavedModel format
                output_tf_dir = os.path.join(output_dir, "tf_model")
                model.save(output_tf_dir)
                
                print(f"Model converted to TensorFlow SavedModel format: {output_tf_dir}")
            except Exception as e:
                print(f"Error converting to TensorFlow format: {e}")
                raise
        
        elif tool_input_format == "pytorch":
            # Example conversion to PyTorch format
            try:
                import tensorflow as tf
                import torch
                import numpy as np
                
                # This is a simplified demonstration - real conversion would be more complex
                # Load TensorFlow model
                tf_model = tf.keras.models.load_model(input_path)
                
                # Extract weights (simplified)
                weights = [layer.get_weights() for layer in tf_model.layers]
                
                # Save weights in a format PyTorch can read
                output_pt_path = os.path.join(output_dir, "model_weights.pt")
                torch.save({"weights": weights}, output_pt_path)
                
                # Save model architecture as JSON
                model_json = tf_model.to_json()
                with open(os.path.join(output_dir, "model_arch.json"), "w") as f:
                    f.write(model_json)
                
                print(f"Model converted to PyTorch format: {output_pt_path}")
            except Exception as e:
                print(f"Error converting to PyTorch format: {e}")
                raise
        
        else:
            print(f"Warning: Conversion to {tool_input_format} not implemented")
    
    print("Pre-processing completed successfully")

except Exception as e:
    print(f"Error in pre-processing: {e}")
    raise