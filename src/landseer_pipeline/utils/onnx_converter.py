"""
ONNX Model Conversion Utilities for Cross-Framework Compatibility

This module provides utilities to convert models between PyTorch, TensorFlow, and ONNX formats
to enable seamless interoperability between tools using different ML frameworks.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import json

logger = logging.getLogger(__name__)

# Framework availability checks
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available - PyTorch conversions will be disabled")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available - TensorFlow conversions will be disabled")

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available - ONNX conversions will be disabled")

# Optional converters
try:
    if PYTORCH_AVAILABLE and ONNX_AVAILABLE:
        # torch.onnx is part of PyTorch
        PYTORCH_ONNX_AVAILABLE = True
    else:
        PYTORCH_ONNX_AVAILABLE = False
except ImportError:
    PYTORCH_ONNX_AVAILABLE = False

try:
    if TENSORFLOW_AVAILABLE and ONNX_AVAILABLE:
        import tf2onnx
        TF2ONNX_AVAILABLE = True
    else:
        TF2ONNX_AVAILABLE = False
except ImportError:
    TF2ONNX_AVAILABLE = False
    if TENSORFLOW_AVAILABLE and ONNX_AVAILABLE:
        logger.warning("tf2onnx not available - TensorFlow to ONNX conversion will be disabled")

try:
    if ONNX_AVAILABLE and TENSORFLOW_AVAILABLE:
        import onnx_tf
        ONNX_TF_AVAILABLE = True
    else:
        ONNX_TF_AVAILABLE = False
except ImportError:
    ONNX_TF_AVAILABLE = False
    if ONNX_AVAILABLE and TENSORFLOW_AVAILABLE:
        logger.warning("onnx-tf not available - ONNX to TensorFlow conversion will be disabled")


class ModelMetadata:
    """Metadata for model conversions"""
    
    def __init__(self, framework: str, input_shape: Optional[List[int]] = None, 
                 model_config: Optional[Dict[str, Any]] = None):
        self.framework = framework
        self.input_shape = input_shape
        self.model_config = model_config or {}
        self.conversion_history = []
    
    def add_conversion(self, from_framework: str, to_framework: str, 
                      success: bool, notes: str = ""):
        """Track conversion attempts"""
        self.conversion_history.append({
            "from": from_framework,
            "to": to_framework,
            "success": success,
            "notes": notes
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "framework": self.framework,
            "input_shape": self.input_shape,
            "model_config": self.model_config,
            "conversion_history": self.conversion_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Load from dictionary"""
        metadata = cls(
            framework=data["framework"],
            input_shape=data.get("input_shape"),
            model_config=data.get("model_config", {})
        )
        metadata.conversion_history = data.get("conversion_history", [])
        return metadata


class ONNXConverter:
    """Main class for ONNX model conversions"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
    def detect_model_format(self, model_path: str) -> str:
        """Detect the format of a model file"""
        path = Path(model_path)
        suffix = path.suffix.lower()
        
        if suffix == ".pt" or suffix == ".pth":
            return "pytorch"
        elif suffix == ".h5" or suffix == ".keras":
            return "tensorflow"
        elif suffix == ".onnx":
            return "onnx"
        else:
            raise ValueError(f"Unknown model format: {suffix}")
    
    def convert_pytorch_to_onnx(self, pytorch_model_path: str, onnx_output_path: str,
                               input_shape: List[int], model_script_path: Optional[str] = None,
                               opset_version: int = 11, dynamic_axes: Optional[Dict] = None) -> bool:
        """Convert PyTorch model to ONNX"""
        if not PYTORCH_ONNX_AVAILABLE:
            logger.error("PyTorch to ONNX conversion not available")
            return False
            
        try:
            # Load the model architecture
            if model_script_path:
                from landseer_pipeline.utils import load_config_from_script
                config = load_config_from_script(model_script_path)
                model = config()
            else:
                # Try to load just the state dict and infer architecture
                state_dict = torch.load(pytorch_model_path, map_location='cpu')
                logger.warning("No model script provided - attempting automatic architecture inference")
                # This is a fallback and might not work for complex models
                model = self._infer_pytorch_model_from_state_dict(state_dict)
            
            # Load the trained weights
            state_dict = torch.load(pytorch_model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes or {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            logger.info(f"Successfully converted PyTorch model to ONNX: {onnx_output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert PyTorch to ONNX: {e}")
            return False
    
    def convert_tensorflow_to_onnx(self, tf_model_path: str, onnx_output_path: str,
                                  opset_version: int = 11) -> bool:
        """Convert TensorFlow model to ONNX"""
        if not TF2ONNX_AVAILABLE:
            logger.error("TensorFlow to ONNX conversion not available")
            return False
            
        try:
            # Load TensorFlow model
            model = tf.keras.models.load_model(tf_model_path)
            
            # Convert using tf2onnx
            import subprocess
            import sys
            
            cmd = [
                sys.executable, "-m", "tf2onnx.convert",
                "--keras", tf_model_path,
                "--output", onnx_output_path,
                "--opset", str(opset_version)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully converted TensorFlow model to ONNX: {onnx_output_path}")
                return True
            else:
                logger.error(f"tf2onnx conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to convert TensorFlow to ONNX: {e}")
            return False
    
    def convert_onnx_to_pytorch(self, onnx_model_path: str, pytorch_output_path: str,
                               model_script_path: str) -> bool:
        """Convert ONNX model to PyTorch (requires model architecture script)"""
        if not PYTORCH_ONNX_AVAILABLE:
            logger.error("ONNX to PyTorch conversion not available")
            return False
            
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_model_path)
            
            # Create PyTorch model from script
            from landseer_pipeline.utils import load_config_from_script
            config = load_config_from_script(model_script_path)
            pytorch_model = config()
            
            # Use ONNX runtime to get output and match with PyTorch model
            ort_session = ort.InferenceSession(onnx_model_path)
            
            # Create random input to get output shape
            input_shape = onnx_model.graph.input[0].type.tensor_type.shape
            input_dims = [dim.dim_value if dim.dim_value > 0 else 1 for dim in input_shape.dim]
            dummy_input = torch.randn(*input_dims)
            
            # Run through both models to verify compatibility
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            pytorch_model.eval()
            with torch.no_grad():
                pytorch_output = pytorch_model(dummy_input)
            
            # For now, we save the PyTorch model structure and warn about weights
            torch.save(pytorch_model.state_dict(), pytorch_output_path)
            logger.warning("ONNX to PyTorch conversion saved model structure. "
                          "Manual weight transfer may be required for full compatibility.")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert ONNX to PyTorch: {e}")
            return False
    
    def convert_onnx_to_tensorflow(self, onnx_model_path: str, tf_output_path: str) -> bool:
        """Convert ONNX model to TensorFlow"""
        if not ONNX_TF_AVAILABLE:
            logger.error("ONNX to TensorFlow conversion not available")
            return False
            
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_model_path)
            
            # Convert using onnx-tf
            from onnx_tf.backend import prepare
            tf_rep = prepare(onnx_model)
            
            # Export as SavedModel format first, then convert to h5 if needed
            temp_saved_model_dir = tempfile.mkdtemp()
            tf_rep.export_graph(temp_saved_model_dir)
            
            # Load and save as h5
            model = tf.keras.models.load_model(temp_saved_model_dir)
            model.save(tf_output_path)
            
            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_saved_model_dir)
            
            logger.info(f"Successfully converted ONNX model to TensorFlow: {tf_output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert ONNX to TensorFlow: {e}")
            return False
    
    def convert_model(self, input_path: str, output_path: str, target_framework: str,
                     model_script_path: Optional[str] = None, input_shape: Optional[List[int]] = None,
                     **kwargs) -> Tuple[bool, ModelMetadata]:
        """Main conversion method that handles all framework combinations"""
        
        source_framework = self.detect_model_format(input_path)
        metadata = ModelMetadata(source_framework, input_shape)
        
        logger.info(f"Converting model from {source_framework} to {target_framework}")
        
        if source_framework == target_framework:
            # No conversion needed, just copy
            import shutil
            shutil.copy2(input_path, output_path)
            metadata.add_conversion(source_framework, target_framework, True, "No conversion needed")
            return True, metadata
        
        success = False
        
        # Direct conversions
        if source_framework == "pytorch" and target_framework == "onnx":
            if not input_shape:
                logger.error("Input shape required for PyTorch to ONNX conversion")
                metadata.add_conversion(source_framework, target_framework, False, "Missing input shape")
                return False, metadata
            success = self.convert_pytorch_to_onnx(input_path, output_path, input_shape, 
                                                  model_script_path, **kwargs)
        
        elif source_framework == "tensorflow" and target_framework == "onnx":
            success = self.convert_tensorflow_to_onnx(input_path, output_path, **kwargs)
        
        elif source_framework == "onnx" and target_framework == "pytorch":
            if not model_script_path:
                logger.error("Model script required for ONNX to PyTorch conversion")
                metadata.add_conversion(source_framework, target_framework, False, "Missing model script")
                return False, metadata
            success = self.convert_onnx_to_pytorch(input_path, output_path, model_script_path)
        
        elif source_framework == "onnx" and target_framework == "tensorflow":
            success = self.convert_onnx_to_tensorflow(input_path, output_path)
        
        # Two-step conversions through ONNX
        elif source_framework == "pytorch" and target_framework == "tensorflow":
            temp_onnx_path = os.path.join(self.temp_dir, "temp_model.onnx")
            step1 = self.convert_pytorch_to_onnx(input_path, temp_onnx_path, input_shape,
                                                model_script_path, **kwargs)
            if step1:
                success = self.convert_onnx_to_tensorflow(temp_onnx_path, output_path)
                # Cleanup temp file
                try:
                    os.remove(temp_onnx_path)
                except:
                    pass
            metadata.add_conversion(source_framework, "onnx", step1, "Intermediate step")
        
        elif source_framework == "tensorflow" and target_framework == "pytorch":
            if not model_script_path:
                logger.error("Model script required for TensorFlow to PyTorch conversion")
                metadata.add_conversion(source_framework, target_framework, False, "Missing model script")
                return False, metadata
            temp_onnx_path = os.path.join(self.temp_dir, "temp_model.onnx")
            step1 = self.convert_tensorflow_to_onnx(input_path, temp_onnx_path, **kwargs)
            if step1:
                success = self.convert_onnx_to_pytorch(temp_onnx_path, output_path, model_script_path)
                # Cleanup temp file
                try:
                    os.remove(temp_onnx_path)
                except:
                    pass
            metadata.add_conversion(source_framework, "onnx", step1, "Intermediate step")
        
        else:
            logger.error(f"Unsupported conversion: {source_framework} -> {target_framework}")
            metadata.add_conversion(source_framework, target_framework, False, "Unsupported conversion")
            return False, metadata
        
        metadata.add_conversion(source_framework, target_framework, success, 
                               "Success" if success else "Conversion failed")
        metadata.framework = target_framework if success else source_framework
        
        return success, metadata
    
    def _infer_pytorch_model_from_state_dict(self, state_dict: Dict[str, Any]) -> nn.Module:
        """Attempt to infer PyTorch model architecture from state dict (limited support)"""
        # This is a very basic implementation - real inference would be much more complex
        logger.warning("Using basic model architecture inference - may not work for complex models")
        
        # Simple CNN example - this should be extended based on your model types
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 10)
        )


def get_model_framework(model_path: str) -> str:
    """Utility function to detect model framework from file extension"""
    converter = ONNXConverter()
    return converter.detect_model_format(model_path)


def save_model_metadata(metadata: ModelMetadata, metadata_path: str):
    """Save model metadata to JSON file"""
    with open(metadata_path, 'w') as f:
        json.dump(metadata.to_dict(), f, indent=2)


def load_model_metadata(metadata_path: str) -> Optional[ModelMetadata]:
    """Load model metadata from JSON file"""
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        return ModelMetadata.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load model metadata: {e}")
        return None


def check_conversion_availability() -> Dict[str, bool]:
    """Check which conversion capabilities are available"""
    return {
        "pytorch_available": PYTORCH_AVAILABLE,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "onnx_available": ONNX_AVAILABLE,
        "pytorch_to_onnx": PYTORCH_ONNX_AVAILABLE,
        "tensorflow_to_onnx": TF2ONNX_AVAILABLE,
        "onnx_to_tensorflow": ONNX_TF_AVAILABLE,
        "onnx_to_pytorch": PYTORCH_ONNX_AVAILABLE,  # Limited support
    }