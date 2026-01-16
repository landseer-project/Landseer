"""
ONNX Model Conversion Utilities for Cross-Framework Compatibility

This module has been refactored to use Docker-based model conversion,
reducing dependency on ML packages in the main Landseer codebase.

The actual conversion logic is now in Docker containers:
- ghcr.io/landseer-project/model_converter_pytorch_to_other:v1
- ghcr.io/landseer-project/model_converter_other_to_pytorch:v1

This file maintains backward compatibility with the existing API.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import json

logger = logging.getLogger(__name__)


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


def get_model_framework(model_path: str) -> str:
    """
    Detect the framework of a model file based on extension.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Framework name: 'pytorch', 'tensorflow', 'onnx', or 'unknown'
    """
    path = Path(model_path)
    
    if not path.exists():
        logger.warning(f"Model path does not exist: {model_path}")
        return "unknown"
    
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in ['.pt', '.pth']:
            return "pytorch"
        elif suffix in ['.h5', '.keras']:
            return "tensorflow"
        elif suffix == '.onnx':
            return "onnx"
        elif suffix == '.pb':
            return "tensorflow"
    elif path.is_dir():
        # Check for SavedModel format
        if (path / 'saved_model.pb').exists():
            return "tensorflow"
        # Look for model files
        for ext, fmt in [('.pt', 'pytorch'), ('.pth', 'pytorch'),
                         ('.onnx', 'onnx'), ('.h5', 'tensorflow')]:
            if list(path.glob(f'*{ext}')):
                return fmt
    
    return "unknown"


class ONNXConverter:
    """
    Model converter that uses Docker containers for actual conversion.
    
    This class maintains the same API as the previous implementation but
    delegates conversion to Docker containers, reducing the need for
    heavy ML framework dependencies in the Landseer environment.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._docker_converter = None
    
    @property
    def docker_converter(self):
        """Lazy-load the Docker model converter."""
        if self._docker_converter is None:
            from .docker_converter import DockerModelConverter
            self._docker_converter = DockerModelConverter(temp_dir=self.temp_dir)
        return self._docker_converter
    
    def detect_model_format(self, model_path: str) -> str:
        """Detect the format of a model file"""
        return get_model_framework(model_path)
    
    def convert_model(self, input_path: str, output_path: str, target_framework: str,
                     source_framework: Optional[str] = None,
                     model_script_path: Optional[str] = None, 
                     input_shape: Optional[List[int]] = None,
                     **kwargs) -> Tuple[bool, ModelMetadata]:
        """
        Convert a model using Docker containers.
        
        This method maintains backward compatibility with the previous
        implementation while using Docker containers for actual conversion.
        
        Args:
            input_path: Path to source model
            output_path: Path for converted model
            target_framework: Target format ('pytorch', 'tensorflow', 'onnx')
            source_framework: Source format (auto-detected if None)
            model_script_path: Path to model architecture script
            input_shape: Input shape for the model
            **kwargs: Additional parameters (opset_version, etc.)
            
        Returns:
            Tuple of (success: bool, metadata: ModelMetadata)
        """
        from .docker_converter import ModelFormat
        
        # Convert framework strings to ModelFormat enum
        format_map = {
            'pytorch': ModelFormat.PYTORCH,
            'tensorflow': ModelFormat.TENSORFLOW,
            'onnx': ModelFormat.ONNX,
        }
        
        target_format = format_map.get(target_framework.lower())
        if target_format is None:
            logger.error(f"Unknown target framework: {target_framework}")
            return False, ModelMetadata("unknown")
        
        source_format = None
        if source_framework:
            source_format = format_map.get(source_framework.lower())
        
        # Get opset version from kwargs
        opset_version = kwargs.get('opset_version', 11)
        
        # Use Docker converter
        success, result_metadata = self.docker_converter.convert_model(
            source_path=input_path,
            target_path=output_path,
            target_format=target_format,
            source_format=source_format,
            input_shape=input_shape,
            model_script=model_script_path,
            opset_version=opset_version
        )
        
        # Build ModelMetadata for backward compatibility
        detected_source = source_framework or self.detect_model_format(input_path)
        metadata = ModelMetadata(
            framework=detected_source,
            input_shape=input_shape
        )
        metadata.add_conversion(
            from_framework=detected_source,
            to_framework=target_framework,
            success=success,
            notes=result_metadata.get('error', '') if not success else 'Converted via Docker container'
        )
        
        return success, metadata
