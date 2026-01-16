"""
Model Handler Package

This package manages model format conversions and framework compatibility 
across pipeline stages. It provides:

1. Automatic framework detection and model loading
2. Docker-based conversion for inter-tool conversions
3. Framework-specific model instantiation for tools
4. ONNX standardization for pipeline storage

Main Components:
- ModelFormatManager: High-level API for pipeline model handling
- ONNXConverter: ONNX conversion utilities with Docker backend
- DockerModelConverter: Low-level Docker-based conversion
- ModelFormat: Enum for supported formats
"""

from .format_manager import ModelFormatManager, get_model_format_manager
from .converter import ONNXConverter, ModelMetadata, get_model_framework
from .docker_converter import (
    DockerModelConverter, 
    ModelFormat, 
    detect_model_format, 
    get_docker_model_converter,
    CONVERTER_IMAGES
)

__all__ = [
    # High-level API
    'ModelFormatManager',
    'get_model_format_manager',
    
    # ONNX converter
    'ONNXConverter',
    'ModelMetadata',
    'get_model_framework',
    
    # Docker converter
    'DockerModelConverter',
    'ModelFormat',
    'detect_model_format',
    'get_docker_model_converter',
    'CONVERTER_IMAGES',
]
