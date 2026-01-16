"""
Container handler package - supports both Docker and Apptainer
"""
from .base import ContainerConfig, ContainerRunner, ContainerImageUtils
from .factory import (get_container_runner, get_container_config, get_available_runtimes, 
                     get_preferred_runtime, get_labels_from_image, get_image_digest, get_container_image_utils)
from .manager import ContainerManager, DockerRunner

# Model converter moved to landseer_pipeline.model_handler
# Import here for backward compatibility
try:
    from ..model_handler import DockerModelConverter, ModelFormat, detect_model_format, get_docker_model_converter
except ImportError:
    # Fallback to local if model_handler not yet set up
    from .model_converter import DockerModelConverter, ModelFormat, detect_model_format, get_docker_model_converter

__all__ = [
    'ContainerConfig',
    'ContainerRunner', 
    'ContainerImageUtils',
    'get_container_runner',
    'get_container_config',
    'get_available_runtimes',
    'get_preferred_runtime',
    'get_labels_from_image',
    'get_image_digest',
    'get_container_image_utils',
    'ContainerManager',
    'DockerRunner',  # Backward compatibility
    'DockerModelConverter',
    'ModelFormat',
    'detect_model_format',
    'get_docker_model_converter'
]