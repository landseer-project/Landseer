"""
Container handler package - supports both Docker and Apptainer
"""
from .base import ContainerConfig, ContainerRunner, ContainerImageUtils
from .factory import (get_container_runner, get_container_config, get_available_runtimes, 
                     get_preferred_runtime, get_labels_from_image, get_image_digest, get_container_image_utils)
from .manager import ContainerManager, DockerRunner

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
    'DockerRunner'  # Backward compatibility
]