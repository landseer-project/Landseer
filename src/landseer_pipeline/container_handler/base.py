"""
Abstract base class for container operations supporting both Docker and Apptainer/Singularity
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any, List
import logging

logger = logging.getLogger(__name__)


class ContainerConfig(ABC):
    """Abstract base class for container configuration"""
    def __init__(self, image: str, command: str, config_script: Optional[str] = None):
        self.image = image
        self.command = command
        self.config_script = config_script
    
    @property
    @abstractmethod
    def image_name(self) -> str:
        """Extract the image name from the full image path"""
        pass
    
    @abstractmethod
    def get_labels(self) -> Dict[str, str]:
        """Get labels from the container image"""
        pass
    
    @abstractmethod
    def validate_image_and_pull(self) -> bool:
        """Validate and pull the container image if necessary"""
        pass


class ContainerRunner(ABC):
    """Abstract base class for container runtime operations"""
    
    def __init__(self, settings: Any):
        self.settings = settings
        self.device = getattr(settings, 'device', 'cpu')
    
    @abstractmethod
    def run_container(self, 
                     image_name: str, 
                     command: Optional[str],
                     environment: Dict[str, str], 
                     volumes: Dict[str, Dict], 
                     gpu_id: Optional[int] = None) -> Tuple[int, str, Any]:
        """
        Run a container with the specified configuration
        
        Returns:
            Tuple of (exit_code, logs, container_handle)
        """
        pass
    
    @abstractmethod
    def cleanup_container(self, container: Any) -> None:
        """Clean up a container instance"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this container runtime is available on the system"""
        pass
    
    @abstractmethod
    def get_runtime_name(self) -> str:
        """Get the name of the container runtime"""
        pass


class ContainerImageUtils(ABC):
    """Abstract base class for container image utilities"""
    
    @staticmethod
    @abstractmethod
    def get_labels_from_image(image: str) -> Dict[str, str]:
        """Get labels from a container image"""
        pass
    
    @staticmethod
    @abstractmethod
    def get_image_digest(image: str) -> str:
        """Get the digest of a container image"""
        pass