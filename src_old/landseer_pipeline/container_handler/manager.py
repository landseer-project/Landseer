"""
Container operations manager for ML Defense Pipeline
"""
import logging
from typing import Dict, Optional, Tuple, Union, Any
from .factory import get_container_runner

logger = logging.getLogger(__name__)


class ContainerManager:
    """Manages container-related operations for the pipeline"""

    def __init__(self, settings, runtime: Optional[str] = None):
        """Initialize container runner based on available runtime"""
        self.settings = settings
        self.runner = get_container_runner(settings, runtime)
        self.device = self.settings.device
        logger.info(f"Using container runtime: {self.runner.get_runtime_name()}")
        logger.info(f"Using device: {self.device}")

    def run_container(self, image_name: str, command: Optional[str],
                      environment: Dict[str, str], volumes: Dict[str, Dict], 
                      gpu_id: Optional[int] = None) -> Tuple[int, str, Any]:
        """
        Run a container with the specified configuration
        
        Args:
            image_name: Container image name
            command: Command to run in the container
            environment: Environment variables
            volumes: Volume mounts
            gpu_id: GPU ID to use (if applicable)
        
        Returns:
            Tuple of (exit_code, logs, container_handle)
        """
        return self.runner.run_container(image_name, command, environment, volumes, gpu_id)

    def cleanup_container(self, container: Any) -> None:
        """Clean up a container instance"""
        self.runner.cleanup_container(container)

    @property
    def runtime_name(self) -> str:
        """Get the name of the container runtime being used"""
        return self.runner.get_runtime_name()

    @property
    def config(self) -> Dict[str, str]:
        """Get configuration information"""
        return {
            'runtime': self.runtime_name,
            'device': self.device
        }


# Backward compatibility alias
DockerRunner = ContainerManager