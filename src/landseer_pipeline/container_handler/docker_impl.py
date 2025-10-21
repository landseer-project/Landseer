"""
Docker implementation of container operations
"""
import os
import logging
from typing import Dict, Optional, Tuple, Any

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None

from .base import ContainerConfig, ContainerRunner, ContainerImageUtils

# Only import docker utils if docker is available
if DOCKER_AVAILABLE:
    from landseer_pipeline.utils.docker import get_labels_from_image, get_image_digest

logger = logging.getLogger(__name__)


class DockerConfig(ContainerConfig):
    """Docker-specific container configuration"""
    
    def __init__(self, *args, **kwargs):
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker is not available on this system. Please install Docker or use Apptainer.")
        super().__init__(*args, **kwargs)
    
    @property
    def image_name(self) -> str:
        if self.image:
            image = self.image.split(":")[0]
            image_name = image.split("/")[-1]
            return image_name
        return ""
    
    def get_labels(self) -> Dict[str, str]:
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker is not available on this system.")
        if self.image:
            labels = DockerImageUtils.get_labels_from_image(self.image)
            if not labels:
                raise ValueError(f"No labels found in Docker image '{self.image}'")
            if "stage" not in labels:
                raise ValueError(f"Label 'stage' not found in Docker image '{self.image}'")
            if "dataset" not in labels:
                raise ValueError(f"Label 'dataset' not found in Docker image '{self.image}'")
            return labels
        return {}
    
    def validate_image_and_pull(self) -> bool:
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker is not available on this system.")
        logger.debug(f"Validating Docker image: {self.image}")
        try:
            client = docker.from_env() 
            # Skip pulling during validation to avoid network issues
            # The image will be pulled when actually used
            logger.debug(f"Docker available, skipping image pull for: {self.image}")
            return True
            
            # Original validation code (commented out for now):
            # logger.warning(f"Attempting to pull docker image...")
            # client.images.pull(self.image)
            # return True
        except Exception as e:
            logger.error(f"Failed to check Docker image '{self.image}': {e}")
            raise ValueError(f"Failed to check Docker image '{self.image}': {e}")


class DockerRunner(ContainerRunner):
    """Docker-specific container runtime operations"""
    
    def __init__(self, settings: Any):
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker is not available on this system. Please install Docker or use Apptainer.")
        super().__init__(settings)
        self.client = docker.from_env()
        logger.info(f"Using Docker with device: {self.device}")
    
    def run_container(self, 
                     image_name: str, 
                     command: Optional[str],
                     environment: Dict[str, str], 
                     volumes: Dict[str, Dict], 
                     gpu_id: Optional[int] = None) -> Tuple[int, str, Any]:
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker is not available on this system.")
        
        container = None
        
        device_requests = None
        if self.device == "cuda":
            device_requests = [docker.types.DeviceRequest(
                count=-1, capabilities=[["gpu"]])]
        
        try:
            container = self.client.containers.run(
                image_name,
                command=command,
                environment=environment,
                volumes=volumes,
                detach=True,
                tty=True,
                stdout=True,
                stderr=True,
                device_requests=device_requests,
            )
            result = container.wait()
            exit_code = result.get("StatusCode", 0)
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            container.remove()
            return exit_code, logs, container
        except Exception as e:
            logger.error(f"Error running Docker container: {e}")
            if container:
                try:
                    container.remove(force=True)
                except:
                    pass
            raise
    
    def cleanup_container(self, container: Any) -> None:
        if not DOCKER_AVAILABLE:
            return
        try:
            if container.status == 'running':
                logger.debug(f"Stopping container: {container.short_id}")
                container.stop(timeout=10)
            
            logger.debug(f"Removing container: {container.short_id}")
            container.remove(force=True)
            
        except Exception as e:
            logger.warning(f"Failed to cleanup container {container.short_id}: {e}")
    
    def is_available(self) -> bool:
        if not DOCKER_AVAILABLE:
            return False
        try:
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False
    
    def get_runtime_name(self) -> str:
        return "docker"


class DockerImageUtils(ContainerImageUtils):
    """Docker-specific image utilities"""
    
    @staticmethod
    def get_labels_from_image(image: str) -> Dict[str, str]:
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker is not available on this system.")
        return get_labels_from_image(image)
    
    @staticmethod
    def get_image_digest(image: str) -> str:
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker is not available on this system.")
        return get_image_digest(image)