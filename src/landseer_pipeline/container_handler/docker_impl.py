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
    from landseer_pipeline.container_handler.docker import get_labels_from_image, get_image_digest

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
        # Increase timeout for large datasets like CelebA (200k+ images)
        self.client = docker.from_env(timeout=600)  # 10 minute timeout
        logger.debug(f"DockerRunner initialized with device: {self.device}")
    
    def run_container(self, 
                     image_name: str, 
                     command: Optional[str],
                     environment: Dict[str, str], 
                     volumes: Dict[str, Dict], 
                     gpu_id: Optional[int] = None,
                     combination_id: Optional[str] = None) -> Tuple[int, str, Any]:
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker is not available on this system.")
        
        container = None
        
        # Use nvidia runtime instead of device_requests to avoid CDI conflicts
        runtime = None
        if self.device == "cuda":
            runtime = "nvidia"
        
        try:
            combo_prefix = f"{combination_id}: " if combination_id else ""
            logger.debug(f"{combo_prefix}Running Docker container: {image_name} with runtime={runtime}")
            
            # Run container as current user to avoid root-owned files in output directories
            import os
            current_uid = os.getuid()
            current_gid = os.getgid()
            
            run_kwargs = {
                'command': command,
                'environment': environment,
                'volumes': volumes,
                'detach': True,
                'tty': True,
                'stdout': True,
                'stderr': True,
                'working_dir': "/app",  # Set working directory to /app where main.py is located
                'shm_size': '2g',  # Increase shared memory for PyTorch DataLoader workers
                #'user': f"{current_uid}:{current_gid}",  # Run as current user to fix file ownership
            }
            
            if runtime:
                run_kwargs['runtime'] = runtime
            
            container = self.client.containers.run(image_name, **run_kwargs)
            result = container.wait()
            exit_code = result.get("StatusCode", 0)
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            logger.debug(f"{combo_prefix}Docker container finished with exit code: {exit_code}")
            container.remove()
            return exit_code, logs, container
        except Exception as e:
            combo_prefix = f"{combination_id}: " if combination_id else ""
            logger.error(f"{combo_prefix}Error running Docker container: {e}")
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