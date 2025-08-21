"""
Docker operations for ML Defense Pipeline
"""
import logging
import torch
import subprocess
import docker
from typing import Dict, Optional, Tuple, Union, Annotated

logger = logging.getLogger(__name__)


class DockerRunner:
    """Manages Docker-related operations for the pipeline"""

    def __init__(self, Settings):
        """Initialize Docker client if available"""
        self.settings = Settings
        self.client = docker.from_env() 
        self.device = self.settings.device
        print(f"Using device: {self.device}")

    def run_container(self, image_name: str, command: Optional[str],
                      environment: Dict[str, str], volumes: Dict[str, Dict], gpu_id) -> Tuple[int, str]:
        container = None
        
        # Use nvidia runtime instead of device_requests to avoid CDI conflicts
        runtime = None
        if self.device == "cuda":
            runtime = "nvidia"
        
        try:
            run_kwargs = {
                'command': command,
                'environment': environment,
                'volumes': volumes,
                'detach': True,
                'tty': True,
                'stdout': True,
                'stderr': True,
                'runtime': runtime,  # Use nvidia runtime instead of device_requests
            }
            # Resource limits (optional)
            shm_size = getattr(self.settings, 'docker_shm_size', None)
            if shm_size:
                run_kwargs['shm_size'] = shm_size
            mem_limit = getattr(self.settings, 'docker_mem_limit', None)
            if mem_limit:
                run_kwargs['mem_limit'] = mem_limit

            logger.debug(f"Running container {image_name} with runtime={runtime} shm_size={run_kwargs.get('shm_size')} mem_limit={run_kwargs.get('mem_limit')}")
            container = self.client.containers.run(image_name, **run_kwargs)
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

    def cleanup_container(self, container) -> None:
        try:
            if container.status == 'running':
                logger.debug(f"Stopping container: {container.short_id}")
                container.stop(timeout=10)
            
            logger.debug(f"Removing container: {container.short_id}")
            container.remove(force=True)
            
        except Exception as e:
            logger.warning(f"Failed to cleanup container {container.short_id}: {e}")

    @property
    def config(self) -> Dict[str, str]:
        return self.stager.config