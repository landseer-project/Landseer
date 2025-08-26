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

    def run_container(self, combination_id: str, image_name: str, command: Optional[str],
                      environment: Dict[str, str], volumes: Dict[str, Dict]) -> Tuple[int, str]:
        container = None
        
        # Use nvidia runtime instead of device_requests to avoid CDI conflicts
        runtime = None
        if self.device == "cuda":
            runtime = "nvidia"
            try:
                self.__sanity_check_for_gpu(environment, image_name)
            except RuntimeError as e:
                logger.error(f"{combination_id}: GPU sanity check failed: {e}")
                return 1, str(e)
        try:
            run_kwargs = {
                'command': command,
                'environment': environment,
                'volumes': volumes,
                'detach': True,
                'tty': True,
                'stdout': True,
                'stderr': True,
                'runtime': "nvidia",  # Use nvidia runtime instead of device_requests
                'shm_size': '1g',
                'mem_limit': '4g',
            }
            # Resource limits for better performance
            # CPU limits to prevent oversubscription
            logger.debug(f"{combination_id}: Running container {image_name} with runtime={runtime} shm_size={run_kwargs.get('shm_size')} mem_limit={run_kwargs.get('mem_limit')}")
            container = self.client.containers.run(image_name, **run_kwargs)
            result = container.wait()
            exit_code = result.get("StatusCode", 0)
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            container.remove()
            return exit_code, logs
        except Exception as e:
            logger.error(f"{combination_id}: Error running Docker container: {e}")
            if container:
                try:
                    container.remove(force=True)
                except:
                    pass
            raise

    def __sanity_check_for_gpu(self, environment: Dict[str, str], image_name: str):
        command = "python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())'"
        run_kwargs = {
            'command': command,
            'environment': environment,
            'detach': False,
            'tty': False,
            'stdout': True,
            'stderr': True,
            'runtime': "nvidia",
            'shm_size': '1g',
            'mem_limit': '4g',
        }
        logger.debug(f"Running GPU sanity check with command: {command}")
        try:
            # Run a throwaway container for the GPU check
            output = self.client.containers.run(image_name, **run_kwargs)
            output = output.decode("utf-8").strip()
            logger.debug(f"GPU check output: {output}")
            
            if "True" not in output:
                raise RuntimeError(f"GPU not available inside container. Output: {output}")
        except Exception as e:
            raise RuntimeError(f"GPU sanity check failed: {e}")

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