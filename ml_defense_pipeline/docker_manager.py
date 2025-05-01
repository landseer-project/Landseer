"""
Docker operations for ML Defense Pipeline
"""
import logging
import torch
import os
import shutil
import subprocess
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger("defense_pipeline")

try:
    import docker
    DOCKER_SDK_AVAILABLE = True
except ImportError:
    DOCKER_SDK_AVAILABLE = False
    logger.warning("Docker SDK not available. Falling back to subprocess for Docker operations.")

class DockerManager:
    """Manages Docker-related operations for the pipeline"""
    
    def __init__(self):
        """Initialize Docker client if available"""
        self.client = docker.from_env() if DOCKER_SDK_AVAILABLE else None


    def run_container(self, image_name: str, command: Optional[str], 
                      environment: Dict[str, str], volumes: Dict[str, Dict]) -> Tuple[int, str]:
        """
        Run a Docker container with the given parameters
        
        Args:
            image_name: Docker image to run
            command: Command to execute in the container (or None to use default)
            environment: Environment variables to set
            volumes: Volume mappings
            
        Returns:
            Tuple of (exit_code, logs)
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                logger.info("Using GPU for Docker container")
            else:
                logger.info("Using CPU for Docker container")

            # TODO: Add support for GPU if needed
            # Check if Docker SDK is available
            # If Docker SDK is available, use it
            # If not, fallback to subprocess
            if self.client:
                from docker.types import DeviceRequest

                device_requests = None
                if device.type == "cuda":
                    device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])]

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
                    remove=True
                )
                
                result = container.wait()
                exit_code = result.get("StatusCode", 0)
                logs = container.logs(stdout=True, stderr=True).decode('utf-8')
                container.remove()
                
                return exit_code, logs
                
            else:
                # Fallback to subprocess
                docker_cmd = ["docker", "run", "--rm"]

                if device.type == "cuda":
                    docker_cmd.append("--gpus")
                    docker_cmd.append("all")
                
                # Add env variables
                for k, v in environment.items():
                    docker_cmd += ["-e", f"{k}={v}"]
                    
                # Add volume mounts
                for host_path, mount in volumes.items():
                    bind_path = mount["bind"]
                    mode = mount.get("mode", "rw")
                    docker_cmd += ["-v", f"{host_path}:{bind_path}:{mode}"]
                    
                docker_cmd.append(image_name)
                
                if command:
                    # Use bash to run multiple commands
                    docker_cmd += ["bash", "-c", command]
                
                logger.debug(f"Running subprocess: {' '.join(docker_cmd)}")
                result = subprocess.run(docker_cmd, capture_output=True, text=True)
                
                return result.returncode, result.stdout + "\n" + result.stderr
                
        except Exception as e:
            logger.error(f"Error running Docker container: {e}")
            raise