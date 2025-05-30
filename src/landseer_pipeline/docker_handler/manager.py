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
        try:
            if self.client:
                device_requests = None
                if self.device == "cuda":
                    device_requests = [docker.types.DeviceRequest(
                        count=-1, capabilities=[["gpu"]])]

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

                return exit_code, logs

            else:
                # Fallback to subprocess
                docker_cmd = ["docker", "run", "--rm"]

                if self.device == "cuda":
                    docker_cmd.append("--gpus")
                    docker_cmd.append("all")

                # Add env variables
                for k, v in environment.items():
                    docker_cmd += ["-e", f"{k}={v}"]

                for host_path, mount in volumes.items():
                    bind_path = mount["bind"]
                    mode = mount.get("mode", "rw")
                    docker_cmd += ["-v", f"{host_path}:{bind_path}:{mode}"]

                docker_cmd.append(image_name)

                if command:
                    docker_cmd += ["bash", "-c", command]

                logger.debug(f"Running subprocess: {' '.join(docker_cmd)}")
                result = subprocess.run(
                    docker_cmd, capture_output=True, text=True)

                return result.returncode, result.stdout + "\n" + result.stderr

        except Exception as e:
            logger.error(f"Error running Docker container: {e}")
            raise

    @property
    def config(self) -> Dict[str, str]:
        return self.stager.config