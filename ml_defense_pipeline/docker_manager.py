"""
Docker operations for ML Defense Pipeline
"""
import logging
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


    def copy_global_requirements(self, context_dir):
        pipeline_requirements = os.path.abspath("scripts/pipeline_requirements.txt")
        target_path = os.path.join(context_dir, "pipeline_requirements.txt")

        if os.path.exists(pipeline_requirements):
            shutil.copy(pipeline_requirements, target_path)
            logger.info(f"Copied fallback requirements.txt to: {target_path}")
        else:
            logger.error(f"pipeline_requirements.txt not found at: {pipeline_requirements}")
            raise FileNotFoundError(f"pipeline_requirements.txt not found at: {pipeline_requirements}")
    
    def build_image(self, tool: Dict) -> str:
        """
        Build or pull Docker image for a tool

        Args:
            tool: Tool configuration dictionary

        Returns:
            Image name/tag to use
        """
        pipeline_dir = os.path.abspath(os.curdir)
        tool_name = tool["tool_name"]
        docker_info = tool.get("docker", {})
        image_name = None

        try:
            if docker_info.get("Dockerfile"):
                dockerfile_path = os.path.abspath(docker_info["Dockerfile"])
                context_dir = os.path.dirname(dockerfile_path)
                image_name = f"{tool_name.lower().replace(' ', '_')}_img"

                if not os.path.isdir(context_dir):
                    raise FileNotFoundError(f"Context directory does not exist: {context_dir}")
                if not os.path.exists(dockerfile_path):
                    raise FileNotFoundError(f"Dockerfile not found at: {dockerfile_path}")

                logger.info(f"Building Docker image for tool '{tool_name}' from {dockerfile_path}...")

                self.copy_global_requirements(context_dir)

                req_path = "pipeline_requirements.txt"
                print(f"Requirement path {req_path}")
                use_temp_dockerfile = False

                if os.path.exists(os.path.join(context_dir, "pipeline_requirements.txt")):
                    # Temporarily create a modified Dockerfile that includes pip install
                    with open(dockerfile_path, 'r') as df:
                        dockerfile_contents = df.read()

                    if "pipeline_requirements.txt" not in dockerfile_contents:
                        dockerfile_contents += f"\nCOPY {req_path} /app/.\nRUN pip install -r /app/pipeline_requirements.txt\n"
                        temp_dockerfile_path = os.path.join(context_dir, "Dockerfile.temp")
                        with open(temp_dockerfile_path, 'w') as tf:
                            tf.write(dockerfile_contents)
                        dockerfile_to_use = "Dockerfile.temp"
                        use_temp_dockerfile = True
                    else:
                        dockerfile_to_use = os.path.basename(dockerfile_path)
                else:
                    dockerfile_to_use = os.path.basename(dockerfile_path)

                # Build the image
                if self.client:
                    self.client.images.build(
                        path=context_dir,
                        dockerfile=dockerfile_to_use,
                        tag=image_name
                    )

                if use_temp_dockerfile:
                    os.remove(temp_dockerfile_path)

                logger.info(f"Successfully built image '{image_name}' for tool {tool_name}.")

            elif docker_info.get("image"):
                image_name = docker_info["image"]
                logger.info(f"Pulling Docker image for tool '{tool_name}': {image_name} ...")

                if self.client:
                    self.client.images.pull(image_name)
                else:
                    subprocess.run(["docker", "pull", image_name], check=True)

                logger.info(f"Successfully pulled image '{image_name}' for tool {tool_name}.")

            else:
                raise ValueError(f"No Docker image or Dockerfile specified for tool {tool_name}.")

        except Exception as e:
            logger.error(f"Failed to build/pull Docker image for tool {tool_name}: {e}")
            raise

        return image_name

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
            if self.client:
                container = self.client.containers.run(
                    image_name, 
                    command=command, 
                    environment=environment, 
                    volumes=volumes, 
                    detach=True,
                    tty=True,            
                    stdout=True,
                    stderr=True
                )
                
                # Wait for container to finish
                result = container.wait()
                exit_code = result.get("StatusCode", 0)
                logs = container.logs(stdout=True, stderr=True).decode('utf-8')
                container.remove()
                
                return exit_code, logs
                
            else:
                # Fallback to subprocess
                docker_cmd = ["docker", "run", "--rm"]
                
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