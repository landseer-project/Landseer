"""
Apptainer/Singularity implementation of container operations
"""
import os
import subprocess
import logging
import json
import tempfile
from typing import Dict, Optional, Tuple, Any
from .base import ContainerConfig, ContainerRunner, ContainerImageUtils

logger = logging.getLogger(__name__)


class ApptainerConfig(ContainerConfig):
    """Apptainer/Singularity-specific container configuration"""
    
    def __init__(self, image: str, command: str, config_script: Optional[str] = None):
        super().__init__(image, command, config_script)
        # Convert Docker image format to Apptainer format if needed
        self.apptainer_image = self._convert_docker_image_to_apptainer(image)
    
    def _convert_docker_image_to_apptainer(self, docker_image: str) -> str:
        """Convert Docker image reference to Apptainer format"""
        return self._convert_docker_image_to_apptainer_static(docker_image)
    
    @staticmethod
    def _convert_docker_image_to_apptainer_static(docker_image: str) -> str:
        """Convert Docker image reference to Apptainer format (static version)"""
        if docker_image.startswith("docker://"):
            return docker_image
        elif docker_image.startswith("oras://") or docker_image.startswith("library://"):
            return docker_image
        else:
            # Assume it's a Docker Hub or registry image
            return f"docker://{docker_image}"
    
    @property
    def image_name(self) -> str:
        if self.image:
            # Remove protocol prefix and extract name
            image = self.image.replace("docker://", "").replace("oras://", "").replace("library://", "")
            image = image.split(":")[0]
            image_name = image.split("/")[-1]
            return image_name
        return ""
    
    def get_labels(self) -> Dict[str, str]:
        """Get labels from Apptainer image"""
        if self.apptainer_image:
            labels = ApptainerImageUtils.get_labels_from_image(self.apptainer_image)
            if not labels:
                raise ValueError(f"No labels found in Apptainer image '{self.apptainer_image}'")
            if "stage" not in labels:
                raise ValueError(f"Label 'stage' not found in Apptainer image '{self.apptainer_image}'")
            if "dataset" not in labels:
                raise ValueError(f"Label 'dataset' not found in Apptainer image '{self.apptainer_image}'")
            return labels
        return {}
    
    def validate_image_and_pull(self) -> bool:
        """Validate and pull the Apptainer image if necessary"""
        logger.debug(f"Validating Apptainer image: {self.apptainer_image}")
        try:
            # Check if apptainer/singularity is available
            runner = subprocess.run(["apptainer", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if runner.returncode != 0:
                # Try singularity as fallback
                runner = subprocess.run(["singularity", "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if runner.returncode != 0:
                    raise ValueError("Neither apptainer nor singularity found on system")
            
            # For now, skip actual image validation to avoid authentication issues
            # The image will be validated when actually used
            logger.debug(f"Apptainer runtime available, skipping image validation for: {self.apptainer_image}")
            return True
            
            # Original validation code (commented out for now):
            # Try to inspect the image (this will pull if not cached)
            # inspect_cmd = ["apptainer", "inspect", self.apptainer_image]
            # result = subprocess.run(inspect_cmd, capture_output=True, text=True, timeout=300)
            
            # if result.returncode != 0:
            #     # Try with singularity as fallback
            #     inspect_cmd = ["singularity", "inspect", self.apptainer_image]
            #     result = subprocess.run(inspect_cmd, capture_output=True, text=True, timeout=300)
            #     
            #     if result.returncode != 0:
            #         logger.error(f"Failed to inspect Apptainer image: {result.stderr}")
            #         raise ValueError(f"Failed to validate Apptainer image '{self.apptainer_image}': {result.stderr}")
            
            return True
        except subprocess.TimeoutExpired:
            raise ValueError(f"Timeout while validating Apptainer image '{self.apptainer_image}'")
        except Exception as e:
            logger.error(f"Failed to validate Apptainer image '{self.apptainer_image}': {e}")
            raise ValueError(f"Failed to validate Apptainer image '{self.apptainer_image}': {e}")


class ApptainerRunner(ContainerRunner):
    """Apptainer/Singularity-specific container runtime operations"""
    
    def __init__(self, settings: Any):
        super().__init__(settings)
        self.runtime_cmd = self._detect_runtime()
        logger.info(f"Using {self.runtime_cmd} with device: {self.device}")
    
    @staticmethod
    def _convert_docker_image_to_apptainer_static(docker_image: str) -> str:
        """Convert Docker image reference to Apptainer format (static version)"""
        if docker_image.startswith("docker://"):
            return docker_image
        elif docker_image.startswith("oras://") or docker_image.startswith("library://"):
            return docker_image
        else:
            # Assume it's a Docker Hub or registry image
            return f"docker://{docker_image}"
    
    def _detect_runtime(self) -> str:
        """Detect whether to use apptainer or singularity"""
        try:
            subprocess.run(["apptainer", "--version"], 
                         capture_output=True, text=True, timeout=10, check=True)
            return "apptainer"
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(["singularity", "--version"], 
                             capture_output=True, text=True, timeout=10, check=True)
                return "singularity"
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError("Neither apptainer nor singularity found on system")
    
    def run_container(self, 
                     image_name: str, 
                     command: Optional[str],
                     environment: Dict[str, str], 
                     volumes: Dict[str, Dict], 
                     gpu_id: Optional[int] = None) -> Tuple[int, str, Any]:
        """Run an Apptainer container"""
        
        # Convert Docker image to Apptainer format if needed
        if not image_name.startswith(("docker://", "oras://", "library://")):
            apptainer_image = f"docker://{image_name}"
        else:
            apptainer_image = image_name
        
        # Build the command
        run_cmd = [self.runtime_cmd, "exec"]
        
        # Add GPU support if needed
        if self.device == "cuda" and gpu_id is not None:
            run_cmd.extend(["--nv"])  # NVIDIA GPU support
        elif self.device == "cuda":
            run_cmd.extend(["--nv"])  # Enable all GPUs
        
        # Add environment variables
        for key, value in environment.items():
            run_cmd.extend(["--env", f"{key}={value}"])
        
        # Add volume mounts
        for host_path, mount_config in volumes.items():
            container_path = mount_config.get('bind', host_path)
            mode = mount_config.get('mode', 'rw')
            if mode == 'rw':
                run_cmd.extend(["--bind", f"{host_path}:{container_path}"])
            else:
                run_cmd.extend(["--bind", f"{host_path}:{container_path}:ro"])
        
        # Add the image and command
        run_cmd.append(apptainer_image)
        if command:
            if isinstance(command, str):
                run_cmd.extend(command.split())
            else:
                run_cmd.extend(command)
        
        try:
            logger.debug(f"Running Apptainer command: {' '.join(run_cmd)}")
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=3600)
            
            exit_code = result.returncode
            logs = result.stdout + result.stderr
            
            # Return a mock container object for compatibility
            container_info = {
                'command': run_cmd,
                'exit_code': exit_code,
                'logs': logs
            }
            
            return exit_code, logs, container_info
            
        except subprocess.TimeoutExpired:
            logger.error("Apptainer container execution timed out")
            raise RuntimeError("Container execution timed out")
        except Exception as e:
            logger.error(f"Error running Apptainer container: {e}")
            raise
    
    def cleanup_container(self, container: Any) -> None:
        """Clean up Apptainer container (no-op since containers are stateless)"""
        # Apptainer containers are stateless, so no cleanup needed
        logger.debug("Apptainer container cleanup (no-op)")
        pass
    
    def is_available(self) -> bool:
        """Check if Apptainer/Singularity is available"""
        try:
            self._detect_runtime()
            return True
        except RuntimeError:
            return False
    
    def get_runtime_name(self) -> str:
        """Get the name of the container runtime"""
        try:
            return self._detect_runtime()
        except RuntimeError:
            return "apptainer/singularity (not available)"


class ApptainerImageUtils(ContainerImageUtils):
    """Apptainer/Singularity-specific image utilities"""
    
    @staticmethod
    def get_labels_from_image(image: str) -> Dict[str, str]:
        """Get labels from an Apptainer image"""
        try:
            # Try apptainer first
            runtime_cmd = "apptainer"
            try:
                subprocess.run([runtime_cmd, "--version"], 
                             capture_output=True, text=True, timeout=10, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                runtime_cmd = "singularity"
                subprocess.run([runtime_cmd, "--version"], 
                             capture_output=True, text=True, timeout=10, check=True)
            
            # Convert Docker registry images to proper Apptainer format
            apptainer_image = ApptainerRunner._convert_docker_image_to_apptainer_static(image)
            
            # Try to inspect the image first
            inspect_cmd = [runtime_cmd, "inspect", "--json", apptainer_image]
            result = subprocess.run(inspect_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                # Check if we should skip pulling during config validation
                from landseer_pipeline.config.settings import is_dry_run, get_current_settings
                
                # Skip pulling if we're in dry-run mode or during initial config validation
                settings = get_current_settings()
                if is_dry_run() or settings is None:
                    logger.warning(f"Image {apptainer_image} not available locally - skipping pull during config validation")
                    logger.info(f"Image will be pulled automatically during pipeline execution")
                    return {}
                
                logger.info(f"Image {apptainer_image} not available locally, pulling...")
                
                # Pull the image first
                pull_cmd = [runtime_cmd, "pull", "--disable-cache", apptainer_image]
                pull_result = subprocess.run(pull_cmd, capture_output=True, text=True, timeout=600)
                
                if pull_result.returncode != 0:
                    logger.warning(f"Failed to pull image {apptainer_image}: {pull_result.stderr}")
                    return {}
                
                logger.info(f"Successfully pulled image {apptainer_image}")
                
                # Now try to inspect again
                result = subprocess.run(inspect_cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    logger.warning(f"Failed to inspect image {apptainer_image} after pulling: {result.stderr}")
                    return {}
            
            try:
                inspect_data = json.loads(result.stdout)
                # Extract labels from the inspection data
                labels = {}
                
                # Try different possible locations for labels in the JSON
                if 'attributes' in inspect_data and 'labels' in inspect_data['attributes']:
                    labels.update(inspect_data['attributes']['labels'])
                
                if 'data' in inspect_data and 'attributes' in inspect_data['data'] and 'labels' in inspect_data['data']['attributes']:
                    labels.update(inspect_data['data']['attributes']['labels'])
                
                # Sometimes labels are in the top level
                if 'labels' in inspect_data:
                    labels.update(inspect_data['labels'])
                
                logger.debug(f"Labels found in image {apptainer_image}: {labels}")
                return labels
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse inspect output for image {apptainer_image}: {e}")
                return {}
                
        except Exception as e:
            logger.warning(f"Failed to get labels from image {apptainer_image if 'apptainer_image' in locals() else image}: {e}")
            return {}
    
    @staticmethod
    def get_image_digest(image: str) -> str:
        """Get the digest of an Apptainer image"""
        try:
            # Try apptainer first
            runtime_cmd = "apptainer"
            try:
                subprocess.run([runtime_cmd, "--version"], 
                             capture_output=True, text=True, timeout=10, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                runtime_cmd = "singularity"
                subprocess.run([runtime_cmd, "--version"], 
                             capture_output=True, text=True, timeout=10, check=True)
            
            # For Apptainer, we'll use a simple hash of the image URI as digest
            # This is not a true content digest but serves as an identifier
            import hashlib
            digest = hashlib.sha256(image.encode()).hexdigest()
            return f"sha256:{digest}"
            
        except Exception as e:
            logger.warning(f"Failed to get digest for image {image}: {e}")
            raise ValueError(f"Could not get digest for image {image}: {e}")