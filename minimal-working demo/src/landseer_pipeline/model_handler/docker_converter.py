"""
Docker-based Model Converter

This module provides model conversion functionality using Docker containers,
replacing the previous package-dependent conversion code. This approach:
1. Reduces dependency on ML packages in the main Landseer codebase
2. Provides consistent conversion behavior across different environments
3. Isolates conversion complexity in dedicated containers

The converter uses two Docker images:
- ghcr.io/landseer-project/model_converter_pytorch_to_other:v1
- ghcr.io/landseer-project/model_converter_other_to_pytorch:v1
"""

import logging
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ModelFormat(str, Enum):
    """Supported model formats for conversion."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    UNKNOWN = "unknown"


# Converter Docker images
CONVERTER_IMAGES = {
    "pytorch_to_other": "ghcr.io/landseer-project/model_converter_pytorch_to_other:v1",
    "other_to_pytorch": "ghcr.io/landseer-project/model_converter_other_to_pytorch:v1",
}


def detect_model_format(model_path: Union[str, Path]) -> ModelFormat:
    """
    Detect the format of a model file based on file extension.
    
    Args:
        model_path: Path to the model file or directory
        
    Returns:
        ModelFormat enum indicating the detected format
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.warning(f"Model path does not exist: {model_path}")
        return ModelFormat.UNKNOWN
    
    if model_path.is_file():
        extension = model_path.suffix.lower()
        
        if extension in ['.pth', '.pt']:
            return ModelFormat.PYTORCH
        elif extension in ['.onnx']:
            return ModelFormat.ONNX
        elif extension in ['.h5', '.hdf5', '.keras']:
            return ModelFormat.TENSORFLOW
        elif extension == '.pb':
            return ModelFormat.TENSORFLOW
    elif model_path.is_dir():
        # Check for SavedModel format
        if (model_path / 'saved_model.pb').exists():
            return ModelFormat.TENSORFLOW
        # Check for model files inside
        for ext, fmt in [('.pt', ModelFormat.PYTORCH), ('.pth', ModelFormat.PYTORCH),
                         ('.onnx', ModelFormat.ONNX), ('.h5', ModelFormat.TENSORFLOW)]:
            if list(model_path.glob(f'*{ext}')):
                return fmt
    
    return ModelFormat.UNKNOWN


class DockerModelConverter:
    """
    Model converter that uses Docker containers to convert between ML frameworks.
    
    This replaces the previous in-code conversion that required heavy ML package
    dependencies (tensorflow, onnx-tf, tf2onnx, etc.) in the main Landseer environment.
    """
    
    def __init__(self, container_runner=None, temp_dir: Optional[str] = None):
        """
        Initialize the Docker model converter.
        
        Args:
            container_runner: Optional container runner instance (DockerRunner/ContainerManager)
            temp_dir: Temporary directory for conversion operations
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._container_runner = container_runner
        self._docker_client = None
    
    @property
    def docker_client(self):
        """Lazy-load Docker client."""
        if self._docker_client is None:
            try:
                import docker
                self._docker_client = docker.from_env()
            except Exception as e:
                logger.error(f"Failed to initialize Docker client: {e}")
                raise RuntimeError("Docker is required for model conversion") from e
        return self._docker_client
    
    def convert_model(
        self,
        source_path: Union[str, Path],
        target_path: Union[str, Path],
        target_format: ModelFormat,
        source_format: Optional[ModelFormat] = None,
        input_shape: Optional[List[int]] = None,
        model_script: Optional[str] = None,
        opset_version: int = 11
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Convert a model from one format to another using Docker containers.
        
        Args:
            source_path: Path to the source model
            target_path: Path where converted model should be saved
            target_format: Target model format (pytorch, tensorflow, onnx)
            source_format: Source model format (auto-detected if None)
            input_shape: Input shape for the model (e.g., [1, 3, 32, 32])
            model_script: Path to model architecture script (for PyTorch state_dict)
            opset_version: ONNX opset version for conversions
            
        Returns:
            Tuple of (success: bool, metadata: dict)
        """
        source_path = Path(source_path)
        target_path = Path(target_path)
        
        # Auto-detect source format if not provided
        if source_format is None:
            source_format = detect_model_format(source_path)
            logger.info(f"Auto-detected source format: {source_format}")
        
        if source_format == ModelFormat.UNKNOWN:
            logger.error(f"Cannot detect format of source model: {source_path}")
            return False, {"error": "Unknown source format"}
        
        # Check if conversion is needed
        if source_format == target_format:
            logger.info(f"No conversion needed, copying {source_path} to {target_path}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.is_dir():
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, target_path)
            return True, {"source_format": source_format.value, "target_format": target_format.value, "converted": False}
        
        # Determine which converter image to use
        if source_format == ModelFormat.PYTORCH:
            converter_image = CONVERTER_IMAGES["pytorch_to_other"]
            target_format_arg = "onnx" if target_format == ModelFormat.ONNX else "tensorflow"
        else:
            converter_image = CONVERTER_IMAGES["other_to_pytorch"]
            target_format_arg = "pytorch"
        
        # Set default input shape for CIFAR-10 if not provided
        if input_shape is None:
            input_shape = [1, 3, 32, 32]
            logger.warning(f"No input shape provided, using default: {input_shape}")
        
        # Prepare conversion
        logger.info(f"Converting model: {source_format.value} -> {target_format.value}")
        logger.info(f"Using converter image: {converter_image}")
        
        try:
            success, metadata = self._run_conversion_container(
                source_path=source_path,
                target_path=target_path,
                converter_image=converter_image,
                source_format=source_format,
                target_format=target_format,
                target_format_arg=target_format_arg,
                input_shape=input_shape,
                model_script=model_script,
                opset_version=opset_version
            )
            
            return success, metadata
            
        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            return False, {"error": str(e)}
    
    def _run_conversion_container(
        self,
        source_path: Path,
        target_path: Path,
        converter_image: str,
        source_format: ModelFormat,
        target_format: ModelFormat,
        target_format_arg: str,
        input_shape: List[int],
        model_script: Optional[str],
        opset_version: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Run the conversion container."""
        
        # Create temporary directories for container I/O
        with tempfile.TemporaryDirectory(prefix="landseer_convert_") as work_dir:
            input_dir = Path(work_dir) / "input"
            output_dir = Path(work_dir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            # Copy source model to input directory
            if source_path.is_dir():
                shutil.copytree(source_path, input_dir / source_path.name)
            else:
                shutil.copy2(source_path, input_dir / source_path.name)
            
            # Copy model script if provided
            script_mount = None
            if model_script and os.path.exists(model_script):
                script_dest = input_dir / "config_model.py"
                shutil.copy2(model_script, script_dest)
                script_mount = "/input/config_model.py"
            
            # Build command arguments
            input_shape_str = ",".join(str(x) for x in input_shape)
            
            if source_format == ModelFormat.PYTORCH:
                command = [
                    "--input", "/input",
                    "--output", "/output",
                    "--target-format", target_format_arg,
                    "--input-shape", input_shape_str,
                    "--opset-version", str(opset_version)
                ]
                if script_mount:
                    command.extend(["--model-script", script_mount])
            else:
                command = [
                    "--input", "/input",
                    "--output", "/output",
                    "--source-format", source_format.value if source_format != ModelFormat.UNKNOWN else "auto"
                ]
                if script_mount:
                    command.extend(["--model-script", script_mount])
            
            # Volume mounts
            volumes = {
                str(input_dir): {"bind": "/input", "mode": "ro"},
                str(output_dir): {"bind": "/output", "mode": "rw"}
            }
            
            # Run the container
            try:
                container = self.docker_client.containers.run(
                    converter_image,
                    command=command,
                    volumes=volumes,
                    detach=True,
                    remove=False,
                    stdout=True,
                    stderr=True
                )
                
                # Wait for completion
                result = container.wait()
                exit_code = result.get("StatusCode", 0)
                logs = container.logs(stdout=True, stderr=True).decode('utf-8')
                
                # Cleanup container
                container.remove()
                
                if exit_code != 0:
                    logger.error(f"Conversion container failed with exit code {exit_code}")
                    logger.error(f"Container logs:\n{logs}")
                    return False, {"error": f"Container exit code: {exit_code}", "logs": logs}
                
                logger.debug(f"Container logs:\n{logs}")
                
                # Check for output files
                output_files = list(output_dir.glob("*"))
                if not output_files:
                    logger.error("No output files produced by conversion")
                    return False, {"error": "No output files produced"}
                
                # Copy output to target path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Find the converted model file
                model_extensions = {'.pt', '.pth', '.onnx', '.h5', '.keras'}
                model_files = [f for f in output_files if f.suffix.lower() in model_extensions]
                
                if model_files:
                    # Copy the model file
                    if target_path.suffix:
                        # Target is a file path
                        shutil.copy2(model_files[0], target_path)
                    else:
                        # Target is a directory
                        target_path.mkdir(parents=True, exist_ok=True)
                        for f in output_files:
                            shutil.copy2(f, target_path / f.name)
                else:
                    # Copy all output files
                    if target_path.suffix:
                        target_path = target_path.parent
                    target_path.mkdir(parents=True, exist_ok=True)
                    for f in output_files:
                        if f.is_file():
                            shutil.copy2(f, target_path / f.name)
                
                # Load metadata if available
                metadata = {
                    "source_format": source_format.value,
                    "target_format": target_format.value,
                    "converted": True,
                    "input_shape": input_shape
                }
                
                metadata_file = output_dir / "conversion_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata.update(json.load(f))
                    except Exception as e:
                        logger.warning(f"Could not load conversion metadata: {e}")
                
                logger.info(f"Model conversion successful: {target_path}")
                return True, metadata
                
            except Exception as e:
                logger.error(f"Failed to run conversion container: {e}")
                return False, {"error": str(e)}
    
    def can_convert(self, source_format: ModelFormat, target_format: ModelFormat) -> bool:
        """
        Check if conversion is possible between two formats.
        
        Args:
            source_format: Source model format
            target_format: Target model format
            
        Returns:
            True if conversion is possible
        """
        if source_format == target_format:
            return True
        
        # PyTorch -> ONNX/TensorFlow
        if source_format == ModelFormat.PYTORCH:
            return target_format in [ModelFormat.ONNX, ModelFormat.TENSORFLOW]
        
        # ONNX/TensorFlow -> PyTorch
        if target_format == ModelFormat.PYTORCH:
            return source_format in [ModelFormat.ONNX, ModelFormat.TENSORFLOW]
        
        # ONNX <-> TensorFlow (via PyTorch as intermediate)
        if source_format in [ModelFormat.ONNX, ModelFormat.TENSORFLOW]:
            return target_format in [ModelFormat.ONNX, ModelFormat.TENSORFLOW, ModelFormat.PYTORCH]
        
        return False
    
    def get_supported_conversions(self) -> Dict[Tuple[str, str], bool]:
        """
        Get a dictionary of all supported conversion pairs.
        
        Returns:
            Dictionary mapping (source_format, target_format) to availability
        """
        supported = {}
        for source in ModelFormat:
            if source == ModelFormat.UNKNOWN:
                continue
            for target in ModelFormat:
                if target == ModelFormat.UNKNOWN:
                    continue
                supported[(source.value, target.value)] = self.can_convert(source, target)
        return supported


# Convenience function for easy import
def get_docker_model_converter(container_runner=None, temp_dir: Optional[str] = None) -> DockerModelConverter:
    """Get a Docker model converter instance."""
    return DockerModelConverter(container_runner=container_runner, temp_dir=temp_dir)
