"""
Model Format Manager for Multi-Framework Pipeline

This module manages model format conversions and framework compatibility across pipeline stages.
It handles:
1. Automatic framework detection and model loading
2. Conversion via Docker containers (for inter-tool conversions)
3. Framework-specific model instantiation for tools
4. Post-processing hooks for pipeline stages

Model conversion between frameworks now uses Docker containers:
- ghcr.io/landseer-project/model_converter_pytorch_to_other:v1
- ghcr.io/landseer-project/model_converter_other_to_pytorch:v1

This reduces dependency on heavy ML packages in the main Landseer environment.
Note: torch is still required for model evaluation since models are loaded in-process.
"""

import logging
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import importlib.util

from .converter import ONNXConverter, ModelMetadata, get_model_framework
from ..container_handler.docker_inspector import get_docker_inspector

logger = logging.getLogger(__name__)


class ModelFormatManager:
    """Manages model format conversions across pipeline stages"""
    
    def __init__(self, temp_dir: Optional[str] = None, input_shape: Optional[Tuple[int, ...]] = None):
        self.temp_dir = temp_dir or "/tmp/landseer_models"
        self.converter = ONNXConverter(temp_dir=self.temp_dir)
        self.metadata_cache = {}
        self.docker_inspector = get_docker_inspector()
        self.input_shape = input_shape
        
        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Standard ONNX storage path in pipeline
        self.onnx_storage_name = "model.onnx"
        self.metadata_storage_name = "model_metadata.json"
    
    def detect_tool_framework_requirement(self, tool_config: Union[Dict[str, Any], Any]) -> Optional[str]:
        """
        Detect what framework a tool requires based on Docker image labels.
        Returns 'pytorch', 'tensorflow', 'onnx', or None if auto-detect
        
        Args:
            tool_config: Either a ToolConfig object or a dictionary with tool configuration
        """
        # Handle both ToolConfig objects and dictionaries
        if hasattr(tool_config, 'container'):
            # ToolConfig object (new schema)
            image_name = tool_config.container.image
            tool_name = tool_config.name
        elif hasattr(tool_config, 'docker'):
            # Legacy ToolConfig object with 'docker' attribute
            image_name = tool_config.docker.get('image', '')
            tool_name = getattr(tool_config, 'name', 'unknown')
        elif isinstance(tool_config, dict):
            # Dictionary format (backward compatibility)
            docker_config = tool_config.get('docker', {}) or tool_config.get('container', {})
            image_name = docker_config.get('image', '')
            tool_name = tool_config.get('name', 'unknown')
        else:
            logger.warning(f"Unsupported tool_config type: {type(tool_config)}")
            return None
        
        if not image_name:
            logger.warning("No container image specified in tool config")
            return None
        
        # Use Docker image inspector to detect framework
        try:
            framework = self.docker_inspector.detect_framework_from_image(image_name)
            if framework:
                logger.debug(f"Tool '{tool_name}': detected framework={framework} from image")
                return framework
            else:
                logger.warning(f"No framework detected for container image '{image_name}', will use auto-detection")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to inspect container image '{image_name}': {e}")            
            return None
    
    def prepare_model_for_tool(self, model_path: str, tool_config: Union[Dict[str, Any], Any],
                              model_script_path: Optional[str] = None) -> Tuple[str, Any]:
        """
        Prepare model in the format required by a specific tool.
        
        Args:
            model_path: Path to current model file or directory containing model
            tool_config: Tool configuration dictionary
            model_script_path: Path to model architecture script
        
        Returns:
            Tuple of (prepared_model_path, updated_tool_config)
        """
        # If model_path is a directory, find the actual model file inside
        resolved_model_path = model_path
        if os.path.isdir(model_path):
            resolved_model_path = self._get_input_model_path(model_path)
            if not resolved_model_path:
                logger.warning(f"No model file found in directory: {model_path}")
                return model_path, tool_config
        
        required_framework = self.detect_tool_framework_requirement(tool_config)
        current_framework = get_model_framework(resolved_model_path)
        
        # Only log when conversion is actually needed
        if required_framework and required_framework != current_framework:
            logger.info(f"Converting model: {current_framework} -> {required_framework}")
        else:
            logger.debug(f"No conversion needed for {current_framework} model")
        
        # If no specific framework required or already in correct format
        if required_framework is None or required_framework == current_framework:
            return resolved_model_path, tool_config
        
        # Convert model to required framework
        output_path = self._get_conversion_output_path(resolved_model_path, required_framework)
        
        success, metadata = self.converter.convert_model(
            resolved_model_path, output_path, 
            source_framework=current_framework,
            target_framework=required_framework,
            model_script_path=model_script_path
        )
        
        if not success:
            logger.error(f"Failed to convert model from {current_framework} to {required_framework}")
            # Fall back to original model
            return resolved_model_path, tool_config
        
        # Update tool config to point to converted model
        updated_config = tool_config.copy()
        if 'docker' in updated_config:
            updated_config['docker'] = updated_config['docker'].copy()
            # Update any model path references in docker config
            if 'volumes' in updated_config['docker']:
                # Handle volume mounts that might reference the model
                pass
        
        return output_path, updated_config
    
    def standardize_model_output(self, model_path: str, output_dir: str,
                                model_script_path: Optional[str] = None) -> str:
        """
        Convert any model output to ONNX format for standardized storage.
        
        Args:
            model_path: Path to model file to standardize
            output_dir: Directory to store standardized model
            model_script_path: Path to model architecture script
        
        Returns:
            Path to standardized ONNX model
        """
        current_framework = get_model_framework(model_path)
        onnx_output_path = os.path.join(output_dir, self.onnx_storage_name)
        metadata_output_path = os.path.join(output_dir, self.metadata_storage_name)
        
        # Use the input shape provided during initialization
        input_shape = self.input_shape
        if input_shape:
            logger.debug(f"Using input shape {input_shape} for ONNX conversion")
        else:
            logger.warning("No input shape available, PyTorch models may fail to convert to ONNX")
        
        if current_framework == "onnx":
            # Already in ONNX format, just copy
            shutil.copy2(model_path, onnx_output_path)
            success = True
            metadata = ModelMetadata("onnx", input_shape)
        else:
            # Convert to ONNX with input shape
            success, metadata = self.converter.convert_model(
                model_path, onnx_output_path,
                target_framework="onnx",
                model_script_path=model_script_path,
                input_shape=input_shape
            )
        
        if success:
            # Save metadata
            with open(metadata_output_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            logger.debug(f"Model standardized to ONNX: {output_dir}")
            return output_dir
        else:
            logger.error(f"Failed to standardize model to ONNX format")
            # Fall back to copying original file
            return None
    
    def load_model_with_framework(self, model_path: str, framework: str,
                                 model_script_path: Optional[str] = None) -> Any:
        """
        Load a model using a specific framework.
        
        Args:
            model_path: Path to model file
            framework: Target framework ('pytorch', 'tensorflow', 'onnx')
            model_script_path: Path to model architecture script
        
        Returns:
            Loaded model object
        """
        current_framework = get_model_framework(model_path)
        
        if current_framework == framework:
            # Direct loading
            if framework == "pytorch":
                import torch
                if model_script_path:
                    # Load architecture and weights
                    spec = importlib.util.spec_from_file_location("model_config", model_script_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    model = config_module.config()
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    return model
                else:
                    return torch.load(model_path, map_location='cpu')
            
            elif framework == "tensorflow":
                import tensorflow as tf
                return tf.keras.models.load_model(model_path)
            
            elif framework == "onnx":
                import onnx
                return onnx.load(model_path)
        
        else:
            # Need conversion first
            temp_path = self._get_conversion_output_path(model_path, framework)
            success, _ = self.converter.convert_model(
                model_path, temp_path,
                source_framework=current_framework,
                target_framework=framework,
                model_script_path=model_script_path
            )
            
            if success:
                return self.load_model_with_framework(temp_path, framework, model_script_path)
            else:
                raise RuntimeError(f"Failed to convert and load model in {framework} format")
    
    def get_pipeline_stage_hooks(self) -> Dict[str, callable]:
        """
        Get post-processing hooks for each pipeline stage.
        
        Returns:
            Dictionary mapping stage names to hook functions
        """
        return {
            'pre_training': self._post_pre_training_hook,
            'during_training': self._post_during_training_hook,
            'post_training': self._post_post_training_hook,
            'deployment': self._post_deployment_hook
        }
    
    def _post_pre_training_hook(self, stage_output_dir: str, tool_outputs: List[str],
                               model_script_path: Optional[str] = None) -> Dict[str, Any]:
        """Post-processing hook for pre-training stage"""
        logger.info("Running post-pre-training model format processing")
        
        # Find model files in outputs
        model_files = self._find_model_files(tool_outputs)
        standardized_models = []
        
        for model_file in model_files:
            standardized_path = self.standardize_model_output(
                model_file, stage_output_dir, model_script_path
            )
            standardized_models.append(standardized_path)
        
        return {
            'standardized_models': standardized_models,
            'stage': 'pre_training'
        }
    
    def _post_during_training_hook(self, stage_output_dir: str, tool_outputs: List[str],
                                  model_script_path: Optional[str] = None) -> Dict[str, Any]:
        """Post-processing hook for during-training stage"""
        logger.info("Running post-during-training model format processing")
        
        model_files = self._find_model_files(tool_outputs)
        standardized_models = []
        
        for model_file in model_files:
            standardized_path = self.standardize_model_output(
                model_file, stage_output_dir, model_script_path
            )
            standardized_models.append(standardized_path)
        
        return {
            'standardized_models': standardized_models,
            'stage': 'during_training'
        }
    
    def _post_post_training_hook(self, stage_output_dir: str, tool_outputs: List[str],
                                model_script_path: Optional[str] = None) -> Dict[str, Any]:
        """Post-processing hook for post-training stage"""
        logger.info("Running post-post-training model format processing")
        
        model_files = self._find_model_files(tool_outputs)
        standardized_models = []
        
        for model_file in model_files:
            standardized_path = self.standardize_model_output(
                model_file, stage_output_dir, model_script_path
            )
            standardized_models.append(standardized_path)
        
        return {
            'standardized_models': standardized_models,
            'stage': 'post_training'
        }
    
    def _post_deployment_hook(self, stage_output_dir: str, tool_outputs: List[str],
                             model_script_path: Optional[str] = None) -> Dict[str, Any]:
        """Post-processing hook for deployment stage"""
        logger.info("Running post-deployment model format processing")
        
        # Deployment stage might not modify models, but standardize any outputs
        model_files = self._find_model_files(tool_outputs)
        standardized_models = []
        
        for model_file in model_files:
            standardized_path = self.standardize_model_output(
                model_file, stage_output_dir, model_script_path
            )
            standardized_models.append(standardized_path)
        
        return {
            'standardized_models': standardized_models,
            'stage': 'deployment'
        }
    
    def _find_model_files(self, output_paths: List[str]) -> List[str]:
        """Find model files in tool output paths"""
        model_extensions = {'.pt', '.pth', '.h5', '.keras', '.onnx', '.pb'}
        model_files = []
        
        for output_path in output_paths:
            path = Path(output_path)
            if path.is_file() and path.suffix.lower() in model_extensions:
                model_files.append(str(path))
            elif path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in model_extensions:
                        model_files.append(str(file_path))
        
        return model_files
    
    def _get_input_model_path(self, tool_input_dir: str) -> Optional[str]:
        """Get model input path from tool input directory"""
        possible_names = ["model.onnx", "model.pt", "model.pth", "model.h5", "model.keras"]
        
        for name in possible_names:
            candidate_path = os.path.join(tool_input_dir, name)
            if os.path.exists(candidate_path):
                return candidate_path
        
        return None
    
    def _get_conversion_output_path(self, input_path: str, target_framework: str) -> str:
        """Generate output path for converted model"""
        input_path_obj = Path(input_path)
        base_name = input_path_obj.stem
        
        extension_map = {
            'pytorch': '.pt',
            'tensorflow': '.h5',
            'onnx': '.onnx'
        }
        
        extension = extension_map.get(target_framework, '.bin')
        output_filename = f"{base_name}_converted{extension}"
        
        return os.path.join(self.temp_dir, output_filename)
    
    def cleanup_temp_files(self):
        """Clean up temporary conversion files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
            logger.info("Cleaned up temporary model conversion files")
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")


# Global instance for easy access
_global_manager = None

def get_model_format_manager(input_shape) -> ModelFormatManager:
    """Get global model format manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = ModelFormatManager(input_shape=input_shape)
    return _global_manager
