"""
Model Format Manager for Multi-Framework Pipeline

This module manages model format conversions and framework compatibility across pipeline stages.
It handles:
1. Automatic framework detection and model loading
2. Conversion to/from ONNX as the intermediate format
3. Framework-specific model instantiation for tools
4. Post-processing hooks for pipeline stages
"""

import logging
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import importlib.util

from .onnx_converter import ONNXConverter, ModelMetadata, get_model_framework
from .docker_inspector import get_docker_inspector

logger = logging.getLogger(__name__)


class ModelFormatManager:
    """Manages model format conversions across pipeline stages"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or "/tmp/landseer_models"
        self.converter = ONNXConverter(temp_dir=self.temp_dir)
        self.metadata_cache = {}
        self.docker_inspector = get_docker_inspector()
        
        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Standard ONNX storage path in pipeline
        self.onnx_storage_name = "model.onnx"
        self.metadata_storage_name = "model_metadata.json"
    
    def detect_tool_framework_requirement(self, tool_config: Dict[str, Any]) -> Optional[str]:
        """
        Detect what framework a tool requires based on Docker image labels.
        Returns 'pytorch', 'tensorflow', 'onnx', or None if auto-detect
        """
        # Get Docker image name from tool config
        docker_config = tool_config.get('docker', {})
        image_name = docker_config.get('image', '')
        
        if not image_name:
            logger.warning("No Docker image specified in tool config")
            return None
        
        # Use Docker image inspector to detect framework
        try:
            framework = self.docker_inspector.detect_framework_from_image(image_name)
            if framework:
                logger.info(f"Detected framework '{framework}' for tool '{tool_config.get('name', 'unknown')}' from Docker image '{image_name}'")
                return framework
            else:
                logger.debug(f"No framework detected for Docker image '{image_name}', will use auto-detection")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to inspect Docker image '{image_name}': {e}")            
            return None
    
    def prepare_model_for_tool(self, model_path: str, tool_config: Dict[str, Any],
                              model_script_path: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare model in the format required by a specific tool.
        
        Args:
            model_path: Path to current model file
            tool_config: Tool configuration dictionary
            model_script_path: Path to model architecture script
        
        Returns:
            Tuple of (prepared_model_path, updated_tool_config)
        """
        required_framework = self.detect_tool_framework_requirement(tool_config)
        current_framework = get_model_framework(model_path)
        
        logger.info(f"Tool requires: {required_framework}, Current model: {current_framework}")
        
        # If no specific framework required or already in correct format
        if required_framework is None or required_framework == current_framework:
            return model_path, tool_config
        
        # Convert model to required framework
        output_path = self._get_conversion_output_path(model_path, required_framework)
        
        success, metadata = self.converter.convert_model(
            model_path, output_path, 
            source_framework=current_framework,
            target_framework=required_framework,
            model_script_path=model_script_path
        )
        
        if not success:
            logger.error(f"Failed to convert model from {current_framework} to {required_framework}")
            # Fall back to original model
            return model_path, tool_config
        
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
        
        if current_framework == "onnx":
            # Already in ONNX format, just copy
            shutil.copy2(model_path, onnx_output_path)
            success = True
            metadata = ModelMetadata("onnx")
        else:
            # Convert to ONNX
            success, metadata = self.converter.convert_model(
                model_path, onnx_output_path,
                source_framework=current_framework,
                target_framework="onnx",
                model_script_path=model_script_path
            )
        
        if success:
            # Save metadata
            with open(metadata_output_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            logger.info(f"Standardized model to ONNX: {onnx_output_path}")
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

def get_model_format_manager() -> ModelFormatManager:
    """Get global model format manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = ModelFormatManager()
    return _global_manager