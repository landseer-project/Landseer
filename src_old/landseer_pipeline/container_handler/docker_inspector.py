"""
Docker Image Label Inspector for Framework Detection

This module provides utilities to inspect Docker images and extract framework
information from image labels. It supports detecting PyTorch, TensorFlow, and
other ML framework requirements from Docker image metadata.
"""

import logging
import subprocess
import json
from typing import Dict, Optional, Any
import re

logger = logging.getLogger(__name__)


class DockerImageInspector:
    """Inspector for Docker image labels and metadata"""
    
    # Standard labels used to indicate ML frameworks
    FRAMEWORK_LABELS = {
        'ml.framework': 'Framework name (pytorch, tensorflow, onnx, etc.)',
        'ml.framework.version': 'Framework version',
        'landseer.framework': 'Landseer-specific framework label',
        'framework': 'Generic framework label'
    }
    
    # Patterns to detect frameworks from various sources
    PYTORCH_PATTERNS = [
        r'pytorch', r'torch', r'torchvision', r'torchaudio'
    ]
    
    TENSORFLOW_PATTERNS = [
        r'tensorflow', r'tf-nightly', r'tf2', r'keras'
    ]
    
    ONNX_PATTERNS = [
        r'onnx', r'onnxruntime'
    ]
    
    def __init__(self):
        self.cache = {}  # Cache inspection results
    
    def inspect_image_labels(self, image_name: str) -> Dict[str, Any]:
        """
        Inspect Docker image labels and metadata.
        
        Args:
            image_name: Name of the Docker image to inspect
        
        Returns:
            Dictionary containing image metadata and labels
        """
        if image_name in self.cache:
            return self.cache[image_name]
        
        try:
            # Use docker inspect to get image metadata
            cmd = ['docker', 'inspect', image_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.warning(f"Failed to inspect Docker image {image_name}: {result.stderr}")
                return {}
            
            # Parse JSON output
            image_data = json.loads(result.stdout)
            if not image_data:
                return {}
            
            metadata = image_data[0]
            
            # Extract relevant information
            inspection_result = {
                'labels': metadata.get('Config', {}).get('Labels') or {},
                'env': metadata.get('Config', {}).get('Env') or [],
                'cmd': metadata.get('Config', {}).get('Cmd') or [],
                'entrypoint': metadata.get('Config', {}).get('Entrypoint') or [],
                'image_name': image_name,
                'created': metadata.get('Created', ''),
                'architecture': metadata.get('Architecture', ''),
                'os': metadata.get('Os', '')
            }
            
            # Cache the result
            self.cache[image_name] = inspection_result
            
            return inspection_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout while inspecting Docker image {image_name}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Docker inspect output for {image_name}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error inspecting Docker image {image_name}: {e}")
            return {}
    
    def detect_framework_from_image(self, image_name: str) -> Optional[str]:
        """
        Detect ML framework from Docker image.
        
        Args:
            image_name: Name of the Docker image
        
        Returns:
            Framework name ('pytorch', 'tensorflow', 'onnx') or None
        """
        metadata = self.inspect_image_labels(image_name)
        if not metadata:
            return None
        
        # 1. Check explicit framework labels
        framework = self._check_framework_labels(metadata.get('labels', {}))
        if framework:
            return framework
        
        # 2. Check environment variables
        framework = self._check_environment_variables(metadata.get('env', []))
        if framework:
            return framework
        
        # 3. Check image name patterns
        framework = self._check_image_name_patterns(image_name)
        if framework:
            return framework
        
        # 4. Check command and entrypoint
        commands = metadata.get('cmd', []) + metadata.get('entrypoint', [])
        framework = self._check_command_patterns(commands)
        if framework:
            return framework        
        return None
    
    def _check_framework_labels(self, labels: Dict[str, str]) -> Optional[str]:
        """Check Docker labels for framework information"""
        if not labels:
            return None
        
        # Check standard framework labels
        for label_key in self.FRAMEWORK_LABELS:
            if label_key in labels:
                framework_value = labels[label_key].lower().strip()
                logger.info(f"Found framework label {label_key}={framework_value}")
                
                # Normalize framework names
                if any(pattern in framework_value for pattern in ['pytorch', 'torch']):
                    return 'pytorch'
                elif any(pattern in framework_value for pattern in ['tensorflow', 'tf']):
                    return 'tensorflow'
                elif 'onnx' in framework_value:
                    return 'onnx'
                else:
                    # Return the raw framework value if it's a known framework
                    known_frameworks = ['pytorch', 'tensorflow', 'onnx', 'sklearn', 'xgboost']
                    if framework_value in known_frameworks:
                        return framework_value
        
        return None
    
    def _check_environment_variables(self, env_vars: list) -> Optional[str]:
        """Check environment variables for framework clues"""
        env_text = ' '.join(env_vars).lower()
        
        if any(re.search(pattern, env_text) for pattern in self.PYTORCH_PATTERNS):
            return 'pytorch'
        elif any(re.search(pattern, env_text) for pattern in self.TENSORFLOW_PATTERNS):
            return 'tensorflow'
        elif any(re.search(pattern, env_text) for pattern in self.ONNX_PATTERNS):
            return 'onnx'
        
        return None
    
    def _check_image_name_patterns(self, image_name: str) -> Optional[str]:
        """Check image name for framework patterns"""
        image_lower = image_name.lower()
        
        # Common official images
        if any(pattern in image_lower for pattern in ['pytorch/pytorch', 'pytorch:']):
            return 'pytorch'
        elif any(pattern in image_lower for pattern in ['tensorflow/tensorflow', 'tensorflow:']):
            return 'tensorflow'
        elif 'onnx' in image_lower:
            return 'onnx'
        
        # Check for framework names in image path/tag
        if any(re.search(pattern, image_lower) for pattern in self.PYTORCH_PATTERNS):
            return 'pytorch'
        elif any(re.search(pattern, image_lower) for pattern in self.TENSORFLOW_PATTERNS):
            return 'tensorflow'
        elif any(re.search(pattern, image_lower) for pattern in self.ONNX_PATTERNS):
            return 'onnx'
        
        return None
    
    def _check_command_patterns(self, commands: list) -> Optional[str]:
        """Check command/entrypoint for framework clues"""
        if not commands:
            return None
        
        cmd_text = ' '.join(str(cmd) for cmd in commands).lower()
        
        if any(re.search(pattern, cmd_text) for pattern in self.PYTORCH_PATTERNS):
            return 'pytorch'
        elif any(re.search(pattern, cmd_text) for pattern in self.TENSORFLOW_PATTERNS):
            return 'tensorflow'
        elif any(re.search(pattern, cmd_text) for pattern in self.ONNX_PATTERNS):
            return 'onnx'
        
        return None
    
    def get_detailed_framework_info(self, image_name: str) -> Dict[str, Any]:
        """
        Get detailed framework information from Docker image.
        
        Args:
            image_name: Name of the Docker image
        
        Returns:
            Dictionary with detailed framework information
        """
        metadata = self.inspect_image_labels(image_name)
        framework = self.detect_framework_from_image(image_name)
        
        return {
            'framework': framework,
            'image_name': image_name,
            'labels': metadata.get('labels', {}),
            'framework_labels': {
                k: v for k, v in metadata.get('labels', {}).items() 
                if any(fw_key in k.lower() for fw_key in ['framework', 'ml', 'ai'])
            },
            'detection_method': self._get_detection_method(image_name, metadata),
            'confidence': self._calculate_confidence(image_name, metadata, framework)
        }
    
    def _get_detection_method(self, image_name: str, metadata: Dict) -> str:
        """Determine how the framework was detected"""
        labels = metadata.get('labels', {})
        
        # Check if detected via labels
        for label_key in self.FRAMEWORK_LABELS:
            if label_key in labels:
                return 'docker_labels'
        
        # Check if detected via environment
        env_vars = metadata.get('env', [])
        env_text = ' '.join(env_vars).lower()
        if any(re.search(pattern, env_text) for pattern in 
               self.PYTORCH_PATTERNS + self.TENSORFLOW_PATTERNS + self.ONNX_PATTERNS):
            return 'environment_variables'
        
        # Check if detected via image name
        if any(re.search(pattern, image_name.lower()) for pattern in 
               self.PYTORCH_PATTERNS + self.TENSORFLOW_PATTERNS + self.ONNX_PATTERNS):
            return 'image_name_pattern'
        
        return 'unknown'
    
    def _calculate_confidence(self, image_name: str, metadata: Dict, framework: Optional[str]) -> float:
        """Calculate confidence score for framework detection"""
        if not framework:
            return 0.0
        
        confidence = 0.0
        
        # High confidence if explicit label exists
        labels = metadata.get('labels', {})
        if any(label_key in labels for label_key in self.FRAMEWORK_LABELS):
            confidence += 0.8
        
        # Medium confidence for image name patterns
        if any(re.search(pattern, image_name.lower()) for pattern in 
               self.PYTORCH_PATTERNS + self.TENSORFLOW_PATTERNS + self.ONNX_PATTERNS):
            confidence += 0.5
        
        # Low confidence for environment variables
        env_vars = metadata.get('env', [])
        env_text = ' '.join(env_vars).lower()
        if any(re.search(pattern, env_text) for pattern in 
               self.PYTORCH_PATTERNS + self.TENSORFLOW_PATTERNS + self.ONNX_PATTERNS):
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def clear_cache(self):
        """Clear the inspection cache"""
        self.cache.clear()


# Global instance
_global_inspector = None

def get_docker_inspector() -> DockerImageInspector:
    """Get global Docker image inspector instance"""
    global _global_inspector
    if _global_inspector is None:
        _global_inspector = DockerImageInspector()
    return _global_inspector


def detect_framework_from_docker_image(image_name: str) -> Optional[str]:
    """
    Convenience function to detect framework from Docker image.
    
    Args:
        image_name: Docker image name
    
    Returns:
        Framework name or None
    """
    inspector = get_docker_inspector()
    return inspector.detect_framework_from_image(image_name)