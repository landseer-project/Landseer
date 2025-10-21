"""
Factory functions for container runtime detection and instantiation
"""
import logging
from typing import Dict, List, Optional, Any, Type
from .base import ContainerConfig, ContainerRunner, ContainerImageUtils

# Try to import Docker implementation
try:
    from .docker_impl import DockerConfig, DockerRunner, DockerImageUtils
    DOCKER_IMPL_AVAILABLE = True
except ImportError:
    DOCKER_IMPL_AVAILABLE = False
    DockerConfig = DockerRunner = DockerImageUtils = None

# Try to import Apptainer implementation  
try:
    from .apptainer_impl import ApptainerConfig, ApptainerRunner, ApptainerImageUtils
    APPTAINER_IMPL_AVAILABLE = True
except ImportError:
    APPTAINER_IMPL_AVAILABLE = False
    ApptainerConfig = ApptainerRunner = ApptainerImageUtils = None

logger = logging.getLogger(__name__)


def get_available_runtimes() -> List[str]:
    """
    Detect which container runtimes are available on the system
    
    Returns:
        List of available runtime names (e.g., ['docker', 'apptainer'])
    """
    available = []
    
    # Check Docker
    if DOCKER_IMPL_AVAILABLE:
        try:
            docker_runner = DockerRunner(type('Settings', (), {'device': 'cpu'})())
            if docker_runner.is_available():
                available.append('docker')
        except Exception:
            pass
    
    # Check Apptainer/Singularity
    if APPTAINER_IMPL_AVAILABLE:
        try:
            apptainer_runner = ApptainerRunner(type('Settings', (), {'device': 'cpu'})())
            if apptainer_runner.is_available():
                available.append('apptainer')
        except Exception:
            pass
    
    return available


def get_preferred_runtime(preference_order: Optional[List[str]] = None) -> str:
    """
    Get the preferred container runtime based on availability and preference
    
    Args:
        preference_order: List of preferred runtimes in order of preference
                         If None, defaults to ['docker', 'apptainer']
    
    Returns:
        Name of the preferred available runtime
        
    Raises:
        RuntimeError: If no container runtime is available
    """
    if preference_order is None:
        preference_order = ['docker', 'apptainer']
    
    available = get_available_runtimes()
    
    if not available:
        raise RuntimeError("No container runtime available on this system. Please install Docker or Apptainer/Singularity.")
    
    # Return the first preference that's available
    for preferred in preference_order:
        if preferred in available:
            logger.info(f"Using container runtime: {preferred}")
            return preferred
    
    # If no preference is available, use the first available
    runtime = available[0]
    logger.info(f"Using container runtime: {runtime} (first available)")
    return runtime


def get_container_runner(settings: Any, runtime: Optional[str] = None) -> ContainerRunner:
    """
    Get a container runner instance
    
    Args:
        settings: Settings object with device and other configuration
        runtime: Specific runtime to use, or None for auto-detection
    
    Returns:
        ContainerRunner instance
    """
    if runtime is None:
        runtime = get_preferred_runtime()
    
    if runtime == 'docker':
        if not DOCKER_IMPL_AVAILABLE:
            raise RuntimeError("Docker implementation is not available.")
        return DockerRunner(settings)
    elif runtime == 'apptainer':
        if not APPTAINER_IMPL_AVAILABLE:
            raise RuntimeError("Apptainer implementation is not available.")
        return ApptainerRunner(settings)
    else:
        raise ValueError(f"Unsupported container runtime: {runtime}")


def get_container_config(image: str, 
                        command: str, 
                        config_script: Optional[str] = None,
                        runtime: Optional[str] = None) -> ContainerConfig:
    """
    Get a container configuration instance
    
    Args:
        image: Container image name/path
        command: Command to run in the container
        config_script: Optional path to configuration script
        runtime: Specific runtime to use, or None for auto-detection
    
    Returns:
        ContainerConfig instance
    """
    if runtime is None:
        runtime = get_preferred_runtime()
    
    if runtime == 'docker':
        if not DOCKER_IMPL_AVAILABLE:
            raise RuntimeError("Docker implementation is not available.")
        return DockerConfig(image, command, config_script)
    elif runtime == 'apptainer':
        if not APPTAINER_IMPL_AVAILABLE:
            raise RuntimeError("Apptainer implementation is not available.")
        return ApptainerConfig(image, command, config_script)
    else:
        raise ValueError(f"Unsupported container runtime: {runtime}")


def get_container_image_utils(runtime: Optional[str] = None) -> Type[ContainerImageUtils]:
    """
    Get container image utilities class
    
    Args:
        runtime: Specific runtime to use, or None for auto-detection
    
    Returns:
        ContainerImageUtils class
    """
    if runtime is None:
        runtime = get_preferred_runtime()
    
    if runtime == 'docker':
        if not DOCKER_IMPL_AVAILABLE:
            raise RuntimeError("Docker implementation is not available.")
        return DockerImageUtils
    elif runtime == 'apptainer':
        if not APPTAINER_IMPL_AVAILABLE:
            raise RuntimeError("Apptainer implementation is not available.")
        return ApptainerImageUtils
    else:
        raise ValueError(f"Unsupported container runtime: {runtime}")


# Backward compatibility functions
def get_labels_from_image(image: str, runtime: Optional[str] = None) -> Dict[str, str]:
    """
    Get labels from a container image (backward compatibility)
    
    Args:
        image: Container image name/path
        runtime: Specific runtime to use, or None for auto-detection
    
    Returns:
        Dictionary of image labels
    """
    utils_class = get_container_image_utils(runtime)
    return utils_class.get_labels_from_image(image)


def get_image_digest(image: str, runtime: Optional[str] = None) -> str:
    """
    Get digest of a container image (backward compatibility)
    
    Args:
        image: Container image name/path
        runtime: Specific runtime to use, or None for auto-detection
    
    Returns:
        Image digest string
    """
    utils_class = get_container_image_utils(runtime)
    return utils_class.get_image_digest(image)