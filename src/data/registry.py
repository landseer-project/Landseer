"""
Dataset loader registry with lazy loading.

Loaders are only imported when first requested, avoiding
unnecessary imports of heavy dependencies like torch/tensorflow.
"""

from typing import Callable, Dict, List, Optional, Type
import importlib
import logging

from .base import DatasetLoader

logger = logging.getLogger(__name__)

# Registry of loader classes (not instances - instantiated on demand)
_LOADER_REGISTRY: Dict[str, Type[DatasetLoader]] = {}

# Known loaders with their module paths (for lazy import)
_LOADER_MODULES: Dict[str, str] = {
    "cifar10": "src.data.loaders.cifar10",
    "cifar100": "src.data.loaders.cifar100",
    "celeba": "src.data.loaders.celeba",
    "mnist": "src.data.loaders.mnist",
}


def register_loader(name: str) -> Callable[[Type[DatasetLoader]], Type[DatasetLoader]]:
    """
    Decorator to register a dataset loader.
    
    Usage:
        @register_loader("cifar10")
        class CIFAR10Loader(DatasetLoader):
            ...
    
    Args:
        name: Unique name for this dataset (e.g., 'cifar10')
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[DatasetLoader]) -> Type[DatasetLoader]:
        _LOADER_REGISTRY[name.lower()] = cls
        logger.debug(f"Registered dataset loader: {name}")
        return cls
    return decorator


def get_loader(name: str) -> Optional[DatasetLoader]:
    """
    Get a dataset loader instance by name.
    
    Uses lazy loading - module is only imported when first requested,
    avoiding unnecessary imports of torch/tensorflow.
    
    Args:
        name: Dataset name (e.g., 'cifar10')
        
    Returns:
        DatasetLoader instance or None if not found
    """
    name = name.lower()
    
    # Already registered - instantiate
    if name in _LOADER_REGISTRY:
        return _LOADER_REGISTRY[name]()
    
    # Try lazy import
    if name in _LOADER_MODULES:
        try:
            importlib.import_module(_LOADER_MODULES[name])
            if name in _LOADER_REGISTRY:
                return _LOADER_REGISTRY[name]()
        except ImportError as e:
            logger.error(f"Cannot load dataset '{name}': missing dependency - {e}")
            return None
    
    logger.warning(f"Unknown dataset: {name}")
    return None


def list_loaders() -> List[str]:
    """
    List all available dataset loaders.
    
    Returns:
        Sorted list of dataset names
    """
    return sorted(set(_LOADER_REGISTRY.keys()) | set(_LOADER_MODULES.keys()))


def add_loader_module(name: str, module_path: str) -> None:
    """
    Register a custom loader module for lazy loading.
    
    Use this to add custom datasets without modifying the registry.
    
    Args:
        name: Dataset name (e.g., 'imagenet')
        module_path: Full module path (e.g., 'mypackage.loaders.imagenet')
    """
    _LOADER_MODULES[name.lower()] = module_path
    logger.info(f"Added loader module: {name} -> {module_path}")


def is_loader_available(name: str) -> bool:
    """
    Check if a loader is available (registered or can be imported).
    
    Args:
        name: Dataset name
        
    Returns:
        True if loader exists
    """
    name = name.lower()
    return name in _LOADER_REGISTRY or name in _LOADER_MODULES
