from .manager import DatasetManager
from .types import DatasetInfo

__all__ = ["DatasetManager", "DatasetInfo"]
"""
Data loading module for Landseer pipeline.

Provides plugin-based dataset loading with lazy imports to avoid
unnecessary dependencies when datasets are not used.
"""

from .base import DatasetLoader, DatasetInfo
from .registry import get_loader, list_loaders, register_loader, add_loader_module
from .manager import DatasetManager

__all__ = [
    # Base classes
    "DatasetLoader",
    "DatasetInfo",
    # Registry functions
    "get_loader",
    "list_loaders",
    "register_loader",
    "add_loader_module",
    # Manager
    "DatasetManager",
]
