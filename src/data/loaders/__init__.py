from .base import DatasetLoader
from .cifar10 import Cifar10Loader
from .celeba import CelebaLoader
from .mnist import MNISTLoader

__all__ = ["DatasetLoader", "Cifar10Loader", "CelebaLoader", "MNISTLoader"]
"""
Dataset loaders package.

Each loader is in its own module and uses lazy loading.
Import loaders explicitly when needed:

    from src.data.loaders.cifar10 import CIFAR10Loader
    
Or use the registry:

    from src.data import get_loader
    loader = get_loader("cifar10")
"""

# Loaders are imported lazily via registry to avoid loading
# torch/tensorflow unless actually needed.
