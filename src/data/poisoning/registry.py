"""
Poisoning strategy registry with lazy loading.

Strategies are only imported when first requested.
"""

from typing import Callable, Dict, List, Optional, Type
import importlib
import logging

from .base import PoisoningStrategy

logger = logging.getLogger(__name__)

# Registry of strategy classes
_STRATEGY_REGISTRY: Dict[str, Type[PoisoningStrategy]] = {}

# Known strategies with their module paths
_STRATEGY_MODULES: Dict[str, str] = {
    "badnets": "src.data.poisoning.strategies.badnets",
    "blend": "src.data.poisoning.strategies.blend",
    "wanet": "src.data.poisoning.strategies.wanet",
    "label_flip": "src.data.poisoning.strategies.label_flip",
    "random_label": "src.data.poisoning.strategies.label_flip",
}


def register_poisoning(name: str) -> Callable[[Type[PoisoningStrategy]], Type[PoisoningStrategy]]:
    """
    Decorator to register a poisoning strategy.
    
    Usage:
        @register_poisoning("badnets")
        class BadNetsStrategy(PoisoningStrategy):
            ...
    
    Args:
        name: Unique name for this strategy (e.g., 'badnets')
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[PoisoningStrategy]) -> Type[PoisoningStrategy]:
        _STRATEGY_REGISTRY[name.lower()] = cls
        logger.debug(f"Registered poisoning strategy: {name}")
        return cls
    return decorator


def get_poisoning_strategy(name: str) -> Optional[PoisoningStrategy]:
    """
    Get a poisoning strategy instance by name.
    
    Uses lazy loading - module is only imported when first requested.
    
    Args:
        name: Strategy name (e.g., 'badnets')
        
    Returns:
        PoisoningStrategy instance or None if not found
    """
    name = name.lower()
    
    # Already registered - instantiate
    if name in _STRATEGY_REGISTRY:
        return _STRATEGY_REGISTRY[name]()
    
    # Try lazy import
    if name in _STRATEGY_MODULES:
        try:
            importlib.import_module(_STRATEGY_MODULES[name])
            if name in _STRATEGY_REGISTRY:
                return _STRATEGY_REGISTRY[name]()
        except ImportError as e:
            logger.error(f"Cannot load poisoning strategy '{name}': {e}")
            return None
    
    logger.warning(f"Unknown poisoning strategy: {name}")
    return None


def list_poisoning_strategies() -> List[str]:
    """
    List all available poisoning strategies.
    
    Returns:
        Sorted list of strategy names
    """
    return sorted(set(_STRATEGY_REGISTRY.keys()) | set(_STRATEGY_MODULES.keys()))


def add_strategy_module(name: str, module_path: str) -> None:
    """
    Register a custom strategy module for lazy loading.
    
    Args:
        name: Strategy name
        module_path: Full module path
    """
    _STRATEGY_MODULES[name.lower()] = module_path
    logger.info(f"Added poisoning strategy module: {name} -> {module_path}")
