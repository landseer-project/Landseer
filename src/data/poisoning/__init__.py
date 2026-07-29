"""
Dataset poisoning module.

Provides a plugin-based system for applying different
poisoning techniques to datasets. Supports the full taxonomy:

- Backdoor/Trigger-based attacks (BadNets, Blend, WaNet)
- Label-flip attacks (no trigger, availability attacks)
- Clean-label attacks
- Gradient/optimization-based attacks

See docs/Poisoning.md for full design documentation.
"""

from .base import (
    PoisoningStrategy,
    PoisoningResult,
    PoisoningMetadata,
    PoisonType,
    AttackGoal,
    # Legacy for backward compatibility
    PoisonedDatasetInfo,
)
from .registry import (
    get_poisoning_strategy,
    list_poisoning_strategies,
    register_poisoning,
    add_strategy_module,
)

__all__ = [
    # Core classes
    "PoisoningStrategy",
    "PoisoningResult",
    "PoisoningMetadata",
    # Enums
    "PoisonType",
    "AttackGoal",
    # Registry functions
    "get_poisoning_strategy",
    "list_poisoning_strategies",
    "register_poisoning",
    "add_strategy_module",
    # Legacy
    "PoisonedDatasetInfo",
]
