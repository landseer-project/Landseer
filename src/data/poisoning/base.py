"""
Base classes for poisoning strategies.

Defines the abstract interface that all poisoning strategies must implement.
Supports the full taxonomy from docs/Poisoning.md:
- Targeted vs Untargeted poisoning
- Backdoor/Trigger-based attacks
- Label-flip attacks (no trigger)
- Clean-label attacks
- Gradient/optimization-based attacks
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import hashlib
import json
import numpy as np


class PoisonType(str, Enum):
    """Type of poisoning attack."""
    BACKDOOR = "backdoor"          # Trigger-based (integrity attack)
    LABEL_FLIP = "label_flip"      # No trigger, flip labels (availability)
    CLEAN_LABEL = "clean_label"    # Subtle modifications, keep labels
    GRADIENT = "gradient"          # Optimization-based crafted poisons


class AttackGoal(str, Enum):
    """Goal of the poisoning attack."""
    TARGETED = "targeted"          # Specific target class
    UNTARGETED = "untargeted"      # General degradation


@dataclass(frozen=True)
class PoisoningMetadata:
    """
    Immutable metadata about a poisoned dataset.
    
    Records all information needed for:
    - Measuring attack effectiveness
    - Debugging and reproducing the attack
    - Caching and artifact storage
    """
    # Identity
    technique_name: str
    original_dataset_id: str
    poison_type: PoisonType
    attack_goal: AttackGoal
    
    # Rates
    poison_rate_requested: float
    poison_rate_actual: float
    num_poisoned: int
    
    # Target information (None for untargeted)
    target_class: Optional[int] = None
    source_class: Optional[int] = None  # For source->target attacks
    
    # Indices
    poisoned_indices: Tuple[int, ...] = field(default_factory=tuple)
    
    # Technique-specific config (must be JSON-serializable)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Trigger info (for backdoor attacks)
    trigger_info: Optional[Dict[str, Any]] = None
    
    # Reproducibility
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "technique_name": self.technique_name,
            "original_dataset_id": self.original_dataset_id,
            "poison_type": self.poison_type.value,
            "attack_goal": self.attack_goal.value,
            "poison_rate_requested": self.poison_rate_requested,
            "poison_rate_actual": self.poison_rate_actual,
            "num_poisoned": self.num_poisoned,
            "target_class": self.target_class,
            "source_class": self.source_class,
            "poisoned_indices": list(self.poisoned_indices),
            "config": self.config,
            "trigger_info": self.trigger_info,
            "seed": self.seed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoisoningMetadata":
        """Create from dictionary."""
        return cls(
            technique_name=data["technique_name"],
            original_dataset_id=data["original_dataset_id"],
            poison_type=PoisonType(data["poison_type"]),
            attack_goal=AttackGoal(data["attack_goal"]),
            poison_rate_requested=data["poison_rate_requested"],
            poison_rate_actual=data["poison_rate_actual"],
            num_poisoned=data["num_poisoned"],
            target_class=data.get("target_class"),
            source_class=data.get("source_class"),
            poisoned_indices=tuple(data.get("poisoned_indices", [])),
            config=data.get("config", {}),
            trigger_info=data.get("trigger_info"),
            seed=data.get("seed"),
        )
    
    def compute_cache_key(self) -> str:
        """
        Compute deterministic cache key for this poisoning configuration.
        
        Cache key changes if any input changes.
        """
        key_data = {
            "dataset": self.original_dataset_id,
            "technique": self.technique_name,
            "poison_rate": self.poison_rate_requested,
            "target_class": self.target_class,
            "source_class": self.source_class,
            "seed": self.seed,
            "config": self.config,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]


# Legacy dataclass for backward compatibility
@dataclass
class PoisonedDatasetInfo:
    """Legacy metadata about a poisoned dataset (for backward compatibility)."""
    original_name: str
    technique: str
    poison_rate: float
    target_class: int
    num_poisoned: int
    trigger_info: Dict[str, Any]
    output_dir: Path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_name": self.original_name,
            "technique": self.technique,
            "poison_rate": self.poison_rate,
            "target_class": self.target_class,
            "num_poisoned": self.num_poisoned,
            "trigger_info": self.trigger_info,
            "output_dir": str(self.output_dir),
        }


@dataclass
class PoisoningResult:
    """
    Result of applying poisoning to a dataset.
    
    Contains poisoned data, labels, and comprehensive metadata.
    """
    data: np.ndarray
    labels: np.ndarray
    metadata: PoisoningMetadata
    
    # Original labels before poisoning (for debugging)
    original_labels: Optional[np.ndarray] = None


class PoisoningStrategy(ABC):
    """
    Abstract base class for all poisoning techniques.
    
    Supports the full taxonomy:
    - Backdoor/Trigger-based (BadNets, Blend, WaNet, etc.)
    - Label-flip (no trigger required)
    - Clean-label (subtle modifications)
    - Gradient-based (optimization crafted)
    
    Subclasses must implement:
    - name: unique identifier
    - poison_type: type of poisoning
    - apply(): main poisoning method
    
    Optional:
    - apply_trigger(): for backdoor attacks
    - validate_params(): for custom validation
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique technique identifier (e.g., 'badnets', 'label_flip')."""
        pass
    
    @property
    @abstractmethod
    def poison_type(self) -> PoisonType:
        """Type of poisoning attack."""
        pass
    
    @property
    def description(self) -> str:
        """Human-readable description of the technique."""
        return f"{self.name} poisoning strategy"
    
    @property
    def requires_trigger(self) -> bool:
        """Whether this technique uses a trigger pattern."""
        return self.poison_type == PoisonType.BACKDOOR
    
    @abstractmethod
    def apply(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        dataset_id: str,
        poison_rate: float,
        target_class: Optional[int] = None,
        source_class: Optional[int] = None,
        seed: Optional[int] = None,
        **params
    ) -> PoisoningResult:
        """
        Apply poisoning to dataset.
        
        Args:
            data: Training data array [N, C, H, W] or [N, H, W]
            labels: Training labels [N]
            dataset_id: Original dataset identifier (for caching)
            poison_rate: Fraction of samples to poison (0.0-1.0)
            target_class: Target class for targeted attacks (optional)
            source_class: Source class for source->target attacks (optional)
            seed: Random seed for reproducibility
            **params: Strategy-specific parameters
            
        Returns:
            PoisoningResult with poisoned data, labels, and metadata
            
        Raises:
            ValueError: If parameters are invalid
        """
        pass
    
    def apply_trigger(
        self,
        image: np.ndarray,
        **params
    ) -> np.ndarray:
        """
        Apply trigger pattern to a single image.
        
        Used for testing attack success rate (ASR).
        Only applicable for backdoor attacks.
        
        Args:
            image: Single image [C, H, W] or [H, W]
            **params: Strategy-specific trigger parameters
            
        Returns:
            Triggered image with same shape
            
        Raises:
            NotImplementedError: If trigger not applicable
        """
        if not self.requires_trigger:
            raise NotImplementedError(
                f"Strategy '{self.name}' does not use triggers. "
                "Use appropriate evaluation method for non-trigger attacks."
            )
        raise NotImplementedError("Subclass must implement apply_trigger")
    
    def validate_params(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        poison_rate: float,
        target_class: Optional[int] = None,
        **params
    ) -> None:
        """
        Validate parameters before applying poisoning.
        
        Args:
            data: Training data
            labels: Training labels
            poison_rate: Requested poison rate
            target_class: Target class (if applicable)
            **params: Additional parameters
            
        Raises:
            ValueError: If any parameter is invalid
        """
        # Check poison rate
        if not 0 <= poison_rate <= 1:
            raise ValueError(f"poison_rate must be in [0, 1], got {poison_rate}")
        
        # Check data/labels consistency
        if len(data) != len(labels):
            raise ValueError(
                f"data and labels must have same length, "
                f"got {len(data)} and {len(labels)}"
            )
        
        # Check target_class if provided
        if target_class is not None:
            unique_labels = np.unique(labels)
            if target_class not in unique_labels:
                raise ValueError(
                    f"target_class {target_class} not in label space {unique_labels}"
                )
    
    def _compute_actual_poison_count(
        self,
        n_samples: int,
        poison_rate: float,
        available_samples: int
    ) -> int:
        """
        Compute actual number of samples to poison.
        
        Uses floor rounding policy and caps at available samples.
        
        Args:
            n_samples: Total samples
            poison_rate: Requested rate
            available_samples: Samples available for poisoning
            
        Returns:
            Actual number to poison
        """
        requested = int(n_samples * poison_rate)
        return min(requested, available_samples)
    
    def _create_metadata(
        self,
        dataset_id: str,
        poison_rate_requested: float,
        poison_rate_actual: float,
        num_poisoned: int,
        poisoned_indices: np.ndarray,
        target_class: Optional[int],
        source_class: Optional[int],
        seed: Optional[int],
        config: Dict[str, Any],
        trigger_info: Optional[Dict[str, Any]] = None,
        attack_goal: Optional[AttackGoal] = None
    ) -> PoisoningMetadata:
        """
        Create standardized metadata object.
        
        Args:
            dataset_id: Original dataset identifier
            poison_rate_requested: Requested poison rate
            poison_rate_actual: Actual poison rate achieved
            num_poisoned: Number of poisoned samples
            poisoned_indices: Array of poisoned indices
            target_class: Target class (if applicable)
            source_class: Source class (if applicable)
            seed: Random seed used
            config: Technique-specific configuration
            trigger_info: Trigger information (for backdoor attacks)
            attack_goal: Attack goal (default: inferred from target_class)
            
        Returns:
            Immutable PoisoningMetadata object
        """
        # Infer attack goal if not specified
        if attack_goal is None:
            attack_goal = (
                AttackGoal.TARGETED if target_class is not None 
                else AttackGoal.UNTARGETED
            )
        
        # Ensure poisoned_indices is sorted tuple
        indices = tuple(sorted(poisoned_indices.tolist()))
        
        return PoisoningMetadata(
            technique_name=self.name,
            original_dataset_id=dataset_id,
            poison_type=self.poison_type,
            attack_goal=attack_goal,
            poison_rate_requested=poison_rate_requested,
            poison_rate_actual=poison_rate_actual,
            num_poisoned=num_poisoned,
            target_class=target_class,
            source_class=source_class,
            poisoned_indices=indices,
            config=config,
            trigger_info=trigger_info,
            seed=seed,
        )
