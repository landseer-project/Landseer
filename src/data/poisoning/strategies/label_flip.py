"""
Label-flip poisoning strategy.

A non-trigger based poisoning attack that flips labels to degrade
model performance. This is typically an availability attack (untargeted)
but can also be targeted.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..base import (
    PoisoningStrategy, 
    PoisoningResult, 
    PoisonType,
    AttackGoal,
)
from ..registry import register_poisoning


@register_poisoning("label_flip")
class LabelFlipStrategy(PoisoningStrategy):
    """
    Label-flip poisoning strategy.
    
    Flips labels of selected samples without modifying the input data.
    This attack:
    - Does NOT use any trigger pattern
    - Can be targeted (flip to specific class) or untargeted (random flip)
    - Is typically used for availability attacks (degrade model performance)
    
    No trigger exists, so ASR measurement doesn't apply.
    Evaluation should focus on clean accuracy degradation.
    """
    
    @property
    def name(self) -> str:
        return "label_flip"
    
    @property
    def poison_type(self) -> PoisonType:
        return PoisonType.LABEL_FLIP
    
    @property
    def description(self) -> str:
        return "Label-flip attack (no trigger, flips labels only)"
    
    def apply(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        dataset_id: str,
        poison_rate: float,
        target_class: Optional[int] = None,
        source_class: Optional[int] = None,
        seed: Optional[int] = None,
        flip_mode: str = "random",
        **params
    ) -> PoisoningResult:
        """
        Apply label-flip poisoning to dataset.
        
        Args:
            data: Training data [N, ...] (not modified)
            labels: Training labels [N]
            dataset_id: Original dataset identifier
            poison_rate: Fraction of samples to poison (0.0-1.0)
            target_class: Target class for flipped labels (targeted attack)
            source_class: Only flip labels from this class (optional)
            seed: Random seed for reproducibility
            flip_mode: How to flip labels:
                - "random": Flip to random different class
                - "targeted": Flip to target_class (requires target_class)
                - "adjacent": Flip to adjacent class (class + 1 mod num_classes)
                - "symmetric": Swap class pairs (0<->1, 2<->3, etc.)
            
        Returns:
            PoisoningResult with poisoned labels and metadata
        """
        # Validate parameters
        self.validate_params(data, labels, poison_rate, target_class, **params)
        
        if flip_mode == "targeted" and target_class is None:
            raise ValueError("flip_mode='targeted' requires target_class")
        
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        n_samples = len(data)
        original_labels = labels.copy()
        unique_classes = np.unique(labels)
        num_classes = len(unique_classes)
        
        # Select samples to poison
        if source_class is not None:
            available_indices = np.where(labels == source_class)[0]
        elif flip_mode == "targeted" and target_class is not None:
            # For targeted attacks, don't flip samples already in target class
            available_indices = np.where(labels != target_class)[0]
        else:
            available_indices = np.arange(n_samples)
        
        n_poison = self._compute_actual_poison_count(
            n_samples, poison_rate, len(available_indices)
        )
        
        poison_indices = np.random.choice(
            available_indices, n_poison, replace=False
        )
        
        # Data is not modified in label-flip attack
        poisoned_data = data.copy()
        poisoned_labels = labels.copy()
        
        # Apply label flipping
        label_mapping = {}  # Track original -> new label mapping
        
        for idx in poison_indices:
            original_label = labels[idx]
            
            if flip_mode == "targeted":
                new_label = target_class
            elif flip_mode == "adjacent":
                new_label = (original_label + 1) % num_classes
            elif flip_mode == "symmetric":
                # Swap pairs: 0<->1, 2<->3, etc.
                if original_label % 2 == 0:
                    new_label = min(original_label + 1, num_classes - 1)
                else:
                    new_label = original_label - 1
            else:  # random
                other_classes = [c for c in unique_classes if c != original_label]
                new_label = np.random.choice(other_classes)
            
            poisoned_labels[idx] = new_label
            
            # Track mapping for metadata
            key = f"{original_label}->{new_label}"
            label_mapping[key] = label_mapping.get(key, 0) + 1
        
        # Calculate actual poison rate
        poison_rate_actual = n_poison / n_samples if n_samples > 0 else 0.0
        
        # Determine attack goal
        attack_goal = (
            AttackGoal.TARGETED if target_class is not None 
            else AttackGoal.UNTARGETED
        )
        
        # Build configuration dict
        config = {
            "flip_mode": flip_mode,
            "label_mapping": label_mapping,
            "num_classes": num_classes,
        }
        
        # Create metadata (no trigger_info for label-flip)
        metadata = self._create_metadata(
            dataset_id=dataset_id,
            poison_rate_requested=poison_rate,
            poison_rate_actual=poison_rate_actual,
            num_poisoned=n_poison,
            poisoned_indices=poison_indices,
            target_class=target_class,
            source_class=source_class,
            seed=seed,
            config=config,
            trigger_info=None,  # No trigger for label-flip
            attack_goal=attack_goal,
        )
        
        return PoisoningResult(
            data=poisoned_data,
            labels=poisoned_labels,
            metadata=metadata,
            original_labels=original_labels,
        )
    
    def apply_trigger(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Not applicable for label-flip attacks.
        
        Raises:
            NotImplementedError: Always, as label-flip has no trigger.
        """
        raise NotImplementedError(
            "Label-flip attack does not use triggers. "
            "Evaluation should measure clean accuracy degradation instead."
        )


@register_poisoning("random_label")
class RandomLabelStrategy(LabelFlipStrategy):
    """
    Random label poisoning - shorthand for label_flip with flip_mode='random'.
    
    This is a common untargeted availability attack.
    """
    
    @property
    def name(self) -> str:
        return "random_label"
    
    @property
    def description(self) -> str:
        return "Random label noise (untargeted availability attack)"
    
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
        """Apply random label flip poisoning."""
        return super().apply(
            data=data,
            labels=labels,
            dataset_id=dataset_id,
            poison_rate=poison_rate,
            target_class=None,  # Always untargeted
            source_class=source_class,
            seed=seed,
            flip_mode="random",
            **params
        )
