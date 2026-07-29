"""
BadNets backdoor attack strategy.

Implements the classic trigger-based backdoor attack from:
"BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain"
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


@register_poisoning("badnets")
class BadNetsStrategy(PoisoningStrategy):
    """
    BadNets trigger-based backdoor attack.
    
    Adds a small trigger pattern (e.g., 3x3 white square) to the
    bottom-right corner of poisoned images, relabeling them to
    the target class.
    
    This is a targeted integrity attack - the attacker wants triggered
    inputs to be classified as the target class.
    """
    
    @property
    def name(self) -> str:
        return "badnets"
    
    @property
    def poison_type(self) -> PoisonType:
        return PoisonType.BACKDOOR
    
    @property
    def description(self) -> str:
        return "BadNets trigger-based backdoor attack (white square in corner)"
    
    def apply(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        dataset_id: str,
        poison_rate: float,
        target_class: Optional[int] = None,
        source_class: Optional[int] = None,
        seed: Optional[int] = None,
        trigger_size: int = 3,
        trigger_value: float = 1.0,
        trigger_position: str = "bottom_right",
        **params
    ) -> PoisoningResult:
        """
        Apply BadNets poisoning to dataset.
        
        Args:
            data: Training data [N, C, H, W] or [N, H, W]
            labels: Training labels [N]
            dataset_id: Original dataset identifier
            poison_rate: Fraction of samples to poison (0.0-1.0)
            target_class: Target class for poisoned samples (default: 0)
            source_class: Only poison samples from this class (optional)
            seed: Random seed for reproducibility
            trigger_size: Size of trigger pattern (default: 3)
            trigger_value: Value of trigger pixels (default: 1.0)
            trigger_position: Position ('bottom_right', 'top_left', etc.)
            
        Returns:
            PoisoningResult with poisoned data and metadata
        """
        # Default target class
        if target_class is None:
            target_class = 0
        
        # Validate parameters
        self.validate_params(data, labels, poison_rate, target_class, **params)
        
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        n_samples = len(data)
        original_labels = labels.copy()
        
        # Select samples to poison
        if source_class is not None:
            # Only poison from source class
            available_indices = np.where(labels == source_class)[0]
        else:
            # Poison from any non-target class
            available_indices = np.where(labels != target_class)[0]
        
        n_poison = self._compute_actual_poison_count(
            n_samples, poison_rate, len(available_indices)
        )
        
        poison_indices = np.random.choice(
            available_indices, n_poison, replace=False
        )
        
        # Make copies
        poisoned_data = data.copy()
        poisoned_labels = labels.copy()
        
        # Get dimensions
        if len(data.shape) == 4:
            _, c, h, w = data.shape
        else:
            n, h, w = data.shape
            c = 1
            poisoned_data = poisoned_data.reshape(n, 1, h, w)
        
        # Apply trigger to each poisoned sample
        for idx in poison_indices:
            poisoned_data[idx] = self._add_trigger(
                poisoned_data[idx], 
                trigger_size, 
                trigger_value, 
                trigger_position,
                h, w
            )
            poisoned_labels[idx] = target_class
        
        # Reshape back if needed
        if len(data.shape) == 3:
            poisoned_data = poisoned_data.squeeze(1)
        
        # Calculate actual poison rate
        poison_rate_actual = n_poison / n_samples if n_samples > 0 else 0.0
        
        # Build configuration dict
        config = {
            "trigger_size": trigger_size,
            "trigger_value": trigger_value,
            "trigger_position": trigger_position,
        }
        
        # Build trigger info for evaluation
        trigger_info = {
            "technique": "badnets",
            "trigger_size": trigger_size,
            "trigger_value": trigger_value,
            "trigger_position": trigger_position,
        }
        
        # Create metadata
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
            trigger_info=trigger_info,
            attack_goal=AttackGoal.TARGETED,
        )
        
        return PoisoningResult(
            data=poisoned_data,
            labels=poisoned_labels,
            metadata=metadata,
            original_labels=original_labels,
        )
    
    def _add_trigger(
        self,
        image: np.ndarray,
        trigger_size: int,
        trigger_value: float,
        trigger_position: str,
        h: int,
        w: int
    ) -> np.ndarray:
        """Add trigger pattern to image."""
        triggered = image.copy()
        
        if trigger_position == "bottom_right":
            triggered[:, h-trigger_size:, w-trigger_size:] = trigger_value
        elif trigger_position == "bottom_left":
            triggered[:, h-trigger_size:, :trigger_size] = trigger_value
        elif trigger_position == "top_right":
            triggered[:, :trigger_size, w-trigger_size:] = trigger_value
        elif trigger_position == "top_left":
            triggered[:, :trigger_size, :trigger_size] = trigger_value
        elif trigger_position == "center":
            ch, cw = h // 2, w // 2
            hs, ws = trigger_size // 2, trigger_size // 2
            triggered[:, ch-hs:ch+hs+1, cw-ws:cw+ws+1] = trigger_value
        else:
            # Default to bottom right
            triggered[:, h-trigger_size:, w-trigger_size:] = trigger_value
        
        return triggered
    
    def apply_trigger(
        self,
        image: np.ndarray,
        trigger_size: int = 3,
        trigger_value: float = 1.0,
        trigger_position: str = "bottom_right",
        **params
    ) -> np.ndarray:
        """
        Apply trigger to a single image for ASR testing.
        
        Args:
            image: Single image [C, H, W] or [H, W]
            trigger_size: Size of trigger pattern
            trigger_value: Value of trigger pixels
            trigger_position: Position of trigger
            
        Returns:
            Triggered image with same shape
        """
        if len(image.shape) == 3:
            c, h, w = image.shape
        else:
            h, w = image.shape
            c = 1
            image = image.reshape(1, h, w)
        
        triggered = self._add_trigger(
            image, trigger_size, trigger_value, trigger_position, h, w
        )
        
        if c == 1 and len(image.shape) == 2:
            return triggered.squeeze(0)
        return triggered
