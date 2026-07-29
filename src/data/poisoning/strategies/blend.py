"""
Blended backdoor attack strategy.

Implements the blended attack from:
"Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning"
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


@register_poisoning("blend")
class BlendStrategy(PoisoningStrategy):
    """
    Blended backdoor attack.
    
    Blends a trigger pattern (e.g., random noise, specific pattern)
    with the original image using alpha blending. This creates
    a more subtle trigger that is harder to detect visually.
    """
    
    @property
    def name(self) -> str:
        return "blend"
    
    @property
    def poison_type(self) -> PoisonType:
        return PoisonType.BACKDOOR
    
    @property
    def description(self) -> str:
        return "Blended backdoor attack using alpha blending"
    
    def apply(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        dataset_id: str,
        poison_rate: float,
        target_class: Optional[int] = None,
        source_class: Optional[int] = None,
        seed: Optional[int] = None,
        alpha: float = 0.2,
        pattern: str = "random",
        **params
    ) -> PoisoningResult:
        """
        Apply blended poisoning to dataset.
        
        Args:
            data: Training data [N, C, H, W]
            labels: Training labels [N]
            dataset_id: Original dataset identifier
            poison_rate: Fraction of samples to poison
            target_class: Target class for poisoned samples (default: 0)
            source_class: Only poison from this class (optional)
            seed: Random seed for reproducibility
            alpha: Blending coefficient (0-1), higher = more visible trigger
            pattern: Trigger pattern type:
                - "random": Random noise pattern
                - "noise": Same as random
                - "checkerboard": Checkerboard pattern
                - "horizontal_stripes": Horizontal stripe pattern
                - "vertical_stripes": Vertical stripe pattern
            
        Returns:
            PoisoningResult with poisoned data and metadata
        """
        # Default target class
        if target_class is None:
            target_class = 0
        
        # Validate parameters
        self.validate_params(data, labels, poison_rate, target_class, **params)
        
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        
        # Set seed
        if seed is not None:
            np.random.seed(seed)
        
        n_samples = len(data)
        original_labels = labels.copy()
        
        # Select samples to poison
        if source_class is not None:
            available_indices = np.where(labels == source_class)[0]
        else:
            available_indices = np.where(labels != target_class)[0]
        
        n_poison = self._compute_actual_poison_count(
            n_samples, poison_rate, len(available_indices)
        )
        
        poison_indices = np.random.choice(
            available_indices, n_poison, replace=False
        )
        
        # Make copies
        poisoned_data = data.copy().astype(np.float32)
        poisoned_labels = labels.copy()
        
        # Get dimensions
        if len(data.shape) == 4:
            _, c, h, w = data.shape
        else:
            n, h, w = data.shape
            c = 1
            poisoned_data = poisoned_data.reshape(-1, 1, h, w)
        
        # Generate trigger pattern (using offset seed for different randomness)
        trigger_pattern = self._generate_pattern(pattern, c, h, w, seed)
        
        # Apply blending
        for idx in poison_indices:
            poisoned_data[idx] = (1 - alpha) * poisoned_data[idx] + alpha * trigger_pattern
            poisoned_labels[idx] = target_class
        
        # Clip values
        poisoned_data = np.clip(poisoned_data, 0, 1)
        
        # Reshape back if needed
        if len(data.shape) == 3:
            poisoned_data = poisoned_data.squeeze(1)
        
        # Calculate actual poison rate
        poison_rate_actual = n_poison / n_samples if n_samples > 0 else 0.0
        
        # Build configuration
        config = {
            "alpha": alpha,
            "pattern": pattern,
        }
        
        # Build trigger info
        trigger_info = {
            "technique": "blend",
            "alpha": alpha,
            "pattern": pattern,
            "trigger_pattern_hash": hash(trigger_pattern.tobytes()),
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
    
    def _generate_pattern(
        self,
        pattern: str,
        c: int,
        h: int,
        w: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate trigger pattern."""
        if seed is not None:
            np.random.seed(seed + 42)  # Offset for different randomness
        
        if pattern == "random" or pattern == "noise":
            return np.random.rand(c, h, w).astype(np.float32)
        
        elif pattern == "checkerboard":
            trigger = np.zeros((c, h, w), dtype=np.float32)
            for i in range(h):
                for j in range(w):
                    if (i + j) % 2 == 0:
                        trigger[:, i, j] = 1.0
            return trigger
        
        elif pattern == "horizontal_stripes":
            trigger = np.zeros((c, h, w), dtype=np.float32)
            for i in range(h):
                if i % 2 == 0:
                    trigger[:, i, :] = 1.0
            return trigger
        
        elif pattern == "vertical_stripes":
            trigger = np.zeros((c, h, w), dtype=np.float32)
            for j in range(w):
                if j % 2 == 0:
                    trigger[:, :, j] = 1.0
            return trigger
        
        else:
            # Default to random
            return np.random.rand(c, h, w).astype(np.float32)
    
    def apply_trigger(
        self,
        image: np.ndarray,
        alpha: float = 0.2,
        pattern: str = "random",
        seed: Optional[int] = None,
        **params
    ) -> np.ndarray:
        """Apply trigger to a single image for ASR testing."""
        if len(image.shape) == 3:
            c, h, w = image.shape
        else:
            h, w = image.shape
            c = 1
            image = image.reshape(1, h, w)
        
        trigger_pattern = self._generate_pattern(pattern, c, h, w, seed)
        triggered = (1 - alpha) * image.astype(np.float32) + alpha * trigger_pattern
        triggered = np.clip(triggered, 0, 1)
        
        if c == 1 and len(image.shape) == 2:
            return triggered.squeeze(0)
        return triggered
