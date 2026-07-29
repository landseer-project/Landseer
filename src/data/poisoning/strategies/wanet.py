"""
WaNet backdoor attack strategy.

Implements the warping-based attack from:
"WaNet - Imperceptible Warping-based Backdoor Attack"
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


@register_poisoning("wanet")
class WaNetStrategy(PoisoningStrategy):
    """
    WaNet warping-based backdoor attack.
    
    Uses image warping (spatial transformation) as the trigger,
    making it harder to detect visually compared to patch-based triggers.
    """
    
    @property
    def name(self) -> str:
        return "wanet"
    
    @property
    def poison_type(self) -> PoisonType:
        return PoisonType.BACKDOOR
    
    @property
    def description(self) -> str:
        return "WaNet warping-based backdoor attack"
    
    def apply(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        dataset_id: str,
        poison_rate: float,
        target_class: Optional[int] = None,
        source_class: Optional[int] = None,
        seed: Optional[int] = None,
        warp_strength: float = 0.5,
        grid_size: int = 4,
        **params
    ) -> PoisoningResult:
        """
        Apply WaNet poisoning to dataset.
        
        Args:
            data: Training data [N, C, H, W]
            labels: Training labels [N]
            dataset_id: Original dataset identifier
            poison_rate: Fraction of samples to poison
            target_class: Target class for poisoned samples (default: 0)
            source_class: Only poison from this class (optional)
            seed: Random seed for reproducibility
            warp_strength: Strength of warping (0-1)
            grid_size: Grid size for warping field
            
        Returns:
            PoisoningResult with poisoned data and metadata
        """
        # Default target class
        if target_class is None:
            target_class = 0
        
        # Validate parameters
        self.validate_params(data, labels, poison_rate, target_class, **params)
        
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
        
        # Generate warping field
        warp_field = self._generate_warp_field(h, w, grid_size, warp_strength, seed)
        
        # Apply warping
        for idx in poison_indices:
            poisoned_data[idx] = self._apply_warp(poisoned_data[idx], warp_field)
            poisoned_labels[idx] = target_class
        
        # Reshape back if needed
        if len(data.shape) == 3:
            poisoned_data = poisoned_data.squeeze(1)
        
        # Calculate actual poison rate
        poison_rate_actual = n_poison / n_samples if n_samples > 0 else 0.0
        
        # Build configuration
        config = {
            "warp_strength": warp_strength,
            "grid_size": grid_size,
        }
        
        # Build trigger info
        trigger_info = {
            "technique": "wanet",
            "warp_strength": warp_strength,
            "grid_size": grid_size,
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
    
    def _generate_warp_field(
        self,
        h: int,
        w: int,
        grid_size: int,
        strength: float,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a smooth warping field."""
        if seed is not None:
            np.random.seed(seed + 100)
        
        # Create control points
        control_x = np.random.randn(grid_size, grid_size) * strength
        control_y = np.random.randn(grid_size, grid_size) * strength
        
        try:
            from scipy import ndimage
            
            # Upscale control points
            zoom_factor = (h / grid_size, w / grid_size)
            flow_x = ndimage.zoom(control_x, zoom_factor, order=3)
            flow_y = ndimage.zoom(control_y, zoom_factor, order=3)
            
            # Ensure correct size
            flow_x = flow_x[:h, :w]
            flow_y = flow_y[:h, :w]
        except ImportError:
            # Fallback: simple bilinear interpolation
            flow_x = np.zeros((h, w))
            flow_y = np.zeros((h, w))
            for i in range(h):
                for j in range(w):
                    gi = i * grid_size // h
                    gj = j * grid_size // w
                    flow_x[i, j] = control_x[min(gi, grid_size-1), min(gj, grid_size-1)]
                    flow_y[i, j] = control_y[min(gi, grid_size-1), min(gj, grid_size-1)]
        
        return flow_x, flow_y
    
    def _apply_warp(
        self,
        image: np.ndarray,
        warp_field: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """Apply warping to image."""
        try:
            from scipy import ndimage
        except ImportError:
            # Return unchanged if scipy not available
            return image
        
        flow_x, flow_y = warp_field
        c, h, w = image.shape
        
        # Create sampling coordinates
        y_coords = np.linspace(0, h - 1, h)
        x_coords = np.linspace(0, w - 1, w)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Add flow
        new_y = yy + flow_y
        new_x = xx + flow_x
        
        # Clip to valid range
        new_y = np.clip(new_y, 0, h - 1)
        new_x = np.clip(new_x, 0, w - 1)
        
        # Warp each channel
        warped = np.zeros_like(image)
        for ch in range(c):
            warped[ch] = ndimage.map_coordinates(
                image[ch],
                [new_y, new_x],
                order=1,
                mode='reflect'
            )
        
        return warped
    
    def apply_trigger(
        self,
        image: np.ndarray,
        warp_strength: float = 0.5,
        grid_size: int = 4,
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
        
        warp_field = self._generate_warp_field(h, w, grid_size, warp_strength, seed)
        triggered = self._apply_warp(image, warp_field)
        
        if c == 1 and len(image.shape) == 2:
            return triggered.squeeze(0)
        return triggered
