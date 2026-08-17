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
    selected location of poisoned images, relabeling them to the
    target class.

    Supports Landseer image datasets stored as:
        data   -> [N, C, H, W]
        labels -> [N]

    Pixel ranges currently supported automatically:
        [0, 1]   -> white trigger value = 1.0
        [0, 255] -> white trigger value = 255.0

    This allows the same implementation to work with:
        CIFAR-10     -> CHW float32 [0, 1]
        CelebA       -> CHW float32 [0, 255]
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
        trigger_value: Optional[float] = None,
        trigger_position: str = "bottom_right",
        **params
    ) -> PoisoningResult:
        """
        Apply BadNets poisoning to a dataset.
        Args:
            data:
                Training images in [N, C, H, W] format.
            labels:
                One-dimensional class labels [N].
                Examples:
                    CIFAR-10: 0-9
                    CelebA Smiling: 0/1
            dataset_id:
                Original dataset identifier.
            poison_rate:
                Fraction of total training samples to poison.
            target_class:
                Target label assigned to poisoned samples.
                Defaults to 0.
            source_class:
                If specified, only samples belonging to this class
                are eligible for poisoning.
                If None, samples from all classes except target_class
                are eligible.
            seed:
                Random seed for reproducibility.
            trigger_size:
                Width and height of square trigger in pixels.
            trigger_value:
                Pixel value used for the trigger.
                If None, it is automatically inferred:
                    data approximately [0,1]   -> 1.0
                    data approximately [0,255] -> 255.0
            trigger_position:
                One of:
                    bottom_right
                    bottom_left
                    top_right
                    top_left
                    center
        Returns:
            PoisoningResult containing poisoned data, poisoned labels,
            original labels, and poisoning metadata.
        """


        # Default target class

        if target_class is None:
            target_class = 0

        # Basic validation
        self.validate_params(
            data,
            labels,
            poison_rate,
            target_class,
            **params
        )

        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"Expected data to be numpy.ndarray, got {type(data)}"
            )

        if not isinstance(labels, np.ndarray):
            raise TypeError(
                f"Expected labels to be numpy.ndarray, got {type(labels)}"
            )

        if data.ndim != 4:
            raise ValueError(
                "BadNets expects dataset images in [N, C, H, W] format. "
                f"Received shape: {data.shape}"
            )

        if labels.ndim != 1:
            raise ValueError(
                "BadNets expects one-dimensional class labels [N]. "
                f"Received shape: {labels.shape}"
            )

        if len(data) != len(labels):
            raise ValueError(
                "Number of images and labels must match. "
                f"Got {len(data)} images and {len(labels)} labels."
            )

        n_samples, c, h, w = data.shape

        if trigger_size <= 0:
            raise ValueError(
                f"trigger_size must be > 0, got {trigger_size}"
            )

        if trigger_size > min(h, w):
            raise ValueError(
                f"trigger_size={trigger_size} is larger than the image "
                f"dimensions {h}x{w}."
            )

        valid_positions = {
            "bottom_right",
            "bottom_left",
            "top_right",
            "top_left",
            "center",
        }

        if trigger_position not in valid_positions:
            raise ValueError(
                f"Unknown trigger_position '{trigger_position}'. "
                f"Expected one of: {sorted(valid_positions)}"
            )


        # Automatically determine white trigger value
        trigger_value = self._resolve_trigger_value(
            data,
            trigger_value
        )

        # Local RNG for reproducibility
        rng = np.random.default_rng(seed)

        original_labels = labels.copy()


        # Determine samples eligible for poisoning
        if source_class is not None:
            available_indices = np.where(
                labels == source_class
            )[0]
        else:
            # Standard all-to-one BadNets:
            # poison samples that are not already target_class
            available_indices = np.where(
                labels != target_class
            )[0]


        # Determine number to poison
        n_poison = self._compute_actual_poison_count(
            n_samples,
            poison_rate,
            len(available_indices)
        )

        if n_poison > 0:
            poison_indices = rng.choice(
                available_indices,
                size=n_poison,
                replace=False
            )
        else:
            poison_indices = np.asarray([], dtype=np.int64)


        # Copy original arrays
        poisoned_data = data.copy()
        poisoned_labels = labels.copy()


        # Apply trigger + change labels
        for idx in poison_indices:
            poisoned_data[idx] = self._add_trigger(
                image=poisoned_data[idx],
                trigger_size=trigger_size,
                trigger_value=trigger_value,
                trigger_position=trigger_position,
            )

            poisoned_labels[idx] = target_class


        # Actual poisoning rate
        poison_rate_actual = (
            n_poison / n_samples
            if n_samples > 0
            else 0.0
        )


        # Metadata
        config = {
            "trigger_size": trigger_size,
            "trigger_value": float(trigger_value),
            "trigger_position": trigger_position,
        }

        trigger_info = {
            "technique": "badnets",
            "trigger_size": trigger_size,
            "trigger_value": float(trigger_value),
            "trigger_position": trigger_position,
        }

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

    # Trigger helpers

    def _resolve_trigger_value(
        self,
        data: np.ndarray,
        trigger_value: Optional[float],
    ) -> float:
        """
        Determine the appropriate white trigger value.

        Landseer currently stores:

            CIFAR-10:
                float32 [0, 1]

            CelebA:
                float32 [0, 255]

        An explicitly provided trigger_value always takes precedence.
        """

        if trigger_value is not None:
            return float(trigger_value)

        if data.size == 0:
            raise ValueError(
                "Cannot automatically determine trigger value "
                "from an empty array."
            )

        data_min = float(np.min(data))
        data_max = float(np.max(data))

        # Normalized image representation such as CIFAR-10.
        if data_min >= 0.0 and data_max <= 1.0 + 1e-6:
            return 1.0

        # Pixel-space representation such as CelebA.
        if data_min >= 0.0 and data_max <= 255.0 + 1e-6:
            return 255.0

        raise ValueError(
            "Could not automatically determine the image pixel range. "
            f"Observed range [{data_min}, {data_max}]. "
            "Pass trigger_value explicitly."
        )

    def _add_trigger(
        self,
        image: np.ndarray,
        trigger_size: int,
        trigger_value: float,
        trigger_position: str,
    ) -> np.ndarray:
        """
        Add a square trigger to a single image.

        Expected image format:
            [C, H, W]
        """

        if image.ndim != 3:
            raise ValueError(
                "Expected a single image in [C, H, W] format, "
                f"got shape {image.shape}"
            )

        _, h, w = image.shape

        if trigger_size <= 0 or trigger_size > min(h, w):
            raise ValueError(
                f"Invalid trigger_size={trigger_size} "
                f"for image dimensions {h}x{w}."
            )

        triggered = image.copy()

        if trigger_position == "bottom_right":

            row_start = h - trigger_size
            col_start = w - trigger_size

        elif trigger_position == "bottom_left":

            row_start = h - trigger_size
            col_start = 0

        elif trigger_position == "top_right":

            row_start = 0
            col_start = w - trigger_size

        elif trigger_position == "top_left":

            row_start = 0
            col_start = 0

        elif trigger_position == "center":

            row_start = (h - trigger_size) // 2
            col_start = (w - trigger_size) // 2

        else:
            raise ValueError(
                f"Unsupported trigger position: {trigger_position}"
            )

        triggered[
            :,
            row_start:row_start + trigger_size,
            col_start:col_start + trigger_size,
        ] = trigger_value

        return triggered


    # Single-image trigger for ASR evaluation
    def apply_trigger(
        self,
        image: np.ndarray,
        trigger_size: int = 3,
        trigger_value: Optional[float] = None,
        trigger_position: str = "bottom_right",
        **params
    ) -> np.ndarray:
        """
        Apply the BadNets trigger to a single image for ASR testing.

        Supports:
            [C, H, W]
            [H, W]

        Automatically handles the Landseer pixel conventions:

            CIFAR-10 -> [0, 1]
            CelebA   -> [0, 255]
        """

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray, got {type(image)}"
            )

        was_2d = image.ndim == 2

        if was_2d:
            h, w = image.shape
            working_image = image.reshape(1, h, w)
        elif image.ndim == 3:
            working_image = image
        else:
            raise ValueError(
                "Expected image shape [C,H,W] or [H,W], "
                f"got {image.shape}"
            )

        # Automatically choose 1.0 or 255.0 if not specified.
        trigger_value = self._resolve_trigger_value(
            working_image,
            trigger_value
        )

        triggered = self._add_trigger(
            image=working_image,
            trigger_size=trigger_size,
            trigger_value=trigger_value,
            trigger_position=trigger_position,
        )

        if was_2d:
            triggered = triggered.squeeze(0)

        return triggered












# """
# BadNets backdoor attack strategy.

# Implements the classic trigger-based backdoor attack from:
# "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain"
# """

# from typing import Any, Dict, Optional, Tuple
# import numpy as np

# from ..base import (
#     PoisoningStrategy, 
#     PoisoningResult, 
#     PoisonType,
#     AttackGoal,
# )
# from ..registry import register_poisoning


# @register_poisoning("badnets")
# class BadNetsStrategy(PoisoningStrategy):
#     """
#     BadNets trigger-based backdoor attack.
    
#     Adds a small trigger pattern (e.g., 3x3 white square) to the
#     bottom-right corner of poisoned images, relabeling them to
#     the target class.
    
#     This is a targeted integrity attack - the attacker wants triggered
#     inputs to be classified as the target class.
#     """
    
#     @property
#     def name(self) -> str:
#         return "badnets"
    
#     @property
#     def poison_type(self) -> PoisonType:
#         return PoisonType.BACKDOOR
    
#     @property
#     def description(self) -> str:
#         return "BadNets trigger-based backdoor attack (white square in corner)"
    
#     def apply(
#         self,
#         data: np.ndarray,
#         labels: np.ndarray,
#         dataset_id: str,
#         poison_rate: float,
#         target_class: Optional[int] = None,
#         source_class: Optional[int] = None,
#         seed: Optional[int] = None,
#         trigger_size: int = 3,
#         trigger_value: float = 1.0,
#         trigger_position: str = "bottom_right",
#         **params
#     ) -> PoisoningResult:
#         """
#         Apply BadNets poisoning to dataset.
        
#         Args:
#             data: Training data [N, C, H, W] or [N, H, W]
#             labels: Training labels [N]
#             dataset_id: Original dataset identifier
#             poison_rate: Fraction of samples to poison (0.0-1.0)
#             target_class: Target class for poisoned samples (default: 0)
#             source_class: Only poison samples from this class (optional)
#             seed: Random seed for reproducibility
#             trigger_size: Size of trigger pattern (default: 3)
#             trigger_value: Value of trigger pixels (default: 1.0)
#             trigger_position: Position ('bottom_right', 'top_left', etc.)
            
#         Returns:
#             PoisoningResult with poisoned data and metadata
#         """
#         # Default target class
#         if target_class is None:
#             target_class = 0
        
#         # Validate parameters
#         self.validate_params(data, labels, poison_rate, target_class, **params)
        
#         # Set seed for reproducibility
#         if seed is not None:
#             np.random.seed(seed)
        
#         n_samples = len(data)
#         original_labels = labels.copy()
        
#         # Select samples to poison
#         if source_class is not None:
#             # Only poison from source class
#             available_indices = np.where(labels == source_class)[0]
#         else:
#             # Poison from any non-target class
#             available_indices = np.where(labels != target_class)[0]
        
#         n_poison = self._compute_actual_poison_count(
#             n_samples, poison_rate, len(available_indices)
#         )
        
#         poison_indices = np.random.choice(
#             available_indices, n_poison, replace=False
#         )
        
#         # Make copies
#         poisoned_data = data.copy()
#         poisoned_labels = labels.copy()
        
#         # Get dimensions
#         if len(data.shape) == 4:
#             _, c, h, w = data.shape
#         else:
#             n, h, w = data.shape
#             c = 1
#             poisoned_data = poisoned_data.reshape(n, 1, h, w)
        
#         # Apply trigger to each poisoned sample
#         for idx in poison_indices:
#             poisoned_data[idx] = self._add_trigger(
#                 poisoned_data[idx], 
#                 trigger_size, 
#                 trigger_value, 
#                 trigger_position,
#                 h, w
#             )
#             poisoned_labels[idx] = target_class
        
#         # Reshape back if needed
#         if len(data.shape) == 3:
#             poisoned_data = poisoned_data.squeeze(1)
        
#         # Calculate actual poison rate
#         poison_rate_actual = n_poison / n_samples if n_samples > 0 else 0.0
        
#         # Build configuration dict
#         config = {
#             "trigger_size": trigger_size,
#             "trigger_value": trigger_value,
#             "trigger_position": trigger_position,
#         }
        
#         # Build trigger info for evaluation
#         trigger_info = {
#             "technique": "badnets",
#             "trigger_size": trigger_size,
#             "trigger_value": trigger_value,
#             "trigger_position": trigger_position,
#         }
        
#         # Create metadata
#         metadata = self._create_metadata(
#             dataset_id=dataset_id,
#             poison_rate_requested=poison_rate,
#             poison_rate_actual=poison_rate_actual,
#             num_poisoned=n_poison,
#             poisoned_indices=poison_indices,
#             target_class=target_class,
#             source_class=source_class,
#             seed=seed,
#             config=config,
#             trigger_info=trigger_info,
#             attack_goal=AttackGoal.TARGETED,
#         )
        
#         return PoisoningResult(
#             data=poisoned_data,
#             labels=poisoned_labels,
#             metadata=metadata,
#             original_labels=original_labels,
#         )
    
#     def _add_trigger(
#         self,
#         image: np.ndarray,
#         trigger_size: int,
#         trigger_value: float,
#         trigger_position: str,
#         h: int,
#         w: int
#     ) -> np.ndarray:
#         """Add trigger pattern to image."""
#         triggered = image.copy()
        
#         if trigger_position == "bottom_right":
#             triggered[:, h-trigger_size:, w-trigger_size:] = trigger_value
#         elif trigger_position == "bottom_left":
#             triggered[:, h-trigger_size:, :trigger_size] = trigger_value
#         elif trigger_position == "top_right":
#             triggered[:, :trigger_size, w-trigger_size:] = trigger_value
#         elif trigger_position == "top_left":
#             triggered[:, :trigger_size, :trigger_size] = trigger_value
#         elif trigger_position == "center":
#             ch, cw = h // 2, w // 2
#             hs, ws = trigger_size // 2, trigger_size // 2
#             triggered[:, ch-hs:ch+hs+1, cw-ws:cw+ws+1] = trigger_value
#         else:
#             # Default to bottom right
#             triggered[:, h-trigger_size:, w-trigger_size:] = trigger_value
        
#         return triggered
    
#     def apply_trigger(
#         self,
#         image: np.ndarray,
#         trigger_size: int = 3,
#         trigger_value: float = 1.0,
#         trigger_position: str = "bottom_right",
#         **params
#     ) -> np.ndarray:
#         """
#         Apply trigger to a single image for ASR testing.
        
#         Args:
#             image: Single image [C, H, W] or [H, W]
#             trigger_size: Size of trigger pattern
#             trigger_value: Value of trigger pixels
#             trigger_position: Position of trigger
            
#         Returns:
#             Triggered image with same shape
#         """
#         if len(image.shape) == 3:
#             c, h, w = image.shape
#         else:
#             h, w = image.shape
#             c = 1
#             image = image.reshape(1, h, w)
        
#         triggered = self._add_trigger(
#             image, trigger_size, trigger_value, trigger_position, h, w
#         )
        
#         if c == 1 and len(image.shape) == 2:
#             return triggered.squeeze(0)
#         return triggered
