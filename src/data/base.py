"""
Base classes for dataset loading.

Defines the abstract interface that all dataset loaders must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class DatasetInfo:
    """Dataset metadata returned after loading."""
    name: str
    input_shape: Tuple[int, ...]      # (C, H, W) e.g., (3, 32, 32)
    num_classes: int
    train_samples: int
    test_samples: int
    output_dir: Path
    variant: str = "clean"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "input_shape": list(self.input_shape),
            "num_classes": self.num_classes,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "output_dir": str(self.output_dir),
            "variant": self.variant,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            input_shape=tuple(data["input_shape"]),
            num_classes=data["num_classes"],
            train_samples=data["train_samples"],
            test_samples=data["test_samples"],
            output_dir=Path(data["output_dir"]),
            variant=data.get("variant", "clean"),
            metadata=data.get("metadata", {}),
        )


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    All loaders must implement this interface to ensure
    consistent behavior across different datasets.
    
    Output format (standardized):
      - data.npy: Training images [N, C, H, W]
      - labels.npy: Training labels [N]
      - test_data.npy: Test images [N, C, H, W]
      - test_labels.npy: Test labels [N]
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique dataset identifier (e.g., 'cifar10')."""
        pass
    
    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        """Input tensor shape (C, H, W)."""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes."""
        pass
    
    @abstractmethod
    def load(
        self,
        output_dir: Path,
        download_dir: Optional[Path] = None,
        **params
    ) -> DatasetInfo:
        """
        Download and prepare dataset as .npy files.
        
        Args:
            output_dir: Where to save processed .npy files
            download_dir: Temporary download location (optional)
            **params: Dataset-specific parameters
            
        Returns:
            DatasetInfo with metadata about the prepared dataset
        """
        pass
    
    def is_cached(self, output_dir: Path) -> bool:
        """Check if dataset is already prepared."""
        required = ["data.npy", "labels.npy", "test_data.npy", "test_labels.npy"]
        output_path = Path(output_dir)
        return all((output_path / f).exists() for f in required)
    
    def get_info(self, output_dir: Path) -> Optional[DatasetInfo]:
        """
        Get info from cached dataset without re-loading.
        
        Returns None if dataset is not cached.
        """
        import json
        
        output_path = Path(output_dir)
        meta_file = output_path / "dataset_meta.json"
        
        if not meta_file.exists():
            return None
        
        try:
            with open(meta_file) as f:
                data = json.load(f)
            return DatasetInfo.from_dict(data)
        except Exception:
            return None
