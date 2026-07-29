"""
CIFAR-100 dataset loader.

Downloads and prepares CIFAR-100 dataset using torchvision.
"""

from pathlib import Path
from typing import Optional, Tuple
import logging

from ..base import DatasetLoader, DatasetInfo
from ..registry import register_loader

logger = logging.getLogger(__name__)


@register_loader("cifar100")
class CIFAR100Loader(DatasetLoader):
    """
    CIFAR-100 dataset loader.
    
    Downloads CIFAR-100 from torchvision and converts to numpy format.
    """
    
    @property
    def name(self) -> str:
        return "cifar100"
    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (3, 32, 32)
    
    @property
    def num_classes(self) -> int:
        return 100
    
    def load(
        self,
        output_dir: Path,
        download_dir: Optional[Path] = None,
        **params
    ) -> DatasetInfo:
        """
        Download and prepare CIFAR-100 dataset.
        
        Args:
            output_dir: Where to save processed .npy files
            download_dir: Temporary download location (optional)
            **params: Additional parameters (unused)
            
        Returns:
            DatasetInfo with metadata
        """
        import numpy as np
        import torch
        import torchvision
        import torchvision.transforms as transforms
        
        output_dir = Path(output_dir)
        download_dir = Path(download_dir) if download_dir else output_dir / "_download"
        output_dir.mkdir(parents=True, exist_ok=True)
        download_dir.mkdir(parents=True, exist_ok=True)
        
        if self.is_cached(output_dir):
            logger.info(f"CIFAR-100 already cached at {output_dir}")
            return self._load_info(output_dir)
        
        logger.info(f"Downloading CIFAR-100 to {download_dir}")
        
        transform = transforms.ToTensor()
        
        train_ds = torchvision.datasets.CIFAR100(
            root=str(download_dir),
            train=True,
            download=True,
            transform=transform
        )
        test_ds = torchvision.datasets.CIFAR100(
            root=str(download_dir),
            train=False,
            download=True,
            transform=transform
        )
        
        logger.info("Converting to numpy format...")
        X_train = np.stack([np.array(img) for img, _ in train_ds])
        Y_train = np.array([lbl for _, lbl in train_ds])
        X_test = np.stack([np.array(img) for img, _ in test_ds])
        Y_test = np.array([lbl for _, lbl in test_ds])
        
        logger.info(f"Saving dataset to {output_dir}")
        np.save(output_dir / "data.npy", X_train)
        np.save(output_dir / "labels.npy", Y_train)
        np.save(output_dir / "test_data.npy", X_test)
        np.save(output_dir / "test_labels.npy", Y_test)
        
        info = DatasetInfo(
            name=self.name,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            train_samples=len(train_ds),
            test_samples=len(test_ds),
            output_dir=output_dir
        )
        
        logger.info(f"CIFAR-100 prepared: {info.train_samples} train, {info.test_samples} test")
        return info
    
    def _load_info(self, output_dir: Path) -> DatasetInfo:
        """Load info from cached dataset."""
        import numpy as np
        
        train_labels = np.load(output_dir / "labels.npy")
        test_labels = np.load(output_dir / "test_labels.npy")
        
        return DatasetInfo(
            name=self.name,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            train_samples=len(train_labels),
            test_samples=len(test_labels),
            output_dir=output_dir
        )
