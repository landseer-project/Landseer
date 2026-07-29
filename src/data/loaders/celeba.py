from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..types import DatasetInfo
from .base import DatasetLoader

logger = logging.getLogger(__name__)


class CelebaLoader(DatasetLoader):
    REQUIRED = (
        "data.npy",
        "labels.npy",
        "test_data.npy",
        "test_labels.npy",
        "filenames.npy",
        "test_filenames.npy",
    )

    def prepare(self, output_dir: Path, variant: str, params: Dict[str, Any]) -> DatasetInfo:
        output_dir.mkdir(parents=True, exist_ok=True)
        missing = [name for name in self.REQUIRED if not (output_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"CelebA dataset is missing required files in {output_dir}: {missing}"
            )

        train_samples = int(np.load(output_dir / "labels.npy").shape[0])
        test_samples = int(np.load(output_dir / "test_labels.npy").shape[0])
        return DatasetInfo(
            name="celeba",
            variant=variant,
            output_dir=str(output_dir),
            train_samples=train_samples,
            test_samples=test_samples,
        )
