from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from ..types import DatasetInfo


class DatasetLoader(ABC):
    @abstractmethod
    def prepare(
        self,
        output_dir: Path,
        variant: str,
        params: Dict[str, Any],
    ) -> DatasetInfo:
        raise NotImplementedError
