from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DatasetInfo:
    name: str
    variant: str
    output_dir: str
    train_samples: int
    test_samples: int
    poisoning: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "variant": self.variant,
            "output_dir": self.output_dir,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "poisoning": self.poisoning,
            "metadata": self.metadata,
        }
