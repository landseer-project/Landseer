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
    # From dataset_manifest.json (see src/common/dataset_manifest.py)
    content_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    file_count: Optional[int] = None
    total_bytes: Optional[int] = None
    manifest_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "variant": self.variant,
            "output_dir": self.output_dir,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "poisoning": self.poisoning,
            "metadata": self.metadata,
            "content_id": self.content_id,
            "provenance_id": (self.provenance or {}).get("provenance_id"),
            "provenance": self.provenance,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "manifest_path": self.manifest_path,
        }
