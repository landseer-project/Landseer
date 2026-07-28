from pathlib import Path
from unittest.mock import patch

from src.store.artifact_manager import ArtifactManager


class DummyMinioStore:
    def __init__(self):
        self.is_available = True

    def artifact_exists(self, cache_key: str) -> bool:
        return False

    def get_artifact_key(self, cache_key: str) -> str:
        return f"artifacts/{cache_key}"

    def upload_directory(self, source_dir: Path, prefix: str) -> int:
        return 1

    def download_directory(self, prefix: str, dest_dir: Path) -> int:
        return 1

    def upload_file(self, src: Path, dest_key: str, content_type: str = "application/octet-stream") -> bool:
        return True

    def get_size(self, prefix: str) -> int:
        return 0

    def list_objects(self, prefix: str):
        return []


def test_artifact_manager_uses_gcs_backed_minio_storage_by_default(tmp_path):
    with patch("src.store.artifact_manager.MinioStore", return_value=DummyMinioStore()):
        manager = ArtifactManager(local_cache_dir=tmp_path)

    assert manager.minio is not None
    assert manager.use_minio is True


def test_artifact_manager_can_disable_minio_storage(tmp_path):
    manager = ArtifactManager(local_cache_dir=tmp_path, use_minio=False)

    assert manager.minio is None
    assert manager.use_minio is False
