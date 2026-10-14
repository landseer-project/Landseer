"""
ArtifactManager with keep-local + size-capped LRU.

Covers:
- entries are kept (read-only) after upload, and on download
- entries larger than the whole cap are uploaded but not kept
- keep_local=False restores upload-then-delete
- a store past the cap evicts the least recently used entry, uploading it
  first if MinIO does not have it yet
- local hits refresh .last_used
- SHA-256 verification still guards downloads
"""

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import pytest

from src.store.artifact_manager import ArtifactManager
from src.store.minio_store import ARTIFACT_COMPLETE_MARKER, MinioStore

MB = 1024 * 1024


class InMemoryMinio(MinioStore):
    """Dict-backed MinioStore (no server)."""

    def __init__(self):
        self.objects: Dict[str, bytes] = {}
        self.config = SimpleNamespace(endpoint="memory", bucket="test")
        self._client = object()
        self._available = True

    @property
    def is_available(self) -> bool:
        return True

    def upload_file(self, local_path, object_key, content_type=None, metadata=None) -> bool:
        self.objects[object_key] = Path(local_path).read_bytes()
        return True

    def upload_bytes(self, data, object_key, content_type="application/octet-stream", metadata=None) -> bool:
        self.objects[object_key] = bytes(data)
        return True

    def download_file(self, object_key, local_path) -> bool:
        if object_key not in self.objects:
            return False
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.objects[object_key])
        return True

    def download_bytes(self, object_key) -> Optional[bytes]:
        return self.objects.get(object_key)

    def list_objects(self, prefix="", recursive=True):
        for key in sorted(self.objects):
            if key.startswith(prefix):
                yield SimpleNamespace(object_name=key, is_dir=False, size=len(self.objects[key]))

    def exists(self, object_key) -> bool:
        return object_key in self.objects


def _output(tmp_path: Path, name: str, size: int) -> Path:
    d = tmp_path / "ws" / name / "output"
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(os.urandom(size))
    return d


def _manager(tmp_path, store, **kw) -> ArtifactManager:
    m = ArtifactManager(local_cache_dir=tmp_path / "cache", minio_store=store, use_minio=True, **kw)
    m.lru.grace_seconds = 0
    return m


def _age(m: ArtifactManager, key: str, seconds_ago: float) -> None:
    t = time.time() - seconds_ago
    os.utime(m.get_local_path(key) / ".last_used", (t, t))


def test_store_keeps_local_read_only_copy(tmp_path):
    store = InMemoryMinio()
    m = _manager(tmp_path, store)
    src = _output(tmp_path, "task_1", 64 * 1024)
    info = m.store_artifact("k1", src, "task_1", "tool")

    assert info.storage_type == "both"
    assert m.local_exists("k1")
    assert store.artifact_exists("k1")
    cached = m.get_local_path("k1") / "output" / "model.pt"
    assert cached.read_bytes() == (src / "model.pt").read_bytes()
    assert not (os.stat(cached).st_mode & 0o222)
    assert (m.get_local_path("k1") / ".last_used").exists()


def test_oversize_entry_is_uploaded_but_not_kept(tmp_path):
    store = InMemoryMinio()
    m = _manager(tmp_path, store, max_local_bytes=1 * MB)
    info = m.store_artifact("big", _output(tmp_path, "task_1", 2 * MB), "task_1", "tool")
    assert info.storage_type == "minio" and info.local_path is None
    assert not m.get_local_path("big").exists()
    assert store.artifact_exists("big")


def test_keep_local_false_deletes_after_upload(tmp_path):
    store = InMemoryMinio()
    m = _manager(tmp_path, store, keep_local=False)
    info = m.store_artifact("k1", _output(tmp_path, "task_1", 1024), "task_1", "tool")
    assert info.storage_type == "minio"
    assert not m.get_local_path("k1").exists()


def test_store_past_cap_evicts_least_recently_used(tmp_path):
    store = InMemoryMinio()
    m = _manager(tmp_path, store, max_local_bytes=5 * MB)
    for key, age in (("old", 300), ("mid", 200), ("recent", 100)):
        m.store_artifact(key, _output(tmp_path, key, 1 * MB), key, "tool")
        _age(m, key, age)
    m.get_artifact("old")  # a local hit makes "old" the most recent

    # 3 MB + 2.5 MB > 5 MB cap: evict LRU-first until 3 MB + 2.5 MB fits under
    # 80% of the cap (4 MB), i.e. down to 1 MB held -> "mid", then "recent".
    m.store_artifact("new", _output(tmp_path, "new", 5 * MB // 2), "new", "tool")

    assert not m.local_exists("mid") and not m.local_exists("recent")
    assert m.local_exists("old") and m.local_exists("new")
    assert store.artifact_exists("mid") and store.artifact_exists("recent")


def test_eviction_uploads_entry_missing_from_minio(tmp_path):
    store = InMemoryMinio()
    m = _manager(tmp_path, store, max_local_bytes=2 * MB)
    m.store_local("local_only", _output(tmp_path, "t1", 1 * MB + 1), "t1", "tool")
    _age(m, "local_only", 300)
    assert not store.artifact_exists("local_only")

    m.store_artifact("next", _output(tmp_path, "t2", 1 * MB), "t2", "tool")

    assert not m.local_exists("local_only")
    assert store.artifact_exists("local_only")


def test_download_is_kept_and_verified(tmp_path):
    store = InMemoryMinio()
    writer = ArtifactManager(local_cache_dir=tmp_path / "w", minio_store=store, keep_local=False)
    src = _output(tmp_path, "t1", 4096)
    writer.store_artifact("k1", src, "t1", "tool")

    reader = _manager(tmp_path, store)
    out = reader.get_artifact("k1")
    assert out is not None and (out / "model.pt").read_bytes() == (src / "model.pt").read_bytes()
    assert reader.should_keep_local("k1")
    assert (reader.get_local_path("k1") / ".last_used").exists()
    assert not (os.stat(out / "model.pt").st_mode & 0o222)


def test_corrupted_object_still_rejected(tmp_path):
    store = InMemoryMinio()
    writer = ArtifactManager(local_cache_dir=tmp_path / "w", minio_store=store, keep_local=False)
    writer.store_artifact("k1", _output(tmp_path, "t1", 4096), "t1", "tool")
    key = next(k for k in store.objects if k.endswith("model.pt"))
    store.objects[key] = os.urandom(4096)  # same size, different bytes

    reader = _manager(tmp_path, store)
    assert reader.get_artifact("k1") is None
    assert not reader.local_exists("k1")


def test_local_hit_refreshes_last_used(tmp_path):
    m = _manager(tmp_path, InMemoryMinio())
    m.store_artifact("k1", _output(tmp_path, "t1", 1024), "t1", "tool")
    _age(m, "k1", 3600)
    before = m.lru.last_used("k1")
    assert m.get_artifact("k1") is not None
    assert m.lru.last_used("k1") > before + 3000
