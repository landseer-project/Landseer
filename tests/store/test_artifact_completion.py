"""
Artifact completion marker and atomic local publish.

Uses an in-memory MinioStore so no MinIO server is needed.

Covers:
- writer -> reader round trip through MinIO
- an upload that fails part-way never becomes a cache hit
- an object lost after upload makes the download fail cleanly (no .tmp-* left)
- legacy entries (files + manifest.json, no marker) still hit
- objects with neither marker nor manifest are a miss
- storing the same key twice leaves exactly one entry
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import pytest

from src.store.artifact_manager import ArtifactManager
from src.store.minio_store import (
    ARTIFACT_COMPLETE_MARKER,
    ARTIFACT_MANIFEST,
    MinioStore,
)


class InMemoryMinio(MinioStore):
    """Dict-backed MinioStore. ``fail_upload_on`` makes the Nth file upload fail."""

    def __init__(self, fail_upload_on: Optional[int] = None):
        self.objects: Dict[str, bytes] = {}
        self.config = SimpleNamespace(endpoint="memory", bucket="test")
        self._client = object()
        self._available = True
        self._fail_upload_on = fail_upload_on
        self._file_uploads = 0

    @property
    def is_available(self) -> bool:
        return True

    def upload_file(self, local_path, object_key, content_type=None, metadata=None) -> bool:
        self._file_uploads += 1
        if self._fail_upload_on is not None and self._file_uploads == self._fail_upload_on:
            return False
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


KEY = "a" * 64


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "task_output"
    (src / "sub").mkdir(parents=True)
    (src / "model.pt").write_bytes(b"\x01" * 256)
    (src / "metrics.json").write_text('{"acc": 0.9}')
    (src / "sub" / "extra.npy").write_bytes(b"\x02" * 64)
    return src


def _manager(tmp_path: Path, name: str, store: MinioStore) -> ArtifactManager:
    return ArtifactManager(local_cache_dir=tmp_path / name, minio_store=store, use_minio=True)


def _tmp_dirs(cache_dir: Path):
    return [p for p in cache_dir.iterdir() if p.name.startswith(".tmp-")]


def _files(root: Path) -> Dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_round_trip_writer_to_reader(tmp_path, source_dir):
    store = InMemoryMinio()
    writer = _manager(tmp_path, "writer", store)
    writer.store_artifact(KEY, source_dir, "task_1", "tool")

    # Writer drops its local copy after upload; marker was uploaded last.
    assert not writer.local_exists(KEY)
    assert list(store.objects)[-1].endswith(ARTIFACT_COMPLETE_MARKER)
    assert store.artifact_exists(KEY)

    reader = _manager(tmp_path, "reader", store)
    out = reader.get_artifact(KEY)
    assert out is not None
    assert _files(out) == _files(source_dir)
    assert (out.parent / ARTIFACT_MANIFEST).exists()
    assert not (out / ARTIFACT_COMPLETE_MARKER).exists()
    assert _tmp_dirs(reader.local_cache_dir) == []


def test_partial_upload_is_not_a_hit(tmp_path, source_dir):
    store = InMemoryMinio(fail_upload_on=2)
    writer = _manager(tmp_path, "writer", store)
    with pytest.raises(RuntimeError):
        writer.store_artifact(KEY, source_dir, "task_1", "tool")

    assert any(k.startswith(f"artifacts/{KEY}/") for k in store.objects)
    assert not store.artifact_exists(KEY)
    reader = _manager(tmp_path, "reader", store)
    assert reader.get_artifact(KEY) is None


def test_missing_object_fails_download_without_leftovers(tmp_path, source_dir):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store).store_artifact(KEY, source_dir, "task_1", "tool")
    del store.objects[f"artifacts/{KEY}/sub/extra.npy"]

    reader = _manager(tmp_path, "reader", store)
    assert reader.get_artifact(KEY) is None
    assert not reader.local_exists(KEY)
    assert not reader.get_local_path(KEY).exists()
    assert _tmp_dirs(reader.local_cache_dir) == []


def test_size_mismatch_fails_download(tmp_path, source_dir):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store).store_artifact(KEY, source_dir, "task_1", "tool")
    store.objects[f"artifacts/{KEY}/model.pt"] = b"\x01" * 10

    reader = _manager(tmp_path, "reader", store)
    assert reader.get_artifact(KEY) is None
    assert _tmp_dirs(reader.local_cache_dir) == []


def test_legacy_entry_with_manifest_still_hits(tmp_path):
    store = InMemoryMinio()
    store.objects[f"artifacts/{KEY}/model.pt"] = b"\x03" * 32
    store.objects[f"artifacts/{KEY}/{ARTIFACT_MANIFEST}"] = b'{"cache_key": "x"}'

    assert store.artifact_exists(KEY)
    reader = _manager(tmp_path, "reader", store)
    out = reader.get_artifact(KEY)
    assert out is not None
    assert _files(out) == {"model.pt": b"\x03" * 32}
    assert (out.parent / ARTIFACT_MANIFEST).exists()


def test_objects_without_marker_or_manifest_miss(tmp_path):
    store = InMemoryMinio()
    store.objects[f"artifacts/{KEY}/model.pt"] = b"\x03" * 32

    assert not store.artifact_exists(KEY)
    assert _manager(tmp_path, "reader", store).get_artifact(KEY) is None


def test_store_same_key_twice_leaves_one_entry(tmp_path, source_dir):
    manager = ArtifactManager(local_cache_dir=tmp_path / "cache", use_minio=False)
    manager.store_local(KEY, source_dir, "task_1", "tool")
    (source_dir / "late.txt").write_text("x")
    info = manager.store_local(KEY, source_dir, "task_1", "tool")

    entries = [p for p in manager.local_cache_dir.iterdir() if not p.name.startswith(".")]
    assert entries == [manager.get_local_path(KEY)]
    assert _tmp_dirs(manager.local_cache_dir) == []
    assert info.local_path == manager.get_local_path(KEY) / "output"
    # First complete copy wins; it is never half-replaced.
    assert not (info.local_path / "late.txt").exists()


def test_empty_output_is_complete(tmp_path):
    store = InMemoryMinio()
    empty = tmp_path / "empty"
    empty.mkdir()
    _manager(tmp_path, "writer", store).store_artifact(KEY, empty, "task_1", "tool")
    assert store.artifact_exists(KEY)
    out = _manager(tmp_path, "reader", store).get_artifact(KEY)
    assert out is not None and _files(out) == {}
