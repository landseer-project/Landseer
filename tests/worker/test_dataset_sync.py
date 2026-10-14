"""
Dataset manifest end to end: backend prep -> MinIO publish -> worker sync.

Covers:
- a dataset prepared before manifests existed gets one on first use (provenance
  fields that cannot be known are null) and is not rewritten when unchanged
- a fresh container prep records provenance (image digest, commit, host, time)
- publish uploads the listed files and the manifest last; same content skips
- a worker sync downloads and verifies; a corrupted or stale local copy is
  replaced (unchanged files reused); a failed sync leaves no staging dirs
- a MinIO dataset without a manifest still downloads (legacy)
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.backend.dataset_publish import publish_dataset
from src.common import dataset_manifest as dsm
from src.data.manager import DatasetManager
from src.worker.cli import Worker


class MemStore:
    """Just enough of MinioStore for datasets."""

    def __init__(self):
        self.objects: Dict[str, bytes] = {}
        self.uploads: List[str] = []
        self.downloads: List[str] = []
        self.is_available = True

    def upload_file(self, local_path, object_key, content_type=None, metadata=None) -> bool:
        self.objects[object_key] = Path(local_path).read_bytes()
        self.uploads.append(object_key)
        return True

    def upload_directory(self, local_dir, prefix, recursive=True) -> int:
        n = 0
        for p in sorted(Path(local_dir).rglob("*")):
            if p.is_file():
                self.upload_file(p, f"{prefix.rstrip('/')}/{p.relative_to(local_dir).as_posix()}")
                n += 1
        return n

    def download_file(self, object_key, local_path) -> bool:
        if object_key not in self.objects:
            return False
        self.downloads.append(object_key)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(self.objects[object_key])
        return True

    def download_directory(self, prefix, local_dir, recursive=True) -> int:
        n = 0
        for key in list(self.objects):
            if key.startswith(prefix.rstrip("/") + "/"):
                self.download_file(key, Path(local_dir) / key[len(prefix.rstrip("/")) + 1:])
                n += 1
        return n

    def download_bytes(self, object_key) -> Optional[bytes]:
        return self.objects.get(object_key)

    def exists(self, object_key) -> bool:
        return object_key in self.objects

    def delete(self, object_key) -> bool:
        return self.objects.pop(object_key, None) is not None


REGISTRY = """
datasets:
  cifar10:
    default_image: ghcr.io/landseer-project/dataset_cifar10:v1
    version: "1"
    required_files: [data.npy, labels.npy, test_data.npy, test_labels.npy]
    variants:
      clean:
        image: ghcr.io/landseer-project/dataset_cifar10:v1
"""


def _write_arrays(d: Path, seed: int = 0) -> None:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.save(d / "data.npy", rng.random((20, 3, 4, 4), dtype=np.float32))
    np.save(d / "labels.npy", rng.integers(0, 10, 20))
    np.save(d / "test_data.npy", rng.random((5, 3, 4, 4), dtype=np.float32))
    np.save(d / "test_labels.npy", rng.integers(0, 10, 5))


@pytest.fixture
def manager(tmp_path, monkeypatch) -> DatasetManager:
    reg = tmp_path / "datasets.yaml"
    reg.write_text(REGISTRY)
    monkeypatch.setenv("LANDSEER_DATASETS_CONFIG", str(reg))
    return DatasetManager(tmp_path / "data")


@pytest.fixture
def legacy_info(manager, tmp_path):
    ds_dir = tmp_path / "data" / "cifar10" / "clean"
    _write_arrays(ds_dir)
    (ds_dir / "config_model.py").write_text("NET = 1\n")
    return manager.prepare_dataset(name="cifar10", variant="clean")


def _worker(tmp_path) -> Worker:
    w = Worker(workspace_dir=tmp_path / "ws", cache_dir=tmp_path / "wcache")
    w._client = MagicMock()
    return w


# ----------------------------------------------------------------------------
# Backend prep
# ----------------------------------------------------------------------------


def test_legacy_dataset_gets_manifest_on_first_use(legacy_info, tmp_path, manager):
    ds_dir = Path(legacy_info.output_dir)
    manifest = dsm.load_manifest(ds_dir)
    assert manifest and manifest["file_count"] == 4
    assert "config_model.py" not in manifest["files"]
    assert legacy_info.content_id == manifest["content_id"]
    prov = manifest["provenance"]
    assert prov["origin"] == "backfilled"
    assert prov["name"] == "cifar10" and prov["variant"] == "clean" and prov["version"] == "1"
    assert prov["configured_image"] == "ghcr.io/landseer-project/dataset_cifar10:v1"
    for unknown in ("image", "image_digest", "landseer_commit", "prep_host", "prepared_at"):
        assert prov[unknown] is None
    assert legacy_info.to_dict()["provenance_id"] == prov["provenance_id"]

    # Unchanged on second use: not rewritten.
    before = (ds_dir / dsm.MANIFEST_NAME).stat().st_mtime_ns
    again = manager.prepare_dataset(name="cifar10", variant="clean")
    assert again.content_id == legacy_info.content_id
    assert (ds_dir / dsm.MANIFEST_NAME).stat().st_mtime_ns == before


def test_fresh_prep_records_provenance(manager, tmp_path, monkeypatch):
    def _fake_prepare(key, output_dir, variant, params, runtime_cfg=None, force_reprepare=False):
        _write_arrays(output_dir, seed=1)

    monkeypatch.setattr(manager, "_prepare_with_container", _fake_prepare)
    monkeypatch.setattr(DatasetManager, "_image_digest",
                        staticmethod(lambda image: "ghcr.io/x@sha256:abc"))
    monkeypatch.setattr(DatasetManager, "_landseer_commit", staticmethod(lambda: "deadbeef"))

    info = manager.prepare_dataset(name="cifar10", variant="clean", subset=0.5)
    prov = dsm.load_manifest(info.output_dir)["provenance"]
    assert prov["origin"] == "prepared"
    assert prov["image"] == "ghcr.io/landseer-project/dataset_cifar10:v1"
    assert prov["image_digest"] == "ghcr.io/x@sha256:abc"
    assert prov["landseer_commit"] == "deadbeef"
    assert prov["params"] == {"subset": 0.5}
    assert prov["prep_host"] and prov["prepared_at"]


def test_edited_dataset_is_reflagged(legacy_info, manager):
    ds_dir = Path(legacy_info.output_dir)
    np.save(ds_dir / "labels.npy", np.arange(20))
    info = manager.prepare_dataset(name="cifar10", variant="clean")
    assert info.content_id != legacy_info.content_id
    assert "content changed" in dsm.load_manifest(ds_dir)["provenance"]["note"]


# ----------------------------------------------------------------------------
# Publish
# ----------------------------------------------------------------------------


def test_publish_uploads_manifest_last_and_skips_same_content(legacy_info):
    store = MemStore()
    info = legacy_info.to_dict()
    assert publish_dataset(store, info, "datasets/cifar10/clean")
    assert store.uploads[-1] == "datasets/cifar10/clean/dataset_manifest.json"
    assert sorted(store.uploads[:-1]) == sorted(
        f"datasets/cifar10/clean/{n}" for n in ("data.npy", "labels.npy", "test_data.npy", "test_labels.npy")
    )
    assert info["manifest_minio_key"] == "datasets/cifar10/clean/dataset_manifest.json"

    store.uploads.clear()
    assert publish_dataset(store, legacy_info.to_dict(), "datasets/cifar10/clean")
    assert store.uploads == []


def test_failed_publish_leaves_no_manifest(legacy_info):
    store = MemStore()
    real = store.upload_file
    store.upload_file = lambda p, k, **kw: False if k.endswith("labels.npy") else real(p, k, **kw)
    assert not publish_dataset(store, legacy_info.to_dict(), "datasets/cifar10/clean")
    assert "datasets/cifar10/clean/dataset_manifest.json" not in store.objects


# ----------------------------------------------------------------------------
# Worker sync
# ----------------------------------------------------------------------------


KEY = "datasets/cifar10/clean"


def _published(legacy_info) -> MemStore:
    store = MemStore()
    assert publish_dataset(store, legacy_info.to_dict(), KEY)
    return store


def _files(d: Path) -> Dict[str, bytes]:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in d.rglob("*")
            if p.is_file() and not p.name.startswith(".")}


def test_sync_downloads_and_verifies(legacy_info, tmp_path):
    store = _published(legacy_info)
    w = _worker(tmp_path)
    dest = w.cache_dir / "datasets" / "cifar10" / "clean"
    out = w._sync_dataset_from_minio(store, KEY, dest)
    assert out == dest
    src = Path(legacy_info.output_dir)
    for rel in ("data.npy", "labels.npy", "test_data.npy", "test_labels.npy"):
        assert (dest / rel).read_bytes() == (src / rel).read_bytes()
    assert w._dataset_identity(dest)[0] == legacy_info.content_id
    assert w._dataset_identity(dest)[1] == legacy_info.provenance["provenance_id"]

    store.downloads.clear()
    assert w._sync_dataset_from_minio(store, KEY, dest) == dest
    assert store.downloads == []  # verified local copy reused


def test_corrupted_local_copy_is_replaced(legacy_info, tmp_path):
    store = _published(legacy_info)
    w = _worker(tmp_path)
    dest = w.cache_dir / "datasets" / "cifar10" / "clean"
    w._sync_dataset_from_minio(store, KEY, dest)

    data = bytearray((dest / "data.npy").read_bytes())
    data[-1] ^= 0xFF
    (dest / "data.npy").write_bytes(bytes(data))
    store.downloads.clear()

    assert _worker(tmp_path)._sync_dataset_from_minio(store, KEY, dest) == dest
    assert store.downloads == [f"{KEY}/data.npy"]  # only the bad file refetched
    assert (dest / "data.npy").read_bytes() == (Path(legacy_info.output_dir) / "data.npy").read_bytes()
    assert not [p for p in dest.parent.iterdir() if p.name.startswith((".tmp-", ".old-"))]


def test_stale_local_copy_is_refreshed(legacy_info, manager, tmp_path):
    store = _published(legacy_info)
    w = _worker(tmp_path)
    dest = w.cache_dir / "datasets" / "cifar10" / "clean"
    w._sync_dataset_from_minio(store, KEY, dest)

    # Backend re-prepares with different labels and republishes.
    src = Path(legacy_info.output_dir)
    np.save(src / "labels.npy", np.arange(20))
    new_info = manager.prepare_dataset(name="cifar10", variant="clean")
    assert publish_dataset(store, new_info.to_dict(), KEY)
    store.downloads.clear()

    w2 = _worker(tmp_path)
    assert w2._sync_dataset_from_minio(store, KEY, dest) == dest
    assert store.downloads == [f"{KEY}/labels.npy"]
    assert dsm.load_manifest(dest)["content_id"] == new_info.content_id
    assert w2._dataset_identity(dest)[0] == new_info.content_id


def test_failed_sync_keeps_old_copy_and_no_staging(legacy_info, tmp_path):
    store = _published(legacy_info)
    w = _worker(tmp_path)
    dest = w.cache_dir / "datasets" / "cifar10" / "clean"
    w._sync_dataset_from_minio(store, KEY, dest)
    (dest / "test_labels.npy").write_bytes(b"broken")
    del store.objects[f"{KEY}/test_labels.npy"]

    assert _worker(tmp_path)._sync_dataset_from_minio(store, KEY, dest) is None
    assert (dest / "data.npy").exists()
    assert not [p for p in dest.parent.iterdir() if p.name.startswith((".tmp-", ".old-"))]


def test_minio_dataset_without_manifest_still_downloads(tmp_path):
    store = MemStore()
    src = tmp_path / "src"
    _write_arrays(src)
    store.upload_directory(src, KEY)
    w = _worker(tmp_path)
    dest = w.cache_dir / "datasets" / "cifar10" / "clean"
    assert w._sync_dataset_from_minio(store, KEY, dest) == dest
    assert (dest / "data.npy").exists()
    assert w._dataset_manifest is None
