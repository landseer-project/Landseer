"""
Artifacts leave out byte-identical copies of the run's dataset.

Uses the in-memory MinioStore from test_artifact_completion.

Covers:
- files identical to the dataset (path, size, sha256) are not uploaded; the
  marker lists them with source=dataset; modified, moved or new files upload
- round trip rebuilds the exact output (many-file directory included) with
  hard links into the dataset copy, or copies when linking fails
- artifacts without dataset entries (old format) download as before
- a reader that predates de-dup (the code at c60c7d0) and a reader without the
  dataset both treat a de-duplicated artifact as a miss
- the resolver syncs an unregistered dataset; a changed dataset copy is a miss
- LANDSEER_ARTIFACT_DATASET_DEDUP=0 uploads everything
- worker: only a verified dataset is registered; the resolver finds known
  copies, syncs a published one and refuses a different version
"""

import errno
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.common import dataset_manifest as dsm
from src.common import fsutil
from src.store.artifact_manager import ArtifactManager, DATASET_SOURCE
from src.store.minio_store import ARTIFACT_COMPLETE_MARKER, ARTIFACT_MANIFEST
from src.worker.cli import Worker
from tests.store.test_artifact_completion import InMemoryMinio

KEY = "d" * 64
LOCATION = {"name": "celeba", "variant": "poisoned", "minio_key": "datasets/celeba/poisoned"}
REPO = Path(__file__).resolve().parents[2]


def _files(root: Path):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def dataset(tmp_path):
    """A small CelebA-like dataset dir and its manifest."""
    d = tmp_path / "datasets" / "celeba" / "poisoned"
    (d / "img_align_celeba").mkdir(parents=True)
    (d / "data.npy").write_bytes(b"\x01" * 4096)
    (d / "labels.npy").write_bytes(b"\x02" * 128)
    (d / "filenames.npy").write_bytes(b"names")
    for i in range(300):
        (d / "img_align_celeba" / f"{i:06d}.jpg").write_bytes(f"img{i}".encode() * 7)
    manifest = dsm.build_manifest(d)
    dsm.write_manifest(d, manifest)
    return d, manifest


@pytest.fixture
def output(tmp_path, dataset):
    """Tool output: dataset copied through, data.npy modified in place, new files."""
    d, _ = dataset
    out = tmp_path / "task_output"
    shutil.copytree(d, out, ignore=shutil.ignore_patterns(dsm.MANIFEST_NAME, dsm.HASH_SIDECAR))
    (out / "data.npy").write_bytes(b"\x09" * 4096)        # same size, other bytes (filtered)
    (out / "filenames.npy").write_bytes(b"names, more")  # other size
    (out / "img_align_celeba" / "000007.jpg").write_bytes(b"triggered")  # watermark-style edit
    (out / "model.pt").write_bytes(b"weights" * 50)
    (out / "moved").mkdir()
    (out / "moved" / "labels.npy").write_bytes(b"\x02" * 128)  # same bytes, other path
    return out


def _manager(tmp_path, name, store, dataset=None, active=True) -> ArtifactManager:
    m = ArtifactManager(local_cache_dir=tmp_path / name, minio_store=store, use_minio=True)
    if dataset is not None:
        m.register_dataset(dataset[0], dataset[1], LOCATION, active=active)
    return m


def _marker(store):
    return json.loads(store.objects[f"artifacts/{KEY}/{ARTIFACT_COMPLETE_MARKER}"])


def _tmp_dirs(cache_dir: Path):
    return [p for p in cache_dir.iterdir() if p.name.startswith(".tmp-")]


def test_identical_files_left_out_modified_kept(tmp_path, dataset, output):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")

    uploaded = {k[len(f"artifacts/{KEY}/"):] for k in store.objects}
    assert "labels.npy" not in uploaded
    assert "img_align_celeba/000001.jpg" not in uploaded
    for kept in ("data.npy", "filenames.npy", "img_align_celeba/000007.jpg", "model.pt", "moved/labels.npy"):
        assert kept in uploaded
    # 300 images + labels.npy minus the edited image
    assert len(uploaded) == len(_files(output)) - 300 + 2  # + manifest.json + marker

    marker = _marker(store)
    assert set(marker["files"]) == set(_files(output))
    entry = marker["files"]["labels.npy"]
    assert entry["source"] == DATASET_SOURCE
    assert entry["dataset_content_id"] == dataset[1]["content_id"]
    assert entry["size"] == 128 and entry["sha256"] == dsm.file_sha256(output / "labels.npy")
    assert "source" not in marker["files"]["data.npy"]
    assert marker["datasets"] == {dataset[1]["content_id"]: LOCATION}


def test_round_trip_hard_links_dataset_copy(tmp_path, dataset, output):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")

    reader = _manager(tmp_path, "reader", store, dataset, active=False)
    out = reader.get_artifact(KEY)
    assert out is not None
    assert _files(out) == _files(output)
    d = dataset[0]
    assert os.stat(out / "labels.npy").st_ino == os.stat(d / "labels.npy").st_ino
    assert os.stat(out / "img_align_celeba" / "000123.jpg").st_ino == os.stat(
        d / "img_align_celeba" / "000123.jpg").st_ino
    assert os.stat(out / "data.npy").st_ino != os.stat(d / "data.npy").st_ino
    assert not os.access(out / "labels.npy", os.W_OK) or os.geteuid() == 0
    assert _tmp_dirs(reader.local_cache_dir) == []


def test_round_trip_copies_when_hard_link_fails(tmp_path, dataset, output, monkeypatch):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")

    def no_link(src, dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(fsutil.os, "link", no_link)
    out = _manager(tmp_path, "reader", store, dataset, active=False).get_artifact(KEY)
    assert out is not None
    assert _files(out) == _files(output)
    assert os.stat(out / "labels.npy").st_ino != os.stat(dataset[0] / "labels.npy").st_ino


def test_old_format_artifact_still_downloads(tmp_path, dataset, output):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store).store_artifact(KEY, output, "t1", "tool")
    assert not any("source" in e for e in _marker(store)["files"].values())
    assert f"artifacts/{KEY}/labels.npy" in store.objects

    out = _manager(tmp_path, "reader", store, dataset).get_artifact(KEY)
    assert out is not None and _files(out) == _files(output)


def _legacy_manager_class(tmp_path):
    """ArtifactManager as it was before de-dup (c60c7d0), loaded from git."""
    try:
        source = subprocess.run(
            ["git", "-C", str(REPO), "show", "c60c7d0:src/store/artifact_manager.py"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history with c60c7d0 not available")
    path = tmp_path / "legacy_artifact_manager.py"
    path.write_text(source)
    # Named inside src.store so its relative imports resolve.
    spec = importlib.util.spec_from_file_location("src.store._legacy_artifact_manager", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ArtifactManager


def test_reader_without_dedup_support_fails_closed(tmp_path, dataset, output):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")

    legacy = _legacy_manager_class(tmp_path)
    reader = legacy(local_cache_dir=tmp_path / "old-reader", minio_store=store, use_minio=True)
    assert reader.get_artifact(KEY) is None
    assert not reader.local_exists(KEY)
    assert _tmp_dirs(reader.local_cache_dir) == []

    # The same old reader still reads artifacts written without de-dup.
    store2 = InMemoryMinio()
    _manager(tmp_path, "writer2", store2).store_artifact(KEY, output, "t1", "tool")
    reader2 = legacy(local_cache_dir=tmp_path / "old-reader2", minio_store=store2, use_minio=True)
    out = reader2.get_artifact(KEY)
    assert out is not None and _files(out) == _files(output)


def test_missing_dataset_is_a_miss(tmp_path, dataset, output):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")

    reader = _manager(tmp_path, "reader", store)
    calls = []
    reader.dataset_resolver = lambda cid, loc: calls.append((cid, loc)) or None
    assert reader.get_artifact(KEY) is None
    assert calls == [(dataset[1]["content_id"], LOCATION)]
    assert not reader.local_exists(KEY)
    assert _tmp_dirs(reader.local_cache_dir) == []


def test_resolver_supplies_unregistered_dataset(tmp_path, dataset, output):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")

    reader = _manager(tmp_path, "reader", store)
    reader.dataset_resolver = lambda cid, loc: dataset[0] if cid == dataset[1]["content_id"] else None
    out = reader.get_artifact(KEY)
    assert out is not None and _files(out) == _files(output)


def test_changed_dataset_copy_is_a_miss(tmp_path, dataset, output):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")

    d = dataset[0]
    (d / "labels.npy").unlink()
    (d / "labels.npy").write_bytes(b"\x03" * 128)
    reader = _manager(tmp_path, "reader", store, dataset, active=False)
    assert reader.get_artifact(KEY) is None
    assert _tmp_dirs(reader.local_cache_dir) == []


def test_dedup_can_be_disabled(tmp_path, dataset, output, monkeypatch):
    monkeypatch.setenv("LANDSEER_ARTIFACT_DATASET_DEDUP", "0")
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")
    assert f"artifacts/{KEY}/labels.npy" in store.objects
    assert "datasets" not in _marker(store)


def test_inactive_dataset_is_not_excluded(tmp_path, dataset, output):
    store = InMemoryMinio()
    writer = _manager(tmp_path, "writer", store, dataset)
    writer.set_active_dataset(None)
    writer.store_artifact(KEY, output, "t1", "tool")
    assert f"artifacts/{KEY}/labels.npy" in store.objects


def test_tool_manifest_json_is_always_uploaded(tmp_path, dataset, output):
    d, _ = dataset
    (d / ARTIFACT_MANIFEST).write_text('{"tool": 1}')
    manifest = dsm.build_manifest(d)
    (output / ARTIFACT_MANIFEST).write_text('{"tool": 1}')
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, (d, manifest)).store_artifact(KEY, output, "t1", "tool")
    assert "source" not in _marker(store)["files"][ARTIFACT_MANIFEST]


# ----------------------------------------------------------------------------
# Worker wiring
# ----------------------------------------------------------------------------

def _worker(tmp_path, store) -> Worker:
    w = Worker(workspace_dir=tmp_path / "ws", cache_dir=tmp_path / "wcache")
    w._client = MagicMock()
    manager = ArtifactManager(local_cache_dir=tmp_path / "wcache", minio_store=store, use_minio=True)
    manager.dataset_resolver = w._local_dataset_for_content
    w._two_level_cache = SimpleNamespace(manager=manager)
    return w


def _publish(store, dataset):
    d, manifest = dataset
    for rel in manifest["files"]:
        store.objects[f"{LOCATION['minio_key']}/{rel}"] = (d / rel).read_bytes()
    store.objects[f"{LOCATION['minio_key']}/{dsm.MANIFEST_NAME}"] = json.dumps(manifest).encode()


def test_worker_registers_only_verified_dataset(tmp_path, dataset):
    w = _worker(tmp_path, InMemoryMinio())
    manager = w._two_level_cache.manager
    w._dataset_manifest = dataset[1]
    w._dataset_info = dict(LOCATION, available=True)

    w._register_cache_dataset(dataset[0], "not-the-manifest-id")
    assert manager._active_dataset is None

    w._register_cache_dataset(dataset[0], dataset[1]["content_id"])
    assert manager._active_dataset == dataset[1]["content_id"]
    assert manager._datasets[dataset[1]["content_id"]][2] == LOCATION

    w._register_cache_dataset(None, None)
    assert manager._active_dataset is None


def test_worker_resolver_uses_known_copy(tmp_path, dataset):
    w = _worker(tmp_path, InMemoryMinio())
    w._run_contexts["run1"] = {"path": dataset[0], "manifest": dataset[1], "info": {}, "model": None}
    assert w._local_dataset_for_content(dataset[1]["content_id"], {}) == dataset[0]


def test_worker_resolver_syncs_published_dataset(tmp_path, dataset, output):
    store = InMemoryMinio()
    _publish(store, dataset)
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")

    w = _worker(tmp_path, store)
    w._dataset_manifest = {"content_id": "active-run"}
    out = w._two_level_cache.manager.get_artifact(KEY)
    assert out is not None and _files(out) == _files(output)
    synced = tmp_path / "wcache" / "datasets" / "celeba" / "poisoned"
    assert dsm.load_manifest(synced)["content_id"] == dataset[1]["content_id"]
    assert os.stat(out / "labels.npy").st_ino == os.stat(synced / "labels.npy").st_ino
    assert w._dataset_manifest == {"content_id": "active-run"}


def test_worker_resolver_refuses_other_version(tmp_path, dataset):
    store = InMemoryMinio()
    _publish(store, dataset)
    w = _worker(tmp_path, store)
    assert w._local_dataset_for_content("0" * 64, LOCATION) is None
    assert w._local_dataset_for_content(dataset[1]["content_id"], dict(LOCATION, name="../x")) is None
    # A different version already in the sync dir is left alone.
    synced = tmp_path / "wcache" / "datasets" / "celeba" / "poisoned"
    synced.mkdir(parents=True)
    dsm.write_manifest(synced, dict(dataset[1], content_id="1" * 64))
    assert w._local_dataset_for_content(dataset[1]["content_id"], LOCATION) is None
    assert dsm.load_manifest(synced)["content_id"] == "1" * 64
