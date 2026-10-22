"""
Corner cases of artifact de-duplication against the run's dataset.

Covers:
- an output that is exactly the dataset (nothing but the cache manifest uploaded)
- empty files, unusual file names, manifest entries without sha256
- a partial upload with left-out files is still not a hit
- tampered markers (unsafe path, no content id) are a miss and write nothing
- link failures: fallback errnos copy, other errors are a miss
- the registered dataset dir deleted: resolver, else miss
- an artifact of dataset A read by a worker whose active run uses dataset B
- hard links keep the cache entry and the dataset independent of each other
- concurrent readers of the same entry
- worker end to end: task run on worker A, cache hit on worker B with no
  dataset objects in MinIO
"""

import errno
import json
import os
import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.common import dataset_manifest as dsm
from src.common import fsutil
from src.store.artifact_manager import ArtifactManager, DATASET_SOURCE
from src.store.minio_store import ARTIFACT_COMPLETE_MARKER, ARTIFACT_MANIFEST
from src.worker.cli import Worker
from src.worker.client import TaskInfo
from src.worker.runner import ExecutionResult
from tests.store.test_artifact_completion import InMemoryMinio
from tests.store.test_artifact_dataset_dedup import (  # noqa: F401 (fixtures)
    KEY,
    LOCATION,
    _files,
    _legacy_manager_class,
    _manager,
    _marker,
    _tmp_dirs,
    dataset,
    output,
)

MARKER_KEY = f"artifacts/{KEY}/{ARTIFACT_COMPLETE_MARKER}"


def _rewrite_marker(store, fn):
    marker = json.loads(store.objects[MARKER_KEY])
    fn(marker)
    store.objects[MARKER_KEY] = json.dumps(marker).encode()


def test_output_identical_to_dataset(tmp_path, dataset):
    d, manifest = dataset
    out = tmp_path / "out"
    shutil.copytree(d, out, ignore=shutil.ignore_patterns(dsm.MANIFEST_NAME, dsm.HASH_SIDECAR))
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, out, "t1", "tool")

    assert sorted(k.rsplit("/", 1)[1] for k in store.objects) == [ARTIFACT_COMPLETE_MARKER, ARTIFACT_MANIFEST]
    assert all(e["source"] == DATASET_SOURCE for e in _marker(store)["files"].values())

    got = _manager(tmp_path, "reader", store, dataset, active=False).get_artifact(KEY)
    assert got is not None and _files(got) == _files(out)
    legacy = _legacy_manager_class(tmp_path)
    assert legacy(local_cache_dir=tmp_path / "old", minio_store=store, use_minio=True).get_artifact(KEY) is None


def test_empty_and_unusual_file_names(tmp_path):
    d = tmp_path / "ds"
    (d / "dir with space").mkdir(parents=True)
    (d / "empty.npy").write_bytes(b"")
    (d / "dir with space" / "ü-ñ.jpg").write_bytes(b"x" * 10)
    (d / "data.npy").write_bytes(b"d" * 100)
    manifest = dsm.build_manifest(d)
    out = tmp_path / "out"
    shutil.copytree(d, out)
    (out / "model.pt").write_bytes(b"m")
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, (d, manifest)).store_artifact(KEY, out, "t1", "tool")

    files = _marker(store)["files"]
    assert files["empty.npy"]["source"] == DATASET_SOURCE
    assert files["dir with space/ü-ñ.jpg"]["source"] == DATASET_SOURCE
    got = _manager(tmp_path, "reader", store, (d, manifest), active=False).get_artifact(KEY)
    assert got is not None and _files(got) == _files(out)


def test_manifest_entry_without_sha256_is_not_excluded(tmp_path, dataset, output):
    d, manifest = dataset
    stripped = dict(manifest, files={r: {"size": e["size"]} for r, e in manifest["files"].items()})
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, (d, stripped)).store_artifact(KEY, output, "t1", "tool")
    assert not any("source" in e for e in _marker(store)["files"].values())


def test_partial_upload_with_left_out_files_is_not_a_hit(tmp_path, dataset, output):
    store = InMemoryMinio(fail_upload_on=2)
    with pytest.raises(RuntimeError):
        _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")
    assert MARKER_KEY not in store.objects
    assert _manager(tmp_path, "reader", store, dataset).get_artifact(KEY) is None


def test_unsafe_marker_path_is_a_miss(tmp_path, output):
    """A marker entry like ../../../escaped.npy must not link files outside the entry."""
    d = tmp_path / "a" / "b" / "c" / "d" / "ds"
    d.mkdir(parents=True)
    (d / "labels.npy").write_bytes(b"" * 128)
    manifest = dsm.build_manifest(d)
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, (d, manifest)).store_artifact(KEY, output, "t1", "tool")

    # Resolves to tmp_path/a/b/escaped.npy in the dataset and to tmp_path/escaped.npy
    # from the reader's staging dir; the source exists with the right bytes.
    bad = "../../../escaped.npy"
    (tmp_path / "a" / "b" / "escaped.npy").write_bytes(b"" * 128)
    _rewrite_marker(store, lambda m: m["files"].__setitem__(bad, dict(m["files"]["labels.npy"])))

    reader = _manager(tmp_path, "reader", store, (d, manifest), active=False)
    assert reader.get_artifact(KEY) is None
    assert not (tmp_path / "escaped.npy").exists()
    assert _tmp_dirs(reader.local_cache_dir) == []

    _rewrite_marker(store, lambda m: m["files"].__setitem__("/abs/escaped.npy", m["files"].pop(bad)))
    assert reader.get_artifact(KEY) is None


def test_marker_entry_without_content_id_is_a_miss(tmp_path, dataset, output):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")
    _rewrite_marker(store, lambda m: m["files"]["labels.npy"].pop("dataset_content_id"))
    assert _manager(tmp_path, "reader", store, dataset, active=False).get_artifact(KEY) is None


@pytest.mark.parametrize("err", [errno.EXDEV, errno.EMLINK, errno.EPERM, errno.EACCES])
def test_link_fallback_errnos_copy(tmp_path, dataset, output, monkeypatch, err):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")

    def no_link(src, dst):
        raise OSError(err, os.strerror(err))

    monkeypatch.setattr(fsutil.os, "link", no_link)
    got = _manager(tmp_path, "reader", store, dataset, active=False).get_artifact(KEY)
    assert got is not None and _files(got) == _files(output)


def test_other_link_error_is_a_miss(tmp_path, dataset, output, monkeypatch):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")

    def broken(src, dst):
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(fsutil.os, "link", broken)
    reader = _manager(tmp_path, "reader", store, dataset, active=False)
    assert reader.get_artifact(KEY) is None
    assert _tmp_dirs(reader.local_cache_dir) == []


def test_deleted_dataset_dir_falls_back_to_resolver(tmp_path, dataset, output):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")
    d, manifest = dataset
    spare = tmp_path / "spare"
    shutil.copytree(d, spare)

    reader = _manager(tmp_path, "reader", store, dataset, active=False)
    shutil.rmtree(d)
    reader.dataset_resolver = lambda cid, loc: None
    assert reader.get_artifact(KEY) is None
    reader.dataset_resolver = lambda cid, loc: spare
    got = reader.get_artifact(KEY)
    assert got is not None and _files(got) == _files(output)


def test_artifact_of_other_dataset(tmp_path, dataset, output):
    """Reader's active run uses dataset B; the artifact was made on dataset A."""
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")

    other = tmp_path / "other_ds"
    other.mkdir()
    (other / "data.npy").write_bytes(b"B" * 10)
    reader = _manager(tmp_path, "reader", store, dataset, active=False)
    reader.register_dataset(other, dsm.build_manifest(other), {}, active=True)
    got = reader.get_artifact(KEY)
    assert got is not None and _files(got) == _files(output)


def test_cache_entry_and_dataset_stay_independent(tmp_path, dataset, output):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")
    d, manifest = dataset
    reader = _manager(tmp_path, "reader", store, dataset, active=False)
    got = reader.get_artifact(KEY)
    expected = _files(got)

    # The dataset still verifies (read-only files, unchanged mtime/content).
    assert dsm.verify_dir(d, manifest).ok
    # Dropping the cache entry leaves the dataset whole ...
    reader.delete_local(KEY)
    assert dsm.verify_dir(d, manifest).ok
    # ... and replacing the dataset (as a sync does) leaves a cached entry whole.
    got = reader.get_artifact(KEY)
    old = tmp_path / "old_ds"
    os.replace(d, old)
    shutil.rmtree(old)
    assert _files(got) == expected


def test_concurrent_readers(tmp_path, dataset, output):
    store = InMemoryMinio()
    _manager(tmp_path, "writer", store, dataset).store_artifact(KEY, output, "t1", "tool")
    results, errors = [], []

    def read():
        try:
            m = _manager(tmp_path, "reader", store, dataset, active=False)
            results.append(m.get_artifact(KEY))
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=read) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(r is not None and _files(r) == _files(output) for r in results)
    assert _tmp_dirs(tmp_path / "reader") == []


# ----------------------------------------------------------------------------
# Worker end to end
# ----------------------------------------------------------------------------

def _task() -> TaskInfo:
    return TaskInfo(
        id="task_1", tool_name="post_noop", tool_image="img:1", tool_command="run",
        tool_runtime=None, tool_is_baseline=False, config={}, priority=1,
        status="pending", task_type="post_training", counter=1, workflows=[],
        pipeline_id="pl", dependency_ids=[],
    )


def _worker(tmp_path: Path, name: str, store, dataset) -> Worker:
    w = Worker(workspace_dir=tmp_path / name / "ws", cache_dir=tmp_path / name / "cache", use_cache=True)
    w._client = MagicMock()
    w._client.get_pipeline_info.return_value = {}
    w._runner = MagicMock()
    w._runner.workspace_dir = tmp_path / name / "ws"
    manager = ArtifactManager(local_cache_dir=tmp_path / name / "cache", minio_store=store, use_minio=True)
    manager.dataset_resolver = w._local_dataset_for_content
    w._two_level_cache = MagicMock()
    w._two_level_cache.manager = manager
    w._two_level_cache.get.side_effect = manager.get_artifact
    w._two_level_cache.put.side_effect = lambda **kw: manager.store_artifact(**kw)
    w._compute_cache_key = lambda *a, **k: KEY
    w._dataset_path = dataset[0]
    w._dataset_manifest = dataset[1]
    w._dataset_info = dict(LOCATION, available=True)
    return w


def test_worker_task_then_cache_hit_on_other_worker(tmp_path, dataset, output):
    store = InMemoryMinio()
    a = _worker(tmp_path, "a", store, dataset)

    def run_task(task, **kwargs):
        out = tmp_path / "a" / "ws" / task.id / "output"
        shutil.copytree(output, out)
        return ExecutionResult(success=True, exit_code=0, execution_time_ms=1, output_path=out)

    a._runner.run_task.side_effect = run_task
    result = a._execute_task(_task())
    assert result.success, result.error_message
    assert MARKER_KEY in store.objects
    assert f"artifacts/{KEY}/labels.npy" not in store.objects
    assert f"artifacts/{KEY}/model.pt" in store.objects

    b = _worker(tmp_path, "b", store, dataset)
    result = b._execute_task(_task())
    assert result.success and result.artifacts["cache_hit"] is True
    b._runner.run_task.assert_not_called()
    assert _files(tmp_path / "b" / "ws" / "task_1" / "output") == _files(output)
