"""
Artifacts leave out unchanged copies of files their ancestor artifacts hold.

Chain used below (like a CelebA run): P = pre-training filter (new data.npy,
wm_test_data.npy), Q = in-training (copies data.npy, adds model.pt, drops
wm_test_data.npy), R = post-training noop (copies everything it was given,
including wm_test_data.npy from its grandparent P).

Covers:
- unchanged files point at the artifact whose objects hold the bytes (P for
  data.npy even in R, found through Q's parent_hashes); changed, moved and new
  files upload
- round trip on a worker with nothing local rebuilds exact outputs with hard
  links into the holder's cache entry; a holder already local is not fetched
- a reader that predates this (c60c7d0) treats such an artifact as a miss
- a holder missing or corrupted in MinIO, or a tampered marker, is a miss
- markers pointing at each other are a miss, not a deadlock
- incomplete ancestors are not referenced; several versions of a path among
  ancestors pick the matching one; parent_hashes cycles terminate
- files matching the dataset still point at the dataset
- evicting the holder locally leaves rebuilt entries intact
"""

import json
import os
import shutil
import threading
from pathlib import Path

import pytest

from src.common import dataset_manifest as dsm
from src.store.artifact_manager import ARTIFACT_SOURCE, DATASET_SOURCE, ArtifactManager
from src.store.minio_store import ARTIFACT_COMPLETE_MARKER, ARTIFACT_MANIFEST
from tests.store.test_artifact_completion import InMemoryMinio
from tests.store.test_artifact_dataset_dedup import (  # noqa: F401 (fixture)
    LOCATION,
    _files,
    _legacy_manager_class,
    _tmp_dirs,
    dataset,
)

P, Q, R = "p" * 64, "q" * 64, "r" * 64


class CountingMinio(InMemoryMinio):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.downloads = []

    def download_file(self, object_key, local_path) -> bool:
        self.downloads.append(object_key)
        return super().download_file(object_key, local_path)


def _write(root: Path, files):
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


@pytest.fixture
def chain(tmp_path):
    p = _write(tmp_path / "P", {
        "data.npy": b"F" * 5000, "labels.npy": b"L" * 100,
        "wm_test_data.npy": b"W" * 3000, "xgbod_report.json": b"{}",
    })
    q = _write(tmp_path / "Q", {
        "data.npy": b"F" * 5000, "labels.npy": b"L" * 100, "model.pt": b"M1" * 400,
        "xgbod_report.json": b"{}",
    })
    r = _write(tmp_path / "R", {
        "data.npy": b"F" * 5000, "labels.npy": b"l" * 100,  # labels changed, same size
        "model.pt": b"M1" * 400, "wm_test_data.npy": b"W" * 3000,
        "moved/data.npy": b"F" * 5000, "post.json": b'{"ok": 1}',
    })
    return {P: (p, []), Q: (q, [P]), R: (r, [Q])}


def _store_chain(tmp_path, store, chain, writer="writer", keys=(P, Q, R)):
    m = ArtifactManager(local_cache_dir=tmp_path / writer, minio_store=store, use_minio=True)
    for key in keys:
        src, parents = chain[key]
        m.store_artifact(key, src, f"task-{key[0]}", "tool", parent_hashes=parents)
    return m


def _marker(store, key):
    return json.loads(store.objects[f"artifacts/{key}/{ARTIFACT_COMPLETE_MARKER}"])


def _uploaded(store, key):
    prefix = f"artifacts/{key}/"
    return {k[len(prefix):] for k in store.objects if k.startswith(prefix)} - {
        ARTIFACT_COMPLETE_MARKER, ARTIFACT_MANIFEST}


def _reader(tmp_path, store, name="reader"):
    return ArtifactManager(local_cache_dir=tmp_path / name, minio_store=store, use_minio=True)


def test_unchanged_files_point_at_holder(tmp_path, chain):
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain)

    assert _uploaded(store, P) == {"data.npy", "labels.npy", "wm_test_data.npy", "xgbod_report.json"}
    assert _uploaded(store, Q) == {"model.pt"}
    assert _uploaded(store, R) == {"labels.npy", "moved/data.npy", "post.json"}

    q, r = _marker(store, Q)["files"], _marker(store, R)["files"]
    assert q["data.npy"]["source"] == ARTIFACT_SOURCE and q["data.npy"]["artifact_cache_key"] == P
    assert r["data.npy"]["artifact_cache_key"] == P            # flattened past Q
    assert r["wm_test_data.npy"]["artifact_cache_key"] == P    # grandparent only
    assert r["model.pt"]["artifact_cache_key"] == Q
    assert "source" not in r["labels.npy"] and "source" not in r["moved/data.npy"]
    assert set(r) == set(_files(chain[R][0]))
    assert r["data.npy"]["sha256"] == dsm.file_sha256(chain[R][0] / "data.npy")


def test_round_trip_from_empty_cache(tmp_path, chain):
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain)

    reader = _reader(tmp_path, store)
    out = reader.get_artifact(R)
    assert out is not None and _files(out) == _files(chain[R][0])
    p_out = reader.get_local_path(P) / "output"
    assert os.stat(out / "data.npy").st_ino == os.stat(p_out / "data.npy").st_ino
    assert os.stat(out / "model.pt").st_ino == os.stat(reader.get_local_path(Q) / "output" / "model.pt").st_ino
    assert _files(reader.get_artifact(Q)) == _files(chain[Q][0])
    assert _tmp_dirs(reader.local_cache_dir) == []


def test_local_holder_is_not_downloaded_again(tmp_path, chain):
    store = CountingMinio()
    _store_chain(tmp_path, store, chain)
    reader = _reader(tmp_path, store)
    assert reader.get_artifact(P) is not None
    store.downloads.clear()

    assert reader.get_artifact(Q) is not None
    assert not any(k.startswith(f"artifacts/{P}/") for k in store.downloads)


def test_reader_without_support_fails_closed(tmp_path, chain):
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain)
    legacy = _legacy_manager_class(tmp_path)
    old = legacy(local_cache_dir=tmp_path / "old", minio_store=store, use_minio=True)
    assert old.get_artifact(P) is not None  # nothing left out of P
    assert old.get_artifact(Q) is None
    assert old.get_artifact(R) is None
    assert _tmp_dirs(old.local_cache_dir) == []


def test_missing_holder_is_a_miss(tmp_path, chain):
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain)
    for k in [k for k in store.objects if k.startswith(f"artifacts/{P}/")]:
        del store.objects[k]
    reader = _reader(tmp_path, store)
    assert reader.get_artifact(R) is None
    assert reader.get_artifact(Q) is None
    assert not reader.local_exists(R)
    assert _tmp_dirs(reader.local_cache_dir) == []


def test_corrupted_holder_is_a_miss(tmp_path, chain):
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain)
    store.objects[f"artifacts/{P}/data.npy"] = b"X" * 5000
    reader = _reader(tmp_path, store)
    assert reader.get_artifact(R) is None
    assert _tmp_dirs(reader.local_cache_dir) == []


def test_changed_local_holder_is_a_miss(tmp_path, chain):
    """A holder entry whose file changed locally fails the child's sha256 check."""
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain)
    reader = _reader(tmp_path, store)
    p_file = reader.get_artifact(P) / "data.npy"
    p_file.chmod(0o644)
    p_file.write_bytes(b"Y" * 5000)
    assert reader.get_artifact(Q) is None


@pytest.mark.parametrize("holder", ["../evil", Q, "", "x" * 64])
def test_tampered_holder_is_a_miss(tmp_path, chain, holder):
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain, keys=(P, Q))
    key = f"artifacts/{Q}/{ARTIFACT_COMPLETE_MARKER}"
    marker = json.loads(store.objects[key])
    marker["files"]["data.npy"]["artifact_cache_key"] = holder
    store.objects[key] = json.dumps(marker).encode()
    reader = _reader(tmp_path, store)
    assert reader.get_artifact(Q) is None
    assert _tmp_dirs(reader.local_cache_dir) == []


def test_unsafe_path_is_a_miss(tmp_path, chain):
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain, keys=(P, Q))
    key = f"artifacts/{Q}/{ARTIFACT_COMPLETE_MARKER}"
    marker = json.loads(store.objects[key])
    marker["files"]["../../../../data.npy"] = marker["files"]["data.npy"]
    store.objects[key] = json.dumps(marker).encode()
    assert _reader(tmp_path, store).get_artifact(Q) is None
    assert not (tmp_path / "data.npy").exists()


def test_incomplete_ancestor_is_not_referenced(tmp_path, chain):
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain, keys=(P,))
    del store.objects[f"artifacts/{P}/{ARTIFACT_COMPLETE_MARKER}"]
    _store_chain(tmp_path, store, chain, writer="writer2", keys=(Q,))
    assert _uploaded(store, Q) == set(_files(chain[Q][0]))


def test_matching_version_among_ancestors(tmp_path, chain):
    """Grandparent has v1, parent v2, child v1 again: the child points at the grandparent."""
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain, keys=(P,))
    q2 = _write(tmp_path / "Q2", {"data.npy": b"G" * 5000})
    r2 = _write(tmp_path / "R2", {"data.npy": b"F" * 5000})
    m = ArtifactManager(local_cache_dir=tmp_path / "w2", minio_store=store, use_minio=True)
    m.store_artifact(Q, q2, "q", "tool", parent_hashes=[P])
    m.store_artifact(R, r2, "r", "tool", parent_hashes=[Q])
    assert _marker(store, R)["files"]["data.npy"]["artifact_cache_key"] == P
    out = _reader(tmp_path, store).get_artifact(R)
    assert out is not None and _files(out) == _files(r2)


def test_parent_hashes_cycle_terminates(tmp_path, chain):
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain, keys=(P,))
    # Corrupt P's manifest so its parent is Q (which will list P): a cycle.
    key = f"artifacts/{P}/{ARTIFACT_MANIFEST}"
    manifest = json.loads(store.objects[key])
    manifest["parent_hashes"] = [Q, "not a key/../x"]
    store.objects[key] = json.dumps(manifest).encode()
    _store_chain(tmp_path, store, chain, writer="writer2", keys=(Q, R))
    out = _reader(tmp_path, store).get_artifact(R)
    assert out is not None and _files(out) == _files(chain[R][0])


def test_dataset_files_still_point_at_dataset(tmp_path, dataset):
    d, manifest = dataset
    store = InMemoryMinio()
    m = ArtifactManager(local_cache_dir=tmp_path / "w", minio_store=store, use_minio=True)
    m.register_dataset(d, manifest, LOCATION)
    p = tmp_path / "Pd"
    shutil.copytree(d, p, ignore=shutil.ignore_patterns(dsm.MANIFEST_NAME, dsm.HASH_SIDECAR))
    (p / "data.npy").write_bytes(b"\x07" * 4096)  # filtered
    q = tmp_path / "Qd"
    shutil.copytree(p, q)
    (q / "model.pt").write_bytes(b"m")
    m.store_artifact(P, p, "p", "tool")
    m.store_artifact(Q, q, "q", "tool", parent_hashes=[P])

    files = _marker(store, Q)["files"]
    assert files["labels.npy"]["source"] == DATASET_SOURCE
    assert files["data.npy"]["source"] == ARTIFACT_SOURCE
    assert _uploaded(store, Q) == {"model.pt"}
    reader = _reader(tmp_path, store)
    reader.register_dataset(d, manifest, LOCATION, active=False)
    out = reader.get_artifact(Q)
    assert out is not None and _files(out) == _files(q)


def test_evicting_holder_keeps_rebuilt_entry(tmp_path, chain):
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain)
    reader = _reader(tmp_path, store)
    out = reader.get_artifact(R)
    reader.delete_local(P)
    reader.delete_local(Q)
    assert _files(out) == _files(chain[R][0])


def test_disabled_uploads_everything(tmp_path, chain, monkeypatch):
    monkeypatch.setenv("LANDSEER_ARTIFACT_DATASET_DEDUP", "0")
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain)
    assert _uploaded(store, R) == set(_files(chain[R][0]))


def test_markers_pointing_at_each_other_are_a_miss(tmp_path, chain):
    """Q's data.npy held by P and (tampered) P's labels.npy held by Q: no deadlock."""
    store = InMemoryMinio()
    _store_chain(tmp_path, store, chain, keys=(P, Q))
    key = f"artifacts/{P}/{ARTIFACT_COMPLETE_MARKER}"
    marker = json.loads(store.objects[key])
    marker["files"]["labels.npy"].update(source=ARTIFACT_SOURCE, artifact_cache_key=Q)
    store.objects[key] = json.dumps(marker).encode()
    del store.objects[f"artifacts/{P}/labels.npy"]

    reader = _reader(tmp_path, store)
    result = []
    t = threading.Thread(target=lambda: result.append(reader.get_artifact(Q)), daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "download deadlocked"
    assert result == [None]
    assert _tmp_dirs(reader.local_cache_dir) == []


# ----------------------------------------------------------------------------
# Worker end to end: parent and child tasks
# ----------------------------------------------------------------------------

def _chain_task(task_id, deps, task_type) -> "TaskInfo":
    from src.worker.client import TaskInfo
    return TaskInfo(
        id=task_id, tool_name=f"tool_{task_id}", tool_image="img:1", tool_command="run",
        tool_runtime=None, tool_is_baseline=False, config={}, priority=1,
        status="pending", task_type=task_type, counter=1, workflows=[],
        pipeline_id="pl", dependency_ids=deps,
    )


def _chain_worker(tmp_path, name, store, dataset):
    from unittest.mock import MagicMock
    from src.worker.cli import Worker
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
    w._dataset_path = dataset[0]
    w._dataset_manifest = dataset[1]
    w._dataset_info = dict(LOCATION, available=True)
    return w


def test_worker_parent_child_tasks_then_hits_elsewhere(tmp_path, dataset):
    """
    Worker A runs pre (filters data.npy) and in (copies it, adds model.pt);
    the in task's artifact points at the pre artifact for data.npy. Worker B
    then gets cache hits for both with identical workspace outputs.
    """
    from src.worker.runner import ExecutionResult

    d, manifest = dataset
    store = InMemoryMinio()
    a = _chain_worker(tmp_path, "a", store, dataset)
    produced = {}

    def run_task(task, **kwargs):
        out = tmp_path / "a" / "ws" / task.id / "output"
        shutil.copytree(d, out, ignore=shutil.ignore_patterns(dsm.MANIFEST_NAME, dsm.HASH_SIDECAR))
        (out / "data.npy").write_bytes(b"\x05" * 4096)  # filtered by pre, passed on by in
        if task.id == "in":
            (out / "model.pt").write_bytes(b"trained" * 20)
        produced[task.id] = _files(out)
        return ExecutionResult(success=True, exit_code=0, execution_time_ms=1, output_path=out)

    a._runner.run_task.side_effect = run_task
    pre, inn = _chain_task("pre", [], "pre_training"), _chain_task("in", ["pre"], "in_training")
    assert a._execute_task(pre).success
    assert a._execute_task(inn).success

    keys = {tid: a._task_outputs[tid][0] for tid in ("pre", "in")}
    assert keys["pre"] != keys["in"]
    in_files = _marker_of(store, keys["in"])["files"]
    assert in_files["data.npy"]["source"] == ARTIFACT_SOURCE
    assert in_files["data.npy"]["artifact_cache_key"] == keys["pre"]
    assert in_files["labels.npy"]["source"] == DATASET_SOURCE
    prefix = f"artifacts/{keys['in']}/"
    assert {k[len(prefix):] for k in store.objects if k.startswith(prefix)} == {
        "model.pt", ARTIFACT_MANIFEST, ARTIFACT_COMPLETE_MARKER}

    b = _chain_worker(tmp_path, "b", store, dataset)
    for task in (_chain_task("pre", [], "pre_training"), _chain_task("in", ["pre"], "in_training")):
        result = b._execute_task(task)
        assert result.success and result.artifacts["cache_hit"] is True
        assert b._task_outputs[task.id][0] == keys[task.id]
        assert _files(tmp_path / "b" / "ws" / task.id / "output") == produced[task.id]
    b._runner.run_task.assert_not_called()

    # Worker C has run nothing: no dataset registered, no parent local. While the
    # dataset is not in MinIO the child is a miss; once published, C syncs it
    # and fetches the parent, and the child hits.
    from tests.store.test_artifact_dataset_dedup import _publish
    c = _chain_worker(tmp_path, "c", store, dataset)
    assert c._two_level_cache.manager.get_artifact(keys["in"]) is None
    _publish(store, dataset)
    out = c._two_level_cache.manager.get_artifact(keys["in"])
    assert out is not None and _files(out) == produced["in"]
    synced = tmp_path / "c" / "cache" / "datasets" / LOCATION["name"] / LOCATION["variant"]
    assert os.stat(out / "labels.npy").st_ino == os.stat(synced / "labels.npy").st_ino


def _marker_of(store, key):
    return json.loads(store.objects[f"artifacts/{key}/{ARTIFACT_COMPLETE_MARKER}"])
