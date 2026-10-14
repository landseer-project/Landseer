"""
Dataset identity in cache keys and lineage checks.

Covers:
- cache keys are identical on hosts with the same files at different paths/mtimes
- dataset_content_id is recorded on executed results and on cache hits
- a parent produced on a different dataset blocks the task (local and remote deps)
- an unknown parent dataset only warns
- manual LANDSEER_DATA_PATH is compared with the backend's dataset at startup
  and on run change, with LANDSEER_ALLOW_DATASET_MISMATCH=1 as the escape hatch
- the backend exposes dataset_content_id / nested cache_key on the task API

No containers or GPUs: the runner, client and cache are mocks.
"""

import logging
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.common import dataset_manifest as dsm
from src.worker.cli import Worker
from src.worker.client import TaskInfo
from src.worker.runner import ExecutionResult


def _content_id(d: Path) -> str:
    """content_id of a dir hashed without a manifest (every file)."""
    return dsm.build_manifest(d)["content_id"]


def _make_dataset(d: Path, fill: bytes = b"\x00") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.npy").write_bytes(fill * 100)
    (d / "labels.npy").write_bytes(fill * 10)
    (d / "test_data.npy").write_bytes(fill * 50)
    (d / "test_labels.npy").write_bytes(fill * 5)
    return d


def _task(task_id="task_2", deps=None, run_id=None) -> TaskInfo:
    return TaskInfo(
        id=task_id,
        tool_name="tool",
        tool_image="img:1",
        tool_command="python main.py",
        tool_runtime=None,
        tool_is_baseline=False,
        config={"p": 1},
        priority=1,
        status="pending",
        task_type="pre_training",
        counter=1,
        workflows=["w1"],
        pipeline_id="pl",
        dependency_ids=list(deps or []),
        run_id=run_id,
    )


@pytest.fixture
def dataset(tmp_path) -> Path:
    return _make_dataset(tmp_path / "data")


@pytest.fixture
def worker(tmp_path, dataset, monkeypatch) -> Worker:
    monkeypatch.delenv("LANDSEER_ALLOW_DATASET_MISMATCH", raising=False)
    monkeypatch.delenv("LANDSEER_PIPELINE_CONFIG", raising=False)
    w = Worker(workspace_dir=tmp_path / "ws", cache_dir=tmp_path / "cache",
               data_path=dataset, use_cache=False)
    w._client = MagicMock()
    w._client.get_dataset_manifest.return_value = None
    w._runner = MagicMock()
    w._runner.workspace_dir = tmp_path / "ws"

    def _run_task(task, **kwargs):
        out = tmp_path / "ws" / task.id / "output"
        out.mkdir(parents=True, exist_ok=True)
        return ExecutionResult(success=True, exit_code=0, execution_time_ms=1, output_path=out)

    w._runner.run_task.side_effect = _run_task
    return w


def _parent_output(tmp_path: Path, name="task_1") -> Path:
    out = tmp_path / "parents" / name / "output"
    out.mkdir(parents=True)
    (out / "model.pt").write_bytes(b"m")
    return out


# ----------------------------------------------------------------------------
# Cache key identity
# ----------------------------------------------------------------------------


def test_cache_key_same_on_hosts_with_same_files(tmp_path, monkeypatch):
    monkeypatch.delenv("LANDSEER_PIPELINE_CONFIG", raising=False)
    host_a = _make_dataset(tmp_path / "host_a" / "datasets" / "cifar10")
    host_b = tmp_path / "host_b" / "elsewhere"
    shutil.copytree(host_a, host_b)
    os.utime(host_b / "data.npy", ns=(10**9, 10**9))
    (host_a / "config_model.py").write_text("NET = 1\n")
    (host_b / "config_model.py").write_text("NET = 1\n")

    keys = []
    for host in (host_a, host_b):
        w = Worker(workspace_dir=tmp_path / f"ws_{host.name}", data_path=host)
        w._dataset_info = {"name": "cifar10", "variant": "clean",
                           "minio_key": f"datasets/{host.name}"}
        w._model_script_path = host / "config_model.py"
        keys.append(w._compute_cache_key(_task(), ["p1"], w._build_cache_context(host)))
    assert keys[0] == keys[1]


def test_cache_key_differs_when_dataset_content_differs(tmp_path):
    a = _make_dataset(tmp_path / "a")
    b = _make_dataset(tmp_path / "b", fill=b"\x01")
    w = Worker(workspace_dir=tmp_path / "ws")
    ka = w._compute_cache_key(_task(), [], w._build_cache_context(a))
    kb = w._compute_cache_key(_task(), [], w._build_cache_context(b))
    assert ka != kb


# ----------------------------------------------------------------------------
# Recording dataset_content_id on results
# ----------------------------------------------------------------------------


def test_executed_task_records_dataset_id(worker, dataset):
    result = worker._execute_task(_task("task_1"))
    sha = _content_id(dataset)
    assert result.success
    assert result.artifacts["dataset_content_id"] == sha
    assert worker._task_dataset_id["task_1"] == sha


def test_cache_hit_records_dataset_id(worker, dataset, tmp_path):
    cached = tmp_path / "cached_output"
    cached.mkdir()
    (cached / "model.pt").write_bytes(b"c")
    worker.use_cache = True
    worker._two_level_cache = MagicMock()
    worker._two_level_cache.get.return_value = cached

    result = worker._execute_task(_task("task_1"))

    assert result.artifacts["cache_hit"] is True
    assert result.artifacts["dataset_content_id"] == _content_id(dataset)
    assert worker._task_dataset_id["task_1"] == _content_id(dataset)
    worker._runner.run_task.assert_not_called()


# ----------------------------------------------------------------------------
# Parent lineage check
# ----------------------------------------------------------------------------


def test_local_parent_on_other_dataset_blocks_task(worker, tmp_path):
    worker._task_outputs["task_1"] = ("k1", _parent_output(tmp_path), [])
    worker._task_dataset_id["task_1"] = "f" * 64

    result = worker._execute_task(_task("task_2", deps=["task_1"]))

    assert not result.success
    assert "parent task_1 was produced on dataset ffffffffffff" in result.error_message
    assert str(worker.data_path) in result.error_message
    worker._runner.run_task.assert_not_called()


def test_local_parent_on_same_dataset_runs(worker, dataset, tmp_path):
    worker._task_outputs["task_1"] = ("k1", _parent_output(tmp_path), [])
    worker._task_dataset_id["task_1"] = _content_id(dataset)

    result = worker._execute_task(_task("task_2", deps=["task_1"]))

    assert result.success
    worker._runner.run_task.assert_called_once()


def test_local_parent_with_unknown_dataset_warns_and_runs(worker, tmp_path, caplog):
    # Older entries were recorded as 3-tuples only, with no hash alongside.
    worker._task_outputs["task_1"] = ("k1", _parent_output(tmp_path), [])

    with caplog.at_level(logging.WARNING):
        result = worker._execute_task(_task("task_2", deps=["task_1"]))

    assert result.success
    worker._runner.run_task.assert_called_once()


def _remote_parent(tmp_path, sha):
    parent = _task("task_1")
    parent.output_path = str(_parent_output(tmp_path))
    parent.cache_key = "k1"
    parent.dataset_content_id = sha
    return parent


def test_remote_parent_on_other_dataset_blocks_task(worker, tmp_path):
    worker._client.get_task.return_value = _remote_parent(tmp_path, "e" * 64)

    result = worker._execute_task(_task("task_2", deps=["task_1"]))

    assert not result.success
    assert "parent task_1 was produced on dataset eeeeeeeeeeee" in result.error_message
    worker._runner.run_task.assert_not_called()


def test_remote_parent_on_same_dataset_runs_and_is_remembered(worker, dataset, tmp_path):
    sha = _content_id(dataset)
    worker._client.get_task.return_value = _remote_parent(tmp_path, sha)

    result = worker._execute_task(_task("task_2", deps=["task_1"]))

    assert result.success
    assert worker._task_dataset_id["task_1"] == sha
    worker._runner.run_task.assert_called_once()


def test_remote_parent_with_unknown_dataset_runs(worker, tmp_path):
    worker._client.get_task.return_value = _remote_parent(tmp_path, None)
    result = worker._execute_task(_task("task_2", deps=["task_1"]))
    assert result.success


# ----------------------------------------------------------------------------
# Manual data path vs backend dataset
# ----------------------------------------------------------------------------


def _other_manifest(tmp_path):
    return dsm.build_manifest(_make_dataset(tmp_path / "backend_copy", fill=b"\x07"),
                              provenance={"name": "cifar10"})


def test_manual_mismatch_detected(worker, tmp_path):
    worker._dataset_info = {"name": "cifar10", "variant": "clean"}
    other = _other_manifest(tmp_path)
    worker._client.get_dataset_manifest.return_value = other
    msg = worker._manual_dataset_mismatch()
    assert msg and "LANDSEER_DATA_PATH" in msg and other["content_id"][:12] in msg
    assert "sha256_mismatch" in msg


def test_manual_match_ok_and_sets_identity(worker, dataset):
    manifest = dsm.build_manifest(dataset, provenance={"name": "cifar10", "variant": "clean"})
    worker._client.get_dataset_manifest.return_value = manifest
    assert worker._manual_dataset_mismatch() is None
    assert worker._dataset_identity(dataset) == (
        manifest["content_id"], manifest["provenance"]["provenance_id"]
    )


def test_manual_extra_files_are_ignored(worker, dataset):
    manifest = dsm.build_manifest(dataset)
    (dataset / "notes.txt").write_text("scratch")
    worker._client.get_dataset_manifest.return_value = manifest
    # The project logger does not propagate to caplog, so watch it directly.
    with patch("src.worker.cli.logger") as log:
        assert worker._manual_dataset_mismatch() is None
    assert any("not in the dataset manifest" in str(c) for c in log.warning.call_args_list)


def test_empty_dataset_dir_has_no_identity(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    w = Worker(workspace_dir=tmp_path / "ws")
    w._dataset_info = {"name": "x"}
    assert w._dataset_identity(empty) == (None, None)
    assert w._dataset_fingerprint(empty) == {"available": False}


def test_manual_backend_without_manifest_warns_and_continues(worker):
    worker._client.get_dataset_manifest.return_value = None
    assert worker._manual_dataset_mismatch() is None


def test_manual_mismatch_override_uses_local_identity(worker, dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("LANDSEER_ALLOW_DATASET_MISMATCH", "1")
    other = _other_manifest(tmp_path)
    worker._client.get_dataset_manifest.return_value = other
    assert worker._manual_dataset_mismatch() is None
    content_id, prov_id = worker._dataset_identity(dataset)
    assert content_id != other["content_id"] and prov_id is None


def test_auto_mode_is_not_checked(tmp_path):
    w = Worker(workspace_dir=tmp_path / "ws")
    w._client = MagicMock()
    assert w._manual_dataset_mismatch() is None
    w._client.get_dataset_manifest.assert_not_called()


def test_start_exits_on_manual_mismatch(worker, tmp_path):
    worker._client.get_dataset_manifest.return_value = _other_manifest(tmp_path)

    with patch.object(worker, "_init_components"), \
         patch.object(worker, "_wait_for_backend", return_value=True), \
         patch.object(worker, "_register", return_value=True), \
         patch.object(worker, "_fetch_dataset", return_value=worker.data_path), \
         patch.object(worker, "_work_loop") as loop:
        assert worker.start() == 1
    loop.assert_not_called()


def test_start_continues_when_backend_has_no_manifest(worker):
    with patch.object(worker, "_init_components"), \
         patch.object(worker, "_wait_for_backend", return_value=True), \
         patch.object(worker, "_register", return_value=True), \
         patch.object(worker, "_fetch_dataset", return_value=worker.data_path), \
         patch.object(worker, "_work_loop") as loop:
        assert worker.start() == 0
    loop.assert_called_once()


def test_run_change_with_mismatched_dataset_fails_every_task(worker, tmp_path):
    worker._client.get_dataset_info.return_value = {"available": True}
    worker._client.get_pipeline_info.return_value = {}
    worker._client.get_dataset_manifest.return_value = _other_manifest(tmp_path)

    r1 = worker._execute_task(_task("task_1", run_id="run_x"))
    r2 = worker._execute_task(_task("task_2", run_id="run_x"))

    assert not r1.success and not r2.success
    assert "LANDSEER_DATA_PATH" in r1.error_message
    worker._runner.run_task.assert_not_called()


def test_results_record_provenance_id_when_manifest_matches(worker, dataset):
    manifest = dsm.build_manifest(dataset, provenance={"name": "cifar10", "image": "img@sha256:1"})
    worker._dataset_manifest = manifest
    result = worker._execute_task(_task("task_1"))
    assert result.artifacts["dataset_content_id"] == manifest["content_id"]
    assert result.artifacts["dataset_provenance_id"] == manifest["provenance"]["provenance_id"]


# ----------------------------------------------------------------------------
# Backend / client plumbing
# ----------------------------------------------------------------------------


def test_task_info_parses_dataset_id():
    info = TaskInfo.from_api_response({"id": "t", "dataset_content_id": "abc"})
    assert info.dataset_content_id == "abc"
    assert TaskInfo.from_api_response({"id": "t"}).dataset_content_id is None


def test_task_to_response_reads_nested_worker_artifacts():
    from src.backend.api.helpers import task_to_response

    task = SimpleNamespace(
        id="task_1",
        tool=SimpleNamespace(
            name="tool",
            container=SimpleNamespace(image="img", command="cmd", runtime=None),
            is_baseline=False,
        ),
        config={},
        priority=1,
        status=SimpleNamespace(value="completed"),
        task_type=SimpleNamespace(value="pre_training"),
        counter=1,
        workflows=[],
        pipeline_id="pl",
        run_id="r",
        dependencies=[],
    )
    state = SimpleNamespace(
        pipeline=None,
        task_metadata={"task_1": {"result": {
            "artifacts": {"cache_key": "ck", "cache_hit": True, "dataset_content_id": "ds",
                          "dataset_provenance_id": "pv"},
            "output_path": "/out",
        }}},
    )
    resp = task_to_response(task, state)
    assert resp.cache_key == "ck"
    assert resp.cache_hit is True
    assert resp.dataset_content_id == "ds"
    assert resp.dataset_provenance_id == "pv"
    assert resp.output_path == "/out"


def test_backend_dataset_summary_reads_info_or_manifest(tmp_path):
    from src.backend.api.routes.dataset import _manifest_summary

    d = _make_dataset(tmp_path / "served")
    manifest = dsm.build_manifest(d, provenance={"name": "cifar10"})
    dsm.write_manifest(d, manifest)
    from_file = _manifest_summary({"output_dir": str(d)})
    assert from_file["content_id"] == manifest["content_id"]
    assert from_file["file_count"] == 4
    assert _manifest_summary({"content_id": "abc", "provenance_id": "p"})["content_id"] == "abc"
    assert _manifest_summary({"output_dir": str(tmp_path / "missing")}) == {}
