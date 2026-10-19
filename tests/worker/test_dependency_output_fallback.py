"""
Dependency outputs after workspace pruning.

Workspaces are pruned by age, but the worker remembers each finished task's
workspace output for downstream tasks. Covers:
- an emptied or deleted remembered workspace (or ancestor) falls back to the cache copy
- a dependency with no usable output anywhere fails the task instead of
  running it on a partial input set
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.worker.cli import Worker
from src.worker.client import TaskInfo
from src.worker.runner import ExecutionResult


def _task(task_id, deps=()) -> TaskInfo:
    return TaskInfo(
        id=task_id, tool_name="tool", tool_image="img:1", tool_command="run",
        tool_runtime=None, tool_is_baseline=False, config={}, priority=1,
        status="pending", task_type="deployment", counter=1, workflows=[],
        pipeline_id="pl", dependency_ids=list(deps),
    )


def _worker(tmp_path: Path) -> Worker:
    w = Worker(workspace_dir=tmp_path / "ws", cache_dir=tmp_path / "cache", use_cache=False)
    w._client = MagicMock()
    w._client.get_dataset_info.return_value = {"available": False}
    w._client.get_pipeline_info.return_value = {}
    w._runner = MagicMock()
    w._runner.workspace_dir = tmp_path / "ws"
    w._runner.run_task.return_value = ExecutionResult(
        success=True, exit_code=0, execution_time_ms=1, output_path=tmp_path / "ws" / "out"
    )
    w._two_level_cache = None
    return w


def _output(root: Path, name: str, filename: str = "model.pt") -> Path:
    out = root / name / "output"
    out.mkdir(parents=True)
    (out / filename).write_bytes(b"data")
    return out


def _cache_entry(cache: Path, key: str) -> Path:
    out = _output(cache, key)
    (cache / key / ".success").touch()
    return out


def _ancestor_dirs(w: Worker) -> list:
    return w._runner.run_task.call_args.kwargs["ancestor_dirs"]


@pytest.mark.parametrize("pruned", ["emptied", "deleted"])
def test_pruned_workspace_falls_back_to_cache_copy(tmp_path, pruned):
    w = _worker(tmp_path)
    ws_out = tmp_path / "ws" / "dep" / "output"
    if pruned == "emptied":
        ws_out.mkdir(parents=True)
    cached = _cache_entry(tmp_path / "cache", "k_dep")
    w._task_outputs["dep"] = ("k_dep", ws_out, [])

    result = w._execute_task(_task("child", ["dep"]))

    assert result.success
    assert _ancestor_dirs(w) == [cached]


def test_workspace_with_files_is_used_directly(tmp_path):
    w = _worker(tmp_path)
    ws_out = _output(tmp_path / "ws", "dep")
    _cache_entry(tmp_path / "cache", "k_dep")
    w._task_outputs["dep"] = ("k_dep", ws_out, [])

    w._execute_task(_task("child", ["dep"]))

    assert _ancestor_dirs(w) == [ws_out]


def test_pruned_grandparent_falls_back_to_its_cache_copy(tmp_path):
    w = _worker(tmp_path)
    gp_ws = _output(tmp_path / "ws", "grandparent")
    parent_ws = _output(tmp_path / "ws", "parent", "weights.pth")
    gp_cached = _cache_entry(tmp_path / "cache", "k_gp")
    w._task_outputs["grandparent"] = ("k_gp", gp_ws, [])
    w._task_outputs["parent"] = ("k_parent", parent_ws, [gp_ws])
    shutil.rmtree(gp_ws.parent)

    result = w._execute_task(_task("child", ["parent"]))

    assert result.success
    assert _ancestor_dirs(w) == [gp_cached, parent_ws]


def test_missing_ancestor_output_fails_without_running(tmp_path):
    w = _worker(tmp_path)
    parent_ws = _output(tmp_path / "ws", "parent")
    w._task_outputs["parent"] = ("k_parent", parent_ws, [tmp_path / "ws" / "gone" / "output"])

    result = w._execute_task(_task("child", ["parent"]))

    assert not result.success
    assert "ancestor of parent" in result.error_message
    w._runner.run_task.assert_not_called()


def test_remote_dependency_without_output_fails_without_running(tmp_path):
    w = _worker(tmp_path)
    ws_out = tmp_path / "ws" / "dep" / "output"
    ws_out.mkdir(parents=True)  # pruned and no cache copy
    dep = _task("dep")
    dep.output_path = str(ws_out)
    dep.cache_key = "k_dep"
    w._client.get_task.return_value = dep

    result = w._execute_task(_task("child", ["dep"]))

    assert not result.success
    assert "dep (no output in workspace, cache or MinIO)" in result.error_message
    w._runner.run_task.assert_not_called()
