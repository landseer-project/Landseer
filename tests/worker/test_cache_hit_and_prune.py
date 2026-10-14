"""
Worker cache-hit materialization and storage pruning with the size-capped cache.

Covers:
- a hit on a kept entry clones it into the workspace and leaves the cache entry
- a real ArtifactManager entry stays intact when the workspace copy is deleted
- runner pruning removes old workspaces but never touches cache entries
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

from src.store.artifact_manager import ArtifactManager
from src.worker.cli import Worker
from src.worker.client import TaskInfo
from src.worker.runner import TaskRunner


def _task(task_id="task_1") -> TaskInfo:
    return TaskInfo(
        id=task_id, tool_name="tool", tool_image="img:1", tool_command="run",
        tool_runtime=None, tool_is_baseline=False, config={}, priority=1,
        status="pending", task_type="pre_training", counter=1, workflows=[],
        pipeline_id="pl", dependency_ids=[],
    )


def _worker(tmp_path: Path) -> Worker:
    w = Worker(workspace_dir=tmp_path / "ws", cache_dir=tmp_path / "cache", use_cache=True)
    w._client = MagicMock()
    w._client.get_dataset_info.return_value = {"available": False}
    w._client.get_pipeline_info.return_value = {}
    w._runner = MagicMock()
    w._runner.workspace_dir = tmp_path / "ws"
    return w


def test_hit_on_kept_entry_clones_and_keeps_cache(tmp_path):
    w = _worker(tmp_path)
    manager = ArtifactManager(local_cache_dir=tmp_path / "cache", use_minio=False)
    src = tmp_path / "produced"
    src.mkdir()
    (src / "model.pt").write_bytes(b"weights" * 100)
    manager.store_local("ck", src, "task_0", "tool")

    w._two_level_cache = MagicMock()
    w._two_level_cache.manager = manager
    w._two_level_cache.get.side_effect = manager.get_artifact
    w._compute_cache_key = lambda *a, **k: "ck"

    result = w._execute_task(_task())

    workspace_out = tmp_path / "ws" / "task_1" / "output"
    assert result.artifacts["cache_hit"] is True
    assert (workspace_out / "model.pt").read_bytes() == b"weights" * 100
    assert manager.local_exists("ck")
    w._runner.run_task.assert_not_called()

    # Removing the workspace copy (re-run, pruning) leaves the cache entry whole.
    import shutil
    shutil.rmtree(workspace_out)
    assert (manager.get_local_path("ck") / "output" / "model.pt").read_bytes() == b"weights" * 100


def test_prune_keeps_cache_entries_and_drops_old_workspaces(tmp_path, monkeypatch):
    monkeypatch.setenv("LANDSEER_WORKSPACE_RETENTION_HOURS", "1")
    monkeypatch.setenv("LANDSEER_CACHE_RETENTION_HOURS", "1")
    ws = tmp_path / "ws"
    cache = tmp_path / "cache"
    old = time.time() - 10 * 3600

    old_task = ws / "task_9"
    (old_task / "output").mkdir(parents=True)
    os.utime(old_task, (old, old))
    entry = cache / ("a" * 64)
    (entry / "output").mkdir(parents=True)
    (entry / "output" / "model.pt").write_bytes(b"m")
    (entry / ".success").touch()
    for p in (entry / "output" / "model.pt", entry / "output", entry):
        os.utime(p, (old, old))
    (cache / "datasets").mkdir()
    os.utime(cache / "datasets", (old, old))

    runner = TaskRunner(workspace_dir=ws, artifact_cache_dir=cache)
    runner._last_prune_ts = 0.0
    runner._maybe_prune_stale_storage()

    assert not old_task.exists()
    assert entry.exists() and (entry / "output" / "model.pt").exists()
    assert (cache / "datasets").exists()
