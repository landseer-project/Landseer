"""
Worker workspace handling for run-scoped task IDs and linked inputs.

Tests cover:
- _is_task_dir_name: legacy and run-scoped task dir names, nothing else
- stale-workspace pruning removes run-scoped task dirs too
- ancestor subfolders are hard-linked into the input, not copied
- a read-only config.json linked in from an ancestor output is replaced by the
  task's own config without touching the ancestor's file
"""

import json
import os
import stat
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.worker.client import TaskInfo
from src.worker.runner import TaskRunner, _is_task_dir_name


def _task(task_id: str, config=None) -> TaskInfo:
    return TaskInfo(
        id=task_id,
        tool_name="clean",
        tool_image="test/image:latest",
        tool_command="python main.py",
        tool_runtime=None,
        tool_is_baseline=False,
        config=config if config is not None else {"stage": "evaluation", "evaluator": "clean"},
        priority=100,
        status="pending",
        task_type="evaluation",
        counter=1,
        workflows=["e807ab0e-comb_001"],
        pipeline_id="run_20261013_232027_e807ab0e",
        dependency_ids=[],
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("task_17", True),
        ("e807ab0e-3f9a2c41d0e7", True),
        ("e807ab0e-3f9a2c41d0e7-2", True),
        ("task_", False),
        ("cache", False),
        ("e807ab0e-3f9a2c41d0e", False),
        ("E807AB0E-3F9A2C41D0E7", False),
        ("logs", False),
    ],
)
def test_is_task_dir_name(name, expected):
    assert _is_task_dir_name(name) is expected


def test_prune_removes_stale_run_scoped_dirs(tmp_path):
    runner = TaskRunner(workspace_dir=tmp_path, gpu_id=None, timeout=60)
    old = time.time() - 96 * 3600  # older than the 72h default retention
    stale = [tmp_path / "task_3", tmp_path / "e807ab0e-3f9a2c41d0e7"]
    keep = [tmp_path / "3b81bf4a-aaaaaaaaaaaa", tmp_path / "not_a_task"]
    for d in stale + keep:
        (d / "output").mkdir(parents=True)
    for d in stale + [tmp_path / "not_a_task"]:
        os.utime(d, (old, old))
    runner._last_prune_ts = 0
    runner._maybe_prune_stale_storage()
    for d in stale:
        assert not d.exists(), d
    for d in keep:
        assert d.exists(), d


def test_linked_readonly_config_json_is_replaced(tmp_path):
    workspace = tmp_path / "ws"
    ancestor = tmp_path / "ancestor_output"
    ancestor.mkdir()
    anc_cfg = ancestor / "config.json"
    anc_cfg.write_text('{"from": "ancestor"}')
    (ancestor / "model.pt").write_bytes(b"weights")
    for f in ancestor.iterdir():
        f.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    runner = TaskRunner(workspace_dir=workspace, gpu_id=None, timeout=60)
    container = Mock()
    container.run.return_value = (0, "ok", "fake-cmd")
    runner._container_runner = container

    task = _task("e807ab0e-3f9a2c41d0e7")
    result = runner.run_task(task, ancestor_dirs=[ancestor])

    assert container.run.called, result.error_message
    in_cfg = workspace / task.id / "input" / "config.json"
    assert json.loads(in_cfg.read_text()) == task.config
    assert anc_cfg.read_text() == '{"from": "ancestor"}'


def test_ancestor_subfolders_are_hard_linked_not_copied(tmp_path, monkeypatch):
    monkeypatch.setenv("LANDSEER_HARDLINK_MACRO", "1")  # as on Gilbreth (env.sh)
    workspace = tmp_path / "ws"
    ancestor = tmp_path / "ancestor_output"
    imgs = ancestor / "img_align_celeba"
    imgs.mkdir(parents=True)
    for i in range(5):
        (imgs / f"{i:06d}.jpg").write_bytes(b"img%d" % i)

    runner = TaskRunner(workspace_dir=workspace, gpu_id=None, timeout=60)
    container = Mock()
    container.run.return_value = (0, "ok", "fake-cmd")
    runner._container_runner = container

    task = _task("e807ab0e-3f9a2c41d0e7")
    runner.run_task(task, ancestor_dirs=[ancestor])

    linked = workspace / task.id / "input" / "img_align_celeba"
    for src in imgs.iterdir():
        dst = linked / src.name
        assert dst.read_bytes() == src.read_bytes()
        # Same inode: no extra files count against file-count quotas.
        assert dst.stat().st_ino == src.stat().st_ino
