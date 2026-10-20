"""Skipped or failed evaluation outputs are neither cached nor replayed from cache."""

import json
from pathlib import Path

import pytest

from src.worker.cli import Worker
from src.worker.client import TaskInfo


def _task(task_type: str = "evaluation") -> TaskInfo:
    return TaskInfo(
        id="eval_1", tool_name="fairness", tool_image="img:1", tool_command="run",
        tool_runtime=None, tool_is_baseline=False, config={}, priority=50,
        status="pending", task_type=task_type, counter=1, workflows=[],
        pipeline_id="pl", dependency_ids=[],
    )


def _output(tmp_path: Path, payload) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    if payload is not None:
        (out / "evaluation_results.json").write_text(json.dumps(payload))
    return out


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"success": True, "skipped": False, "metrics": {"m": 0.9}}, False),
        ({"success": True, "skipped": True, "metrics": {"m": -1.0}}, True),
        ({"success": False, "error": "boom", "metrics": {}}, True),
        (None, True),  # no results file: container crashed
    ],
)
def test_eval_not_cacheable(tmp_path, payload, expected):
    out = _output(tmp_path, payload)
    assert Worker._eval_not_cacheable(None, _task(), out) is expected


def test_non_evaluation_tasks_always_cacheable(tmp_path):
    out = _output(tmp_path, None)
    assert Worker._eval_not_cacheable(None, _task("during_training"), out) is False
