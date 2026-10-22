"""Evaluators with `input: training_input` get exactly the in-training task's input.

The worker rebuilds the in-training task's ancestor list from backend task
metadata, as it does for every task, so the evaluator's /input matches what
training received even when the tasks ran on other workers.
"""

from pathlib import Path
from unittest.mock import MagicMock

from src.worker.cli import Worker
from src.worker.client import TaskInfo
from src.worker.runner import ExecutionResult


def _info(task_id, task_type, deps=(), output=None, config=None) -> TaskInfo:
    return TaskInfo(
        id=task_id, tool_name=task_id, tool_image="img:1", tool_command="run",
        tool_runtime=None, tool_is_baseline=False, config=config or {}, priority=1,
        status="completed", task_type=task_type, counter=1, workflows=[],
        pipeline_id="pl", dependency_ids=list(deps), output_path=str(output) if output else None,
    )


def _out(root: Path, name: str) -> Path:
    out = root / name / "output"
    out.mkdir(parents=True)
    (out / f"{name}.bin").write_bytes(b"x")
    return out


def _worker(tmp_path: Path, backend: dict) -> Worker:
    w = Worker(workspace_dir=tmp_path / "ws", cache_dir=tmp_path / "cache", use_cache=False)
    w._client = MagicMock()
    w._client.get_dataset_info.return_value = {"available": False}
    w._client.get_pipeline_info.return_value = {}
    w._client.get_task.side_effect = backend.get
    w._runner = MagicMock()
    w._runner.workspace_dir = tmp_path / "ws"
    w._runner.run_task.return_value = ExecutionResult(
        success=True, exit_code=0, execution_time_ms=1, output_path=tmp_path / "ws" / "eval_out"
    )
    w._two_level_cache = None
    return w


def _chain(tmp_path, with_training=True):
    """pre1 -> pre2 -> in1 -> post1 -> dep1 (each with its own output dir)."""
    outs = {n: _out(tmp_path / "remote", n) for n in ("pre1", "pre2", "in1", "post1", "dep1")}
    backend = {
        "pre1": _info("pre1", "pre_training", (), outs["pre1"]),
        "pre2": _info("pre2", "pre_training", ("pre1",), outs["pre2"]),
        "in1": _info("in1", "in_training", ("pre2",), outs["in1"]),
        "post1": _info("post1", "post_training", ("in1",), outs["post1"]),
        "dep1": _info("dep1", "deployment", ("post1",), outs["dep1"]),
    }
    if not with_training:
        backend["post1"] = _info("post1", "post_training", ("pre2",), outs["post1"])
        del backend["in1"]
    return outs, backend


def _eval(scope):
    return _info("eval1", "evaluation", ("dep1",), config={"input_scope": scope} if scope else {})


def test_training_input_scope_gets_only_pre_training_outputs(tmp_path):
    outs, backend = _chain(tmp_path)
    w = _worker(tmp_path, backend)
    result = w._execute_task(_eval("training_input"))
    assert result.success
    assert w._runner.run_task.call_args.kwargs["ancestor_dirs"] == [outs["pre1"], outs["pre2"]]


def test_default_scope_gets_full_chain(tmp_path):
    outs, backend = _chain(tmp_path)
    w = _worker(tmp_path, backend)
    w._execute_task(_eval(None))
    assert w._runner.run_task.call_args.kwargs["ancestor_dirs"] == [
        outs["pre1"], outs["pre2"], outs["in1"], outs["post1"], outs["dep1"],
    ]


def test_matches_in_training_task_own_ancestry(tmp_path):
    """The rebuilt list is exactly what the in-training task itself was given."""
    outs, backend = _chain(tmp_path)
    w = _worker(tmp_path, backend)
    w._execute_task(backend["in1"])
    in_training_input = w._runner.run_task.call_args.kwargs["ancestor_dirs"]

    w2 = _worker(tmp_path, backend)
    w2._execute_task(_eval("training_input"))
    assert w2._runner.run_task.call_args.kwargs["ancestor_dirs"] == in_training_input


def test_no_in_training_task_fails(tmp_path):
    _, backend = _chain(tmp_path, with_training=False)
    w = _worker(tmp_path, backend)
    result = w._execute_task(_eval("training_input"))
    assert not result.success and "no in-training task" in result.error_message
    w._runner.run_task.assert_not_called()


def test_missing_pre_training_output_fails(tmp_path):
    outs, backend = _chain(tmp_path)
    for f in outs["pre1"].iterdir():
        f.unlink()  # pruned workspace, no cache copy
    w = _worker(tmp_path, backend)
    result = w._execute_task(_eval("training_input"))
    assert not result.success
    w._runner.run_task.assert_not_called()
