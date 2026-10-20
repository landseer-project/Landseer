"""
Adapter chain execution tests (src/worker/adapters.py).

Tests cover:
- task_adapters: absent, malformed and valid task config entries
- Order of input adapter -> tool -> output adapter and their mounts
- Input adapter files are visible to the tool; tool outputs are visible to the output adapter
- Output adapter files replace tool files with the same name in the task output
- Failures: each step stops the chain and is named in the error message
- Adapter environment (LANDSEER_ADAPTER, per-adapter env) and image pulls
- TaskRunner.run_task dispatches to the chain only for tasks with adapters
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

from src.worker.adapters import run_adapter_chain, task_adapters
from src.worker.client import TaskInfo
from src.worker.runner import TaskRunner

IN_ADAPTER = {"name": "torch-to-tf", "image": "test/adapter:v1", "command": "python3 main.py --direction input", "env": {"A": "1"}}
OUT_ADAPTER = {"name": "tf-to-torch", "image": "test/adapter:v1", "command": "python3 main.py --direction output", "env": {}}


class FakeRunner:
    """Container runner double: records calls and runs a Python side effect per image/command."""

    def __init__(self, effects: Optional[Dict[str, object]] = None, exit_codes: Optional[Dict[str, int]] = None):
        self.calls: List[dict] = []
        self.effects = effects or {}
        self.exit_codes = exit_codes or {}
        self.pulled: List[str] = []
        self.pull_ok = True

    @staticmethod
    def _key(command: str) -> str:
        if "--direction input" in command:
            return "input"
        if "--direction output" in command:
            return "output"
        return "tool"

    def pull_image(self, image: str) -> bool:
        self.pulled.append(image)
        return self.pull_ok

    def run(self, image, command, input_dir, output_dir, env=None, extra_mounts=None, model_script_path=None):
        key = self._key(command)
        self.calls.append({
            "key": key,
            "image": image,
            "input_dir": Path(input_dir),
            "output_dir": Path(output_dir),
            "env": dict(env or {}),
            "input_files": sorted(str(p.relative_to(input_dir)) for p in Path(input_dir).rglob("*") if p.is_file()),
        })
        # Real runners add these to the dict they receive.
        if env is not None:
            env["INPUT_DIR"] = "/input"
        effect = self.effects.get(key)
        if effect:
            effect(Path(input_dir), Path(output_dir))
        return self.exit_codes.get(key, 0), f"{key} log", f"run {image} {command}"


def _input_adapter_effect(in_dir: Path, out_dir: Path) -> None:
    (out_dir / "model_tf").mkdir(parents=True)
    (out_dir / "model_tf" / "model.keras").write_text("initial")
    (out_dir / "model_tf" / "weight_map.json").write_text("{}")


def _tool_effect(in_dir: Path, out_dir: Path) -> None:
    assert (in_dir / "model_tf" / "model.keras").read_text() == "initial"
    (out_dir / "model_tf").mkdir(parents=True)
    (out_dir / "model_tf" / "model.keras").write_text("trained")
    (out_dir / "model.pt").write_text("tool-placeholder")
    (out_dir / "metrics.json").write_text("{}")


def _output_adapter_effect(in_dir: Path, out_dir: Path) -> None:
    # Sees the tool's trained model and the input adapter's weight map side by side.
    assert (in_dir / "model_tf" / "model.keras").read_text() == "trained"
    assert (in_dir / "model_tf" / "weight_map.json").exists()
    (out_dir / "model.pt").write_text("converted")


@pytest.fixture
def workspace(tmp_path: Path):
    task_dir = tmp_path / "task_1"
    input_dir = task_dir / "input"
    output_dir = task_dir / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (input_dir / "data.npy").write_text("data")
    (input_dir / "config_model.py").write_text("def config(): pass")
    return task_dir, input_dir, output_dir


def _chain(runner, workspace, adapters, **kwargs):
    task_dir, input_dir, output_dir = workspace
    return run_adapter_chain(
        runner,
        adapters=adapters,
        tool_image="test/tool:v1",
        tool_command="python3 main.py",
        task_dir=task_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        env=kwargs.pop("env", {"PYTHONPATH": "/input:/app"}),
        **kwargs,
    )


class TestTaskAdapters:
    @pytest.mark.parametrize("config", [None, {}, {"stage": "x"}, {"adapters": None}, {"adapters": {}}])
    def test_no_adapters(self, config):
        assert task_adapters(config) == {}

    def test_valid(self):
        assert task_adapters({"adapters": {"input": IN_ADAPTER}}) == {"input": IN_ADAPTER}

    def test_ignores_unknown_positions(self):
        assert task_adapters({"adapters": {"input": IN_ADAPTER, "middle": OUT_ADAPTER}}) == {"input": IN_ADAPTER}

    def test_not_a_mapping(self):
        with pytest.raises(ValueError):
            task_adapters({"adapters": ["input"]})

    def test_missing_image(self):
        with pytest.raises(ValueError, match="output adapter"):
            task_adapters({"adapters": {"output": {"name": "x"}}})


class TestRunAdapterChain:
    def test_full_chain_order_and_outputs(self, workspace):
        task_dir, input_dir, output_dir = workspace
        runner = FakeRunner(effects={
            "input": _input_adapter_effect, "tool": _tool_effect, "output": _output_adapter_effect,
        })
        exit_code, logs, commands, error = _chain(runner, workspace, {"input": IN_ADAPTER, "output": OUT_ADAPTER})

        assert (exit_code, error) == (0, None)
        assert [c["key"] for c in runner.calls] == ["input", "tool", "output"]
        # Input adapter: task input -> its own staging dir.
        assert runner.calls[0]["input_dir"] == input_dir
        assert runner.calls[0]["output_dir"] == task_dir / "adapters" / "input"
        # Tool: unchanged contract, task input -> task output.
        assert runner.calls[1]["input_dir"] == input_dir
        assert runner.calls[1]["output_dir"] == output_dir
        assert "model_tf/model.keras" in runner.calls[1]["input_files"]
        # Output adapter: task input (with tool outputs) -> its own staging dir.
        assert runner.calls[2]["input_dir"] == input_dir
        assert runner.calls[2]["output_dir"] == task_dir / "adapters" / "output"
        # Output adapter's model.pt replaces the tool's; other tool outputs are kept.
        assert (output_dir / "model.pt").read_text() == "converted"
        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "model_tf" / "model.keras").read_text() == "trained"
        # Logs and commands from all three containers.
        assert "input log" in logs and "tool log" in logs and "output log" in logs
        assert len(commands.splitlines()) == 3
        # Staging dirs are removed once their files are overlaid.
        assert not (task_dir / "adapters").exists()

    def test_input_adapter_only(self, workspace):
        _, _, output_dir = workspace
        runner = FakeRunner(effects={"input": _input_adapter_effect})
        exit_code, _, _, error = _chain(runner, workspace, {"input": IN_ADAPTER})
        assert (exit_code, error) == (0, None)
        assert [c["key"] for c in runner.calls] == ["input", "tool"]

    def test_output_adapter_only(self, workspace):
        _, _, output_dir = workspace
        runner = FakeRunner(effects={"output": lambda i, o: (o / "model.pt").write_text("converted")})
        exit_code, _, _, error = _chain(runner, workspace, {"output": OUT_ADAPTER})
        assert (exit_code, error) == (0, None)
        assert [c["key"] for c in runner.calls] == ["tool", "output"]
        assert (output_dir / "model.pt").read_text() == "converted"

    def test_input_adapter_failure_stops_chain(self, workspace):
        runner = FakeRunner(exit_codes={"input": 3})
        exit_code, logs, _, error = _chain(runner, workspace, {"input": IN_ADAPTER, "output": OUT_ADAPTER})
        assert exit_code == 3
        assert error == "Input adapter 'torch-to-tf' exited with code 3"
        assert [c["key"] for c in runner.calls] == ["input"]
        assert error in logs

    def test_tool_failure_skips_output_adapter(self, workspace):
        runner = FakeRunner(effects={"input": _input_adapter_effect}, exit_codes={"tool": 1})
        exit_code, _, _, error = _chain(runner, workspace, {"input": IN_ADAPTER, "output": OUT_ADAPTER})
        assert exit_code == 1
        assert error == "Container exited with code 1"
        assert [c["key"] for c in runner.calls] == ["input", "tool"]

    def test_output_adapter_failure(self, workspace):
        runner = FakeRunner(exit_codes={"output": 2})
        exit_code, _, _, error = _chain(runner, workspace, {"output": OUT_ADAPTER})
        assert exit_code == 2
        assert error == "Output adapter 'tf-to-torch' exited with code 2"

    def test_adapter_env(self, workspace):
        env = {"PYTHONPATH": "/input:/app"}
        runner = FakeRunner()
        _chain(runner, workspace, {"input": IN_ADAPTER, "output": OUT_ADAPTER}, env=env)
        in_env, tool_env, out_env = (c["env"] for c in runner.calls)
        assert in_env["LANDSEER_ADAPTER"] == "input" and in_env["A"] == "1"
        assert in_env["LANDSEER_TOOL_IMAGE"] == "test/tool:v1"
        assert out_env["LANDSEER_ADAPTER"] == "output" and "A" not in out_env
        assert "LANDSEER_ADAPTER" not in tool_env
        assert all(e["PYTHONPATH"] == "/input:/app" for e in (in_env, tool_env, out_env))
        # Runner mutations do not leak into the caller's env or between steps.
        assert env == {"PYTHONPATH": "/input:/app"}
        assert "INPUT_DIR" not in tool_env

    def test_pulls_adapter_images(self, workspace):
        runner = FakeRunner()
        _chain(runner, workspace, {"input": IN_ADAPTER, "output": OUT_ADAPTER}, pull_image=runner.pull_image)
        assert runner.pulled == ["test/adapter:v1", "test/adapter:v1"]

    def test_pull_failure(self, workspace):
        runner = FakeRunner()
        runner.pull_ok = False
        exit_code, _, _, error = _chain(runner, workspace, {"input": IN_ADAPTER}, pull_image=runner.pull_image)
        assert exit_code == -1
        assert "input adapter image" in error
        assert runner.calls == []

    def test_model_path_written_by_tool_passes(self, workspace):
        _, _, output_dir = workspace
        runner = FakeRunner(effects={
            "input": _input_adapter_effect, "tool": _tool_effect, "output": _output_adapter_effect,
        })
        out = {**OUT_ADAPTER, "model_path": "model_tf/model.keras"}
        exit_code, _, _, error = _chain(runner, workspace, {"input": IN_ADAPTER, "output": out})
        assert (exit_code, error) == (0, None)
        assert (output_dir / "model.pt").read_text() == "converted"

    def test_tool_that_saved_elsewhere_fails_instead_of_converting_untrained_model(self, workspace):
        """H1: the input adapter's untrained model is also under /input; without the
        check the output adapter would convert it and the task would succeed."""
        _, _, output_dir = workspace

        def tool_saves_elsewhere(in_dir: Path, out_dir: Path) -> None:
            (out_dir / "tf_out").mkdir(parents=True)
            (out_dir / "tf_out" / "model.keras").write_text("trained")

        runner = FakeRunner(effects={"input": _input_adapter_effect, "tool": tool_saves_elsewhere})
        out = {**OUT_ADAPTER, "model_path": "model_tf/model.keras"}
        exit_code, logs, _, error = _chain(runner, workspace, {"input": IN_ADAPTER, "output": out})
        assert exit_code == -1
        assert "/output/model_tf/model.keras" in error and "tf-to-torch" in error
        assert [c["key"] for c in runner.calls] == ["input", "tool"]
        assert not (output_dir / "model.pt").exists()
        assert error in logs

    def test_tool_that_saved_nothing_fails(self, workspace):
        runner = FakeRunner(effects={"input": _input_adapter_effect})
        out = {**OUT_ADAPTER, "model_path": "model_tf/model.keras"}
        exit_code, _, _, error = _chain(runner, workspace, {"input": IN_ADAPTER, "output": out}, pull_image=runner.pull_image)
        assert exit_code == -1
        assert "did not write its model" in error
        # Checked before pulling the output adapter image.
        assert runner.pulled == ["test/adapter:v1"]

    def test_without_model_path_output_adapter_still_runs(self, workspace):
        """Older task configs carry no model_path; the chain runs as before."""
        runner = FakeRunner(effects={"input": _input_adapter_effect})
        exit_code, _, _, error = _chain(runner, workspace, {"input": IN_ADAPTER, "output": OUT_ADAPTER})
        assert (exit_code, error) == (0, None)
        assert [c["key"] for c in runner.calls] == ["input", "tool", "output"]

    def test_rerun_clears_stale_adapter_outputs(self, workspace):
        task_dir, _, output_dir = workspace
        stale = task_dir / "adapters" / "output" / "stale.txt"
        stale.parent.mkdir(parents=True)
        stale.write_text("old")
        runner = FakeRunner()
        _chain(runner, workspace, {"output": OUT_ADAPTER})
        assert not stale.exists()
        assert not (output_dir / "stale.txt").exists()


def _task(config: dict) -> TaskInfo:
    return TaskInfo(
        id="task_adapters_1",
        tool_name="in-freeat",
        tool_image="test/tool:v1",
        tool_command="python3 main.py",
        tool_runtime=None,
        tool_is_baseline=False,
        config=config,
        priority=100,
        status="pending",
        task_type="in_training",
        counter=1,
        workflows=[],
        pipeline_id="pipeline_1",
        dependency_ids=[],
    )


class TestTaskRunnerDispatch:
    def _runner(self, tmp_path: Path, fake: FakeRunner) -> TaskRunner:
        task_runner = TaskRunner(workspace_dir=tmp_path / "ws", gpu_id=None, runtime="docker")
        task_runner._container_runner = fake
        return task_runner

    def test_task_with_adapters_runs_chain(self, tmp_path):
        fake = FakeRunner(effects={
            "input": _input_adapter_effect, "tool": _tool_effect, "output": _output_adapter_effect,
        })
        task_runner = self._runner(tmp_path, fake)
        result = task_runner.run_task(_task({"stage": "during_training", "adapters": {"input": IN_ADAPTER, "output": OUT_ADAPTER}}))
        assert result.success, result.error_message
        assert [c["key"] for c in fake.calls] == ["input", "tool", "output"]
        assert (result.output_path / "model.pt").read_text() == "converted"
        # Tool image pulled by run_task, adapter images by the chain.
        assert fake.pulled == ["test/tool:v1", "test/adapter:v1", "test/adapter:v1"]

    def test_task_without_adapters_runs_tool_only(self, tmp_path):
        fake = FakeRunner()
        task_runner = self._runner(tmp_path, fake)
        result = task_runner.run_task(_task({"stage": "during_training"}))
        assert result.success
        assert [c["key"] for c in fake.calls] == ["tool"]

    def test_adapter_failure_reported(self, tmp_path):
        fake = FakeRunner(exit_codes={"output": 5})
        task_runner = self._runner(tmp_path, fake)
        result = task_runner.run_task(_task({"adapters": {"output": OUT_ADAPTER}}))
        assert not result.success
        assert result.exit_code == 5
        assert result.error_message == "Output adapter 'tf-to-torch' exited with code 5"

    def test_apptainer_runs_all_three_containers(self, tmp_path):
        """Same chain under Apptainer: three `apptainer exec` calls with the same binds, no docker pull."""
        task_runner = TaskRunner(workspace_dir=tmp_path / "ws", gpu_id=None, runtime="apptainer")
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")
        with patch("src.worker.runner.subprocess.run", return_value=completed) as run:
            result = task_runner.run_task(_task({"adapters": {"input": IN_ADAPTER, "output": OUT_ADAPTER}}))
        assert result.success, result.error_message
        commands = [call.args[0] for call in run.call_args_list if call.args and call.args[0][0] == "apptainer"]
        assert len(commands) == 3
        assert commands[0][-5:] == ["docker://test/adapter:v1", "python3", "main.py", "--direction", "input"]
        assert commands[1][-3:] == ["docker://test/tool:v1", "python3", "main.py"]
        assert commands[2][-5:] == ["docker://test/adapter:v1", "python3", "main.py", "--direction", "output"]
        task_dir = tmp_path / "ws" / "task_adapters_1"
        binds = [" ".join(c) for c in commands]
        assert all(f"{(task_dir / 'input').absolute()}:/input:ro" in b for b in binds)
        assert f"{(task_dir / 'adapters' / 'input').absolute()}:/output:rw" in binds[0]
        assert f"{(task_dir / 'output').absolute()}:/output:rw" in binds[1]
        assert f"{(task_dir / 'adapters' / 'output').absolute()}:/output:rw" in binds[2]

    def test_malformed_adapters_fail_task(self, tmp_path):
        fake = FakeRunner()
        task_runner = self._runner(tmp_path, fake)
        result = task_runner.run_task(_task({"adapters": {"input": {"name": "no-image"}}}))
        assert not result.success
        assert "input adapter" in result.error_message
        assert fake.calls == []
