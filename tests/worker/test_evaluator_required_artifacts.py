"""Tests for evaluator required_artifacts materialization and skip behavior."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.worker.client import TaskInfo
from src.worker.runner import TaskRunner


def _eval_task(
    task_id: str = "eval_1",
    required=None,
    metrics=None,
    artifact_roots=None,
) -> TaskInfo:
    return TaskInfo(
        id=task_id,
        tool_name="fairness",
        tool_image="test/eval:latest",
        tool_command="python evaluate.py",
        tool_runtime=None,
        tool_is_baseline=False,
        config={
            "required_artifacts": required or [],
            "metrics": metrics or ["demographic_parity"],
            "artifact_roots": artifact_roots or [],
        },
        priority=50,
        status="pending",
        task_type="evaluation",
        counter=1,
        workflows=[],
        pipeline_id="pipeline_1",
        dependency_ids=[],
    )


def _mock_container(runner: TaskRunner) -> MagicMock:
    mock = MagicMock()
    mock.run.return_value = (0, "ok", "docker run ...")
    mock.pull_image.return_value = True
    runner._container_runner = mock
    return mock


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


class TestMaterializeRequiredArtifacts:
    def test_copies_from_dataset_dir(self, workspace: Path, tmp_path: Path):
        runner = TaskRunner(workspace_dir=workspace)
        mock = _mock_container(runner)

        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        (data_dir / "sensitive_attributes.npy").write_bytes(b"attrs")
        (data_dir / "data.npy").write_bytes(b"data")

        task = _eval_task(required=["sensitive_attributes.npy"])
        result = runner.run_task(task, input_path=data_dir)

        assert result.success
        assert (workspace / task.id / "input" / "sensitive_attributes.npy").exists()
        mock.run.assert_called_once()

    def test_copies_from_eval_artifacts_dir(
        self, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = TaskRunner(workspace_dir=workspace)
        mock = _mock_container(runner)

        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        (data_dir / "data.npy").write_bytes(b"data")

        sidecar = tmp_path / "eval_artifacts"
        shadow = sidecar / "celeba_shadow"
        shadow.mkdir(parents=True)
        (shadow / "model.pt").write_bytes(b"shadow")
        monkeypatch.setenv("LANDSEER_EVAL_ARTIFACTS_DIR", str(sidecar))

        task = _eval_task(required=["celeba_shadow"])
        result = runner.run_task(task, input_path=data_dir)

        assert result.success
        assert (workspace / task.id / "input" / "celeba_shadow" / "model.pt").read_bytes() == b"shadow"
        mock.run.assert_called_once()

    def test_copies_from_yaml_artifact_roots(self, workspace: Path, tmp_path: Path):
        runner = TaskRunner(workspace_dir=workspace)
        mock = _mock_container(runner)

        data_dir = tmp_path / "dataset"
        data_dir.mkdir()

        sidecar = tmp_path / "yaml_roots"
        shadow = sidecar / "celeba_shadow"
        shadow.mkdir(parents=True)
        (shadow / "model.pt").write_bytes(b"from-yaml")

        task = _eval_task(
            required=["celeba_shadow"],
            artifact_roots=[str(sidecar)],
        )
        result = runner.run_task(task, input_path=data_dir)

        assert result.success
        assert (workspace / task.id / "input" / "celeba_shadow" / "model.pt").read_bytes() == b"from-yaml"
        mock.run.assert_called_once()

    def test_keeps_tool_output_if_already_present(self, workspace: Path, tmp_path: Path):
        runner = TaskRunner(workspace_dir=workspace)
        mock = _mock_container(runner)

        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        (data_dir / "watermark_key.json").write_text('{"from":"dataset"}')

        dep_out = workspace / "dep" / "output"
        dep_out.mkdir(parents=True)
        (dep_out / "watermark_key.json").write_text('{"from":"tool"}')

        task = _eval_task(required=["watermark_key.json"])
        result = runner.run_task(task, input_path=data_dir, ancestor_dirs=[dep_out])

        assert result.success
        payload = (workspace / task.id / "input" / "watermark_key.json").read_text()
        assert json.loads(payload)["from"] == "tool"
        mock.run.assert_called_once()

    def test_skips_when_missing(self, workspace: Path, tmp_path: Path):
        runner = TaskRunner(workspace_dir=workspace)
        mock = _mock_container(runner)

        data_dir = tmp_path / "dataset"
        data_dir.mkdir()

        task = _eval_task(required=["missing_sidecar.bin"], metrics=["demographic_parity"])
        result = runner.run_task(task, input_path=data_dir)

        assert result.success
        assert result.artifacts.get("skipped") is True
        mock.run.assert_not_called()

        results_path = workspace / task.id / "output" / "evaluation_results.json"
        payload = json.loads(results_path.read_text())
        assert payload["skipped"] is True
        assert "missing_sidecar.bin" in payload["skip_reason"]
        assert payload["metrics"]["demographic_parity"] == -1.0

    def test_rejects_path_traversal(self, workspace: Path, tmp_path: Path):
        runner = TaskRunner(workspace_dir=workspace)
        mock = _mock_container(runner)

        data_dir = tmp_path / "dataset"
        data_dir.mkdir()

        task = _eval_task(required=["../outside.txt"])
        result = runner.run_task(task, input_path=data_dir)

        assert result.success
        assert result.artifacts.get("skipped") is True
        mock.run.assert_not_called()

    def test_non_evaluation_tasks_ignore_required_artifacts(
        self, workspace: Path, tmp_path: Path
    ):
        runner = TaskRunner(workspace_dir=workspace)
        mock = _mock_container(runner)

        data_dir = tmp_path / "dataset"
        data_dir.mkdir()

        task = _eval_task(required=["does_not_exist.npy"])
        task.task_type = "during_training"
        result = runner.run_task(task, input_path=data_dir)

        assert result.success
        mock.run.assert_called_once()


class TestJudgeApiKey:
    def test_forwards_key_to_evaluation_tasks(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("JUDGE_API_KEY", "secret-value")
        runner = TaskRunner(workspace_dir=workspace)
        mock = _mock_container(runner)

        result = runner.run_task(_eval_task())

        assert result.success
        assert mock.run.call_args.kwargs["env"]["JUDGE_API_KEY"] == "secret-value"
        config = json.loads((workspace / "eval_1" / "input" / "config.json").read_text())
        assert "JUDGE_API_KEY" not in config

    def test_skips_other_task_types(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("JUDGE_API_KEY", "secret-value")
        runner = TaskRunner(workspace_dir=workspace)
        mock = _mock_container(runner)

        task = _eval_task()
        task.task_type = "during_training"
        result = runner.run_task(task)

        assert result.success
        assert "JUDGE_API_KEY" not in mock.run.call_args.kwargs["env"]

    def test_docker_command_omits_key_value(self, workspace: Path, monkeypatch: pytest.MonkeyPatch):
        from unittest.mock import patch

        from src.worker.runner import DockerRunner

        monkeypatch.setenv("JUDGE_API_KEY", "secret-value")
        runner = DockerRunner(workspace_dir=workspace, gpu_id=None)
        input_dir = workspace / "input"
        output_dir = workspace / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        with patch("src.worker.runner.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.poll.return_value = 0
            proc.returncode = 0
            proc.stdout.readline.return_value = ""
            proc.stderr.readline.return_value = ""
            mock_popen.return_value = proc
            _exit, _logs, cmd = runner.run(
                image="test/eval:latest",
                command="python evaluate.py",
                input_dir=input_dir,
                output_dir=output_dir,
                env={"JUDGE_API_KEY": "secret-value", "FOO": "bar"},
            )

        argv = mock_popen.call_args[0][0]
        assert "JUDGE_API_KEY" in argv
        assert "JUDGE_API_KEY=secret-value" not in argv
        assert "FOO=bar" in argv
        assert "secret-value" not in cmd


class TestTaskConfigNotDumpedToEnv:
    """task.config must not be dumped into subprocess env (Apptainer PathLike crash)."""

    def test_non_string_config_values_omitted_from_env(self, workspace: Path):
        runner = TaskRunner(workspace_dir=workspace)
        mock = _mock_container(runner)
        runner._materialize_required_artifacts = MagicMock(return_value=[])

        task = _eval_task(
            required=["model.pt"],
            metrics=["demographic_parity", "equalized_odds"],
        )
        task.config["_run_use_cache"] = True
        task.config["epochs"] = 10
        task.config["name"] = "fairness_eval"

        result = runner.run_task(task)
        assert result.success

        passed_env = mock.run.call_args.kwargs["env"]
        assert all(isinstance(v, str) for v in passed_env.values())
        assert "metrics" not in passed_env
        assert "required_artifacts" not in passed_env
        assert "_run_use_cache" not in passed_env
        assert "epochs" not in passed_env
        # String config keys are also omitted — config lives in config.json.
        assert "name" not in passed_env
        assert "PYTHONPATH" in passed_env

        config = json.loads((workspace / "eval_1" / "input" / "config.json").read_text())
        assert config["metrics"] == ["demographic_parity", "equalized_odds"]
        assert config["_run_use_cache"] is True
        assert config["name"] == "fairness_eval"

    def test_caller_string_env_still_forwarded(self, workspace: Path):
        runner = TaskRunner(workspace_dir=workspace)
        mock = _mock_container(runner)
        runner._materialize_required_artifacts = MagicMock(return_value=[])

        task = _eval_task()
        result = runner.run_task(task, env={"CUSTOM_FLAG": "yes", "BAD_LIST": [1, 2]})
        assert result.success

        passed_env = mock.run.call_args.kwargs["env"]
        assert passed_env["CUSTOM_FLAG"] == "yes"
        assert "BAD_LIST" not in passed_env
