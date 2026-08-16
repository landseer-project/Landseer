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
