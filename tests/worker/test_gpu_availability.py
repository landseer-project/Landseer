"""
Additional tests for GPU availability and detection.

These tests verify that:
1. GPUs are properly detected and passed to containers
2. Tasks that require GPUs get them
3. Tasks that don't need GPUs can run without them
4. GPU availability is correctly reported
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.worker.runner import DockerRunner, TaskRunner
from src.worker.client import TaskInfo


def _run_docker(runner, input_dir, output_dir):
    """
    Run DockerRunner.run with subprocess mocked and return the docker argv.

    DockerRunner.run probes `docker info` via subprocess.run (stubbed here to
    report an nvidia runtime) and launches the container via subprocess.Popen.
    """
    proc = MagicMock()
    proc.poll.return_value = 0
    proc.returncode = 0
    proc.stdout.readline.return_value = ""
    proc.stderr.readline.return_value = ""
    with patch("src.worker.runner.subprocess.run") as mock_run, \
            patch("src.worker.runner.subprocess.Popen") as mock_popen:
        mock_run.return_value = MagicMock(returncode=0, stdout='{"nvidia": {}, "runc": {}}', stderr="")
        mock_popen.return_value = proc
        runner.run(
            image="test/image:latest",
            command="python main.py",
            input_dir=input_dir,
            output_dir=output_dir,
        )
        assert mock_popen.called, "docker run should be launched via subprocess.Popen"
        return mock_popen.call_args[0][0]


def _env_vars(call_args):
    return [call_args[i + 1] for i, arg in enumerate(call_args) if arg == "-e"]


class TestGPUAvailability:
    """Tests for GPU availability and detection."""

    def _dirs(self, temp_workspace):
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return input_dir, output_dir

    def test_gpu_flag_format_correct(self, temp_workspace):
        """
        GPU flags should be in the correct format for Docker.

        Expected: a single `--runtime=nvidia` argument and unquoted env values
        `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=all`,
        `CUDA_VISIBLE_DEVICES=0` (no quotes embedded in the strings).
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        call_args = _run_docker(runner, *self._dirs(temp_workspace))

        assert "--runtime=nvidia" in call_args, \
            f"Should pass --runtime=nvidia as a single argument: {call_args}"
        env_vars = _env_vars(call_args)
        for expected in ("NVIDIA_VISIBLE_DEVICES=all",
                         "NVIDIA_DRIVER_CAPABILITIES=all",
                         "CUDA_VISIBLE_DEVICES=0"):
            assert expected in env_vars, f"Expected env {expected!r}, got {env_vars}"
        for v in env_vars:
            if "VISIBLE_DEVICES" in v:
                assert '"' not in v and "'" not in v, f"GPU env should not contain quotes: {v}"

    def test_gpu_uses_nvidia_runtime_not_gpus_flag(self, temp_workspace):
        """
        GPU access uses --runtime=nvidia (CDI-compatible), not --gpus.

        The host is configured for CDI + nvidia runtime; mixing --gpus with
        that configuration conflicts, so --gpus must not be emitted.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        call_args = _run_docker(runner, *self._dirs(temp_workspace))

        assert "--runtime=nvidia" in call_args, \
            "Should set --runtime=nvidia when a GPU is assigned"
        assert "--gpus" not in call_args, \
            "Should NOT use --gpus (conflicts with CDI/nvidia runtime config)"

    def test_cuda_visible_devices_matches_gpu_id(self, temp_workspace):
        """
        CUDA_VISIBLE_DEVICES should match the GPU ID assigned to the runner.

        NVIDIA_VISIBLE_DEVICES is always `all` (needed for CDI visibility);
        CUDA_VISIBLE_DEVICES pins the logical device the tool uses.
        """
        for gpu_id in [0, 1, 2, 3]:
            runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=gpu_id)
            call_args = _run_docker(runner, *self._dirs(temp_workspace))

            env_vars = _env_vars(call_args)
            cuda_vars = [v for v in env_vars if v.startswith("CUDA_VISIBLE_DEVICES=")]
            assert cuda_vars == [f"CUDA_VISIBLE_DEVICES={gpu_id}"], \
                f"CUDA_VISIBLE_DEVICES should be exactly {gpu_id}, got {cuda_vars}"
            assert "NVIDIA_VISIBLE_DEVICES=all" in env_vars, \
                f"NVIDIA_VISIBLE_DEVICES should be 'all', got {env_vars}"

    def test_no_gpu_flags_when_gpu_id_none(self, temp_workspace):
        """No GPU-related flags should be set when gpu_id is None."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=None)
        call_args = _run_docker(runner, *self._dirs(temp_workspace))

        assert "--gpus" not in call_args, "Should not have --gpus flag"
        assert "--runtime=nvidia" not in call_args, "Should not have nvidia runtime without a GPU"

        env_vars = _env_vars(call_args)
        gpu_vars = [v for v in env_vars
                    if v.startswith(("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"))]
        assert gpu_vars == [], f"Should not set GPU env vars, got {gpu_vars}"
    
    def test_task_runner_passes_gpu_to_container_runner(self, temp_workspace):
        """TaskRunner should pass gpu_id to the container runner."""
        task_runner = TaskRunner(
            workspace_dir=temp_workspace,
            gpu_id=1
        )
        
        # Verify the container runner has the correct GPU ID
        assert task_runner._container_runner.gpu_id == 1, \
            "Container runner should have gpu_id=1"
    
    def test_task_runner_no_gpu_when_none(self, temp_workspace):
        """TaskRunner should work without GPU when gpu_id is None."""
        task_runner = TaskRunner(
            workspace_dir=temp_workspace,
            gpu_id=None
        )
        
        # Should still initialize
        assert task_runner._container_runner is not None, \
            "Container runner should be initialized even without GPU"
        assert task_runner._container_runner.gpu_id is None, \
            "Container runner should have gpu_id=None"


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    return tmp_path / "workspace"
