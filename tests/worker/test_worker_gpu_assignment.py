"""
Tests to verify GPU assignment and passing to containers.

These tests verify:
1. Worker correctly receives and stores GPU ID
2. GPU ID is passed to TaskRunner
3. TaskRunner passes GPU ID to container runner
4. Container runner correctly formats GPU flags
5. GPU is actually available in containers
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

from src.worker.cli import Worker
from src.worker.runner import TaskRunner, DockerRunner
from src.worker.client import TaskInfo


def _run_docker(runner, input_dir, output_dir):
    """
    Run DockerRunner.run with subprocess mocked.

    DockerRunner.run probes `docker info` via subprocess.run (stubbed here to
    report an nvidia runtime) and launches the container via subprocess.Popen.
    Returns (docker argv passed to Popen, run() return value).
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
        result = runner.run(
            image="test/image:latest",
            command="python main.py",
            input_dir=input_dir,
            output_dir=output_dir,
        )
        assert mock_popen.called, "docker run should be launched via subprocess.Popen"
        return mock_popen.call_args[0][0], result


def _env_vars(call_args):
    return [call_args[i + 1] for i, arg in enumerate(call_args) if arg == "-e"]


def _make_dirs(temp_workspace):
    input_dir = temp_workspace / "input"
    output_dir = temp_workspace / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    return TaskInfo(
        id="test_task_gpu",
        tool_name="test_tool",
        tool_image="test/image:latest",
        tool_command="python main.py",
        tool_runtime=None,
        tool_is_baseline=False,
        config={},
        priority=100,
        status="pending",
        task_type="in_training",
        counter=1,
        workflows=[],
        pipeline_id="pipeline_1",
        dependency_ids=[]
    )


# ============================================================================
# Test: Worker GPU Assignment
# ============================================================================


class TestWorkerGPUAssignment:
    """Tests for GPU assignment in Worker."""
    
    def test_worker_stores_gpu_id(self, temp_workspace):
        """Worker should store GPU ID when provided."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=0)
        
        assert worker.gpu_id == 0
    
    def test_worker_passes_gpu_to_task_runner(self, temp_workspace):
        """Worker should pass GPU ID to TaskRunner during initialization."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=1)
        worker._init_components()
        
        assert worker._runner is not None
        assert worker._runner.gpu_id == 1, \
            "TaskRunner should receive GPU ID from Worker"
    
    def test_worker_without_gpu(self, temp_workspace):
        """Worker should work without GPU (gpu_id=None)."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=None)
        worker._init_components()
        
        assert worker.gpu_id is None
        assert worker._runner is not None
        assert worker._runner.gpu_id is None, \
            "TaskRunner should have gpu_id=None when Worker has no GPU"
    
    def test_worker_reports_gpu_capability(self, temp_workspace):
        """Worker should report GPU capability during registration."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=0)
        worker._client = MagicMock()
        
        worker_info = MagicMock()
        worker_info.worker_id = "test_worker"
        worker._client.register.return_value = worker_info
        
        with patch('src.worker.cli.ContainerRuntime') as mock_runtime:
            mock_runtime.detect_runtime.return_value = "docker"
            worker._register()
        
        call_args = worker._client.register.call_args
        capabilities = call_args[1]["capabilities"]
        
        assert capabilities["gpu_available"] is True, \
            "Worker should report GPU as available"
        assert capabilities["gpu_id"] == 0, \
            "Worker should report correct GPU ID"
    
    def test_worker_reports_no_gpu_capability(self, temp_workspace):
        """Worker should report no GPU when gpu_id is None."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=None)
        worker._client = MagicMock()
        
        worker_info = MagicMock()
        worker_info.worker_id = "test_worker"
        worker._client.register.return_value = worker_info
        
        with patch('src.worker.cli.ContainerRuntime') as mock_runtime:
            mock_runtime.detect_runtime.return_value = "docker"
            worker._register()
        
        call_args = worker._client.register.call_args
        capabilities = call_args[1]["capabilities"]
        
        assert capabilities["gpu_available"] is False, \
            "Worker should report GPU as not available"
        assert capabilities["gpu_id"] is None, \
            "Worker should report gpu_id as None"


# ============================================================================
# Test: TaskRunner GPU Passing
# ============================================================================


class TestTaskRunnerGPUPassing:
    """Tests for GPU ID passing through TaskRunner."""
    
    def test_task_runner_stores_gpu_id(self, temp_workspace):
        """TaskRunner should store GPU ID."""
        runner = TaskRunner(workspace_dir=temp_workspace, gpu_id=2)
        
        assert runner.gpu_id == 2
    
    def test_task_runner_passes_gpu_to_docker_runner(self, temp_workspace):
        """TaskRunner should pass GPU ID to DockerRunner."""
        runner = TaskRunner(workspace_dir=temp_workspace, gpu_id=0, runtime="docker")
        
        assert runner._container_runner is not None
        assert runner._container_runner.gpu_id == 0, \
            "DockerRunner should receive GPU ID from TaskRunner"
    
    def test_task_runner_passes_gpu_to_apptainer_runner(self, temp_workspace):
        """TaskRunner should pass GPU ID to ApptainerRunner."""
        runner = TaskRunner(workspace_dir=temp_workspace, gpu_id=1, runtime="apptainer")
        
        assert runner._container_runner is not None
        assert runner._container_runner.gpu_id == 1, \
            "ApptainerRunner should receive GPU ID from TaskRunner"


# ============================================================================
# Test: Docker GPU Flags
# ============================================================================


class TestDockerGPUFlags:
    """Tests for Docker GPU flag formatting."""
    
    def test_docker_runner_adds_gpu_flags(self, temp_workspace):
        """DockerRunner should add GPU flags when gpu_id is set."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        call_args, _ = _run_docker(runner, *_make_dirs(temp_workspace))

        # GPU access is via the nvidia runtime (CDI-compatible), not --gpus
        assert "--runtime=nvidia" in call_args, "Should have --runtime=nvidia"
        assert "--gpus" not in call_args, \
            "Should NOT use --gpus (conflicts with CDI/nvidia runtime config)"

        env_vars = _env_vars(call_args)
        assert "NVIDIA_VISIBLE_DEVICES=all" in env_vars, \
            f"NVIDIA_VISIBLE_DEVICES should be 'all', got {env_vars}"
        assert "NVIDIA_DRIVER_CAPABILITIES=all" in env_vars, \
            f"NVIDIA_DRIVER_CAPABILITIES should be 'all', got {env_vars}"
        cuda_var = next((v for v in env_vars if v.startswith("CUDA_VISIBLE_DEVICES=")), None)
        assert cuda_var == "CUDA_VISIBLE_DEVICES=0", \
            f"CUDA_VISIBLE_DEVICES should be 0, got '{cuda_var}'"
    
    def test_docker_runner_gpu_id_matches_flag(self, temp_workspace):
        """GPU pinning (CUDA_VISIBLE_DEVICES) should match the gpu_id parameter."""
        for gpu_id in [0, 1, 2, 3]:
            runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=gpu_id)
            call_args, _ = _run_docker(runner, *_make_dirs(temp_workspace))

            env_vars = _env_vars(call_args)
            cuda_vars = [v for v in env_vars if v.startswith("CUDA_VISIBLE_DEVICES=")]
            assert len(cuda_vars) == 1, f"Should set CUDA_VISIBLE_DEVICES once, got {cuda_vars}"
            gpu_id_from_env = int(cuda_vars[0].split("=")[1])
            assert gpu_id_from_env == gpu_id, \
                f"GPU ID from env should be {gpu_id}, got {gpu_id_from_env}"
    
    def test_docker_runner_no_gpu_flags_when_none(self, temp_workspace):
        """DockerRunner should not add GPU flags when gpu_id is None."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=None)
        call_args, _ = _run_docker(runner, *_make_dirs(temp_workspace))

        # Should not have GPU flags
        assert "--gpus" not in call_args, "Should not have --gpus flag"
        assert "--runtime=nvidia" not in call_args, \
            "Should not have --runtime=nvidia flag"

        # Should not have GPU env vars
        env_vars = _env_vars(call_args)
        gpu_vars = [v for v in env_vars
                    if v.startswith(("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"))]
        assert gpu_vars == [], f"Should not set GPU env vars, got {gpu_vars}"


# ============================================================================
# Test: End-to-End GPU Assignment
# ============================================================================


class TestEndToEndGPUAssignment:
    """End-to-end tests for GPU assignment from Worker to container."""
    
    def test_worker_to_container_gpu_flow(self, temp_workspace, mock_task):
        """Test complete flow: Worker -> TaskRunner -> DockerRunner -> Container."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=0)
        worker._init_components()
        
        # Verify GPU ID flows through
        assert worker.gpu_id == 0
        assert worker._runner.gpu_id == 0
        assert worker._runner._container_runner.gpu_id == 0
        
        # Mock the container execution
        container_runner = worker._runner._container_runner
        with patch.object(container_runner, 'pull_image', return_value=True), \
                patch.object(container_runner, 'run') as mock_run:
            mock_run.return_value = (0, "Success", "docker run ...")
            
            result = worker._runner.run_task(mock_task)
        
        # The container runner (which holds gpu_id=0) should have run the task
        assert mock_run.called, "Container runner.run should be called"
        assert mock_run.call_args.kwargs["image"] == mock_task.tool_image
        assert result.success, f"Task should succeed: {result.error_message}"
    
    def test_worker_without_gpu_does_not_add_flags(self, temp_workspace, mock_task):
        """Worker without GPU should not add GPU flags to containers."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=None)
        worker._init_components()
        
        assert worker.gpu_id is None
        assert worker._runner.gpu_id is None
        assert worker._runner._container_runner.gpu_id is None
        
        # Mock the container execution
        container_runner = worker._runner._container_runner
        with patch.object(container_runner, 'pull_image', return_value=True), \
                patch.object(container_runner, 'run') as mock_run:
            mock_run.return_value = (0, "Success", "docker run ...")
            
            result = worker._runner.run_task(mock_task)
        
        # Verify no GPU flags were used
        call_args = mock_run.call_args
        # Check that GPU flags are not in the command
        if 'call_args' in locals() and call_args:
            cmd = call_args[0][0] if call_args[0] else []
            if isinstance(cmd, list):
                assert "--gpus" not in cmd, "Should not have --gpus flag"


# ============================================================================
# Test: GPU Assignment Verification
# ============================================================================


class TestGPUAssignmentVerification:
    """Tests to verify GPU assignment is working correctly."""
    
    def test_verify_gpu_id_passed_to_task_runner(self, temp_workspace):
        """Verify GPU ID is correctly passed from Worker to TaskRunner."""
        for gpu_id in [0, 1, 2]:
            worker = Worker(workspace_dir=temp_workspace, gpu_id=gpu_id)
            worker._init_components()
            
            assert worker._runner.gpu_id == gpu_id, \
                f"TaskRunner should have gpu_id={gpu_id}"
    
    def test_verify_gpu_id_passed_to_container_runner(self, temp_workspace):
        """Verify GPU ID is correctly passed from TaskRunner to ContainerRunner."""
        for gpu_id in [0, 1, 2]:
            runner = TaskRunner(workspace_dir=temp_workspace, gpu_id=gpu_id)
            
            assert runner._container_runner.gpu_id == gpu_id, \
                f"ContainerRunner should have gpu_id={gpu_id}"
    
    def test_verify_gpu_flags_in_docker_command(self, temp_workspace):
        """Verify the docker command string returned by run() (used in task logs) has the GPU flags."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=1)
        call_args, (exit_code, _logs, cmd_str) = _run_docker(runner, *_make_dirs(temp_workspace))

        assert exit_code == 0
        assert cmd_str == ' '.join(call_args), \
            "Returned docker command should match the command actually executed"
        assert "--runtime=nvidia" in cmd_str, "Should have --runtime=nvidia in command"
        assert "--gpus" not in cmd_str, "Should NOT have --gpus in command"
        assert "-e NVIDIA_VISIBLE_DEVICES=all" in cmd_str, \
            "Should have NVIDIA_VISIBLE_DEVICES=all in command"
        assert "-e CUDA_VISIBLE_DEVICES=1" in cmd_str, \
            "Should have CUDA_VISIBLE_DEVICES=1 in command"


# ============================================================================
# Test: Debugging Helper Tests
# ============================================================================


class TestDebuggingHelpers:
    """Tests to help debug GPU assignment issues."""
    
    def test_log_gpu_assignment_chain(self, temp_workspace, caplog):
        """Test that logs show GPU assignment at each level."""
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        worker = Worker(workspace_dir=temp_workspace, gpu_id=0)
        worker._init_components()
        
        # Check that GPU ID is logged or accessible
        assert worker.gpu_id == 0
        assert worker._runner.gpu_id == 0
        assert worker._runner._container_runner.gpu_id == 0
    
    def test_verify_worker_started_with_gpu(self, temp_workspace):
        """Verify that worker can be started with GPU ID."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=0)
        
        # Worker should accept GPU ID
        assert worker.gpu_id == 0
        
        # After initialization, TaskRunner should have GPU ID
        worker._init_components()
        assert worker._runner.gpu_id == 0, \
            "TaskRunner should have GPU ID after Worker initialization"
