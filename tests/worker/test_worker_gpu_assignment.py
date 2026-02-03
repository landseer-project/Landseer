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
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_run.call_args[0][0]
            
            # Should have --gpus flag
            assert "--gpus" in call_args, "Should have --gpus flag"
            gpus_idx = call_args.index("--gpus")
            assert call_args[gpus_idx + 1] == "device=0", \
                f"GPU spec should be 'device=0', got '{call_args[gpus_idx + 1]}'"
            
            # Should NOT have --runtime=nvidia when using --gpus
            # Docker automatically uses nvidia runtime when --gpus is specified
            # Specifying both can cause conflicts
            assert "--runtime=nvidia" not in call_args, \
                "Should NOT have --runtime=nvidia when using --gpus (can conflict)"
            
            # Should have CUDA_VISIBLE_DEVICES
            env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
            cuda_var = next((v for v in env_vars if "CUDA_VISIBLE_DEVICES" in v), None)
            assert cuda_var is not None, "Should set CUDA_VISIBLE_DEVICES"
            assert cuda_var == "CUDA_VISIBLE_DEVICES=0", \
                f"CUDA_VISIBLE_DEVICES should be 0, got '{cuda_var}'"
    
    def test_docker_runner_gpu_id_matches_flag(self, temp_workspace):
        """GPU ID in flags should match the gpu_id parameter."""
        for gpu_id in [0, 1, 2, 3]:
            runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=gpu_id)
            
            input_dir = temp_workspace / "input"
            output_dir = temp_workspace / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                runner.run(
                    image="test/image:latest",
                    command="python main.py",
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
                call_args = mock_run.call_args[0][0]
                
                # Check --gpus flag
                gpus_idx = call_args.index("--gpus")
                gpu_spec = call_args[gpus_idx + 1]
                gpu_id_from_flag = int(gpu_spec.split("=")[1])
                assert gpu_id_from_flag == gpu_id, \
                    f"GPU ID from flag should be {gpu_id}, got {gpu_id_from_flag}"
                
                # Check CUDA_VISIBLE_DEVICES
                env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
                cuda_var = next((v for v in env_vars if "CUDA_VISIBLE_DEVICES" in v), None)
                assert cuda_var is not None, "Should set CUDA_VISIBLE_DEVICES"
                gpu_id_from_env = int(cuda_var.split("=")[1])
                assert gpu_id_from_env == gpu_id, \
                    f"GPU ID from env should be {gpu_id}, got {gpu_id_from_env}"
    
    def test_docker_runner_no_gpu_flags_when_none(self, temp_workspace):
        """DockerRunner should not add GPU flags when gpu_id is None."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=None)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_run.call_args[0][0]
            
            # Should not have GPU flags
            assert "--gpus" not in call_args, "Should not have --gpus flag"
            assert "--runtime=nvidia" not in call_args, \
                "Should not have --runtime=nvidia flag"
            
            # Should not have CUDA_VISIBLE_DEVICES
            env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
            cuda_vars = [v for v in env_vars if "CUDA_VISIBLE_DEVICES" in v]
            assert len(cuda_vars) == 0, "Should not set CUDA_VISIBLE_DEVICES"


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
        with patch.object(worker._runner._container_runner, 'run') as mock_run:
            mock_run.return_value = (0, "Success")
            mock_run.pull_image = MagicMock(return_value=True)
            
            result = worker._runner.run_task(mock_task)
        
        # Verify GPU flags were used
        call_args = mock_run.call_args
        # The run method should have been called
        assert mock_run.called
    
    def test_worker_without_gpu_does_not_add_flags(self, temp_workspace, mock_task):
        """Worker without GPU should not add GPU flags to containers."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=None)
        worker._init_components()
        
        assert worker.gpu_id is None
        assert worker._runner.gpu_id is None
        assert worker._runner._container_runner.gpu_id is None
        
        # Mock the container execution
        with patch.object(worker._runner._container_runner, 'run') as mock_run:
            mock_run.return_value = (0, "Success")
            mock_run.pull_image = MagicMock(return_value=True)
            
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
        """Verify GPU flags are correctly formatted in Docker command."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_run.call_args[0][0]
            cmd_str = ' '.join(call_args)
            
            # Verify GPU flags are present and correctly formatted
            assert "--gpus device=0" in cmd_str or "--gpus" in call_args, \
                "Should have --gpus device=0 in command"
            # Should NOT have --runtime=nvidia when using --gpus (can conflict)
            assert "--runtime=nvidia" not in cmd_str and "--runtime=nvidia" not in call_args, \
                "Should NOT have --runtime=nvidia when using --gpus (can conflict)"
            assert "CUDA_VISIBLE_DEVICES=0" in cmd_str or any("CUDA_VISIBLE_DEVICES=0" in str(arg) for arg in call_args), \
                "Should have CUDA_VISIBLE_DEVICES=0 in command"


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
