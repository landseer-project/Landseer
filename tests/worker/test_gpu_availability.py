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


class TestGPUAvailability:
    """Tests for GPU availability and detection."""
    
    def test_gpu_flag_format_correct(self, temp_workspace):
        """
        GPU flag should be in correct format for Docker.
        
        Docker expects: --gpus device=0
        Not: --gpus "device=0" (with quotes as part of string)
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_run.call_args[0][0]
            
            # Find --gpus flag
            gpus_idx = call_args.index("--gpus")
            gpu_spec = call_args[gpus_idx + 1]
            
            # Should NOT have quotes as part of the string
            assert not gpu_spec.startswith('"'), \
                f"GPU spec should not start with quote: {gpu_spec}"
            assert not gpu_spec.endswith('"'), \
                f"GPU spec should not end with quote: {gpu_spec}"
            
            # Should be in format: device=0
            assert gpu_spec == "device=0", \
                f"GPU spec should be 'device=0', got '{gpu_spec}'"
    
    def test_gpu_runtime_not_needed_with_gpus_flag(self, temp_workspace):
        """
        When using --gpus flag, --runtime=nvidia is NOT needed.
        
        Docker automatically uses nvidia runtime when --gpus is specified.
        Specifying both can cause conflicts.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_run.call_args[0][0]
            
            # Should have --gpus flag
            assert "--gpus" in call_args, \
                "Should have --gpus flag when GPU is available"
            
            # Should NOT have --runtime=nvidia (can conflict with --gpus)
            # Docker automatically uses nvidia runtime when --gpus is specified
            assert "--runtime=nvidia" not in call_args, \
                "Should NOT set --runtime=nvidia when using --gpus (can cause conflicts)"
    
    def test_cuda_visible_devices_matches_gpu_id(self, temp_workspace):
        """
        CUDA_VISIBLE_DEVICES should match the GPU ID passed to Docker.
        
        This ensures the container sees the correct GPU.
        """
        for gpu_id in [0, 1, 2, 3]:
            runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=gpu_id)
            
            input_dir = temp_workspace / "input"
            output_dir = temp_workspace / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with patch('subprocess.run') as mock_run:
                runner.run(
                    image="test/image:latest",
                    command="python main.py",
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
                call_args = mock_run.call_args[0][0]
                
                # Extract GPU ID from --gpus flag
                gpus_idx = call_args.index("--gpus")
                gpu_spec = call_args[gpus_idx + 1]
                gpu_id_from_flag = int(gpu_spec.split("=")[1])
                
                # Extract GPU ID from CUDA_VISIBLE_DEVICES
                env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
                cuda_var = next((v for v in env_vars if "CUDA_VISIBLE_DEVICES" in v), None)
                assert cuda_var is not None, "CUDA_VISIBLE_DEVICES should be set"
                gpu_id_from_env = int(cuda_var.split("=")[1])
                
                # Both should match the expected GPU ID
                assert gpu_id_from_flag == gpu_id, \
                    f"GPU ID from flag should be {gpu_id}, got {gpu_id_from_flag}"
                assert gpu_id_from_env == gpu_id, \
                    f"GPU ID from env should be {gpu_id}, got {gpu_id_from_env}"
    
    def test_no_gpu_flags_when_gpu_id_none(self, temp_workspace):
        """No GPU-related flags should be set when gpu_id is None."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=None)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_run.call_args[0][0]
            
            # Should not have GPU flags
            assert "--gpus" not in call_args, "Should not have --gpus flag"
            assert "--runtime=nvidia" not in call_args, "Should not have nvidia runtime (not needed without --gpus)"
            
            # Should not have CUDA_VISIBLE_DEVICES
            env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
            cuda_vars = [v for v in env_vars if "CUDA_VISIBLE_DEVICES" in v]
            assert len(cuda_vars) == 0, "Should not set CUDA_VISIBLE_DEVICES"
    
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
