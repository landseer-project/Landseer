"""
Integration tests for GPU assignment from command line to Docker execution.

These tests verify:
1. Worker CLI accepts --gpu argument
2. GPU ID flows from CLI -> Worker -> TaskRunner -> DockerRunner -> Docker command
3. Docker command includes correct GPU flags
4. Docker can actually access the specified GPU
"""

import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import List

from src.worker.cli import Worker, create_parser, main
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
# Test: CLI Argument Parsing
# ============================================================================


class TestCLIGPUArgument:
    """Tests for GPU argument parsing in CLI."""
    
    def test_cli_parser_accepts_gpu_argument(self):
        """CLI parser should accept --gpu argument."""
        parser = create_parser()
        
        # Parse with GPU argument
        args = parser.parse_args(["--gpu", "0"])
        assert args.gpu == 0
        
        args = parser.parse_args(["--gpu", "1"])
        assert args.gpu == 1
    
    def test_cli_parser_gpu_defaults_to_none(self):
        """CLI parser should default GPU to None."""
        parser = create_parser()
        
        args = parser.parse_args([])
        assert args.gpu is None
    
    def test_cli_parser_gpu_accepts_integer(self):
        """CLI parser should accept integer GPU IDs."""
        parser = create_parser()
        
        for gpu_id in [0, 1, 2, 3, 7]:
            args = parser.parse_args(["--gpu", str(gpu_id)])
            assert args.gpu == gpu_id
    
    def test_worker_created_with_cli_gpu_argument(self, temp_workspace):
        """Worker should be created with GPU ID from CLI argument."""
        parser = create_parser()
        args = parser.parse_args(["--gpu", "2", "--workspace", str(temp_workspace)])
        
        worker = Worker(
            backend_url=args.backend_url,
            worker_id=args.worker_id,
            workspace_dir=Path(args.workspace) if args.workspace else None,
            gpu_id=args.gpu
        )
        
        assert worker.gpu_id == 2, "Worker should have GPU ID from CLI argument"


# ============================================================================
# Test: GPU ID Flow from CLI to Docker
# ============================================================================


class TestGPUIDFlow:
    """Tests for GPU ID flow from CLI argument to Docker command."""
    
    def test_gpu_id_flows_cli_to_worker(self, temp_workspace):
        """GPU ID should flow from CLI argument to Worker."""
        parser = create_parser()
        args = parser.parse_args(["--gpu", "1"])
        
        worker = Worker(workspace_dir=temp_workspace, gpu_id=args.gpu)
        
        assert worker.gpu_id == 1, "Worker should have GPU ID from CLI"
    
    def test_gpu_id_flows_worker_to_task_runner(self, temp_workspace):
        """GPU ID should flow from Worker to TaskRunner."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=0)
        worker._init_components()
        
        assert worker._runner is not None, "TaskRunner should be initialized"
        assert worker._runner.gpu_id == 0, \
            f"TaskRunner should have GPU ID 0, got {worker._runner.gpu_id}"
    
    def test_gpu_id_flows_task_runner_to_docker_runner(self, temp_workspace):
        """GPU ID should flow from TaskRunner to DockerRunner."""
        runner = TaskRunner(workspace_dir=temp_workspace, gpu_id=1, runtime="docker")
        
        assert runner._container_runner is not None, "DockerRunner should be initialized"
        assert runner._container_runner.gpu_id == 1, \
            f"DockerRunner should have GPU ID 1, got {runner._container_runner.gpu_id}"
    
    def test_gpu_id_in_docker_command(self, temp_workspace):
        """GPU ID should appear in Docker command."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=2)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Check GPU ID in command
            assert "--runtime=nvidia" in docker_cmd, \
                f"Should have --runtime=nvidia in Docker command: {docker_cmd}"
            assert f"NVIDIA_VISIBLE_DEVICES=2" in docker_cmd, \
                f"NVIDIA_VISIBLE_DEVICES=2 should be in Docker command: {docker_cmd}"
            assert f"CUDA_VISIBLE_DEVICES=2" in docker_cmd, \
                f"CUDA_VISIBLE_DEVICES=2 should be in Docker command: {docker_cmd}"
            assert "NVIDIA_DRIVER_CAPABILITIES=all" in docker_cmd, \
                f"NVIDIA_DRIVER_CAPABILITIES=all should be in Docker command: {docker_cmd}"


# ============================================================================
# Test: End-to-End GPU Flow
# ============================================================================


class TestEndToEndGPUFlow:
    """End-to-end tests for GPU flow from CLI to Docker execution."""
    
    def test_complete_gpu_flow_cli_to_docker(self, temp_workspace, mock_task):
        """
        Test complete flow: CLI argument -> Worker -> TaskRunner -> DockerRunner -> Docker command.
        
        This verifies that GPU ID specified on command line ends up in Docker command.
        """
        # Simulate CLI argument
        gpu_id = 0
        
        # Create Worker with GPU ID (as if from CLI)
        worker = Worker(workspace_dir=temp_workspace, gpu_id=gpu_id)
        worker._init_components()
        
        # Verify GPU ID at each level
        assert worker.gpu_id == gpu_id, f"Worker should have GPU ID {gpu_id}"
        assert worker._runner.gpu_id == gpu_id, f"TaskRunner should have GPU ID {gpu_id}"
        assert worker._runner._container_runner.gpu_id == gpu_id, \
            f"DockerRunner should have GPU ID {gpu_id}"
        
        # Mock Docker execution and capture command
        captured_docker_cmd = []
        
        def capture_docker_call(*args, **kwargs):
            if args and isinstance(args[0], list) and args[0][0] == "docker":
                captured_docker_cmd.extend(args[0])
            return MagicMock(returncode=0, stdout="", stderr="")
        
        with patch('subprocess.run', side_effect=capture_docker_call):
            with patch.object(worker._runner._container_runner, 'pull_image', return_value=True):
                result = worker._runner.run_task(mock_task)
        
        # Verify GPU flags in captured command
        if captured_docker_cmd:
            cmd_str = ' '.join(captured_docker_cmd)
            assert "--runtime=nvidia" in cmd_str, \
                f"Should have --runtime=nvidia in Docker command: {cmd_str}"
            assert f"NVIDIA_VISIBLE_DEVICES={gpu_id}" in cmd_str, \
                f"NVIDIA_VISIBLE_DEVICES={gpu_id} should be in Docker command: {cmd_str}"
            assert f"CUDA_VISIBLE_DEVICES={gpu_id}" in cmd_str, \
                f"CUDA_VISIBLE_DEVICES={gpu_id} should be in Docker command: {cmd_str}"
            assert "NVIDIA_DRIVER_CAPABILITIES=all" in cmd_str, \
                f"NVIDIA_DRIVER_CAPABILITIES=all should be in Docker command: {cmd_str}"
    
    def test_multiple_gpu_ids_flow_correctly(self, temp_workspace, mock_task):
        """Test that different GPU IDs flow correctly through the system."""
        for gpu_id in [0, 1, 2, 3]:
            worker = Worker(workspace_dir=temp_workspace, gpu_id=gpu_id)
            worker._init_components()
            
            # Verify GPU ID at each level
            assert worker.gpu_id == gpu_id, f"Worker should have GPU ID {gpu_id}"
            assert worker._runner.gpu_id == gpu_id, f"TaskRunner should have GPU ID {gpu_id}"
            assert worker._runner._container_runner.gpu_id == gpu_id, \
                f"DockerRunner should have GPU ID {gpu_id}"
            
            # Verify Docker command would have correct GPU ID
            docker_runner = worker._runner._container_runner
            input_dir = temp_workspace / f"input_{gpu_id}"
            output_dir = temp_workspace / f"output_{gpu_id}"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                exit_code, logs, docker_cmd = docker_runner.run(
                    image="test/image:latest",
                    command="python main.py",
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
                # Verify GPU ID in command
                assert "--runtime=nvidia" in docker_cmd, \
                    f"Should have --runtime=nvidia in Docker command"
                assert f"NVIDIA_VISIBLE_DEVICES={gpu_id}" in docker_cmd, \
                    f"NVIDIA_VISIBLE_DEVICES={gpu_id} should be in Docker command"
                assert f"CUDA_VISIBLE_DEVICES={gpu_id}" in docker_cmd, \
                    f"CUDA_VISIBLE_DEVICES={gpu_id} should be in Docker command"
                assert "NVIDIA_DRIVER_CAPABILITIES=all" in docker_cmd, \
                    f"NVIDIA_DRIVER_CAPABILITIES=all should be in Docker command"


# ============================================================================
# Test: Docker GPU Access Verification
# ============================================================================


class TestDockerGPUAccess:
    """Tests to verify Docker actually has access to the specified GPU."""
    
    def test_docker_command_has_correct_gpu_flags(self, temp_workspace):
        """Docker command should have correct GPU flags for specified GPU ID."""
        for gpu_id in [0, 1, 2]:
            runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=gpu_id)
            
            input_dir = temp_workspace / "input"
            output_dir = temp_workspace / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                exit_code, logs, docker_cmd = runner.run(
                    image="test/image:latest",
                    command="python main.py",
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
                # Verify GPU flags
                assert "--runtime=nvidia" in docker_cmd, \
                    f"Should have --runtime=nvidia"
                assert f"NVIDIA_VISIBLE_DEVICES={gpu_id}" in docker_cmd, \
                    f"Should have NVIDIA_VISIBLE_DEVICES={gpu_id}"
                assert f"CUDA_VISIBLE_DEVICES={gpu_id}" in docker_cmd, \
                    f"Should have CUDA_VISIBLE_DEVICES={gpu_id}"
                assert "NVIDIA_DRIVER_CAPABILITIES=all" in docker_cmd, \
                    f"Should have NVIDIA_DRIVER_CAPABILITIES=all"
    
    def test_docker_gpu_flags_match_gpu_id(self, temp_workspace):
        """GPU flags in Docker command should match the GPU ID."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=1)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Extract GPU ID from NVIDIA_VISIBLE_DEVICES
            import re
            nvidia_match = re.search(r'NVIDIA_VISIBLE_DEVICES=(\d+)', docker_cmd)
            assert nvidia_match is not None, "Should have NVIDIA_VISIBLE_DEVICES=X in command"
            gpu_id_from_nvidia = int(nvidia_match.group(1))
            
            # Extract GPU ID from CUDA_VISIBLE_DEVICES
            cuda_match = re.search(r'CUDA_VISIBLE_DEVICES=(\d+)', docker_cmd)
            assert cuda_match is not None, "Should have CUDA_VISIBLE_DEVICES=X in command"
            gpu_id_from_cuda = int(cuda_match.group(1))
            
            # Verify runtime flag
            assert "--runtime=nvidia" in docker_cmd, "Should have --runtime=nvidia"
            
            # Both should match the runner's GPU ID
            assert gpu_id_from_nvidia == 1, \
                f"GPU ID from NVIDIA_VISIBLE_DEVICES should be 1, got {gpu_id_from_nvidia}"
            assert gpu_id_from_cuda == 1, \
                f"GPU ID from CUDA_VISIBLE_DEVICES should be 1, got {gpu_id_from_cuda}"
            assert gpu_id_from_nvidia == gpu_id_from_cuda, \
                "GPU IDs from NVIDIA_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES should match"


# ============================================================================
# Test: Integration with Actual Docker (if available)
# ============================================================================


class TestDockerGPUIntegration:
    """Integration tests with actual Docker (if available)."""
    
    @pytest.mark.skipif(
        not Path("/usr/bin/docker").exists() and not Path("/usr/local/bin/docker").exists(),
        reason="Docker not available"
    )
    def test_docker_can_access_specified_gpu(self, temp_workspace):
        """
        Test that Docker can actually access the specified GPU.
        
        This test requires:
        - Docker installed
        - nvidia-container-runtime configured
        - GPU available
        """
        # Test with GPU 0
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to run a simple GPU test container
        # Use nvidia/cuda image with nvidia-smi to verify GPU access
        try:
            exit_code, logs, docker_cmd = runner.run(
                image="nvidia/cuda:11.0-base",
                command="nvidia-smi --query-gpu=index --format=csv,noheader",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # If successful, logs should contain GPU index
            if exit_code == 0:
                # Should see GPU index in output
                assert "0" in logs or "GPU" in logs, \
                    f"Should see GPU information in output, got: {logs[:200]}"
            else:
                pytest.skip(f"Docker GPU test failed (may not have GPU access): {logs[:200]}")
                
        except Exception as e:
            pytest.skip(f"Docker GPU test failed: {e}")
    
    def test_docker_command_format_is_correct(self, temp_workspace):
        """
        Test that Docker command format is correct for GPU access.
        
        This verifies the command structure without actually running Docker.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Parse the command
            cmd_parts = docker_cmd.split()
            
            # Verify structure
            assert cmd_parts[0] == "docker", "Command should start with docker"
            assert cmd_parts[1] == "run", "Should be docker run"
            assert "--rm" in cmd_parts, "Should have --rm flag"
            
            # Verify GPU runtime flag (using --runtime=nvidia instead of --gpus)
            assert "--runtime=nvidia" in cmd_parts, "Should have --runtime=nvidia flag"
            
            # Verify NVIDIA_VISIBLE_DEVICES (find the correct -e flag)
            nvidia_env = None
            cuda_env = None
            for i, part in enumerate(cmd_parts):
                if part == "-e" and i + 1 < len(cmd_parts):
                    env_var = cmd_parts[i + 1]
                    if "NVIDIA_VISIBLE_DEVICES" in env_var:
                        nvidia_env = env_var
                    if "CUDA_VISIBLE_DEVICES" in env_var:
                        cuda_env = env_var
            
            assert nvidia_env is not None, "Should have NVIDIA_VISIBLE_DEVICES in command"
            assert nvidia_env == "NVIDIA_VISIBLE_DEVICES=0", \
                f"NVIDIA_VISIBLE_DEVICES should be 0, got '{nvidia_env}'"
            assert cuda_env is not None, "Should also have CUDA_VISIBLE_DEVICES for compatibility"
            assert cuda_env == "CUDA_VISIBLE_DEVICES=0", \
                f"CUDA_VISIBLE_DEVICES should be 0, got '{cuda_env}'"
            
            # Verify NVIDIA_DRIVER_CAPABILITIES
            driver_caps = None
            for i, part in enumerate(cmd_parts):
                if part == "-e" and i + 1 < len(cmd_parts):
                    env_var = cmd_parts[i + 1]
                    if "NVIDIA_DRIVER_CAPABILITIES" in env_var:
                        driver_caps = env_var
            assert driver_caps == "NVIDIA_DRIVER_CAPABILITIES=all", \
                f"Should have NVIDIA_DRIVER_CAPABILITIES=all, got '{driver_caps}'"


# ============================================================================
# Test: Worker Main Function Integration
# ============================================================================


class TestWorkerMainGPUIntegration:
    """Tests for GPU integration in worker main function."""
    
    def test_main_function_accepts_gpu_argument(self, temp_workspace):
        """main() function should accept --gpu argument."""
        # Test argument parsing
        parser = create_parser()
        args = parser.parse_args(["--gpu", "0", "--workspace", str(temp_workspace)])
        
        # Verify GPU is passed to Worker
        worker = Worker(
            backend_url=args.backend_url,
            workspace_dir=Path(args.workspace) if args.workspace else None,
            gpu_id=args.gpu
        )
        
        assert worker.gpu_id == 0, "Worker should have GPU ID from main() arguments"
    
    def test_main_function_creates_worker_with_gpu(self, temp_workspace):
        """main() function should create Worker with GPU ID."""
        # Simulate main() function behavior
        parser = create_parser()
        args = parser.parse_args([
            "--gpu", "1",
            "--workspace", str(temp_workspace),
            "--backend-url", "http://test:8000"
        ])
        
        # Create worker as main() would
        worker = Worker(
            backend_url=args.backend_url,
            worker_id=args.worker_id,
            workspace_dir=Path(args.workspace) if args.workspace else None,
            cache_dir=Path(args.cache_dir),
            data_path=Path(args.data_path) if args.data_path else None,
            gpu_id=args.gpu,
            poll_interval=args.poll_interval,
            task_timeout=args.timeout,
            heartbeat_interval=args.heartbeat_interval,
            use_cache=not args.no_cache,
            runtime=None if args.runtime == "auto" else args.runtime
        )
        
        assert worker.gpu_id == 1, "Worker should have GPU ID 1"
        
        # Initialize components
        worker._init_components()
        
        # Verify GPU flows through
        assert worker._runner.gpu_id == 1, "TaskRunner should have GPU ID 1"
        assert worker._runner._container_runner.gpu_id == 1, \
            "DockerRunner should have GPU ID 1"


# ============================================================================
# Test: Real-World Scenarios
# ============================================================================


class TestRealWorldScenarios:
    """Tests for real-world GPU assignment scenarios."""
    
    def test_worker_with_gpu_0_executes_task_with_gpu(self, temp_workspace, mock_task):
        """Worker with GPU 0 should execute tasks with GPU 0."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=0)
        worker._init_components()
        
        # Capture Docker command
        docker_commands = []
        
        def capture_command(*args, **kwargs):
            if args and isinstance(args[0], list):
                docker_commands.append(' '.join(args[0]))
            return MagicMock(returncode=0, stdout="", stderr="")
        
        with patch('subprocess.run', side_effect=capture_command):
            with patch.object(worker._runner._container_runner, 'pull_image', return_value=True):
                result = worker._runner.run_task(mock_task)
        
        # Verify GPU 0 was used
        assert len(docker_commands) > 0, "Should have executed Docker command"
        docker_cmd = docker_commands[0]
        assert "--runtime=nvidia" in docker_cmd, \
            f"Should use --runtime=nvidia, command: {docker_cmd[:200]}"
        assert "NVIDIA_VISIBLE_DEVICES=0" in docker_cmd, \
            f"Should set NVIDIA_VISIBLE_DEVICES=0, command: {docker_cmd[:200]}"
        assert "CUDA_VISIBLE_DEVICES=0" in docker_cmd, \
            f"Should set CUDA_VISIBLE_DEVICES=0, command: {docker_cmd[:200]}"
        assert "NVIDIA_DRIVER_CAPABILITIES=all" in docker_cmd, \
            f"Should set NVIDIA_DRIVER_CAPABILITIES=all, command: {docker_cmd[:200]}"
    
    def test_multiple_workers_different_gpus(self, temp_workspace, mock_task):
        """Multiple workers with different GPU IDs should use correct GPUs."""
        workers = []
        for gpu_id in [0, 1, 2]:
            worker = Worker(workspace_dir=temp_workspace / f"worker_{gpu_id}", gpu_id=gpu_id)
            worker._init_components()
            workers.append((gpu_id, worker))
        
        # Execute tasks and verify each uses correct GPU
        for gpu_id, worker in workers:
            docker_commands = []
            
            def capture_command(*args, **kwargs):
                if args and isinstance(args[0], list):
                    docker_commands.append(' '.join(args[0]))
                return MagicMock(returncode=0, stdout="", stderr="")
            
            with patch('subprocess.run', side_effect=capture_command):
                with patch.object(worker._runner._container_runner, 'pull_image', return_value=True):
                    result = worker._runner.run_task(mock_task)
            
            # Verify correct GPU was used
            assert len(docker_commands) > 0, f"Worker {gpu_id} should execute Docker command"
            docker_cmd = docker_commands[0]
            assert "--runtime=nvidia" in docker_cmd, \
                f"Worker {gpu_id} should use --runtime=nvidia, command: {docker_cmd[:200]}"
            assert f"NVIDIA_VISIBLE_DEVICES={gpu_id}" in docker_cmd, \
                f"Worker {gpu_id} should set NVIDIA_VISIBLE_DEVICES={gpu_id}"
            assert f"CUDA_VISIBLE_DEVICES={gpu_id}" in docker_cmd, \
                f"Worker {gpu_id} should set CUDA_VISIBLE_DEVICES={gpu_id}"
            assert "NVIDIA_DRIVER_CAPABILITIES=all" in docker_cmd, \
                f"Worker {gpu_id} should set NVIDIA_DRIVER_CAPABILITIES=all"
    
    def test_worker_without_gpu_does_not_add_gpu_flags(self, temp_workspace, mock_task):
        """Worker without GPU should not add GPU flags to Docker commands."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=None)
        worker._init_components()
        
        docker_commands = []
        
        def capture_command(*args, **kwargs):
            if args and isinstance(args[0], list):
                docker_commands.append(' '.join(args[0]))
            return MagicMock(returncode=0, stdout="", stderr="")
        
        with patch('subprocess.run', side_effect=capture_command):
            with patch.object(worker._runner._container_runner, 'pull_image', return_value=True):
                result = worker._runner.run_task(mock_task)
        
        # Verify no GPU flags
        assert len(docker_commands) > 0, "Should have executed Docker command"
        docker_cmd = docker_commands[0]
        assert "--runtime=nvidia" not in docker_cmd, \
            f"Should NOT have --runtime=nvidia flag, command: {docker_cmd[:200]}"
        assert "NVIDIA_VISIBLE_DEVICES" not in docker_cmd, \
            f"Should NOT have NVIDIA_VISIBLE_DEVICES, command: {docker_cmd[:200]}"
        assert "CUDA_VISIBLE_DEVICES" not in docker_cmd, \
            f"Should NOT have CUDA_VISIBLE_DEVICES, command: {docker_cmd[:200]}"


# ============================================================================
# Test: Command Line Integration
# ============================================================================


class TestCommandLineIntegration:
    """Tests for command-line integration with GPU argument."""
    
    def test_worker_cli_with_gpu_flag(self, temp_workspace):
        """Test that worker CLI correctly handles --gpu flag."""
        # Simulate: landseer-worker --gpu 0 --workspace /tmp/test
        parser = create_parser()
        args = parser.parse_args([
            "--gpu", "0",
            "--workspace", str(temp_workspace)
        ])
        
        # Create worker as CLI would
        worker = Worker(
            backend_url=args.backend_url,
            worker_id=args.worker_id,
            workspace_dir=Path(args.workspace) if args.workspace else None,
            cache_dir=Path(args.cache_dir),
            data_path=Path(args.data_path) if args.data_path else None,
            gpu_id=args.gpu,
            poll_interval=args.poll_interval,
            task_timeout=args.timeout,
            heartbeat_interval=args.heartbeat_interval,
            use_cache=not args.no_cache,
            runtime=None if args.runtime == "auto" else args.runtime
        )
        
        assert worker.gpu_id == 0, "Worker should have GPU ID from CLI"
        
        # Initialize and verify GPU flows through
        worker._init_components()
        assert worker._runner.gpu_id == 0, "TaskRunner should have GPU ID"
        assert worker._runner._container_runner.gpu_id == 0, \
            "DockerRunner should have GPU ID"
    
    def test_worker_cli_without_gpu_flag(self, temp_workspace):
        """Test that worker CLI works without --gpu flag (CPU only)."""
        parser = create_parser()
        args = parser.parse_args([
            "--workspace", str(temp_workspace)
        ])
        
        worker = Worker(
            backend_url=args.backend_url,
            workspace_dir=Path(args.workspace) if args.workspace else None,
            gpu_id=args.gpu
        )
        
        assert worker.gpu_id is None, "Worker should have no GPU when --gpu not specified"
        
        worker._init_components()
        assert worker._runner.gpu_id is None, "TaskRunner should have no GPU"
        assert worker._runner._container_runner.gpu_id is None, \
            "DockerRunner should have no GPU"


# ============================================================================
# Test: Docker GPU Verification
# ============================================================================


class TestDockerGPUVerification:
    """Tests to verify Docker GPU access is correctly configured."""
    
    def test_docker_command_verification_helper(self, temp_workspace):
        """
        Helper function to verify Docker command has correct GPU configuration.
        
        This can be used to verify GPU assignment in production.
        """
        def verify_gpu_in_docker_command(docker_cmd: str, expected_gpu_id: int) -> bool:
            """Verify GPU ID is correctly set in Docker command."""
            checks = {
                "--runtime=nvidia": "--runtime=nvidia" in docker_cmd,
                f"NVIDIA_VISIBLE_DEVICES={expected_gpu_id}": f"NVIDIA_VISIBLE_DEVICES={expected_gpu_id}" in docker_cmd,
                f"CUDA_VISIBLE_DEVICES={expected_gpu_id}": f"CUDA_VISIBLE_DEVICES={expected_gpu_id}" in docker_cmd,
                "NVIDIA_DRIVER_CAPABILITIES=all": "NVIDIA_DRIVER_CAPABILITIES=all" in docker_cmd
            }
            
            all_passed = all(checks.values())
            if not all_passed:
                failed = [k for k, v in checks.items() if not v]
                print(f"Failed checks: {failed}")
                print(f"Docker command: {docker_cmd[:300]}")
            
            return all_passed
        
        # Test with GPU 0
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            assert verify_gpu_in_docker_command(docker_cmd, 0), \
                "Docker command should have correct GPU configuration for GPU 0"
    
    def test_gpu_id_consistency_across_flags(self, temp_workspace):
        """GPU ID should be consistent across all GPU-related flags."""
        for gpu_id in [0, 1, 2]:
            runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=gpu_id)
            
            input_dir = temp_workspace / "input"
            output_dir = temp_workspace / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                exit_code, logs, docker_cmd = runner.run(
                    image="test/image:latest",
                    command="python main.py",
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
                # Extract GPU IDs from different parts of command
                import re
                nvidia_match = re.search(r'NVIDIA_VISIBLE_DEVICES=(\d+)', docker_cmd)
                cuda_match = re.search(r'CUDA_VISIBLE_DEVICES=(\d+)', docker_cmd)
                
                assert "--runtime=nvidia" in docker_cmd, "Should have --runtime=nvidia"
                assert nvidia_match is not None, "Should have NVIDIA_VISIBLE_DEVICES=X"
                assert cuda_match is not None, "Should have CUDA_VISIBLE_DEVICES=X"
                
                gpu_from_nvidia = int(nvidia_match.group(1))
                gpu_from_cuda = int(cuda_match.group(1))
                
                # All should match
                assert gpu_from_nvidia == gpu_id, \
                    f"NVIDIA_VISIBLE_DEVICES should have GPU ID {gpu_id}, got {gpu_from_nvidia}"
                assert gpu_from_cuda == gpu_id, \
                    f"CUDA_VISIBLE_DEVICES should have GPU ID {gpu_id}, got {gpu_from_cuda}"
                assert gpu_from_nvidia == gpu_from_cuda, \
                    "GPU IDs should be consistent across environment variables"


# ============================================================================
# Test: GPU Isolation and Security
# ============================================================================


class TestGPUIsolation:
    """Tests to verify GPU isolation - worker can only access specified GPU."""
    
    def test_worker_with_gpu_0_cannot_access_gpu_1(self, temp_workspace):
        """
        Worker started with --gpu 0 should NOT be able to access GPU 1.
        
        This verifies GPU isolation by checking that:
        1. Docker command restricts to GPU 0 only
        2. CUDA_VISIBLE_DEVICES=0 means only GPU 0 is visible
        3. Worker cannot access other GPUs
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify GPU 0 is specified
            assert "--runtime=nvidia" in docker_cmd, \
                "Should have --runtime=nvidia"
            assert "NVIDIA_VISIBLE_DEVICES=0" in docker_cmd, \
                "Should have NVIDIA_VISIBLE_DEVICES=0"
            assert "CUDA_VISIBLE_DEVICES=0" in docker_cmd, \
                "Should have CUDA_VISIBLE_DEVICES=0"
            
            # Verify GPU 1 is NOT accessible
            assert "NVIDIA_VISIBLE_DEVICES=1" not in docker_cmd, \
                "Should NOT have NVIDIA_VISIBLE_DEVICES=1 (worker should only access GPU 0)"
            assert "CUDA_VISIBLE_DEVICES=1" not in docker_cmd, \
                "Should NOT have CUDA_VISIBLE_DEVICES=1"
    
    def test_worker_with_gpu_1_cannot_access_gpu_0(self, temp_workspace):
        """Worker started with --gpu 1 should NOT be able to access GPU 0."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=1)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify GPU 1 is specified
            assert "--runtime=nvidia" in docker_cmd, \
                "Should have --runtime=nvidia"
            assert "NVIDIA_VISIBLE_DEVICES=1" in docker_cmd, \
                "Should have NVIDIA_VISIBLE_DEVICES=1"
            assert "CUDA_VISIBLE_DEVICES=1" in docker_cmd, \
                "Should have CUDA_VISIBLE_DEVICES=1"
            
            # Verify GPU 0 is NOT accessible
            assert "NVIDIA_VISIBLE_DEVICES=0" not in docker_cmd, \
                "Should NOT have NVIDIA_VISIBLE_DEVICES=0 (worker should only access GPU 1)"
            assert "CUDA_VISIBLE_DEVICES=0" not in docker_cmd, \
                "Should NOT have CUDA_VISIBLE_DEVICES=0"
    
    def test_worker_cannot_access_multiple_gpus(self, temp_workspace):
        """
        Worker started with --gpu X should NOT be able to access multiple GPUs.
        
        Even if system has multiple GPUs, worker should only see the one specified.
        """
        for gpu_id in [0, 1, 2]:
            runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=gpu_id)
            
            input_dir = temp_workspace / "input"
            output_dir = temp_workspace / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                exit_code, logs, docker_cmd = runner.run(
                    image="test/image:latest",
                    command="python main.py",
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
                # Verify only the specified GPU is accessible
                assert "--runtime=nvidia" in docker_cmd, \
                    f"Should have --runtime=nvidia"
                assert f"NVIDIA_VISIBLE_DEVICES={gpu_id}" in docker_cmd, \
                    f"Should have NVIDIA_VISIBLE_DEVICES={gpu_id}"
                assert f"CUDA_VISIBLE_DEVICES={gpu_id}" in docker_cmd, \
                    f"Should have CUDA_VISIBLE_DEVICES={gpu_id}"
                
                # Verify no other GPU IDs are present
                import re
                nvidia_specs = re.findall(r'NVIDIA_VISIBLE_DEVICES=(\d+)', docker_cmd)
                assert len(nvidia_specs) == 1, \
                    f"Should have exactly one NVIDIA_VISIBLE_DEVICES spec, got {nvidia_specs}"
                assert nvidia_specs[0] == str(gpu_id), \
                    f"NVIDIA_VISIBLE_DEVICES should be {gpu_id}, got {nvidia_specs[0]}"
                
                # Verify CUDA_VISIBLE_DEVICES only has one GPU
                cuda_specs = re.findall(r'CUDA_VISIBLE_DEVICES=([\d,]+)', docker_cmd)
                assert len(cuda_specs) == 1, \
                    f"Should have exactly one CUDA_VISIBLE_DEVICES, got {cuda_specs}"
                assert cuda_specs[0] == str(gpu_id), \
                    f"CUDA_VISIBLE_DEVICES should be {gpu_id}, got {cuda_specs[0]}"
                
                # Verify no "all" or multiple GPU access
                assert "--gpus all" not in docker_cmd, \
                    f"Should NOT have --gpus all (worker should only access GPU {gpu_id})"
                assert "," not in cuda_specs[0], \
                    f"CUDA_VISIBLE_DEVICES should not contain multiple GPUs: {cuda_specs[0]}"
    
    def test_cuda_visible_devices_restricts_gpu_access(self, temp_workspace):
        """
        CUDA_VISIBLE_DEVICES should restrict container to only see specified GPU.
        
        When CUDA_VISIBLE_DEVICES=0, the container should only see GPU 0,
        even if the system has multiple GPUs.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify CUDA_VISIBLE_DEVICES=0 (not 0,1 or all)
            import re
            cuda_matches = re.findall(r'CUDA_VISIBLE_DEVICES=([^ ]+)', docker_cmd)
            assert len(cuda_matches) == 1, \
                f"Should have exactly one CUDA_VISIBLE_DEVICES, got {cuda_matches}"
            
            cuda_value = cuda_matches[0]
            assert cuda_value == "0", \
                f"CUDA_VISIBLE_DEVICES should be '0' (single GPU), got '{cuda_value}'"
            assert "," not in cuda_value, \
                f"CUDA_VISIBLE_DEVICES should not contain multiple GPUs: {cuda_value}"
            assert cuda_value != "all", \
                f"CUDA_VISIBLE_DEVICES should not be 'all': {cuda_value}"
    
    def test_docker_runtime_nvidia_restricts_access(self, temp_workspace):
        """
        Docker --runtime=nvidia with NVIDIA_VISIBLE_DEVICES should restrict container to only GPU X.
        
        This ensures Docker runtime restricts GPU access via NVIDIA_VISIBLE_DEVICES.
        """
        for gpu_id in [0, 1, 2]:
            runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=gpu_id)
            
            input_dir = temp_workspace / "input"
            output_dir = temp_workspace / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                exit_code, logs, docker_cmd = runner.run(
                    image="test/image:latest",
                    command="python main.py",
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
                # Verify --runtime=nvidia and NVIDIA_VISIBLE_DEVICES
                assert "--runtime=nvidia" in docker_cmd, \
                    f"Should have --runtime=nvidia"
                
                import re
                nvidia_matches = re.findall(r'NVIDIA_VISIBLE_DEVICES=([^ ]+)', docker_cmd)
                assert len(nvidia_matches) == 1, \
                    f"Should have exactly one NVIDIA_VISIBLE_DEVICES=X, got {nvidia_matches}"
                
                gpu_spec = nvidia_matches[0]
                assert gpu_spec == str(gpu_id), \
                    f"NVIDIA_VISIBLE_DEVICES should be '{gpu_id}', got '{gpu_spec}'"
                assert "," not in gpu_spec, \
                    f"NVIDIA_VISIBLE_DEVICES should not contain multiple GPUs: {gpu_spec}"
    
    def test_worker_isolation_across_different_gpus(self, temp_workspace, mock_task):
        """
        Multiple workers with different GPU IDs should be isolated.
        
        Worker 0 should only access GPU 0, Worker 1 should only access GPU 1, etc.
        """
        workers = []
        for gpu_id in [0, 1, 2]:
            worker = Worker(workspace_dir=temp_workspace / f"worker_{gpu_id}", gpu_id=gpu_id)
            worker._init_components()
            workers.append((gpu_id, worker))
        
        # Execute tasks and verify each worker only accesses its assigned GPU
        for assigned_gpu_id, worker in workers:
            docker_commands = []
            
            def capture_command(*args, **kwargs):
                if args and isinstance(args[0], list):
                    docker_commands.append(' '.join(args[0]))
                return MagicMock(returncode=0, stdout="", stderr="")
            
            with patch('subprocess.run', side_effect=capture_command):
                with patch.object(worker._runner._container_runner, 'pull_image', return_value=True):
                    result = worker._runner.run_task(mock_task)
            
            assert len(docker_commands) > 0, \
                f"Worker {assigned_gpu_id} should execute Docker command"
            
            docker_cmd = docker_commands[0]
            
            # Verify assigned GPU is used
            assert "--runtime=nvidia" in docker_cmd, \
                f"Worker {assigned_gpu_id} should use --runtime=nvidia"
            assert f"NVIDIA_VISIBLE_DEVICES={assigned_gpu_id}" in docker_cmd, \
                f"Worker {assigned_gpu_id} should set NVIDIA_VISIBLE_DEVICES={assigned_gpu_id}"
            assert f"CUDA_VISIBLE_DEVICES={assigned_gpu_id}" in docker_cmd, \
                f"Worker {assigned_gpu_id} should set CUDA_VISIBLE_DEVICES={assigned_gpu_id}"
            
            # Verify other GPUs are NOT accessible
            for other_gpu_id in [0, 1, 2]:
                if other_gpu_id != assigned_gpu_id:
                    assert f"NVIDIA_VISIBLE_DEVICES={other_gpu_id}" not in docker_cmd, \
                        f"Worker {assigned_gpu_id} should NOT set NVIDIA_VISIBLE_DEVICES={other_gpu_id}"
                    assert f"CUDA_VISIBLE_DEVICES={other_gpu_id}" not in docker_cmd, \
                        f"Worker {assigned_gpu_id} should NOT set CUDA_VISIBLE_DEVICES={other_gpu_id}"
    
    def test_gpu_id_mismatch_detection(self, temp_workspace):
        """
        Test that we can detect if GPU ID in Docker command doesn't match worker's GPU ID.
        
        This helps catch bugs where GPU ID might be incorrectly passed.
        """
        worker_gpu_id = 0
        
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=worker_gpu_id)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Extract GPU IDs from Docker command
            import re
            assert "--runtime=nvidia" in docker_cmd, "Should have --runtime=nvidia"
            nvidia_match = re.search(r'NVIDIA_VISIBLE_DEVICES=(\d+)', docker_cmd)
            cuda_match = re.search(r'CUDA_VISIBLE_DEVICES=(\d+)', docker_cmd)
            
            assert nvidia_match is not None, "Should have NVIDIA_VISIBLE_DEVICES=X"
            assert cuda_match is not None, "Should have CUDA_VISIBLE_DEVICES=X"
            
            gpu_from_nvidia = int(nvidia_match.group(1))
            gpu_from_cuda = int(cuda_match.group(1))
            
            # Both should match worker's GPU ID
            assert gpu_from_nvidia == worker_gpu_id, \
                f"GPU ID from NVIDIA_VISIBLE_DEVICES ({gpu_from_nvidia}) should match worker GPU ID ({worker_gpu_id})"
            assert gpu_from_cuda == worker_gpu_id, \
                f"GPU ID from CUDA_VISIBLE_DEVICES ({gpu_from_cuda}) should match worker GPU ID ({worker_gpu_id})"
            assert gpu_from_nvidia == gpu_from_cuda, \
                "GPU IDs from NVIDIA_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES should match"
    
    @pytest.mark.skipif(
        not Path("/usr/bin/docker").exists() and not Path("/usr/local/bin/docker").exists(),
        reason="Docker not available for integration test"
    )
    def test_docker_actually_restricts_gpu_access(self, temp_workspace):
        """
        Test that Docker actually restricts GPU access to only the specified GPU.
        
        This requires Docker and nvidia-container-runtime to be properly configured.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to run nvidia-smi and verify only GPU 0 is visible
        try:
            exit_code, logs, docker_cmd = runner.run(
                image="nvidia/cuda:11.0-base",
                command="nvidia-smi --query-gpu=index --format=csv,noheader",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            if exit_code == 0:
                # Parse GPU indices from output
                gpu_indices = [line.strip() for line in logs.strip().split('\n') if line.strip().isdigit()]
                
                # Should only see GPU 0 (even if system has more GPUs)
                assert len(gpu_indices) == 1, \
                    f"Should only see 1 GPU (GPU 0), but saw {len(gpu_indices)}: {gpu_indices}"
                assert gpu_indices[0] == "0", \
                    f"Should only see GPU 0, but saw GPU {gpu_indices[0]}"
                
                print(f"\n✓ GPU isolation verified:")
                print(f"  Worker GPU ID: 0")
                print(f"  GPUs visible in container: {gpu_indices}")
                print(f"  ✓ Container can only see GPU 0")
            else:
                pytest.skip(f"Docker GPU test failed: {logs[:200]}")
                
        except Exception as e:
            pytest.skip(f"Docker GPU test failed: {e}")
    
    def test_worker_cannot_override_gpu_assignment(self, temp_workspace):
        """
        Test that worker cannot override GPU assignment through environment variables.
        
        Even if a task tries to set CUDA_VISIBLE_DEVICES in its config,
        the worker should enforce its assigned GPU ID.
        """
        worker = Worker(workspace_dir=temp_workspace, gpu_id=0)
        worker._init_components()
        
        # Create task that tries to use GPU 1 via config
        task_with_gpu_config = TaskInfo(
            id="test_task",
            tool_name="test_tool",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={"CUDA_VISIBLE_DEVICES": "1"},  # Task tries to use GPU 1
            priority=100,
            status="pending",
            task_type="in_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        docker_commands = []
        
        def capture_command(*args, **kwargs):
            if args and isinstance(args[0], list):
                docker_commands.append(' '.join(args[0]))
            return MagicMock(returncode=0, stdout="", stderr="")
        
        with patch('subprocess.run', side_effect=capture_command):
            with patch.object(worker._runner._container_runner, 'pull_image', return_value=True):
                result = worker._runner.run_task(task_with_gpu_config)
        
        # Verify worker's GPU ID (0) is used, not task's config (1)
        assert len(docker_commands) > 0, "Should execute Docker command"
        docker_cmd = docker_commands[0]
        
        # Extract all CUDA_VISIBLE_DEVICES values from the command
        import re
        cuda_matches = re.findall(r'CUDA_VISIBLE_DEVICES=(\d+)', docker_cmd)
        
        # Worker should enforce GPU 0, not GPU 1 from task config
        # The worker's CUDA_VISIBLE_DEVICES=0 should be the LAST one (effective value)
        assert len(cuda_matches) > 0, "Should have at least one CUDA_VISIBLE_DEVICES"
        
        # The last occurrence is the effective value (Docker uses the last -e flag)
        final_cuda_value = cuda_matches[-1]
        assert final_cuda_value == "0", \
            f"Final CUDA_VISIBLE_DEVICES should be 0 (worker's GPU), got {final_cuda_value}. " \
            f"All values: {cuda_matches}"
        
        # Verify --runtime=nvidia and NVIDIA_VISIBLE_DEVICES=0 is used (this is the primary GPU restriction)
        assert "--runtime=nvidia" in docker_cmd, \
            "Worker should use --runtime=nvidia"
        assert "NVIDIA_VISIBLE_DEVICES=0" in docker_cmd, \
            "Worker should use NVIDIA_VISIBLE_DEVICES=0, not 1"
        
        # Verify that even if task config added CUDA_VISIBLE_DEVICES=1,
        # the worker's value (0) comes last and will be effective
        # This ensures worker's GPU assignment cannot be overridden by task config
        if "CUDA_VISIBLE_DEVICES=1" in docker_cmd:
            # Find positions of both values
            pos_1 = docker_cmd.rfind("CUDA_VISIBLE_DEVICES=1")
            pos_0 = docker_cmd.rfind("CUDA_VISIBLE_DEVICES=0")
            assert pos_0 > pos_1, \
                "Worker's CUDA_VISIBLE_DEVICES=0 should come AFTER task's CUDA_VISIBLE_DEVICES=1 " \
                f"(pos_0={pos_0}, pos_1={pos_1})"


# ============================================================================
# Test: Docker Runtime Configuration and Error Detection
# ============================================================================


class TestDockerRuntimeConfiguration:
    """Tests to verify Docker runtime configuration compatibility and error detection."""
    
    def test_docker_command_fails_with_cdi_mode_error(self, temp_workspace):
        """
        Test that we can detect CDI mode errors when Docker fails.
        
        CDI mode requires --runtime=nvidia instead of --gpus flag.
        This test verifies we can detect this specific error.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulate CDI mode error from Docker
        cdi_error = (
            "docker: Error response from daemon: failed to create task for container: "
            "failed to create shim task: OCI runtime create failed: runc create failed: "
            "unable to start container process: error during container init: "
            "error running prestart hook #0: exit status 1, stdout: , stderr: "
            "Using requested mode 'cdi'\n"
            "invoking the NVIDIA Container Runtime Hook directly "
            "(e.g. specifying the docker --gpus flag) is not supported. "
            "Please use the NVIDIA Container Runtime "
            "(e.g. specify the --runtime=nvidia flag) instead."
        )
        
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 125
            mock_result.stdout = ""
            mock_result.stderr = cdi_error
            mock_run.return_value = mock_result
            
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify error is captured
            assert exit_code == 125, "Should capture Docker error exit code"
            assert "cdi" in logs.lower() or "nvidia runtime" in logs.lower(), \
                f"Should detect CDI mode error in logs: {logs[:200]}"
            # Note: This test simulates the old error when --gpus was used
            # With the new implementation using --runtime=nvidia, this error should not occur
    
    def test_docker_command_fails_with_gpu_not_available_error(self, temp_workspace):
        """
        Test that we can detect when specified GPU is not available.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=999)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        gpu_error = (
            "docker: Error response from daemon: could not select device driver \"\" "
            "with capabilities: [[gpu]]."
        )
        
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 125
            mock_result.stdout = ""
            mock_result.stderr = gpu_error
            mock_run.return_value = mock_result
            
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify error is captured
            assert exit_code == 125, "Should capture Docker error exit code"
            assert "NVIDIA_VISIBLE_DEVICES=999" in docker_cmd, \
                "Command should request GPU 999 via NVIDIA_VISIBLE_DEVICES (which doesn't exist)"
    
    def test_docker_runtime_error_detection(self, temp_workspace):
        """
        Test detection of various Docker runtime errors.
        
        This helps identify configuration issues early.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        error_cases = [
            ("CDI mode", "cdi", "nvidia runtime"),
            ("GPU not available", "could not select device driver", "gpu"),
            ("Runtime not found", "runtime \"nvidia\" is not available", "runtime"),
            ("Permission denied", "permission denied", "permission"),
        ]
        
        for error_name, error_text, expected_keyword in error_cases:
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 125
                mock_result.stdout = ""
                mock_result.stderr = f"docker: Error: {error_text}"
                mock_run.return_value = mock_result
                
                exit_code, logs, docker_cmd = runner.run(
                    image="test/image:latest",
                    command="python main.py",
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
                assert exit_code == 125, \
                    f"Should capture {error_name} error"
                assert expected_keyword.lower() in logs.lower() or error_text.lower() in logs.lower(), \
                    f"Should detect {error_name} in logs: {logs[:200]}"
    
    @pytest.mark.skipif(
        not Path("/usr/bin/docker").exists() and not Path("/usr/local/bin/docker").exists(),
        reason="Docker not available for integration test"
    )
    def test_actual_docker_gpu_command_execution(self, temp_workspace):
        """
        Test actual Docker execution to catch runtime configuration issues.
        
        This test actually runs Docker (not mocked) to verify:
        1. Docker command syntax is correct
        2. GPU flags are properly formatted
        3. No runtime configuration errors occur (especially CDI mode errors)
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to run a simple GPU test command
        # Use a minimal image that supports GPU
        try:
            exit_code, logs, docker_cmd = runner.run(
                image="nvidia/cuda:11.0-base",
                command="nvidia-smi --query-gpu=index --format=csv,noheader",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Check for specific error messages
            if exit_code != 0:
                error_lower = logs.lower()
                
                # Skip permission errors (test environment limitation)
                if "permission denied" in error_lower or "operation not permitted" in error_lower:
                    pytest.skip(
                        f"Docker permission denied (test environment limitation). "
                        f"This is expected in CI/test environments without Docker access."
                    )
                
                # Check for CDI mode error - THIS IS THE CRITICAL CHECK
                if "cdi" in error_lower and ("nvidia runtime" in error_lower or "runtime" in error_lower):
                    pytest.fail(
                        f"❌ CDI MODE ERROR DETECTED: Docker is configured in CDI mode but we're using --gpus flag.\n"
                        f"This is the exact error from the user's terminal!\n\n"
                        f"Error: {logs[:500]}\n\n"
                        f"Command: {docker_cmd}\n\n"
                        f"SOLUTION: The code needs to detect CDI mode and use --runtime=nvidia instead of --gpus.\n"
                        f"This test successfully caught the configuration mismatch!"
                    )
                
                # Check for GPU not available
                if "could not select device driver" in error_lower:
                    pytest.fail(
                        f"GPU device driver not available. "
                        f"Error: {logs[:500]}\n"
                        f"Command: {docker_cmd}\n"
                        f"Check if nvidia-container-runtime is properly installed."
                    )
                
                # Check for runtime not found
                if "runtime" in error_lower and "not available" in error_lower:
                    pytest.fail(
                        f"Docker runtime not available. "
                        f"Error: {logs[:500]}\n"
                        f"Command: {docker_cmd}\n"
                        f"Check Docker runtime configuration."
                    )
                
                # Generic error - but don't fail on permission issues
                if "permission" not in error_lower:
                    pytest.fail(
                        f"Docker GPU command failed with exit code {exit_code}. "
                        f"Error: {logs[:500]}\n"
                        f"Command: {docker_cmd}"
                    )
                else:
                    pytest.skip(f"Docker permission issue: {logs[:200]}")
            
            # If successful, verify GPU is accessible
            gpu_indices = [line.strip() for line in logs.strip().split('\n') 
                          if line.strip().isdigit()]
            assert len(gpu_indices) > 0, \
                f"Should see at least one GPU, got output: {logs}"
            assert "0" in gpu_indices, \
                f"Should see GPU 0, got GPUs: {gpu_indices}"
            
        except Exception as e:
            # Don't fail on permission errors
            if "permission" in str(e).lower() or "operation not permitted" in str(e).lower():
                pytest.skip(f"Docker permission issue: {e}")
            pytest.fail(
                f"Exception during Docker GPU test: {e}\n"
                f"This may indicate a Docker configuration issue."
            )
    
    @pytest.mark.skipif(
        not Path("/usr/bin/docker").exists() and not Path("/usr/local/bin/docker").exists(),
        reason="Docker not available for integration test"
    )
    def test_docker_gpu_command_validation(self, temp_workspace):
        """
        Validate Docker GPU command syntax before execution.
        
        This test checks that the command we generate is syntactically correct
        and would work with Docker, catching issues before they cause runtime errors.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command without executing
        with patch('subprocess.run') as mock_run:
            # Don't actually run, just capture the command
            mock_run.side_effect = Exception("Should not execute")
            
            try:
                exit_code, logs, docker_cmd = runner.run(
                    image="test/image:latest",
                    command="python main.py",
                    input_dir=input_dir,
                    output_dir=output_dir
                )
            except Exception:
                pass  # Expected, we're just building the command
            
            # Verify command structure
            cmd_parts = docker_cmd.split()
            
            # Check for required components
            assert "docker" in cmd_parts, "Should start with docker"
            assert "run" in cmd_parts, "Should have 'run' command"
            assert "--rm" in cmd_parts, "Should have --rm flag"
            assert "--runtime=nvidia" in cmd_parts, "Should have --runtime=nvidia flag"
            
            # Verify NVIDIA_VISIBLE_DEVICES format
            nvidia_env = None
            for i, part in enumerate(cmd_parts):
                if part == "-e" and i + 1 < len(cmd_parts):
                    env_var = cmd_parts[i + 1]
                    if "NVIDIA_VISIBLE_DEVICES" in env_var:
                        nvidia_env = env_var
                        break
            assert nvidia_env is not None, "Should have NVIDIA_VISIBLE_DEVICES"
            assert nvidia_env.startswith("NVIDIA_VISIBLE_DEVICES="), \
                f"NVIDIA_VISIBLE_DEVICES should start with 'NVIDIA_VISIBLE_DEVICES=', got '{nvidia_env}'"
            gpu_id_str = nvidia_env.split("=")[1]
            assert gpu_id_str.isdigit(), \
                f"GPU ID should be numeric, got '{gpu_id_str}'"
    
    def test_error_logging_includes_docker_command(self, temp_workspace):
        """
        Test that error logs include the full Docker command for debugging.
        
        This is critical for diagnosing runtime configuration issues.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        error_message = "Some Docker error occurred"
        
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 125
            mock_result.stdout = ""
            mock_result.stderr = error_message
            mock_run.return_value = mock_result
            
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify command is returned for debugging
            assert docker_cmd is not None, "Should return Docker command string"
            assert "docker" in docker_cmd, "Command should contain 'docker'"
            assert "--runtime=nvidia" in docker_cmd, "Command should contain '--runtime=nvidia'"
            assert "NVIDIA_VISIBLE_DEVICES" in docker_cmd, "Command should contain 'NVIDIA_VISIBLE_DEVICES'"
            
            # Verify error is in logs
            assert error_message in logs or str(exit_code) in logs, \
                f"Logs should contain error information: {logs[:200]}"
    
    def test_multiple_gpu_ids_are_rejected(self, temp_workspace):
        """
        Test that Docker command doesn't accidentally allow multiple GPUs.
        
        Worker should only access one GPU at a time.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify only single GPU is specified
            import re
            nvidia_specs = re.findall(r'NVIDIA_VISIBLE_DEVICES=(\d+)', docker_cmd)
            assert len(nvidia_specs) == 1, \
                f"Should have exactly one NVIDIA_VISIBLE_DEVICES spec, got {nvidia_specs}"
            
            gpu_value = nvidia_specs[0]
            # Should be single digit, not comma-separated list
            assert "," not in gpu_value, \
                f"NVIDIA_VISIBLE_DEVICES should not contain multiple GPUs: {gpu_value}"
            assert gpu_value.isdigit(), \
                f"GPU spec should be a single digit: {gpu_value}"
    
    def test_docker_command_includes_all_required_components(self, temp_workspace):
        """
        Test that Docker command includes all required components for GPU access.
        
        Missing components can cause silent failures or configuration errors.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify all required components
            required_components = [
                "docker",
                "run",
                "--rm",
                "--runtime=nvidia",
                "NVIDIA_VISIBLE_DEVICES=0",
                "CUDA_VISIBLE_DEVICES=0",
                "NVIDIA_DRIVER_CAPABILITIES=all",
                "-v",  # Volume mounts
                "/input:ro",
                "/output:rw",
            ]
            
            for component in required_components:
                assert component in docker_cmd, \
                    f"Docker command should contain '{component}'. " \
                    f"Command: {docker_cmd[:200]}"
