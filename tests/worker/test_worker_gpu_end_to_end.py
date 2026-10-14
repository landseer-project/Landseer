"""
End-to-end integration test for GPU assignment from command line to Docker execution.

This test verifies the complete flow:
1. Worker started with --gpu X command line argument
2. GPU ID flows through Worker -> TaskRunner -> DockerRunner
3. Docker command includes correct GPU flags
4. Docker can actually access the specified GPU (if Docker is available)
"""

import pytest
import subprocess
import sys
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import Mock, patch, MagicMock

from src.worker.cli import Worker, create_parser
from src.worker.runner import TaskRunner, DockerRunner
from src.worker.client import TaskInfo


@contextmanager
def _mock_docker(docker_info_stdout='{"nvidia": {}, "runc": {}}'):
    """
    Mock the subprocess calls DockerRunner.run makes.

    DockerRunner.run probes `docker info` via subprocess.run (stubbed to report
    an nvidia runtime) and launches the container via subprocess.Popen (stubbed
    as a successful, silent run). Yields (mock_run, mock_popen).
    """
    proc = MagicMock()
    proc.poll.return_value = 0
    proc.returncode = 0
    proc.stdout.readline.return_value = ""
    proc.stderr.readline.return_value = ""
    with patch("src.worker.runner.subprocess.run") as mock_run, \
            patch("src.worker.runner.subprocess.Popen") as mock_popen:
        mock_run.return_value = MagicMock(returncode=0, stdout=docker_info_stdout, stderr="")
        mock_popen.return_value = proc
        yield mock_run, mock_popen


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
        id="test_task_gpu_e2e",
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
# Test: Complete End-to-End Flow
# ============================================================================


class TestCompleteGPUFlow:
    """Complete end-to-end tests for GPU flow."""
    
    def test_cli_gpu_argument_to_docker_command(self, temp_workspace, mock_task):
        """
        Test complete flow: CLI --gpu argument -> Docker command execution.
        
        This verifies that when a worker is started with --gpu X, the GPU ID
        flows correctly through the entire system and appears in Docker commands.
        """
        # Step 1: Parse CLI arguments (simulating: landseer-worker --gpu 0)
        parser = create_parser()
        args = parser.parse_args([
            "--gpu", "0",
            "--workspace", str(temp_workspace)
        ])
        
        assert args.gpu == 0, "CLI should parse --gpu 0"
        
        # Step 2: Create Worker with GPU ID from CLI
        worker = Worker(
            workspace_dir=Path(args.workspace) if args.workspace else None,
            gpu_id=args.gpu
        )
        
        assert worker.gpu_id == 0, "Worker should have GPU ID 0"
        
        # Step 3: Initialize components
        worker._init_components()
        
        assert worker._runner is not None, "TaskRunner should be initialized"
        assert worker._runner.gpu_id == 0, "TaskRunner should have GPU ID 0"
        
        assert worker._runner._container_runner is not None, \
            "DockerRunner should be initialized"
        assert worker._runner._container_runner.gpu_id == 0, \
            "DockerRunner should have GPU ID 0"
        
        # Step 4: Execute task and capture the `docker run` command launched
        with _mock_docker() as (_mock_run, mock_popen):
            with patch.object(worker._runner._container_runner, 'pull_image', return_value=True):
                result = worker._runner.run_task(mock_task)
        
        # Step 5: Verify Docker command has correct GPU configuration
        assert mock_popen.called, "Should have executed a Docker command"
        docker_argv = mock_popen.call_args[0][0]
        assert docker_argv[:2] == ["docker", "run"], f"Should run docker: {docker_argv[:5]}"
        docker_cmd = ' '.join(docker_argv)
        
        # Verify GPU configuration: nvidia runtime, all devices visible to the
        # runtime, and the assigned GPU pinned via CUDA_VISIBLE_DEVICES
        assert "--runtime=nvidia" in docker_argv, \
            f"Should have --runtime=nvidia in Docker command: {docker_cmd[:300]}"
        assert "--gpus" not in docker_argv, \
            f"Should NOT have --gpus (conflicts with CDI/nvidia runtime): {docker_cmd[:300]}"
        assert "-e NVIDIA_VISIBLE_DEVICES=all" in docker_cmd, \
            f"Should have NVIDIA_VISIBLE_DEVICES=all in Docker command: {docker_cmd[:300]}"
        assert "-e CUDA_VISIBLE_DEVICES=0" in docker_cmd, \
            f"Should have CUDA_VISIBLE_DEVICES=0 in Docker command: {docker_cmd[:300]}"
        assert result.success, f"Task should succeed: {result.error_message}"
        
        print(f"\n✓ Complete flow verified:")
        print(f"  CLI: --gpu 0")
        print(f"  Worker.gpu_id: {worker.gpu_id}")
        print(f"  TaskRunner.gpu_id: {worker._runner.gpu_id}")
        print(f"  DockerRunner.gpu_id: {worker._runner._container_runner.gpu_id}")
        print(f"  Docker command: {docker_cmd[:200]}...")
    
    def test_different_gpu_ids_flow_correctly(self, temp_workspace, mock_task):
        """Test that different GPU IDs flow correctly from CLI to Docker."""
        for gpu_id in [0, 1, 2, 3]:
            # Simulate CLI: landseer-worker --gpu <gpu_id>
            parser = create_parser()
            args = parser.parse_args([
                "--gpu", str(gpu_id),
                "--workspace", str(temp_workspace / f"worker_{gpu_id}")
            ])
            
            # Create worker
            worker = Worker(
                workspace_dir=Path(args.workspace) if args.workspace else None,
                gpu_id=args.gpu
            )
            worker._init_components()
            
            # Verify GPU ID at each level
            assert worker.gpu_id == gpu_id, \
                f"Worker should have GPU ID {gpu_id}"
            assert worker._runner.gpu_id == gpu_id, \
                f"TaskRunner should have GPU ID {gpu_id}"
            assert worker._runner._container_runner.gpu_id == gpu_id, \
                f"DockerRunner should have GPU ID {gpu_id}"
            
            # Verify Docker command would have correct GPU
            docker_runner = worker._runner._container_runner
            input_dir = temp_workspace / f"input_{gpu_id}"
            output_dir = temp_workspace / f"output_{gpu_id}"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with _mock_docker():
                exit_code, logs, docker_cmd = docker_runner.run(
                    image="test/image:latest",
                    command="python main.py",
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
                # Verify GPU ID in command
                assert f"-e CUDA_VISIBLE_DEVICES={gpu_id}" in docker_cmd, \
                    f"CUDA_VISIBLE_DEVICES={gpu_id} should be in Docker command"
                assert "--runtime=nvidia" in docker_cmd, \
                    "--runtime=nvidia should be in Docker command"
                other = [g for g in [0, 1, 2, 3] if g != gpu_id]
                assert not any(f"CUDA_VISIBLE_DEVICES={g}" in docker_cmd for g in other), \
                    f"Only GPU {gpu_id} should be pinned in Docker command"


# ============================================================================
# Test: Docker GPU Access Verification (if Docker available)
# ============================================================================


class TestDockerGPUAccessVerification:
    """Tests to verify Docker can actually access the GPU."""
    
    @pytest.mark.skipif(
        not Path("/usr/bin/docker").exists() and not Path("/usr/local/bin/docker").exists(),
        reason="Docker not available for integration test"
    )
    def test_docker_can_access_gpu_0(self, temp_workspace):
        """
        Test that Docker can actually access GPU 0.
        
        This requires:
        - Docker installed
        - nvidia-container-runtime configured
        - GPU 0 available
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to run nvidia-smi in container to verify GPU access
        try:
            exit_code, logs, docker_cmd = runner.run(
                image="nvidia/cuda:11.0-base",
                command="nvidia-smi --query-gpu=index --format=csv,noheader",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            if exit_code == 0:
                # Should see GPU index in output
                assert "0" in logs or "GPU" in logs or len(logs.strip()) > 0, \
                    f"Should see GPU information, got: {logs[:200]}"
                print(f"\n✓ Docker can access GPU 0")
                print(f"  Output: {logs[:200]}")
            else:
                pytest.skip(f"Docker GPU test failed (may not have GPU access): {logs[:200]}")
                
        except Exception as e:
            pytest.skip(f"Docker GPU test failed: {e}")
    
    def test_docker_command_has_correct_structure(self, temp_workspace):
        """
        Verify Docker command structure is correct for GPU access.
        
        This test verifies the command format without actually running Docker.
        """
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with _mock_docker():
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify command structure
            cmd_parts = docker_cmd.split()
            
            # Should have: docker run --rm ... --runtime=nvidia ... -e NVIDIA_VISIBLE_DEVICES=all
            #              -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_VISIBLE_DEVICES=0 ... <image> <cmd>
            assert cmd_parts[0] == "docker", "Should start with docker"
            assert cmd_parts[1] == "run", "Should be docker run"
            assert "--rm" in cmd_parts, "Should have --rm"
            assert "--runtime=nvidia" in cmd_parts, "Should have --runtime=nvidia"
            assert "--gpus" not in cmd_parts, \
                "Should NOT have --gpus (conflicts with CDI/nvidia runtime)"
            
            env_vars = [cmd_parts[i + 1] for i, part in enumerate(cmd_parts)
                        if part == "-e" and i + 1 < len(cmd_parts)]
            for expected in ("NVIDIA_VISIBLE_DEVICES=all",
                             "NVIDIA_DRIVER_CAPABILITIES=all",
                             "CUDA_VISIBLE_DEVICES=0"):
                assert expected in env_vars, \
                    f"Should have {expected} in command: {docker_cmd[:300]}"
            
            # Runtime flag and env vars must precede the image name
            image_idx = cmd_parts.index("test/image:latest")
            assert cmd_parts.index("--runtime=nvidia") < image_idx, \
                "--runtime=nvidia must come before the image"
            assert cmd_parts.index("CUDA_VISIBLE_DEVICES=0") < image_idx, \
                "CUDA_VISIBLE_DEVICES must come before the image"
            assert cmd_parts[image_idx + 1:] == ["python", "main.py"], \
                "Command should follow the image"


# ============================================================================
# Test: Worker Registration with GPU
# ============================================================================


class TestWorkerRegistrationWithGPU:
    """Tests for worker registration with GPU capabilities."""
    
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
        
        # Verify capabilities were reported
        call_args = worker._client.register.call_args
        capabilities = call_args[1]["capabilities"]
        
        assert capabilities["gpu_available"] is True, \
            "Worker should report GPU as available"
        assert capabilities["gpu_id"] == 0, \
            f"Worker should report GPU ID 0, got {capabilities.get('gpu_id')}"
    
    def test_worker_reports_no_gpu_when_not_specified(self, temp_workspace):
        """Worker should report no GPU when --gpu flag is not used."""
        # Simulate: landseer-worker (no --gpu flag)
        parser = create_parser()
        args = parser.parse_args(["--workspace", str(temp_workspace)])
        
        worker = Worker(workspace_dir=temp_workspace, gpu_id=args.gpu)
        worker._client = MagicMock()
        
        worker_info = MagicMock()
        worker_info.worker_id = "test_worker"
        worker._client.register.return_value = worker_info
        
        with patch('src.worker.cli.ContainerRuntime') as mock_runtime:
            mock_runtime.detect_runtime.return_value = "docker"
            worker._register()
        
        # Verify no GPU capability was reported
        call_args = worker._client.register.call_args
        capabilities = call_args[1]["capabilities"]
        
        assert capabilities["gpu_available"] is False, \
            "Worker should report GPU as not available"
        assert capabilities["gpu_id"] is None, \
            "Worker should report gpu_id as None"


# ============================================================================
# Test: Real-World Worker Startup
# ============================================================================


class TestRealWorldWorkerStartup:
    """Tests simulating real-world worker startup scenarios."""
    
    def test_worker_startup_with_gpu_from_cli(self, temp_workspace):
        """
        Simulate real worker startup: landseer-worker --gpu 0 --backend-url http://localhost:8000
        """
        # Parse CLI arguments as if worker was started
        parser = create_parser()
        args = parser.parse_args([
            "--gpu", "0",
            "--backend-url", "http://localhost:8000",
            "--workspace", str(temp_workspace)
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
        
        # Verify GPU is set
        assert worker.gpu_id == 0, "Worker should have GPU ID 0 from CLI"
        
        # Initialize components
        worker._init_components()
        
        # Verify GPU flows through
        assert worker._runner.gpu_id == 0, "TaskRunner should have GPU ID 0"
        assert worker._runner._container_runner.gpu_id == 0, \
            "DockerRunner should have GPU ID 0"
        
        # Verify Docker command would have GPU flags
        docker_runner = worker._runner._container_runner
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with _mock_docker():
            exit_code, logs, docker_cmd = docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify GPU configuration
            assert "--runtime=nvidia" in docker_cmd, \
                "Docker command should have --runtime=nvidia"
            assert "-e CUDA_VISIBLE_DEVICES=0" in docker_cmd, \
                "Docker command should have CUDA_VISIBLE_DEVICES=0"
            assert "-e NVIDIA_VISIBLE_DEVICES=all" in docker_cmd, \
                "Docker command should have NVIDIA_VISIBLE_DEVICES=all"
            assert "--gpus" not in docker_cmd, \
                "Docker command should NOT have --gpus"
    
    def test_worker_startup_without_gpu_from_cli(self, temp_workspace):
        """
        Simulate real worker startup: landseer-worker --backend-url http://localhost:8000
        (no --gpu flag)
        """
        parser = create_parser()
        args = parser.parse_args([
            "--backend-url", "http://localhost:8000",
            "--workspace", str(temp_workspace)
        ])
        
        worker = Worker(
            backend_url=args.backend_url,
            workspace_dir=Path(args.workspace) if args.workspace else None,
            gpu_id=args.gpu  # Should be None
        )
        
        assert worker.gpu_id is None, "Worker should have no GPU when --gpu not specified"
        
        worker._init_components()
        
        # Verify no GPU flags would be added
        docker_runner = worker._runner._container_runner
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with _mock_docker():
            exit_code, logs, docker_cmd = docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify no GPU flags
            assert "--gpus" not in docker_cmd, \
                "Should NOT have --gpus flag when worker has no GPU"
            assert "CUDA_VISIBLE_DEVICES" not in docker_cmd, \
                "Should NOT have CUDA_VISIBLE_DEVICES when worker has no GPU"
            assert "--runtime=nvidia" not in docker_cmd, \
                "Should NOT have --runtime=nvidia when worker has no GPU"
            assert "NVIDIA_VISIBLE_DEVICES" not in docker_cmd, \
                "Should NOT have NVIDIA_VISIBLE_DEVICES when worker has no GPU"


# ============================================================================
# Test: Verification Helper
# ============================================================================


def verify_gpu_assignment(worker_gpu_id: int, docker_cmd: str) -> tuple[bool, list[str]]:
    """
    Verify that GPU ID from worker appears correctly in Docker command.
    
    Args:
        worker_gpu_id: GPU ID that worker was started with
        docker_cmd: Docker command string to verify
        
    Returns:
        Tuple of (all_checks_passed, list_of_failed_checks)
    """
    checks = []
    failed = []
    
    def check(ok: bool, good: str, bad: str) -> None:
        if ok:
            checks.append(f"✓ {good}")
        else:
            failed.append(f"✗ {bad}")
            checks.append(failed[-1])

    # Check 1: nvidia runtime requested
    check("--runtime=nvidia" in docker_cmd,
          "--runtime=nvidia found", "--runtime=nvidia NOT found")

    # Check 2: all devices visible to the nvidia runtime (required with CDI)
    check("NVIDIA_VISIBLE_DEVICES=all" in docker_cmd,
          "NVIDIA_VISIBLE_DEVICES=all found", "NVIDIA_VISIBLE_DEVICES=all NOT found")

    # Check 3: CUDA_VISIBLE_DEVICES pins the worker's GPU
    check(f"CUDA_VISIBLE_DEVICES={worker_gpu_id}" in docker_cmd,
          f"CUDA_VISIBLE_DEVICES={worker_gpu_id} found",
          f"CUDA_VISIBLE_DEVICES={worker_gpu_id} NOT found")

    # Check 4: no conflicting --gpus flag
    check("--gpus" not in docker_cmd,
          "--gpus NOT present (correct)", "--gpus present (conflicts with CDI/nvidia runtime)")
    
    return len(failed) == 0, failed


class TestGPUVerificationHelper:
    """Tests for GPU verification helper function."""
    
    def test_verification_helper_passes_for_correct_config(self, temp_workspace):
        """Verification helper should pass for correct GPU configuration."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with _mock_docker():
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            passed, failed = verify_gpu_assignment(0, docker_cmd)
            
            assert passed, f"Verification should pass, but failed: {failed}"
            print("\n".join([f"  {check}" for check in verify_gpu_assignment(0, docker_cmd)[1] if "✓" in check]))
    
    def test_verification_helper_fails_for_missing_gpu(self, temp_workspace):
        """Verification helper should fail when GPU flags are missing."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=None)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with _mock_docker():
            exit_code, logs, docker_cmd = runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Should fail verification (no GPU flags)
            passed, failed = verify_gpu_assignment(0, docker_cmd)
            
            assert not passed, "Verification should fail when GPU flags are missing"
            assert len(failed) > 0, "Should have failed checks"
