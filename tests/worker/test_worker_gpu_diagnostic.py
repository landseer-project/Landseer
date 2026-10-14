"""
Diagnostic tests to identify GPU assignment issues.

These tests help debug why containers don't see GPUs even when workers
are supposed to have them assigned.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.worker.cli import Worker
from src.worker.runner import TaskRunner, DockerRunner
from src.worker.client import TaskInfo


NVIDIA_DOCKER_INFO = MagicMock(returncode=0, stdout='{"nvidia": {}, "runc": {}}', stderr="")


def _run_docker(runner, input_dir, output_dir, run_side_effect=None):
    """
    Run DockerRunner.run with subprocess mocked and return the docker argv.

    DockerRunner.run probes `docker info` (and `which nvidia-container-runtime`)
    via subprocess.run and launches the container via subprocess.Popen.
    By default the probe reports an nvidia runtime; pass run_side_effect to
    simulate other host configurations.
    """
    proc = MagicMock()
    proc.poll.return_value = 0
    proc.returncode = 0
    proc.stdout.readline.return_value = ""
    proc.stderr.readline.return_value = ""
    with patch("src.worker.runner.subprocess.run") as mock_run, \
            patch("src.worker.runner.subprocess.Popen") as mock_popen:
        if run_side_effect is not None:
            mock_run.side_effect = run_side_effect
        else:
            mock_run.return_value = NVIDIA_DOCKER_INFO
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


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


class TestGPUDiagnostic:
    """Diagnostic tests for GPU assignment issues."""
    
    def test_verify_gpu_id_flows_from_worker_to_container(self, temp_workspace):
        """
        Verify GPU ID flows correctly: Worker -> TaskRunner -> DockerRunner -> Container.
        
        This test helps identify where GPU assignment might be breaking.
        """
        gpu_id = 0
        
        # Step 1: Create Worker with GPU
        worker = Worker(workspace_dir=temp_workspace, gpu_id=gpu_id)
        assert worker.gpu_id == gpu_id, f"Step 1 FAILED: Worker.gpu_id should be {gpu_id}"
        
        # Step 2: Initialize components
        worker._init_components()
        assert worker._runner is not None, "Step 2 FAILED: TaskRunner should be initialized"
        assert worker._runner.gpu_id == gpu_id, \
            f"Step 2 FAILED: TaskRunner.gpu_id should be {gpu_id}, got {worker._runner.gpu_id}"
        
        # Step 3: Verify DockerRunner has GPU ID
        assert worker._runner._container_runner is not None, \
            "Step 3 FAILED: ContainerRunner should be initialized"
        assert worker._runner._container_runner.gpu_id == gpu_id, \
            f"Step 3 FAILED: DockerRunner.gpu_id should be {gpu_id}, got {worker._runner._container_runner.gpu_id}"
        
        # Step 4: Verify GPU flags are added to Docker command
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        docker_runner = worker._runner._container_runner
        
        call_args = _run_docker(docker_runner, input_dir, output_dir)

        # GPU access is via the nvidia runtime; --gpus is not used
        if "--runtime=nvidia" not in call_args:
            pytest.fail(f"Step 4 FAILED: --runtime=nvidia not found in Docker command: {call_args}")
        assert "--gpus" not in call_args, \
            f"Step 4 FAILED: --gpus should not be used (conflicts with CDI/nvidia runtime): {call_args}"

        env_vars = _env_vars(call_args)
        if "NVIDIA_VISIBLE_DEVICES=all" not in env_vars:
            pytest.fail(f"Step 4 FAILED: NVIDIA_VISIBLE_DEVICES=all not found in Docker command: {env_vars}")

        # CUDA_VISIBLE_DEVICES pins the assigned GPU
        cuda_var = next((v for v in env_vars if v.startswith("CUDA_VISIBLE_DEVICES=")), None)
        if cuda_var is None:
            pytest.fail(f"Step 4 FAILED: CUDA_VISIBLE_DEVICES not found in Docker command")

        expected_cuda = f"CUDA_VISIBLE_DEVICES={gpu_id}"
        assert cuda_var == expected_cuda, \
            f"Step 4 FAILED: CUDA_VISIBLE_DEVICES should be '{expected_cuda}', got '{cuda_var}'"
    
    def test_detect_if_worker_started_without_gpu(self, temp_workspace):
        """
        Detect if worker was started without GPU ID.
        
        This is a common issue - worker might be started without --gpu flag.
        """
        # Simulate worker started without --gpu flag
        worker_no_gpu = Worker(workspace_dir=temp_workspace, gpu_id=None)
        worker_no_gpu._init_components()
        
        # This worker should NOT have GPU flags
        assert worker_no_gpu.gpu_id is None, \
            "Worker started without GPU should have gpu_id=None"
        assert worker_no_gpu._runner.gpu_id is None, \
            "TaskRunner should have gpu_id=None when Worker has no GPU"
        
        # Verify no GPU flags would be added
        docker_runner = worker_no_gpu._runner._container_runner
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        call_args = _run_docker(docker_runner, input_dir, output_dir)

        # Should NOT have GPU flags
        assert "--gpus" not in call_args, \
            "Worker without GPU should NOT have --gpus flag"
        assert "--runtime=nvidia" not in call_args, \
            "Worker without GPU should NOT have --runtime=nvidia flag"
        env_vars = _env_vars(call_args)
        assert not any(v.startswith(("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")) for v in env_vars), \
            f"Worker without GPU should NOT set GPU env vars: {env_vars}"
    
    def test_verify_gpu_id_not_lost_during_task_execution(self, temp_workspace):
        """
        Verify GPU ID is not lost during task execution.
        
        Sometimes GPU ID might be set initially but lost during execution.
        """
        worker = Worker(workspace_dir=temp_workspace, gpu_id=0)
        worker._init_components()
        
        task = TaskInfo(
            id="test_task",
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
        
        # Before execution
        assert worker._runner.gpu_id == 0, \
            "GPU ID should be 0 before task execution"
        
        # During execution (mock)
        container_runner = worker._runner._container_runner
        with patch.object(container_runner, 'pull_image', return_value=True), \
                patch.object(container_runner, 'run') as mock_run:
            mock_run.return_value = (0, "Success", "docker run ...")
            
            result = worker._runner.run_task(task)
            
            # The task should have reached the container runner
            assert mock_run.called, "Container runner.run should be called during task execution"
            
            # Verify GPU ID is still set on every layer
            assert worker._runner.gpu_id == 0, \
                "GPU ID should still be 0 after task execution"
            assert container_runner.gpu_id == 0, \
                "DockerRunner GPU ID should still be 0 after task execution"
    
    def test_check_docker_runtime_availability(self, temp_workspace):
        """
        Check if Docker runtime supports GPU (nvidia runtime).
        
        Even if GPU ID is set correctly, Docker might not have nvidia runtime.
        """
        # This test would require actual Docker check
        # For now, we just verify the code sets the flag correctly
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Host reports the nvidia runtime in `docker info`
        call_args = _run_docker(runner, input_dir, output_dir)
        assert "--runtime=nvidia" in call_args, \
            "DockerRunner should set --runtime=nvidia when GPU is available"

        # Host does NOT report an nvidia runtime and nvidia-container-runtime is
        # not on PATH: DockerRunner logs a warning but still requests the nvidia
        # runtime (it may be configured at the daemon level).
        def no_nvidia(cmd, *args, **kwargs):
            if cmd[:2] == ["docker", "info"]:
                return MagicMock(returncode=0, stdout='{"runc": {}}', stderr="")
            return MagicMock(returncode=1, stdout="", stderr="")

        call_args = _run_docker(runner, input_dir, output_dir, run_side_effect=no_nvidia)
        assert "--runtime=nvidia" in call_args, \
            "DockerRunner should still set --runtime=nvidia when runtime detection fails"


def test_generate_gpu_diagnostic_report(temp_workspace):
    """
    Generate a diagnostic report for GPU assignment.
    
    This can be used to debug GPU issues in production.
    """
    report = []
    
    # Test 1: Worker initialization
    worker = Worker(workspace_dir=temp_workspace, gpu_id=0)
    report.append(f"✓ Worker initialized with gpu_id={worker.gpu_id}")
    
    # Test 2: Component initialization
    worker._init_components()
    report.append(f"✓ TaskRunner initialized with gpu_id={worker._runner.gpu_id}")
    report.append(f"✓ DockerRunner initialized with gpu_id={worker._runner._container_runner.gpu_id}")
    
    # Test 3: Docker command generation
    docker_runner = worker._runner._container_runner
    input_dir = temp_workspace / "input"
    output_dir = temp_workspace / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    call_args = _run_docker(docker_runner, input_dir, output_dir)
    cmd_str = ' '.join(call_args)

    report.append(f"✓ Docker command generated: {cmd_str[:200]}...")

    if "--runtime=nvidia" in call_args:
        report.append("✓ --runtime=nvidia flag found")
    else:
        report.append("✗ --runtime=nvidia flag NOT found")

    if "--gpus" in call_args:
        report.append("✗ --gpus flag present (conflicts with CDI/nvidia runtime)")
    else:
        report.append("✓ --gpus flag absent (GPU selected via env)")

    env_vars = _env_vars(call_args)
    if "NVIDIA_VISIBLE_DEVICES=all" in env_vars:
        report.append("✓ NVIDIA_VISIBLE_DEVICES=all found")
    else:
        report.append("✗ NVIDIA_VISIBLE_DEVICES=all NOT found")

    cuda_vars = [v for v in env_vars if v.startswith("CUDA_VISIBLE_DEVICES=")]
    if cuda_vars == [f"CUDA_VISIBLE_DEVICES={worker.gpu_id}"]:
        report.append(f"✓ CUDA_VISIBLE_DEVICES found: {cuda_vars[0]}")
    else:
        report.append(f"✗ CUDA_VISIBLE_DEVICES wrong or missing: {cuda_vars}")
    
    # Print report
    print("\n".join(report))
    
    # All checks should pass
    assert all("✓" in line for line in report), \
        "Some GPU flags are missing - check the diagnostic report above"
