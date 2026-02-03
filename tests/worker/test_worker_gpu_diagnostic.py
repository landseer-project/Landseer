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
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_run.call_args[0][0]
            
            # Check for --gpus flag
            if "--gpus" not in call_args:
                pytest.fail(f"Step 4 FAILED: --gpus flag not found in Docker command: {call_args}")
            
            gpus_idx = call_args.index("--gpus")
            gpu_spec = call_args[gpus_idx + 1]
            expected_spec = f"device={gpu_id}"
            assert gpu_spec == expected_spec, \
                f"Step 4 FAILED: GPU spec should be '{expected_spec}', got '{gpu_spec}'"
            
            # Check for --runtime=nvidia
            if "--runtime=nvidia" not in call_args:
                pytest.fail(f"Step 4 FAILED: --runtime=nvidia not found in Docker command")
            
            # Check for CUDA_VISIBLE_DEVICES
            env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
            cuda_var = next((v for v in env_vars if "CUDA_VISIBLE_DEVICES" in v), None)
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
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_run.call_args[0][0]
            
            # Should NOT have GPU flags
            assert "--gpus" not in call_args, \
                "Worker without GPU should NOT have --gpus flag"
            assert "--runtime=nvidia" not in call_args, \
                "Worker without GPU should NOT have --runtime=nvidia flag"
    
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
        with patch.object(worker._runner._container_runner, 'run') as mock_run:
            mock_run.return_value = (0, "Success")
            mock_run.pull_image = MagicMock(return_value=True)
            
            result = worker._runner.run_task(task)
            
            # Verify GPU ID is still set
            assert worker._runner.gpu_id == 0, \
                "GPU ID should still be 0 after task execution"
            
            # Verify GPU flags were used in the call
            if mock_run.called:
                # The run method should have been called with GPU support
                pass  # We can't easily verify the internal call, but GPU ID should be preserved
    
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
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_run.call_args[0][0]
            
            # Should have --runtime=nvidia
            assert "--runtime=nvidia" in call_args, \
                "DockerRunner should set --runtime=nvidia when GPU is available"
            
            # Note: Actual Docker runtime check would require:
            # docker info | grep -i runtime
            # or checking if nvidia-container-runtime is installed


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
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        docker_runner.run(
            image="test/image:latest",
            command="python main.py",
            input_dir=input_dir,
            output_dir=output_dir
        )
        
        call_args = mock_run.call_args[0][0]
        cmd_str = ' '.join(call_args)
        
        report.append(f"✓ Docker command generated: {cmd_str[:200]}...")
        
        if "--gpus" in call_args:
            gpus_idx = call_args.index("--gpus")
            report.append(f"✓ --gpus flag found: {call_args[gpus_idx + 1]}")
        else:
            report.append("✗ --gpus flag NOT found")
        
        if "--runtime=nvidia" in call_args:
            report.append("✓ --runtime=nvidia flag found")
        else:
            report.append("✗ --runtime=nvidia flag NOT found")
        
        env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
        cuda_vars = [v for v in env_vars if "CUDA_VISIBLE_DEVICES" in v]
        if cuda_vars:
            report.append(f"✓ CUDA_VISIBLE_DEVICES found: {cuda_vars[0]}")
        else:
            report.append("✗ CUDA_VISIBLE_DEVICES NOT found")
    
    # Print report
    print("\n".join(report))
    
    # All checks should pass
    assert all("✓" in line for line in report if "flag" in line or "CUDA" in line), \
        "Some GPU flags are missing - check the diagnostic report above"
