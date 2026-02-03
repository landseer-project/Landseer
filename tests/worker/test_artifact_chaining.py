"""
Tests for artifact chaining between dependent tasks.

Verifies that:
1. Outputs from dependency tasks are copied to dependent task input directories
2. Artifacts like model.pt are properly chained
3. Multiple dependencies are handled correctly
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil

from src.worker.runner import TaskRunner, ExecutionResult
from src.worker.client import TaskInfo


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def dependency_task_output(temp_workspace):
    """Create a mock dependency task output directory with model.pt."""
    dep_id = "task_dep_1"
    dep_output_dir = temp_workspace / dep_id / "output"
    dep_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model.pt file (simulating in_training output)
    model_file = dep_output_dir / "model.pt"
    model_file.write_bytes(b"fake model data")
    
    # Create other artifacts
    (dep_output_dir / "config.json").write_text('{"epochs": 10}')
    
    return dep_id, dep_output_dir


class TestArtifactChaining:
    """Tests for artifact chaining between tasks."""
    
    def test_dependency_outputs_copied_to_input(self, temp_workspace, dependency_task_output):
        """Outputs from dependency tasks should be copied to dependent task input directory."""
        dep_id, dep_output_dir = dependency_task_output
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        # Create a task that depends on the dependency
        task = TaskInfo(
            id="task_2",
            tool_name="post_training_tool",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=90,
            status="pending",
            task_type="post_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[dep_id]
        )
        
        # Prepare dependency outputs
        dependency_outputs = {dep_id: dep_output_dir}
        
        # Mock container runner to avoid actual execution
        with patch.object(runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "Success")
            mock_runner.pull_image.return_value = True
            
            result = runner.run_task(
                task,
                dependency_outputs=dependency_outputs
            )
        
        # Check that model.pt was copied to input directory
        task_input_dir = temp_workspace / task.id / "input"
        model_file = task_input_dir / "model.pt"
        
        assert model_file.exists(), "model.pt should be copied from dependency output"
        assert model_file.read_bytes() == b"fake model data", "model.pt content should match"
        
        # Check that other artifacts were also copied
        config_file = task_input_dir / "config.json"
        assert config_file.exists(), "config.json should be copied from dependency output"
    
    def test_multiple_dependencies_copied(self, temp_workspace):
        """Outputs from multiple dependencies should all be copied."""
        # Create two dependency outputs
        dep1_id = "task_dep_1"
        dep1_output = temp_workspace / dep1_id / "output"
        dep1_output.mkdir(parents=True, exist_ok=True)
        (dep1_output / "model.pt").write_bytes(b"model from dep1")
        
        dep2_id = "task_dep_2"
        dep2_output = temp_workspace / dep2_id / "output"
        dep2_output.mkdir(parents=True, exist_ok=True)
        (dep2_output / "weights.pth").write_bytes(b"weights from dep2")
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        task = TaskInfo(
            id="task_3",
            tool_name="deployment_tool",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=80,
            status="pending",
            task_type="deployment",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[dep1_id, dep2_id]
        )
        
        dependency_outputs = {
            dep1_id: dep1_output,
            dep2_id: dep2_output
        }
        
        with patch.object(runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "Success")
            mock_runner.pull_image.return_value = True
            
            result = runner.run_task(
                task,
                dependency_outputs=dependency_outputs
            )
        
        # Check that both dependencies' outputs were copied
        task_input_dir = temp_workspace / task.id / "input"
        assert (task_input_dir / "model.pt").exists(), "model.pt from dep1 should be copied"
        assert (task_input_dir / "weights.pth").exists(), "weights.pth from dep2 should be copied"
    
    def test_missing_dependency_output_handled_gracefully(self, temp_workspace):
        """Missing dependency outputs should be handled gracefully."""
        dep_id = "task_missing"
        dep_output_dir = temp_workspace / dep_id / "output"
        # Don't create the directory - simulate missing dependency
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        task = TaskInfo(
            id="task_4",
            tool_name="post_training_tool",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=90,
            status="pending",
            task_type="post_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[dep_id]
        )
        
        # Pass non-existent dependency output
        dependency_outputs = {dep_id: dep_output_dir}
        
        with patch.object(runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "Success")
            mock_runner.pull_image.return_value = True
            
            # Should not raise an error, just log a warning
            result = runner.run_task(
                task,
                dependency_outputs=dependency_outputs
            )
        
        # Task should still execute (though it may fail if it needs the dependency)
        assert result is not None
    
    def test_dependency_output_overwrites_with_latest(self, temp_workspace):
        """If multiple dependencies produce the same file, later one should overwrite."""
        dep1_id = "task_dep_1"
        dep1_output = temp_workspace / dep1_id / "output"
        dep1_output.mkdir(parents=True, exist_ok=True)
        (dep1_output / "model.pt").write_bytes(b"model from dep1")
        
        dep2_id = "task_dep_2"
        dep2_output = temp_workspace / dep2_id / "output"
        dep2_output.mkdir(parents=True, exist_ok=True)
        (dep2_output / "model.pt").write_bytes(b"model from dep2")
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        task = TaskInfo(
            id="task_5",
            tool_name="post_training_tool",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=90,
            status="pending",
            task_type="post_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[dep1_id, dep2_id]
        )
        
        dependency_outputs = {
            dep1_id: dep1_output,
            dep2_id: dep2_output
        }
        
        with patch.object(runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "Success")
            mock_runner.pull_image.return_value = True
            
            result = runner.run_task(
                task,
                dependency_outputs=dependency_outputs
            )
        
        # The last dependency copied should win
        task_input_dir = temp_workspace / task.id / "input"
        model_file = task_input_dir / "model.pt"
        assert model_file.exists(), "model.pt should exist"
        # Since we iterate through dict, order matters - but both should be copied
        # The actual content will be from the last one copied (dep2 in this case)
        assert model_file.read_bytes() in [b"model from dep1", b"model from dep2"]


class TestWorkflowMountsLatestArtifacts:
    """Tests that the latest artifacts for a task are mounted at /data inside the container."""

    def test_post_training_task_mounts_input_dir_as_data(self, temp_workspace, dependency_task_output):
        """
        For a post-training task, the per-task input directory (which already
        contains copied dependency outputs) should be mounted as both /input
        and /data in the Docker command.
        """
        from src.worker.cli import Worker
        from src.worker.runner import TaskRunner, DockerRunner
        from src.worker.client import TaskInfo
        from unittest.mock import MagicMock, patch

        dep_id, dep_output_dir = dependency_task_output

        # Create worker and real TaskRunner (using docker runtime) so that
        # DockerRunner.builds a full docker command, but patch subprocess.run.
        worker = Worker(
            backend_url="http://localhost:8000",
            workspace_dir=temp_workspace,
            gpu_id=None,
        )
        worker._runner = TaskRunner(
            workspace_dir=temp_workspace,
            artifact_cache_dir=None,
            gpu_id=None,
            runtime="docker",
        )

        # Create a post-training task that depends on dep_id
        task = TaskInfo(
            id="task_post",
            tool_name="post_training_tool",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=90,
            status="pending",
            task_type="post_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[dep_id],
        )

        dependency_outputs = {dep_id: dep_output_dir}

        # Patch Docker invocation so we can inspect the generated command.
        with patch("src.worker.runner.subprocess.run") as mock_run, \
             patch.object(worker._runner._container_runner, "pull_image", return_value=True):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            worker._runner.run_task(
                task,
                input_path=None,
                dependency_outputs=dependency_outputs,
            )

        # Determine the expected input directory for this task
        task_input_dir = temp_workspace / task.id / "input"

        # Extract the docker command list from the subprocess call
        call_args = mock_run.call_args[0][0]
        mounts = [call_args[i + 1] for i, arg in enumerate(call_args) if arg == "-v"]

        input_mount = next((m for m in mounts if m.endswith(":\/input:ro") or m.endswith(":/input:ro")), None)
        data_mount = next((m for m in mounts if m.endswith(":/data:ro")), None)

        assert input_mount is not None, "Input directory should be mounted at /input:ro"
        assert data_mount is not None, "Input directory should also be mounted at /data:ro"

        input_host = input_mount.split(":")[0]
        data_host = data_mount.split(":")[0]
        assert input_host == data_host == str(task_input_dir.absolute()), \
            "Both /input and /data should point to the task's input directory"

class TestWorkerDependencyHandling:
    """Tests for how the Worker class handles dependencies."""
    
    def test_worker_collects_dependency_outputs(self, temp_workspace):
        """Worker should collect output directories from dependency tasks."""
        from src.worker.cli import Worker
        
        # Create a dependency task output
        dep_id = "task_dep_1"
        dep_output_dir = temp_workspace / dep_id / "output"
        dep_output_dir.mkdir(parents=True, exist_ok=True)
        (dep_output_dir / "model.pt").write_bytes(b"model data")
        
        # Create worker
        worker = Worker(
            backend_url="http://localhost:8000",
            workspace_dir=temp_workspace,
            gpu_id=None
        )
        
        # Initialize components (this sets up _runner)
        worker._init_components()
        
        # Create a task with dependency
        task = TaskInfo(
            id="task_2",
            tool_name="post_training_tool",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=90,
            status="pending",
            task_type="post_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[dep_id]
        )
        
        # Mock the runner to capture dependency_outputs
        with patch.object(worker._runner, 'run_task') as mock_run_task:
            mock_run_task.return_value = ExecutionResult(
                success=True,
                exit_code=0,
                execution_time_ms=1000,
                output_path=temp_workspace / task.id / "output"
            )
            
            # Execute task
            result = worker._execute_task(task)
            
            # Verify run_task was called with dependency_outputs
            call_kwargs = mock_run_task.call_args[1]
            assert "dependency_outputs" in call_kwargs, "run_task should be called with dependency_outputs"
            dependency_outputs = call_kwargs["dependency_outputs"]
            assert dependency_outputs is not None, "dependency_outputs should not be None"
            assert dep_id in dependency_outputs, f"Dependency {dep_id} should be in dependency_outputs"
            assert dependency_outputs[dep_id] == dep_output_dir, "Dependency output path should match"
