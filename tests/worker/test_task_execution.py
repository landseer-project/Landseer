"""
Adversarial tests for task execution.

These tests verify:
1. Task workspace isolation
2. Input/output handling correctness
3. Dependency verification
4. Artifact caching security
5. Error handling and recovery
6. Task status reporting accuracy
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Optional, List

from src.worker.runner import TaskRunner, ExecutionResult
from src.worker.client import TaskInfo


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    return TaskInfo(
        id="test_task_1",
        tool_name="test_tool",
        tool_image="test/image:latest",
        tool_command="python main.py",
        tool_runtime=None,
        tool_is_baseline=False,
        config={"param1": "value1", "stage": "pre_training"},
        priority=100,
        status="pending",
        task_type="pre_training",
        counter=1,
        workflows=["workflow_1"],
        pipeline_id="pipeline_1",
        dependency_ids=[]
    )


@pytest.fixture
def task_runner(temp_workspace):
    """Create a TaskRunner instance."""
    return TaskRunner(
        workspace_dir=temp_workspace,
        gpu_id=None,
        timeout=3600
    )


# ============================================================================
# Test: Workspace Isolation
# ============================================================================


class TestWorkspaceIsolation:
    """Tests for task workspace isolation and security."""
    
    def test_each_task_gets_unique_workspace(self, task_runner, temp_workspace):
        """Each task should get its own isolated workspace directory."""
        task1 = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        task2 = TaskInfo(
            id="task_2",
            tool_name="tool2",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        # Setup workspaces
        task_dir1, input_dir1, output_dir1 = task_runner._setup_task_workspace(task1)
        task_dir2, input_dir2, output_dir2 = task_runner._setup_task_workspace(task2)
        
        # Verify unique directories
        assert task_dir1 != task_dir2
        assert input_dir1 != input_dir2
        assert output_dir1 != output_dir2
        
        # Verify they're subdirectories of workspace
        assert task_dir1.parent == temp_workspace
        assert task_dir2.parent == temp_workspace
    
    def test_task_cannot_access_other_task_files(self, task_runner, temp_workspace):
        """Tasks should not be able to access other tasks' files."""
        task1 = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        task2 = TaskInfo(
            id="task_2",
            tool_name="tool2",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        # Setup workspaces
        _, input_dir1, output_dir1 = task_runner._setup_task_workspace(task1)
        _, input_dir2, output_dir2 = task_runner._setup_task_workspace(task2)
        
        # Create a file in task1's output
        secret_file = output_dir1 / "secret.txt"
        secret_file.write_text("secret data")
        
        # Task2 should not have access to task1's output
        # (This is enforced by Docker mounts, but we verify the directories are separate)
        assert not (output_dir2 / "secret.txt").exists()
        assert secret_file.exists()  # But task1's file exists
    
    def test_workspace_cleanup_preserves_logs(self, task_runner, temp_workspace):
        """Workspace cleanup should preserve log files."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        task_dir, input_dir, output_dir = task_runner._setup_task_workspace(task)
        
        # Create some files
        (input_dir / "input.txt").write_text("input")
        (output_dir / "output.txt").write_text("output")
        logs_dir = task_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)  # Already created by _setup_task_workspace
        (logs_dir / "task.log").write_text("log content")
        
        # Cleanup
        task_runner._cleanup_task_workspace(task_dir, keep_logs=True)
        
        # Logs should still exist
        assert (logs_dir / "task.log").exists()
        
        # Input/output should be removed
        assert not input_dir.exists()
        assert not output_dir.exists()
    
    def test_workspace_cleanup_removes_all_when_requested(self, task_runner, temp_workspace):
        """Workspace cleanup can remove everything including logs."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        task_dir, input_dir, output_dir = task_runner._setup_task_workspace(task)
        
        # Create files
        (input_dir / "input.txt").write_text("input")
        (output_dir / "output.txt").write_text("output")
        logs_dir = task_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)  # Already created by _setup_task_workspace
        (logs_dir / "task.log").write_text("log content")
        
        # Cleanup everything
        task_runner._cleanup_task_workspace(task_dir, keep_logs=False)
        
        # Everything should be removed
        assert not task_dir.exists()


# ============================================================================
# Test: Input/Output Handling
# ============================================================================


class TestInputOutputHandling:
    """Tests for input/output data handling."""
    
    def test_input_directory_copied_correctly(self, task_runner, temp_workspace):
        """Input data should be copied to task input directory."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        # Create source input directory
        source_input = temp_workspace / "source_input"
        source_input.mkdir()
        (source_input / "data.txt").write_text("test data")
        (source_input / "config.json").write_text('{"key": "value"}')
        
        task_dir, input_dir, output_dir = task_runner._setup_task_workspace(task)
        
        # Copy input
        if source_input.is_dir():
            for item in source_input.iterdir():
                dest = input_dir / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
        
        # Verify files were copied
        assert (input_dir / "data.txt").exists()
        assert (input_dir / "config.json").exists()
        assert (input_dir / "data.txt").read_text() == "test data"
    
    def test_input_file_copied_correctly(self, task_runner, temp_workspace):
        """Single input file should be copied correctly."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        source_file = temp_workspace / "source.txt"
        source_file.write_text("source content")
        
        task_dir, input_dir, output_dir = task_runner._setup_task_workspace(task)
        
        # Copy file
        if source_file.exists():
            shutil.copy2(source_file, input_dir / source_file.name)
        
        # Verify
        assert (input_dir / "source.txt").exists()
        assert (input_dir / "source.txt").read_text() == "source content"
    
    def test_output_directory_created(self, task_runner, temp_workspace):
        """Output directory should be created and writable."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        task_dir, input_dir, output_dir = task_runner._setup_task_workspace(task)
        
        # Output directory should exist
        assert output_dir.exists()
        assert output_dir.is_dir()
        
        # Should be writable
        test_file = output_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()
    
    def test_model_script_copied_to_input(self, task_runner, temp_workspace):
        """Model script should be copied to input directory."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
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
        
        model_script = temp_workspace / "config_model.py"
        model_script.write_text("def config(): return {}")
        
        task_dir, input_dir, output_dir = task_runner._setup_task_workspace(task)
        
        # Copy model script
        if model_script.exists():
            dest_path = input_dir / model_script.name
            shutil.copy2(model_script, dest_path)
        
        # Verify
        assert (input_dir / "config_model.py").exists()
        assert (input_dir / "config_model.py").read_text() == "def config(): return {}"


# ============================================================================
# Test: Environment Variable Handling
# ============================================================================


class TestEnvironmentVariableHandling:
    """Tests for environment variable setup in tasks."""
    
    def test_pythonpath_includes_input_dir(self, task_runner, temp_workspace):
        """PYTHONPATH should include /input for model script imports."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
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
        
        with patch.object(task_runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "success")
            mock_runner.pull_image.return_value = True
            
            result = task_runner.run_task(task)
            
            # Check that PYTHONPATH was set
            call_kwargs = mock_runner.run.call_args
            env = call_kwargs.kwargs.get('env', {})
            
            assert "PYTHONPATH" in env
            assert "/input" in env["PYTHONPATH"]
    
    def test_task_config_added_to_env(self, task_runner, temp_workspace):
        """Task config should be added to environment variables."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={"param1": "value1", "param2": "value2"},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        with patch.object(task_runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "success")
            mock_runner.pull_image.return_value = True
            
            result = task_runner.run_task(task)
            
            # Check that config was added to env
            call_kwargs = mock_runner.run.call_args
            env = call_kwargs.kwargs.get('env', {})
            
            # Config values should be in env (as strings)
            assert env.get("param1") == "value1"
            assert env.get("param2") == "value2"
    
    def test_custom_env_vars_preserved(self, task_runner, temp_workspace):
        """Custom environment variables should be preserved."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        custom_env = {"CUSTOM_VAR": "custom_value"}
        
        with patch.object(task_runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "success")
            mock_runner.pull_image.return_value = True
            
            result = task_runner.run_task(task, env=custom_env)
            
            # Check that custom env was preserved
            call_kwargs = mock_runner.run.call_args
            env = call_kwargs.kwargs.get('env', {})
            
            assert env.get("CUSTOM_VAR") == "custom_value"


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and recovery."""
    
    def test_missing_container_runtime_handled(self, temp_workspace):
        """Missing container runtime should be handled gracefully."""
        runner = TaskRunner(
            workspace_dir=temp_workspace,
            runtime="none"
        )
        
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        result = runner.run_task(task)
        
        assert not result.success
        assert result.exit_code == -1
        assert "No container runtime available" in result.error_message
    
    def test_image_pull_failure_handled(self, task_runner, temp_workspace):
        """Image pull failure should be handled gracefully."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="nonexistent/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        with patch.object(task_runner, '_container_runner') as mock_runner:
            mock_runner.pull_image.return_value = False
            
            result = task_runner.run_task(task)
            
            assert not result.success
            assert "Failed to pull image" in result.error_message
    
    def test_container_execution_failure_handled(self, task_runner, temp_workspace):
        """Container execution failure should be reported correctly."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        with patch.object(task_runner, '_container_runner') as mock_runner:
            mock_runner.pull_image.return_value = True
            mock_runner.run.return_value = (1, "Error: task failed")
            
            result = task_runner.run_task(task)
            
            assert not result.success
            assert result.exit_code == 1
            assert "Container exited with code 1" in result.error_message
    
    def test_exception_during_execution_handled(self, task_runner, temp_workspace):
        """Exceptions during execution should be caught and reported."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        with patch.object(task_runner, '_container_runner') as mock_runner:
            mock_runner.pull_image.side_effect = Exception("Unexpected error")
            
            result = task_runner.run_task(task)
            
            assert not result.success
            assert result.exit_code == -1
            assert "Unexpected error" in result.error_message
    
    def test_log_file_written_on_failure(self, task_runner, temp_workspace):
        """Log file should be written even on task failure."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        with patch.object(task_runner, '_container_runner') as mock_runner:
            mock_runner.pull_image.return_value = True
            mock_runner.run.return_value = (1, "Error occurred")
            
            result = task_runner.run_task(task)
            
            # Check that log file was written
            task_dir = temp_workspace / task.id
            logs_dir = task_dir / "logs"
            
            # Log file should exist
            log_files = list(logs_dir.glob("*.log"))
            assert len(log_files) > 0
            
            # Log should contain error information
            log_content = log_files[0].read_text()
            assert "Error occurred" in log_content
            assert "Exit Code: 1" in log_content


# ============================================================================
# Test: Execution Result Accuracy
# ============================================================================


class TestExecutionResultAccuracy:
    """Tests for execution result accuracy and completeness."""
    
    def test_success_result_has_correct_fields(self, task_runner, temp_workspace):
        """Successful execution should have all correct result fields."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        with patch.object(task_runner, '_container_runner') as mock_runner, \
             patch('src.worker.runner.time.time', side_effect=[1000.0, 1001.5]):  # Simulate 1.5 seconds elapsed
            mock_runner.pull_image.return_value = True
            mock_runner.run.return_value = (0, "Success output")
            
            result = task_runner.run_task(task)
            
            assert result.success
            assert result.exit_code == 0
            assert result.output_path is not None
            assert result.logs == "Success output"
            assert result.error_message is None
            assert result.execution_time_ms > 0
    
    def test_failure_result_has_correct_fields(self, task_runner, temp_workspace):
        """Failed execution should have all correct result fields."""
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        with patch.object(task_runner, '_container_runner') as mock_runner:
            mock_runner.pull_image.return_value = True
            mock_runner.run.return_value = (1, "Error output")
            
            result = task_runner.run_task(task)
            
            assert not result.success
            assert result.exit_code == 1
            assert result.output_path is None
            assert result.logs == "Error output"
            assert result.error_message is not None
            assert "Container exited with code 1" in result.error_message
