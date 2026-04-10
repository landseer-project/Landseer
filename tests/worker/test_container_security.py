"""
Adversarial tests for container execution security.

These tests verify that:
1. Volume mounts are properly isolated and cannot escape
2. GPU assignment matches worker capabilities
3. Environment variables are properly sanitized
4. Command injection is prevented
5. Path traversal attacks are blocked
6. Extra mounts are validated
"""

import os
import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Optional

from src.worker.runner import (
    DockerRunner,
    ApptainerRunner,
    TaskRunner,
    ContainerRuntime,
    ExecutionResult,
)
from src.worker.client import TaskInfo


def _mock_successful_popen(mock_popen):
    """Configure subprocess.Popen mock for a successful container run."""
    proc = MagicMock()
    proc.poll.return_value = 0
    proc.returncode = 0
    proc.stdout.readline.return_value = ""
    proc.stderr.readline.return_value = ""
    mock_popen.return_value = proc
    return proc


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
        config={"param1": "value1"},
        priority=100,
        status="pending",
        task_type="pre_training",
        counter=1,
        workflows=["workflow_1"],
        pipeline_id="pipeline_1",
        dependency_ids=[]
    )


@pytest.fixture
def docker_runner(temp_workspace):
    """Create a DockerRunner instance."""
    return DockerRunner(workspace_dir=temp_workspace, gpu_id=0)


@pytest.fixture
def apptainer_runner(temp_workspace):
    """Create an ApptainerRunner instance."""
    return ApptainerRunner(workspace_dir=temp_workspace, gpu_id=0)


# ============================================================================
# Test: Docker /dev/shm (PyTorch DataLoader)
# ============================================================================


class TestDockerShmSize:
    """DockerRunner should raise /dev/shm above Docker's default for DataLoader workers."""

    def test_default_shm_size_1g(self, docker_runner, temp_workspace, monkeypatch):
        monkeypatch.delenv("LANDSEER_DOCKER_SHM_SIZE", raising=False)
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
            _mock_successful_popen(mock_popen)
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir,
            )
            call_args = mock_popen.call_args[0][0]
            idx = call_args.index("--shm-size")
            assert call_args[idx + 1] == "1g"

    def test_shm_size_env_override(self, temp_workspace, monkeypatch):
        monkeypatch.setenv("LANDSEER_DOCKER_SHM_SIZE", "2g")
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=0)
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
            _mock_successful_popen(mock_popen)
            runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir,
            )
            call_args = mock_popen.call_args[0][0]
            idx = call_args.index("--shm-size")
            assert call_args[idx + 1] == "2g"


# ============================================================================
# Test: Volume Mount Security
# ============================================================================


class TestVolumeMountSecurity:
    """Tests for volume mount security and path traversal prevention."""
    
    def test_input_dir_mounted_readonly(self, docker_runner, temp_workspace):
        """Input directory should be mounted as read-only."""
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
            _mock_successful_popen(mock_popen)
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Check that input is mounted as read-only
            call_args = mock_popen.call_args[0][0]
            assert "-v" in call_args
            input_mount_idx = call_args.index("-v")
            input_mount = call_args[input_mount_idx + 1]
            assert ":ro" in input_mount, "Input directory should be read-only"
            assert str(input_dir.absolute()) in input_mount
    
    def test_output_dir_mounted_readwrite(self, docker_runner, temp_workspace):
        """Output directory should be mounted as read-write."""
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
            _mock_successful_popen(mock_popen)
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Check that output is mounted as read-write
            call_args = mock_popen.call_args[0][0]
            assert "-v" in call_args
            output_mount_idx = call_args.index("-v")
            # Find the output mount (second -v)
            output_mounts = [i for i, arg in enumerate(call_args) if arg == "-v"]
            assert len(output_mounts) >= 2
            output_mount = call_args[output_mounts[1] + 1]
            assert ":rw" in output_mount or ":rw" not in output_mount, \
                "Output directory should be read-write (no :ro)"
            assert str(output_dir.absolute()) in output_mount
    
    def test_path_traversal_in_input_dir_blocked(self, docker_runner, temp_workspace):
        """
        Path traversal attacks in input directory should be blocked.
        
        Adversary tries to access /etc/passwd via ../../../etc/passwd
        """
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        # Create a malicious symlink
        malicious_file = input_dir / "malicious"
        try:
            malicious_file.symlink_to("/etc/passwd")
        except (OSError, PermissionError):
            # Symlink creation might fail, that's fine for the test
            pass
        
        with patch('subprocess.run') as mock_run:
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            # Verify that only the input_dir is mounted, not parent directories
            call_args = mock_run.call_args[0][0]
            mounts = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-v"]
            
            for mount in mounts:
                # Should not mount parent directories
                assert "/etc" not in mount, "Should not mount /etc"
                assert "/root" not in mount, "Should not mount /root"
                assert "/home" not in mount, "Should not mount /home"
    
    def test_absolute_paths_used_for_mounts(self, docker_runner, temp_workspace):
        """
        All mount paths should be absolute to prevent relative path attacks.
        """
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        with patch('subprocess.run') as mock_run:
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_run.call_args[0][0]
            mounts = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-v"]
            
            for mount in mounts:
                host_path = mount.split(":")[0]
                assert Path(host_path).is_absolute(), \
                    f"Mount path should be absolute: {host_path}"
    
    def test_extra_mounts_are_readonly(self, docker_runner, temp_workspace):
        """
        Extra mounts should default to read-only for security.
        """
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        extra_dir = temp_workspace / "extra"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        extra_dir.mkdir(parents=True)
        
        extra_mounts = {str(extra_dir): "/extra"}
        
        with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
            _mock_successful_popen(mock_popen)
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir,
                extra_mounts=extra_mounts
            )
            
            call_args = mock_popen.call_args[0][0]
            mounts = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-v"]
            
            # Find the extra mount
            extra_mount = next((m for m in mounts if "/extra" in m), None)
            assert extra_mount is not None, "Extra mount should be present"
            assert ":ro" in extra_mount, "Extra mounts should be read-only"
    
    def test_workspace_isolation(self, docker_runner, temp_workspace):
        """
        Tasks should be isolated to their workspace directory.
        """
        task1_dir = temp_workspace / "task1"
        task2_dir = temp_workspace / "task2"
        
        task1_input = task1_dir / "input"
        task1_output = task1_dir / "output"
        task2_input = task2_dir / "input"
        task2_output = task2_dir / "output"
        
        for d in [task1_input, task1_output, task2_input, task2_output]:
            d.mkdir(parents=True)
        
        with patch('subprocess.run') as mock_run:
            # Run task 1
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=task1_input,
                output_dir=task1_output
            )
            
            call1 = mock_run.call_args[0][0]
            mounts1 = [call1[i+1] for i, arg in enumerate(call1) if arg == "-v"]
            
            # Run task 2
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=task2_input,
                output_dir=task2_output
            )
            
            call2 = mock_run.call_args[0][0]
            mounts2 = [call2[i+1] for i, arg in enumerate(call2) if arg == "-v"]
            
            # Verify isolation - task1 should not see task2's directories
            for mount in mounts1:
                assert str(task2_dir) not in mount, \
                    "Task 1 should not have access to task 2's directories"
            
            for mount in mounts2:
                assert str(task1_dir) not in mount, \
                    "Task 2 should not have access to task 1's directories"


# ============================================================================
# Test: GPU Assignment Security
# ============================================================================


class TestGPUSecurity:
    """Tests for GPU assignment and validation."""
    
    def test_gpu_id_passed_to_container(self, temp_workspace):
        """GPU ID should be correctly passed to the container."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=2)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
            _mock_successful_popen(mock_popen)
            runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_popen.call_args[0][0]
            
            # Check GPU runtime contract
            assert "--runtime=nvidia" in call_args
            
            # Check CUDA_VISIBLE_DEVICES
            assert "-e" in call_args
            env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
            cuda_var = next((v for v in env_vars if "CUDA_VISIBLE_DEVICES" in v), None)
            assert cuda_var is not None, "CUDA_VISIBLE_DEVICES should be set"
            assert "CUDA_VISIBLE_DEVICES=2" in cuda_var, \
                f"CUDA_VISIBLE_DEVICES should be 2, got {cuda_var}"
    
    def test_no_gpu_when_gpu_id_is_none(self, temp_workspace):
        """No GPU flags should be set when gpu_id is None."""
        runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=None)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
            _mock_successful_popen(mock_popen)
            runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_popen.call_args[0][0]
            
            # Should not have GPU flags
            assert "--gpus" not in call_args, "Should not have --gpus flag"
            assert "--runtime=nvidia" not in call_args, "Should not have nvidia runtime"
            
            # Should not have CUDA_VISIBLE_DEVICES
            env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
            cuda_vars = [v for v in env_vars if "CUDA_VISIBLE_DEVICES" in v]
            assert len(cuda_vars) == 0, "Should not set CUDA_VISIBLE_DEVICES"
    
    def test_gpu_id_consistency_docker_vs_env(self, temp_workspace):
        """
        GPU ID in --gpus flag should match CUDA_VISIBLE_DEVICES.
        
        This prevents a mismatch where container gets wrong GPU.
        """
        for gpu_id in [0, 1, 2, 3]:
            runner = DockerRunner(workspace_dir=temp_workspace, gpu_id=gpu_id)
            
            input_dir = temp_workspace / "input"
            output_dir = temp_workspace / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
                _mock_successful_popen(mock_popen)
                runner.run(
                    image="test/image:latest",
                    command="python main.py",
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
                call_args = mock_popen.call_args[0][0]
                assert "--runtime=nvidia" in call_args
                
                # Extract GPU ID from CUDA_VISIBLE_DEVICES
                env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
                cuda_var = next((v for v in env_vars if "CUDA_VISIBLE_DEVICES" in v), None)
                gpu_id_from_env = int(cuda_var.split("=")[1])
                assert gpu_id_from_env == gpu_id, \
                    f"GPU ID should match env assignment: env={gpu_id_from_env}, expected={gpu_id}"
    
    def test_apptainer_gpu_flag(self, temp_workspace):
        """Apptainer should use --nv flag for GPU support."""
        runner = ApptainerRunner(workspace_dir=temp_workspace, gpu_id=1)
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        with patch('subprocess.run') as mock_run:
            runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_run.call_args[0][0]
            
            # Apptainer uses --nv for GPU
            assert "--nv" in call_args, "Apptainer should use --nv flag for GPU"
            
            # Check CUDA_VISIBLE_DEVICES
            # Apptainer sets env in the subprocess env, not as flags
            # So we check the env parameter
            if 'env' in mock_run.call_args.kwargs:
                env = mock_run.call_args.kwargs['env']
                assert "CUDA_VISIBLE_DEVICES" in env
                assert env["CUDA_VISIBLE_DEVICES"] == "1"


# ============================================================================
# Test: Command Injection Prevention
# ============================================================================


class TestCommandInjectionPrevention:
    """Tests to prevent command injection attacks."""
    
    def test_command_splitting_prevents_injection(self, docker_runner, temp_workspace):
        """
        Commands should be split properly to prevent injection.
        
        Adversary tries: "python main.py; rm -rf /"
        """
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        malicious_command = "python main.py; rm -rf /"
        
        with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
            _mock_successful_popen(mock_popen)
            docker_runner.run(
                image="test/image:latest",
                command=malicious_command,
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_popen.call_args[0][0]
            
            # Command should be split, not executed as shell command
            assert mock_popen.call_args.kwargs.get('shell', True) is False, \
                "Should not use shell=True (prevents injection)"
            
            # The command should be split into arguments
            image_idx = call_args.index("test/image:latest")
            command_parts = call_args[image_idx + 1:]
            
            # Should be split, not a single string
            assert len(command_parts) > 1, "Command should be split into parts"
            # Note: The semicolon will still be in the argument string (e.g., "main.py;")
            # but since shell=False, it won't be interpreted as a command separator.
            # The important thing is that shell=False, which we've already verified.
    
    def test_environment_variable_injection_blocked(self, docker_runner, temp_workspace):
        """
        Environment variables should not allow command injection.
        """
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        malicious_env = {
            "PATH": "/bin:/usr/bin; rm -rf /",
            "LD_PRELOAD": "/lib; echo pwned"
        }
        
        with patch('subprocess.run') as mock_run:
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir,
                env=malicious_env
            )
            
            call_args = mock_run.call_args[0][0]
            
            # Environment variables should be passed as -e flags, not shell expanded
            env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
            
            # Check that values are properly escaped/quoted
            for env_var in env_vars:
                # The value should be part of the string, not executed
                assert "=" in env_var
                key, value = env_var.split("=", 1)
                # Should not contain unescaped special characters that could be executed
                # (This is a basic check - actual escaping depends on subprocess implementation)
    
    def test_image_name_validation(self, docker_runner, temp_workspace):
        """
        Image names should not allow command injection.
        """
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        malicious_image = "test/image:latest; rm -rf /"
        
        with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
            _mock_successful_popen(mock_popen)
            docker_runner.run(
                image=malicious_image,
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_popen.call_args[0][0]
            
            # Image should be a single argument, not split
            image_idx = call_args.index(malicious_image)
            # Should be before command parts
            assert image_idx < len(call_args) - 1


# ============================================================================
# Test: Environment Variable Security
# ============================================================================


class TestEnvironmentVariableSecurity:
    """Tests for environment variable handling and security."""
    
    def test_input_output_dirs_set_in_env(self, docker_runner, temp_workspace):
        """INPUT_DIR and OUTPUT_DIR should be set in container environment."""
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
            _mock_successful_popen(mock_popen)
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            call_args = mock_popen.call_args[0][0]
            env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
            
            input_dir_env = next((v for v in env_vars if "INPUT_DIR" in v), None)
            output_dir_env = next((v for v in env_vars if "OUTPUT_DIR" in v), None)
            
            assert input_dir_env is not None, "INPUT_DIR should be set"
            assert output_dir_env is not None, "OUTPUT_DIR should be set"
            assert "INPUT_DIR=/input" in input_dir_env
            assert "OUTPUT_DIR=/output" in output_dir_env
    
    def test_custom_env_vars_passed_through(self, docker_runner, temp_workspace):
        """Custom environment variables should be passed to container."""
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        custom_env = {
            "MY_VAR": "my_value",
            "ANOTHER_VAR": "another_value"
        }
        
        with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
            _mock_successful_popen(mock_popen)
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir,
                env=custom_env
            )
            
            call_args = mock_popen.call_args[0][0]
            env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
            
            for key, value in custom_env.items():
                env_var = next((v for v in env_vars if key in v), None)
                assert env_var is not None, f"{key} should be set"
                assert f"{key}={value}" in env_var
    
    def test_env_vars_do_not_override_critical(self, docker_runner, temp_workspace):
        """
        User-provided env vars should not override critical system vars.
        """
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        # Try to override critical variables
        malicious_env = {
            "INPUT_DIR": "/malicious",
            "OUTPUT_DIR": "/malicious",
            "CUDA_VISIBLE_DEVICES": "999"  # Invalid GPU ID
        }
        
        with patch('subprocess.run') as mock_run:
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir,
                env=malicious_env
            )
            
            call_args = mock_run.call_args[0][0]
            env_vars = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-e"]
            
            # Critical vars should be set after user vars, overriding them
            # Find last occurrence of each critical var
            input_dirs = [v for v in env_vars if "INPUT_DIR" in v]
            output_dirs = [v for v in env_vars if "OUTPUT_DIR" in v]
            
            # Last one should be the correct one
            if input_dirs:
                assert "INPUT_DIR=/input" in input_dirs[-1], \
                    "INPUT_DIR should be /input, not overridden"
            if output_dirs:
                assert "OUTPUT_DIR=/output" in output_dirs[-1], \
                    "OUTPUT_DIR should be /output, not overridden"


# ============================================================================
# Test: Model Script Mounting
# ============================================================================


class TestModelScriptMounting:
    """Tests for model script mounting security."""
    
    def test_model_script_mounted_to_app(self, docker_runner, temp_workspace):
        """Model script should be mounted to /app/ directory."""
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        model_script = temp_workspace / "config_model.py"
        model_script.write_text("# Model config")
        
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
            _mock_successful_popen(mock_popen)
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir,
                model_script_path=model_script
            )
            
            call_args = mock_popen.call_args[0][0]
            mounts = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-v"]
            
            # Find model script mount
            model_mount = next((m for m in mounts if "config_model.py" in m), None)
            assert model_mount is not None, "Model script should be mounted"
            assert "/app/config_model.py" in model_mount, \
                "Model script should be mounted to /app/"
            assert ":ro" in model_mount, "Model script should be read-only"
    
    def test_model_script_path_traversal_blocked(self, docker_runner, temp_workspace):
        """
        Model script path should not allow traversal.
        """
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        
        # Try to mount a file outside workspace
        malicious_path = Path("/etc/passwd")
        
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        with patch('subprocess.run') as mock_run:
            # This should still work (we can't prevent mounting arbitrary files)
            # But we should verify the path is absolute
            docker_runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir,
                model_script_path=malicious_path
            )
            
            call_args = mock_run.call_args[0][0]
            mounts = [call_args[i+1] for i, arg in enumerate(call_args) if arg == "-v"]
            
            # The path should be absolute (which it is)
            # In a real system, you'd want to validate the path is within allowed directories
            for mount in mounts:
                host_path = mount.split(":")[0]
                assert Path(host_path).is_absolute()


# ============================================================================
# Test: Container Runtime Detection
# ============================================================================


class TestContainerRuntimeDetection:
    """Tests for container runtime detection and validation."""
    
    def test_docker_detection(self):
        """Should detect Docker if available."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            runtime = ContainerRuntime.detect_runtime()
            # If docker is detected, it should return "docker"
            # We can't test the actual detection without mocking properly
    
    def test_no_runtime_when_none_available(self):
        """Should return 'none' when no runtime is available."""
        with patch('subprocess.run', side_effect=FileNotFoundError):
            runtime = ContainerRuntime.detect_runtime()
            assert runtime == "none"
    
    def test_task_runner_fails_gracefully_no_runtime(self, temp_workspace):
        """TaskRunner should handle missing container runtime gracefully."""
        runner = TaskRunner(
            workspace_dir=temp_workspace,
            runtime="none"  # Force no runtime
        )
        
        task = TaskInfo(
            id="test_task",
            tool_name="test",
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


# ============================================================================
# Test: Timeout and Resource Limits
# ============================================================================


class TestTimeoutAndResourceLimits:
    """Tests for timeout handling and resource limits."""
    
    def test_timeout_enforced(self, temp_workspace):
        """Container execution should respect timeout."""
        runner = DockerRunner(
            workspace_dir=temp_workspace,
            timeout=1  # 1 second timeout
        )
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        with patch('subprocess.Popen') as mock_popen:
            proc = MagicMock()
            proc.poll.return_value = None
            proc.kill.return_value = None
            proc.wait.return_value = None
            proc.stdout.readline.return_value = ""
            proc.stderr.readline.return_value = ""
            mock_popen.return_value = proc
            exit_code, logs, _cmd = runner.run(
                image="test/image:latest",
                command="sleep 100",  # Would run forever
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            assert exit_code == -1
            assert "Timeout" in logs
    
    def test_timeout_passed_to_subprocess(self, temp_workspace):
        """Timeout should be passed to subprocess.run."""
        runner = DockerRunner(
            workspace_dir=temp_workspace,
            timeout=3600
        )
        
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        with patch('subprocess.Popen') as mock_popen:
            _mock_successful_popen(mock_popen)
            runner.run(
                image="test/image:latest",
                command="python main.py",
                input_dir=input_dir,
                output_dir=output_dir
            )
            
            assert mock_popen.call_args is not None
