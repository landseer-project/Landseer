"""
Task runner for executing Landseer pipeline tasks.

This module handles the actual execution of tasks, including:
- Container execution (Docker/Apptainer)
- GPU allocation
- Input/output handling
- Logging and result collection
"""

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..common import get_logger
from .client import TaskInfo

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of a task execution."""
    success: bool
    exit_code: int
    execution_time_ms: int
    output_path: Optional[Path] = None
    logs: str = ""
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


class ContainerRuntime:
    """
    Abstract interface for container runtimes.
    
    Supports both Docker and Apptainer/Singularity.
    """
    
    @staticmethod
    def detect_runtime() -> str:
        """
        Detect which container runtime is available.
        
        Returns:
            'docker', 'apptainer', 'singularity', or 'none'
        """
        # Check for Docker
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "docker"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Check for Apptainer
        try:
            result = subprocess.run(
                ["apptainer", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "apptainer"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Check for Singularity
        try:
            result = subprocess.run(
                ["singularity", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "singularity"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        return "none"
    
    @staticmethod
    def is_available() -> bool:
        """Check if any container runtime is available."""
        return ContainerRuntime.detect_runtime() != "none"


class DockerRunner:
    """
    Docker container runner.
    
    Handles pulling images, running containers with proper mounts,
    GPU support, and log collection.
    """
    
    def __init__(
        self,
        workspace_dir: Path,
        gpu_id: Optional[int] = None,
        timeout: int = 3600  # 1 hour default
    ):
        """
        Initialize Docker runner.
        
        Args:
            workspace_dir: Directory for task workspaces
            gpu_id: GPU ID to use (None for CPU-only)
            timeout: Container execution timeout in seconds
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_id = gpu_id
        self.timeout = timeout
    
    def pull_image(self, image: str) -> bool:
        """
        Pull a Docker image if not present.
        
        Args:
            image: Docker image name
            
        Returns:
            True if image is available
        """
        logger.info(f"Ensuring image is available: {image}")
        
        try:
            # Check if image exists locally
            result = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.debug(f"Image already present: {image}")
                return True
            
            # Pull the image
            logger.info(f"Pulling image: {image}")
            result = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes for pull
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to pull image {image}: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout pulling image: {image}")
            return False
        except Exception as e:
            logger.error(f"Error pulling image {image}: {e}")
            return False
    
    def run(
        self,
        image: str,
        command: str,
        input_dir: Path,
        output_dir: Path,
        env: Optional[Dict[str, str]] = None,
        extra_mounts: Optional[Dict[str, str]] = None,
        model_script_path: Optional[Path] = None
    ) -> Tuple[int, str, str]:
        """
        Run a container.
        
        Args:
            image: Docker image name
            command: Command to run in container
            input_dir: Input directory to mount
            output_dir: Output directory to mount
            env: Environment variables
            extra_mounts: Additional volume mounts (host_path -> container_path)
            model_script_path: Path to model config script to mount in container
            
        Returns:
            Tuple of (exit_code, logs, docker_command)
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build docker command
        docker_cmd = ["docker", "run", "--rm"]
        
        # Add GPU support if available
        # Use --runtime=nvidia with NVIDIA_VISIBLE_DEVICES env var (compatible with CDI mode)
        # This approach avoids conflicts with CDI mode configuration
        if self.gpu_id is not None:
            # Try multiple methods to detect nvidia runtime availability
            nvidia_runtime_available = False
            
            # Method 1: Check docker info for runtimes (try JSON format first, then string)
            try:
                # Try JSON format first (more reliable)
                result = subprocess.run(
                    ["docker", "info", "--format", "{{json .Runtimes}}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    import json
                    try:
                        runtimes = json.loads(result.stdout.strip())
                        nvidia_runtime_available = "nvidia" in runtimes
                        if nvidia_runtime_available:
                            logger.debug(f"nvidia runtime found in Docker runtimes: {list(runtimes.keys())}")
                        else:
                            logger.debug(f"nvidia runtime not in Docker runtimes. Available: {list(runtimes.keys())}")
                    except (json.JSONDecodeError, ValueError):
                        # Fallback: check as string (for older Docker versions)
                        nvidia_runtime_available = "nvidia" in result.stdout.lower()
                        if nvidia_runtime_available:
                            logger.debug("nvidia runtime found (string match)")
                        else:
                            logger.debug(f"nvidia runtime not found. Output: {result.stdout[:200]}")
                elif result.returncode == 0:
                    # Empty output - try alternative format
                    result2 = subprocess.run(
                        ["docker", "info"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result2.returncode == 0:
                        nvidia_runtime_available = "nvidia" in result2.stdout.lower() or "runtime.*nvidia" in result2.stdout.lower()
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                logger.debug(f"Could not check Docker runtimes via docker info: {e}")
            
            # Method 2: Try to test if nvidia runtime works by checking if it's executable
            if not nvidia_runtime_available:
                try:
                    result = subprocess.run(
                        ["which", "nvidia-container-runtime"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        logger.debug("nvidia-container-runtime found in PATH, assuming runtime is available")
                        nvidia_runtime_available = True
                except Exception:
                    pass
            
            # Use nvidia runtime if available, otherwise log warning but proceed
            # (Docker might still work if runtime is configured at daemon level)
            if nvidia_runtime_available:
                docker_cmd.extend(["--runtime=nvidia"])
                logger.debug(f"Using --runtime=nvidia for GPU access")
            else:
                logger.warning(
                    f"⚠️  nvidia runtime not detected in Docker configuration. "
                    f"GPU access may fail. "
                    f"If GPU access fails, check:\n"
                    f"  1. nvidia-container-runtime is installed\n"
                    f"  2. Docker daemon is configured with nvidia runtime\n"
                    f"  3. Docker daemon is restarted after configuration\n"
                    f"  Check: docker info | grep -i runtime"
                )
                # Still try to use it - might work if runtime is configured at daemon level
                docker_cmd.extend(["--runtime=nvidia"])
        else:
            logger.warning(f"DockerRunner.run() called with gpu_id=None - GPU will NOT be available in container!")
        
        # Add volume mounts
        # - /input: current task's working input directory (latest dataset/model/additional files)
        # - /output: where the tool should write its outputs
        # - /data: canonical path for tools that expect dataset/model under /data
        docker_cmd.extend([
            "-v", f"{input_dir.absolute()}:/input:ro",
            "-v", f"{output_dir.absolute()}:/output:rw",
            "-v", f"{input_dir.absolute()}:/data:ro",
        ])
        
        # Mount model config script directly to /app/ where containers expect it
        # This is critical: containers import "from config_model import config"
        # and expect config_model.py to be in /app/ alongside main.py
        if model_script_path and model_script_path.exists():
            docker_cmd.extend([
                "-v", f"{model_script_path.absolute()}:/app/{model_script_path.name}:ro"
            ])
            logger.debug(f"Mounting model script: {model_script_path} -> /app/{model_script_path.name}")
        
        # Add extra mounts
        if extra_mounts:
            for host_path, container_path in extra_mounts.items():
                docker_cmd.extend(["-v", f"{host_path}:{container_path}:ro"])
        
        # Add environment variables
        if env:
            for key, value in env.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
        
        # Add common environment variables
        docker_cmd.extend([
            "-e", "INPUT_DIR=/input",
            "-e", "OUTPUT_DIR=/output",
            # Evaluator/tool containers in this repo commonly use WORKSPACE=/workspace
            # and then read/write under $WORKSPACE/input and $WORKSPACE/output.
            # We mount host input/output at /input and /output, so set WORKSPACE=/
            # to make $WORKSPACE/input == /input and $WORKSPACE/output == /output.
            "-e", "WORKSPACE=/",
        ])
        
        # Add GPU-related environment variables.
        # IMPORTANT for this host's configuration (CDI + nvidia runtime):
        # - GPUs only become visible in the container when NVIDIA_VISIBLE_DEVICES=all
        # - We still use CUDA_VISIBLE_DEVICES to pin a specific logical device for the tool
        if self.gpu_id is not None:
            docker_cmd.extend([
                "-e",
                "NVIDIA_VISIBLE_DEVICES=all",
                "-e",
                "NVIDIA_DRIVER_CAPABILITIES=all",
                "-e",
                f"CUDA_VISIBLE_DEVICES={self.gpu_id}",
            ])
        
        # Add image and command
        docker_cmd.append(image)
        if command:
            docker_cmd.extend(command.split())
        
        # Log the full command for debugging
        cmd_str = ' '.join(docker_cmd)
        logger.info(f"Running Docker command: {cmd_str}")
        logger.debug(f"Full Docker command: {cmd_str}")
        
        # Log GPU status
        if self.gpu_id is not None:
            logger.info(
                f"GPU support enabled: device={self.gpu_id}, "
                f"NVIDIA_VISIBLE_DEVICES=all, CUDA_VISIBLE_DEVICES={self.gpu_id}, runtime=nvidia"
            )
        else:
            logger.warning("GPU support DISABLED - gpu_id is None. Containers will NOT have GPU access!")
        
        # Use Popen + streaming readers so the container is never blocked on full stdout/stderr
        # pipes. With capture_output=True, heavy tool output (e.g. XGBoost) can fill the pipe
        # and deadlock; streaming avoids that and matches "manual docker run" behavior.
        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []
        chunk_lock = threading.Lock()

        def read_pipe(pipe: Any, out_chunks: List[str], label: str) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    with chunk_lock:
                        out_chunks.append(line)
            except (ValueError, OSError, BrokenPipeError):
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        try:
            proc = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                bufsize=1,  # line-buffered
            )
            t_stdout = threading.Thread(target=read_pipe, args=(proc.stdout, stdout_chunks, "stdout"))
            t_stderr = threading.Thread(target=read_pipe, args=(proc.stderr, stderr_chunks, "stderr"))
            t_stdout.daemon = True
            t_stderr.daemon = True
            t_stdout.start()
            t_stderr.start()

            start = time.monotonic()
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed >= self.timeout:
                    proc.kill()
                    proc.wait()
                    t_stdout.join(timeout=2.0)
                    t_stderr.join(timeout=2.0)
                    logger.error(f"Container execution timed out after {self.timeout}s")
                    logs = "".join(stdout_chunks) + "".join(stderr_chunks)
                    return -1, f"Timeout after {self.timeout} seconds", cmd_str
                time.sleep(1)
            t_stdout.join(timeout=5.0)
            t_stderr.join(timeout=5.0)
            logs = "".join(stdout_chunks) + "".join(stderr_chunks)
            returncode = proc.returncode

            # Check for GPU-related errors and provide helpful diagnostics
            if returncode != 0 and self.gpu_id is not None:
                error_lower = logs.lower()
                if "no cuda gpus are available" in error_lower or ("cuda" in error_lower and "not available" in error_lower):
                    logger.error(
                        f"❌ GPU ACCESS FAILED in container!\n"
                        f"   GPU ID: {self.gpu_id}\n"
                        f"   Runtime: nvidia\n"
                        f"   NVIDIA_VISIBLE_DEVICES: {self.gpu_id}\n"
                        f"   CUDA_VISIBLE_DEVICES: {self.gpu_id}\n"
                        f"   Docker Command: {cmd_str[:300]}...\n"
                        f"   This may indicate:\n"
                        f"   1. nvidia-container-runtime not properly installed/configured\n"
                        f"   2. Docker daemon not configured for GPU access\n"
                        f"   3. GPU {self.gpu_id} not accessible to Docker\n"
                        f"   4. nvidia runtime not available (check: docker info | grep -i runtime)\n"
                        f"   Container output: {logs[:500]}"
                    )
            return returncode, logs, cmd_str

        except Exception as e:
            logger.error(f"Container execution failed: {e}")
            return -1, str(e), cmd_str


class ApptainerRunner:
    """
    Apptainer/Singularity container runner.
    
    Similar to DockerRunner but for HPC environments.
    """
    
    def __init__(
        self,
        workspace_dir: Path,
        gpu_id: Optional[int] = None,
        timeout: int = 7200,
        runtime: str = "apptainer"
    ):
        """
        Initialize Apptainer runner.
        
        Args:
            workspace_dir: Directory for task workspaces
            gpu_id: GPU ID to use (None for CPU-only)
            timeout: Container execution timeout in seconds
            runtime: Runtime to use ('apptainer' or 'singularity')
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_id = gpu_id
        self.timeout = timeout
        self.runtime = runtime
    
    def convert_docker_image(self, docker_image: str) -> str:
        """
        Convert Docker image reference to Apptainer format.
        
        Args:
            docker_image: Docker image name (e.g., ghcr.io/org/image:tag)
            
        Returns:
            Apptainer image reference
        """
        # Apptainer can pull directly from Docker registries
        return f"docker://{docker_image}"
    
    def run(
        self,
        image: str,
        command: str,
        input_dir: Path,
        output_dir: Path,
        env: Optional[Dict[str, str]] = None,
        extra_mounts: Optional[Dict[str, str]] = None,
        model_script_path: Optional[Path] = None
    ) -> Tuple[int, str, str]:
        """
        Run a container using Apptainer/Singularity.
        
        Args:
            image: Docker image name (will be converted)
            command: Command to run in container
            input_dir: Input directory to mount
            output_dir: Output directory to mount
            env: Environment variables
            extra_mounts: Additional volume mounts
            model_script_path: Path to model config script to mount in container
            
        Returns:
            Tuple of (exit_code, logs, command_string)
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert image reference
        apptainer_image = self.convert_docker_image(image)
        
        # Build command
        apptainer_cmd = [self.runtime, "exec"]
        
        # Add GPU support if available
        if self.gpu_id is not None:
            apptainer_cmd.append("--nv")  # NVIDIA GPU support
        else:
            logger.warning(f"ApptainerRunner.run() called with gpu_id=None - GPU will NOT be available in container!")
        
        # Add volume mounts
        apptainer_cmd.extend([
            "--bind", f"{input_dir.absolute()}:/input:ro",
            "--bind", f"{output_dir.absolute()}:/output:rw"
        ])
        
        # Mount model config script directly to /app/ where containers expect it
        if model_script_path and model_script_path.exists():
            apptainer_cmd.extend([
                "--bind", f"{model_script_path.absolute()}:/app/{model_script_path.name}:ro"
            ])
            logger.debug(f"Mounting model script: {model_script_path} -> /app/{model_script_path.name}")
        
        # Add extra mounts
        if extra_mounts:
            for host_path, container_path in extra_mounts.items():
                apptainer_cmd.extend(["--bind", f"{host_path}:{container_path}"])
        
        # Set environment variables
        env_vars = env or {}
        env_vars["INPUT_DIR"] = "/input"
        env_vars["OUTPUT_DIR"] = "/output"
        # See DockerRunner comment above: align /workspace/{input,output} defaults
        # to our mounted /input and /output paths.
        env_vars["WORKSPACE"] = "/"
        if self.gpu_id is not None:
            env_vars["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
        # Add image and command
        apptainer_cmd.append(apptainer_image)
        if command:
            apptainer_cmd.extend(command.split())
        
        cmd_str = ' '.join(apptainer_cmd)
        logger.debug(f"Running Apptainer command: {cmd_str}")
        
        # Log GPU status
        if self.gpu_id is not None:
            logger.info(f"GPU support enabled: device={self.gpu_id}, CUDA_VISIBLE_DEVICES={self.gpu_id}")
        else:
            logger.warning("GPU support DISABLED - gpu_id is None. Containers will NOT have GPU access!")
        
        try:
            # Set environment for the subprocess
            run_env = os.environ.copy()
            run_env.update(env_vars)
            
            result = subprocess.run(
                apptainer_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=run_env
            )
            
            logs = result.stdout + result.stderr
            return result.returncode, logs, cmd_str
            
        except subprocess.TimeoutExpired:
            logger.error(f"Container execution timed out after {self.timeout}s")
            return -1, f"Timeout after {self.timeout} seconds", cmd_str
        except Exception as e:
            logger.error(f"Container execution failed: {e}")
            return -1, str(e), cmd_str


class TaskRunner:
    """
    Main task runner that orchestrates task execution.
    
    Handles:
    - Selecting appropriate container runtime
    - Setting up workspace directories
    - Executing tasks
    - Collecting results
    """
    
    def __init__(
        self,
        workspace_dir: Path,
        artifact_cache_dir: Optional[Path] = None,
        gpu_id: Optional[int] = None,
        timeout: int = 7200,
        runtime: Optional[str] = None
    ):
        """
        Initialize the task runner.
        
        Args:
            workspace_dir: Base directory for task workspaces
            artifact_cache_dir: Directory for artifact caching
            gpu_id: GPU ID to use (None for CPU-only)
            timeout: Task execution timeout in seconds
            runtime: Container runtime to use (auto-detect if None)
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.artifact_cache_dir = Path(artifact_cache_dir) if artifact_cache_dir else None
        self.gpu_id = gpu_id
        self.timeout = timeout
        
        # Auto-detect or use specified runtime
        if runtime:
            self.runtime = runtime
        else:
            self.runtime = ContainerRuntime.detect_runtime()
        
        if self.runtime == "none":
            logger.warning("No container runtime detected. Task execution may fail.") # TODO: This should be an error and the worker should not start and should communicate the error to the host
        else:
            logger.info(f"Using container runtime: {self.runtime}")
        
        # Initialize appropriate runner
        self._init_container_runner()
    
    def _init_container_runner(self) -> None:
        """Initialize the container runner based on detected runtime."""
        if self.runtime == "docker":
            self._container_runner = DockerRunner(
                workspace_dir=self.workspace_dir,
                gpu_id=self.gpu_id,
                timeout=self.timeout
            )
        elif self.runtime in ("apptainer", "singularity"):
            self._container_runner = ApptainerRunner(
                workspace_dir=self.workspace_dir,
                gpu_id=self.gpu_id,
                timeout=self.timeout,
                runtime=self.runtime
            )
        else:
            self._container_runner = None
    
    def _setup_task_workspace(self, task: TaskInfo) -> Tuple[Path, Path, Path]:
        """
        Set up workspace directories for a task.
        
        Args:
            task: Task to set up workspace for
            
        Returns:
            Tuple of (task_dir, input_dir, output_dir)
        """
        task_dir = self.workspace_dir / task.id
        input_dir = task_dir / "input"
        output_dir = task_dir / "output"
        logs_dir = task_dir / "logs"
        
        # Clean up old input/output directories on re-runs (but keep logs)
        for d in [input_dir, output_dir]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        
        # Create directories
        for d in [task_dir, input_dir, output_dir, logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        return task_dir, input_dir, output_dir
    
    def _cleanup_task_workspace(self, task_dir: Path, keep_logs: bool = True) -> None:
        """
        Clean up task workspace after execution.
        
        Args:
            task_dir: Task directory to clean up
            keep_logs: Whether to preserve log files
        """
        if not task_dir.exists():
            return
        
        if keep_logs:
            # Only remove input/output directories, keep logs
            for subdir in ["input", "output"]:
                subdir_path = task_dir / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path, ignore_errors=True)
        else:
            # Remove entire task directory
            shutil.rmtree(task_dir, ignore_errors=True)
    
    def _write_task_log(
        self,
        task_dir: Path,
        task: TaskInfo,
        result: ExecutionResult,
        docker_command: Optional[str] = None
    ) -> None:
        """
        Write task execution log.
        
        Args:
            task_dir: Task directory
            task: Task information
            result: Execution result
            docker_command: Optional Docker command that was executed
        """
        logs_dir = task_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_dir / f"{task.task_type}_{task.tool_name.replace(' ', '_')}.log"
        
        log_content = [
            f"Task ID: {task.id}",
            f"Tool: {task.tool_name}",
            f"Type: {task.task_type}",
            f"Image: {task.tool_image}",
            f"Command: {task.tool_command}",
            f"Exit Code: {result.exit_code}",
            f"Execution Time: {result.execution_time_ms}ms",
            f"Success: {result.success}",
        ]
        
        # Add GPU information
        if self.gpu_id is not None:
            log_content.append(f"GPU ID: {self.gpu_id}")
        else:
            log_content.append("GPU ID: None (CPU only)")
        
        # Add Docker command if available
        if docker_command:
            log_content.extend([
                "",
                "=== Docker Command ===",
                docker_command,
            ])
        
        log_content.extend([
            "",
            "=== Container Output ===",
            result.logs,
            ""
        ])
        
        if result.error_message:
            log_content.extend([
                "=== Error ===",
                result.error_message,
                ""
            ])
        
        log_file.write_text("\n".join(log_content))
    
    def run_task(
        self,
        task: TaskInfo,
        input_path: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        extra_mounts: Optional[Dict[str, str]] = None,
        model_script_path: Optional[Path] = None,
        dependency_outputs: Optional[Dict[str, Path]] = None
    ) -> ExecutionResult:
        """
        Execute a task.
        
        Args:
            task: Task to execute
            input_path: Path to input data (optional)
            env: Additional environment variables
            extra_mounts: Additional volume mounts
            model_script_path: Path to model config script (e.g., config_model.py)
            dependency_outputs: Dict mapping dependency task IDs to their output directories
            
        Returns:
            ExecutionResult with execution details
        """
        logger.info(f"Starting task execution: {task.id} ({task.tool_name})")
        
        start_time = time.time()
        
        # Check if container runtime is available
        if self._container_runner is None:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                execution_time_ms=0,
                error_message=f"No container runtime available (detected: {self.runtime})"
            )
        
        # Setup workspace
        task_dir, input_dir, output_dir = self._setup_task_workspace(task)
        
        try:
            # Copy/link input data if provided
            if input_path and input_path.exists():
                if input_path.is_dir():
                    # Copy or symlink input directory contents
                    for item in input_path.iterdir():
                        dest = input_dir / item.name
                        if item.is_file():
                            shutil.copy2(item, dest)
                        elif item.is_dir():
                            # Use dirs_exist_ok=True to handle re-runs
                            shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(input_path, input_dir / input_path.name)
            
            # Copy outputs from dependent tasks to input directory
            # This is critical for artifact chaining (e.g., model.pt from in_training -> post_training)
            if dependency_outputs:
                for dep_task_id, dep_output_path in dependency_outputs.items():
                    if dep_output_path and dep_output_path.exists():
                        logger.info(f"Copying outputs from dependency {dep_task_id} to input directory")
                        if dep_output_path.is_dir():
                            # Copy all files from dependency output directory
                            for item in dep_output_path.iterdir():
                                dest = input_dir / item.name
                                if item.is_file():
                                    shutil.copy2(item, dest)
                                    logger.debug(f"Copied {item.name} from dependency {dep_task_id}")
                                elif item.is_dir():
                                    shutil.copytree(item, dest, dirs_exist_ok=True)
                                    logger.debug(f"Copied directory {item.name} from dependency {dep_task_id}")
                        else:
                            # Single file output
                            shutil.copy2(dep_output_path, input_dir / dep_output_path.name)
                            logger.debug(f"Copied {dep_output_path.name} from dependency {dep_task_id}")
                    else:
                        logger.warning(f"Dependency {dep_task_id} output not found at {dep_output_path}")
            
            # Copy model script to input directory if provided
            # This allows containers to import config_model
            if model_script_path and model_script_path.exists():
                dest_path = input_dir / model_script_path.name
                shutil.copy2(model_script_path, dest_path)
                logger.debug(f"Copied model script to: {dest_path}")
            
            # Build environment
            task_env = env or {}
            task_env.update(task.config)  # Add task config to environment
            
            # Add PYTHONPATH to include /input so containers can import config_model
            # This is a fallback in case the direct mount to /app/ doesn't work
            # Prepend /input to existing PYTHONPATH if any
            existing_pythonpath = task_env.get("PYTHONPATH", "")
            pythonpath_parts = ["/input", "/app"]
            if existing_pythonpath:
                pythonpath_parts.append(existing_pythonpath)
            task_env["PYTHONPATH"] = ":".join(pythonpath_parts)
            logger.debug(f"Set PYTHONPATH={task_env['PYTHONPATH']}")
            
            # Pull image if using Docker
            if self.runtime == "docker":
                if not self._container_runner.pull_image(task.tool_image):
                    return ExecutionResult(
                        success=False,
                        exit_code=-1,
                        execution_time_ms=int((time.time() - start_time) * 1000),
                        error_message=f"Failed to pull image: {task.tool_image}"
                    )
            
            # Run the container
            # Handle both old (exit_code, logs) and new (exit_code, logs, docker_command) return formats
            container_result = self._container_runner.run(
                image=task.tool_image,
                command=task.tool_command,
                input_dir=input_dir,
                output_dir=output_dir,
                env=task_env,
                extra_mounts=extra_mounts,
                model_script_path=model_script_path
            )
            
            # Unpack result (handle both formats for backward compatibility)
            if len(container_result) == 3:
                exit_code, logs, docker_command = container_result
            else:
                exit_code, logs = container_result
                docker_command = None
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Create result
            success = exit_code == 0
            result = ExecutionResult(
                success=success,
                exit_code=exit_code,
                execution_time_ms=execution_time_ms,
                output_path=output_dir if success else None,
                logs=logs,
                error_message=None if success else f"Container exited with code {exit_code}"
            )
            
            # Write log with Docker command
            self._write_task_log(task_dir, task, result, docker_command=docker_command)
            
            if success:
                logger.info(f"Task {task.id} completed successfully in {execution_time_ms}ms")
            else:
                logger.error(f"Task {task.id} failed with exit code {exit_code}")
            
            return result
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.exception(f"Task {task.id} execution error: {e}")
            
            result = ExecutionResult(
                success=False,
                exit_code=-1,
                execution_time_ms=execution_time_ms,
                error_message=str(e)
            )
            
            self._write_task_log(task_dir, task, result, docker_command=None)
            return result
    
    def run_task_with_cache(
        self,
        task: TaskInfo,
        cache_db: "ArtifactCacheDB",
        input_path: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        parent_hashes: Optional[List[str]] = None
    ) -> ExecutionResult:
        """
        Execute a task with artifact caching.
        
        Checks cache before execution and stores results after successful execution.
        
        Args:
            task: Task to execute
            cache_db: Artifact cache database
            input_path: Path to input data
            env: Additional environment variables
            parent_hashes: Parent artifact hashes for cache key computation
            
        Returns:
            ExecutionResult with execution details
        """
        # Compute cache key
        cache_key = cache_db.compute_task_hash(task, parent_hashes or [])
        
        # Check cache
        cached_path = cache_db.get_cached_artifact(cache_key)
        if cached_path and cached_path.exists():
            logger.info(f"Cache hit for task {task.id}: {cache_key[:12]}")
            return ExecutionResult(
                success=True,
                exit_code=0,
                execution_time_ms=0,
                output_path=cached_path,
                artifacts={"cache_hit": True, "cache_key": cache_key}
            )
        
        # Cache miss - execute task
        logger.info(f"Cache miss for task {task.id}: {cache_key[:12]}")
        result = self.run_task(task, input_path, env)
        
        # Store in cache if successful
        if result.success and result.output_path:
            cache_db.store_artifact(cache_key, result.output_path, task)
            result.artifacts["cache_key"] = cache_key
            result.artifacts["cache_hit"] = False
        
        return result
