"""
Command-line interface for Landseer worker.

The worker connects to the Landseer backend scheduler, claims tasks,
executes them in containers, and reports results back.
"""

import argparse
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from ..common import get_logger
from .client import LandseerClient, TaskInfo
from .db import ArtifactCacheDB, CacheManager
from .runner import TaskRunner, ExecutionResult, ContainerRuntime

# Import two-level cache if available
try:
    from ..store import TwoLevelCache, CacheConfig
    TWO_LEVEL_CACHE_AVAILABLE = True
except ImportError:
    TWO_LEVEL_CACHE_AVAILABLE = False
    TwoLevelCache = None
    CacheConfig = None

logger = get_logger(__name__)


class Worker:
    """
    Landseer Worker - executes pipeline tasks.
    
    The worker operates in a loop:
    1. Connect to backend and register
    2. Claim available tasks
    3. Execute tasks in containers
    4. Report results back to backend
    5. Repeat until no more tasks or shutdown
    """
    
    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        worker_id: Optional[str] = None,
        workspace_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        data_path: Optional[Path] = None,
        gpu_id: Optional[int] = None,
        poll_interval: float = 5.0,
        task_timeout: int = 3600,
        heartbeat_interval: float = 30.0,
        use_cache: bool = True,
        runtime: Optional[str] = None
    ):
        """
        Initialize the worker.
        
        Args:
            backend_url: URL of the Landseer backend API
            worker_id: Worker identifier (auto-generated if not provided)
            workspace_dir: Directory for task workspaces
            cache_dir: Directory for artifact caching
            data_path: Path to input data directory (dataset, model config)
            gpu_id: GPU ID to use (None for CPU-only)
            poll_interval: Seconds between task polls when idle
            task_timeout: Task execution timeout in seconds
            heartbeat_interval: Seconds between heartbeats
            use_cache: Whether to use artifact caching
            runtime: Container runtime to use (auto-detect if None)
        """
        self.backend_url = backend_url
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        
        # Directories
        self.workspace_dir = workspace_dir or Path(f"/tmp/landseer_worker_{self.worker_id}")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = cache_dir or Path("/tmp/landseer_cache")
        self.data_path = data_path  # Path to input data (manual override)
        self._dataset_path: Optional[Path] = None  # Auto-fetched from backend
        
        # Configuration
        self.gpu_id = gpu_id
        self.poll_interval = poll_interval
        self.task_timeout = task_timeout
        self.heartbeat_interval = heartbeat_interval
        self.use_cache = use_cache
        self.runtime = runtime
        
        # State
        self._running = False
        self._current_task: Optional[TaskInfo] = None
        self._last_heartbeat = 0.0
        self._tasks_completed = 0
        self._tasks_failed = 0
        
        # Components (initialized on start)
        self._client: Optional[LandseerClient] = None
        self._runner: Optional[TaskRunner] = None
        self._cache: Optional[CacheManager] = None
        self._two_level_cache: Optional["TwoLevelCache"] = None
        self._use_minio: bool = os.environ.get("LANDSEER_USE_MINIO", "true").lower() == "true"
        self._model_script_path: Optional[Path] = None  # Path to model config script (e.g., config_model.py)
        
        # Signal handling
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._running = False
    
    def _init_components(self) -> None:
        """Initialize worker components."""
        # HTTP client
        self._client = LandseerClient(
            backend_url=self.backend_url,
            timeout=30.0,
            retry_attempts=3
        )
        
        # Task runner
        self._runner = TaskRunner(
            workspace_dir=self.workspace_dir,
            artifact_cache_dir=self.cache_dir if self.use_cache else None,
            gpu_id=self.gpu_id,
            timeout=self.task_timeout,
            runtime=self.runtime
        )
        
        # Cache manager - prefer two-level cache with MinIO if available
        if self.use_cache:
            if TWO_LEVEL_CACHE_AVAILABLE and self._use_minio:
                cache_config = CacheConfig(
                    local_cache_dir=self.cache_dir,
                    use_minio=True
                )
                self._two_level_cache = TwoLevelCache(cache_config)
                logger.info("Using two-level cache (local + MinIO)")
            else:
                self._cache = CacheManager(self.cache_dir)
                logger.info("Using local-only cache")
    
    def _fetch_dataset(self) -> Optional[Path]:
        """
        Fetch dataset information from backend and download if needed.
        
        The backend prepares the dataset on startup and uploads to MinIO.
        This method:
        1. Gets dataset info from backend
        2. If MinIO available, downloads dataset to local cache
        3. Gets model script path for container execution
        4. Returns path to dataset
        
        Returns:
            Path to dataset directory, or None if not available
        """
        # If manual data_path specified, use it
        if self.data_path and self.data_path.exists():
            logger.info(f"Using manual data path: {self.data_path}")
            return self.data_path
        
        # Try to get dataset info from backend
        try:
            dataset_info = self._client.get_dataset_info()
        except Exception as e:
            logger.warning(f"Failed to get dataset info from backend: {e}")
            return None
        
        if not dataset_info.get("available"):
            logger.info("No dataset available from backend")
            return None
        
        # Get model script path (for container execution)
        model_script = dataset_info.get("model_script")
        if model_script:
            model_path = Path(model_script)
            if model_path.exists():
                self._model_script_path = model_path
                logger.info(f"Model config script: {model_path}")
            else:
                logger.warning(f"Model script path not found: {model_script}")
        
        # Check if we can use local path (same machine as backend)
        local_path = dataset_info.get("local_path")
        if local_path:
            local_dir = Path(local_path)
            if local_dir.exists() and (local_dir / "data.npy").exists():
                logger.info(f"Using local dataset path: {local_dir}")
                return local_dir
        
        # Try to download from MinIO
        minio_key = dataset_info.get("minio_key")
        if minio_key and dataset_info.get("minio_available") and self._two_level_cache:
            try:
                # Download to cache directory
                dataset_name = dataset_info.get("name", "dataset")
                variant = dataset_info.get("variant", "clean")
                download_dir = self.cache_dir / "datasets" / dataset_name / variant
                download_dir.mkdir(parents=True, exist_ok=True)
                
                # Check if already downloaded
                if (download_dir / "data.npy").exists():
                    logger.info(f"Dataset already cached at: {download_dir}")
                    return download_dir
                
                logger.info(f"Downloading dataset from MinIO: {minio_key}")
                
                # Use the store from two-level cache
                if hasattr(self._two_level_cache, '_minio_store'):
                    self._two_level_cache._minio_store.download_directory(
                        minio_key, download_dir
                    )
                    logger.info(f"Dataset downloaded to: {download_dir}")
                    return download_dir
                    
            except Exception as e:
                logger.warning(f"Failed to download dataset from MinIO: {e}")
        
        logger.warning("Dataset not available locally or via MinIO")
        return None
    
    def _wait_for_backend(self, max_retries: int = 30, retry_delay: float = 2.0) -> bool:
        """
        Wait for backend to become available.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if backend is available
        """
        logger.info(f"Waiting for backend at {self.backend_url}...")
        
        for attempt in range(max_retries):
            try:
                if self._client.is_backend_available():
                    logger.info("Backend is available")
                    return True
            except Exception as e:
                logger.debug(f"Backend not ready: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        logger.error(f"Backend not available after {max_retries} attempts")
        return False
    
    def _register(self) -> bool:
        """
        Register worker with the backend.
        
        Returns:
            True if registration successful
        """
        try:
            # Detect capabilities
            capabilities = {
                "runtime": ContainerRuntime.detect_runtime(),
                "gpu_available": self.gpu_id is not None,
                "gpu_id": self.gpu_id
            }
            
            worker_info = self._client.register(
                worker_id=self.worker_id,
                capabilities=capabilities
            )
            
            self.worker_id = worker_info.worker_id
            logger.info(f"Registered as worker: {self.worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register: {e}")
            return False
    
    def _send_heartbeat(self) -> None:
        """Send heartbeat to backend if interval has passed."""
        now = time.time()
        if now - self._last_heartbeat >= self.heartbeat_interval:
            try:
                status = "busy" if self._current_task else "idle"
                self._client.heartbeat(status=status)
                self._last_heartbeat = now
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
    
    def _execute_task(self, task: TaskInfo) -> ExecutionResult:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
        """
        self._current_task = task
        logger.info(f"Executing task: {task.id} ({task.tool_name})")
        
        try:
            # Compute cache key
            parent_hashes = []  # TODO: Track parent hashes through the pipeline
            cache_key = self._compute_cache_key(task, parent_hashes)
            
            # Check cache first - using two-level cache if available
            if self.use_cache:
                cached_path = self._check_cache(cache_key, task, parent_hashes)
                
                if cached_path:
                    logger.info(f"Cache hit for task {task.id}")
                    return ExecutionResult(
                        success=True,
                        exit_code=0,
                        execution_time_ms=0,
                        output_path=cached_path,
                        artifacts={"cache_hit": True, "cache_key": cache_key}
                    )
            
            # Execute the task with data directory mounted
            # Priority: manual data_path > fetched dataset_path
            extra_mounts = {}
            data_dir = self.data_path if (self.data_path and self.data_path.exists()) else self._dataset_path
            if data_dir and data_dir.exists():
                extra_mounts[str(data_dir.absolute())] = "/data"
                logger.debug(f"Mounting data directory: {data_dir} -> /data")
            
            result = self._runner.run_task(
                task,
                input_path=data_dir,
                extra_mounts=extra_mounts if extra_mounts else None,
                model_script_path=self._model_script_path
            )
            
            # Store in cache if successful
            if self.use_cache and result.success and result.output_path:
                self._store_in_cache(
                    cache_key=cache_key,
                    task=task,
                    output_path=result.output_path,
                    execution_time_ms=result.execution_time_ms,
                    parent_hashes=parent_hashes
                )
                result.artifacts["cache_key"] = cache_key
            
            return result
            
        finally:
            self._current_task = None
    
    def _compute_cache_key(self, task: TaskInfo, parent_hashes: List[str]) -> str:
        """Compute cache key for a task."""
        import hashlib
        import json
        
        identity = {
            "tool_name": task.tool_name,
            "tool_image": task.tool_image,
            "tool_command": task.tool_command,
            "config": task.config,
            "parents": sorted(parent_hashes)
        }
        json_str = json.dumps(identity, sort_keys=True, separators=(',', ':'))
        return hashlib.blake2s(json_str.encode()).hexdigest()
    
    def _check_cache(
        self,
        cache_key: str,
        task: TaskInfo,
        parent_hashes: List[str]
    ) -> Optional[Path]:
        """Check cache for a task output."""
        # Prefer two-level cache
        if self._two_level_cache:
            return self._two_level_cache.get(cache_key)
        
        # Fall back to local-only cache
        if self._cache:
            return self._cache.check_cache(task, parent_hashes)
        
        return None
    
    def _store_in_cache(
        self,
        cache_key: str,
        task: TaskInfo,
        output_path: Path,
        execution_time_ms: int,
        parent_hashes: List[str]
    ) -> None:
        """Store task output in cache."""
        # Prefer two-level cache
        if self._two_level_cache:
            self._two_level_cache.put(
                cache_key=cache_key,
                source_dir=output_path,
                task_id=task.id,
                tool_name=task.tool_name,
                parent_hashes=parent_hashes,
                metadata={"execution_time_ms": execution_time_ms}
            )
            return
        
        # Fall back to local-only cache
        if self._cache:
            self._cache.store_result(
                task=task,
                output_path=output_path,
                execution_time_ms=execution_time_ms,
                parent_hashes=parent_hashes
            )
    
    def _report_result(self, task: TaskInfo, result: ExecutionResult) -> None:
        """
        Report task result to backend.
        
        Args:
            task: Completed task
            result: Execution result
        """
        try:
            if result.success:
                self._client.report_task_completed(
                    task_id=task.id,
                    execution_time_ms=result.execution_time_ms,
                    result=result.artifacts
                )
                self._tasks_completed += 1
            else:
                self._client.report_task_failed(
                    task_id=task.id,
                    error_message=result.error_message or "Unknown error",
                    execution_time_ms=result.execution_time_ms
                )
                self._tasks_failed += 1
                
        except Exception as e:
            logger.error(f"Failed to report result for task {task.id}: {e}")
    
    def _work_loop(self) -> None:
        """Main work loop - claim and execute tasks."""
        idle_count = 0
        max_idle = 10  # Number of idle polls before longer wait
        
        while self._running:
            self._send_heartbeat()
            
            try:
                # Try to claim a task
                task = self._client.claim_task()
                
                if task:
                    idle_count = 0
                    
                    # Execute the task
                    result = self._execute_task(task)
                    
                    # Report result
                    self._report_result(task, result)
                    
                else:
                    idle_count += 1
                    
                    # Check if all tasks are done
                    progress = self._client.get_progress()
                    
                    if progress.get("is_complete"):
                        logger.info("All tasks completed. Worker shutting down.")
                        break
                    
                    # Exponential backoff when idle
                    wait_time = min(self.poll_interval * (1.5 ** min(idle_count, max_idle)), 60)
                    logger.debug(f"No tasks available, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    
            except KeyboardInterrupt:
                logger.info("Interrupted, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in work loop: {e}")
                time.sleep(self.poll_interval)
    
    def start(self) -> int:
        """
        Start the worker.
        
        Returns:
            Exit code (0 for success)
        """
        logger.info(f"Starting Landseer Worker: {self.worker_id}")
        logger.info(f"Backend URL: {self.backend_url}")
        logger.info(f"Workspace: {self.workspace_dir}")
        logger.info(f"Cache: {self.cache_dir if self.use_cache else 'disabled'}")
        logger.info(f"GPU: {self.gpu_id if self.gpu_id is not None else 'None (CPU only)'}")
        
        # Initialize components
        self._init_components()
        
        # Wait for backend
        if not self._wait_for_backend():
            return 1
        
        # Register
        if not self._register():
            return 1
        
        # Fetch dataset from backend
        self._dataset_path = self._fetch_dataset()
        if self._dataset_path:
            logger.info(f"Dataset available at: {self._dataset_path}")
        else:
            logger.warning("No dataset available - tasks may fail if data is required")
        
        # Start work loop
        self._running = True
        logger.info("Worker started, entering work loop...")
        
        try:
            self._work_loop()
        except Exception as e:
            logger.exception(f"Worker crashed: {e}")
            return 1
        finally:
            self._cleanup()
        
        logger.info(f"Worker finished. Completed: {self._tasks_completed}, Failed: {self._tasks_failed}")
        return 0
    
    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        
        logger.info("Worker cleanup complete")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the worker CLI."""
    parser = argparse.ArgumentParser(
        description="Landseer Worker - ML Security Pipeline Task Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start worker with defaults
  landseer-worker

  # Connect to specific backend
  landseer-worker --backend-url http://scheduler:8000

  # Use specific GPU
  landseer-worker --gpu 0

  # Custom workspace
  landseer-worker --workspace /data/landseer_work

  # Disable caching
  landseer-worker --no-cache
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    # Connection options
    conn_group = parser.add_argument_group("Connection Options")
    conn_group.add_argument(
        "--backend-url",
        type=str,
        default=os.environ.get("LANDSEER_BACKEND_URL", "http://localhost:8000"),
        help="Backend API URL (default: http://localhost:8000, env: LANDSEER_BACKEND_URL)",
    )
    conn_group.add_argument(
        "--worker-id",
        type=str,
        default=os.environ.get("LANDSEER_WORKER_ID"),
        help="Unique worker identifier (auto-generated if not provided, env: LANDSEER_WORKER_ID)",
    )
    
    # Execution options
    exec_group = parser.add_argument_group("Execution Options")
    exec_group.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="ID",
        help="GPU ID to use (default: CPU only)",
    )
    exec_group.add_argument(
        "--timeout",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="Task execution timeout in seconds (default: 3600)",
    )
    exec_group.add_argument(
        "--runtime",
        type=str,
        choices=["docker", "apptainer", "singularity", "auto"],
        default="auto",
        help="Container runtime to use (default: auto-detect)",
    )
    
    # Storage options
    storage_group = parser.add_argument_group("Storage Options")
    storage_group.add_argument(
        "--workspace",
        type=str,
        default=os.environ.get("LANDSEER_WORKSPACE"),
        metavar="DIR",
        help="Workspace directory for task execution (env: LANDSEER_WORKSPACE)",
    )
    storage_group.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("LANDSEER_CACHE_DIR", "/tmp/landseer_cache"),
        metavar="DIR",
        help="Artifact cache directory (default: /tmp/landseer_cache, env: LANDSEER_CACHE_DIR)",
    )
    storage_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable artifact caching",
    )
    storage_group.add_argument(
        "--data-path",
        type=str,
        default=os.environ.get("LANDSEER_DATA_PATH"),
        metavar="DIR",
        help="Override: manual path to dataset directory. If not specified, "
             "dataset is automatically fetched from backend/MinIO (env: LANDSEER_DATA_PATH)",
    )
    storage_group.add_argument(
        "--no-minio",
        action="store_true",
        help="Disable MinIO remote storage (use local-only cache)",
    )
    storage_group.add_argument(
        "--minio-endpoint",
        type=str,
        default=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
        metavar="HOST:PORT",
        help="MinIO server endpoint (default: localhost:9000, env: MINIO_ENDPOINT)",
    )
    
    # Timing options
    timing_group = parser.add_argument_group("Timing Options")
    timing_group.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Seconds between task polls when idle (default: 5.0)",
    )
    timing_group.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Seconds between heartbeats (default: 30.0)",
    )
    
    # Debug options
    debug_group = parser.add_argument_group("Debug Options")
    debug_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    debug_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the worker CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Configure logging
    from ..common.pylogger import set_global_log_level
    import logging
    log_level = getattr(logging, args.log_level.upper())
    set_global_log_level(log_level)
    
    if args.debug:
        set_global_log_level(logging.DEBUG)
    
    # Determine runtime
    runtime = None if args.runtime == "auto" else args.runtime
    
    # Configure MinIO via environment
    if args.no_minio:
        os.environ["LANDSEER_USE_MINIO"] = "false"
    else:
        os.environ["LANDSEER_USE_MINIO"] = "true"
        os.environ["MINIO_ENDPOINT"] = args.minio_endpoint
    
    # Create and start worker
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
        runtime=runtime
    )
    
    return worker.start()


if __name__ == "__main__":
    sys.exit(main())
