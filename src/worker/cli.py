"""
Command-line interface for Landseer worker.

The worker connects to the Landseer backend scheduler, claims tasks,
executes them in containers, and reports results back.
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..common import get_logger, init_sentry
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
DEBUG_LOG_PATH = "/share/landseer/workspace-ayushi/Landseer/.cursor/debug-26edf1.log"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": "26edf1",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass


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
        task_timeout: int = 7200,
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
        self.workspace_dir = workspace_dir or Path(f"/data/landseer/workers/{self.worker_id}")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = cache_dir or Path("/data/landseer/cache")
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
        self._task_heartbeat_thread: Optional[threading.Thread] = None
        self._task_heartbeat_stop = threading.Event()
        self._tasks_completed = 0
        self._tasks_failed = 0
        # Maps task_id -> (cache_key, output_path, ancestor_paths) so that
        # downstream tasks can:
        #   (a) build correct parent_hashes for cache-key computation
        #   (b) locate artifacts from cache-hit deps that skipped the workspace
        #   (c) accumulate the full ordered ancestry chain (earliest stage first)
        #       so every task in the pipeline sees ALL upstream artifacts, not
        #       just its direct parent's output.
        # ancestor_paths is a List[Path] containing the output dirs of every
        # task in this task's upstream lineage (grandparents, parents, etc.)
        # in pipeline order — later entries override on filename collision.
        self._task_outputs: Dict[str, tuple] = {}  # task_id -> (cache_key, Path|None, List[Path])
        
        # Components (initialized on start)
        self._client: Optional[LandseerClient] = None
        self._runner: Optional[TaskRunner] = None
        self._cache: Optional[CacheManager] = None
        self._two_level_cache: Optional["TwoLevelCache"] = None
        self._use_minio: bool = os.environ.get("LANDSEER_USE_MINIO", "true").lower() == "true"
        self._model_script_path: Optional[Path] = None  # Path to model config script (e.g., config_model.py)
        self._dataset_info: Dict[str, Any] = {}
        self._active_run_id: Optional[str] = None
        
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
        # If manual data_path specified, prefer it for dataset arrays, but still
        # try to fetch backend dataset metadata so we can resolve model_script_path.
        if self.data_path and self.data_path.exists():
            logger.info(f"Using manual data path: {self.data_path}")
            # In manual dataset mode, prefer explicit YAML config for model script
            # so we don't couple to whichever pipeline the backend currently has loaded.
            cfg_path_raw = os.environ.get("LANDSEER_PIPELINE_CONFIG")
            if self._model_script_path is None and cfg_path_raw:
                try:
                    cfg_path = Path(cfg_path_raw).expanduser().resolve()
                    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                    model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
                    model_script = model_cfg.get("script") if isinstance(model_cfg, dict) else None
                    if model_script:
                        model_path = Path(model_script)
                        if not model_path.is_absolute():
                            model_path = (cfg_path.parent / model_path).resolve()
                        if model_path.exists():
                            self._model_script_path = model_path
                            logger.info(f"Model config script (pipeline yaml): {model_path}")
                        else:
                            logger.warning(
                                f"Model script from LANDSEER_PIPELINE_CONFIG not found: {model_path}"
                            )
                except Exception as e:
                    logger.warning(
                        f"Failed to resolve model script from LANDSEER_PIPELINE_CONFIG={cfg_path_raw}: {e}"
                    )

            try:
                dataset_info = self._client.get_dataset_info()
                if isinstance(dataset_info, dict):
                    self._dataset_info = dataset_info
                    model_script = dataset_info.get("model_script")
                    if model_script:
                        model_path = Path(model_script)
                        if model_path.exists():
                            self._model_script_path = model_path
                            logger.info(f"Model config script: {model_path}")
                        else:
                            logger.warning(f"Model script path not found: {model_script}")
            except Exception as e:
                logger.debug(
                    f"Manual data path mode: failed to fetch dataset metadata for model script: {e}"
                )
            if self._model_script_path is None:
                try:
                    pipeline_info = self._client.get_pipeline_info()
                    model = pipeline_info.get("model") if isinstance(pipeline_info, dict) else None
                    model_script = model.get("script") if isinstance(model, dict) else None
                    if model_script:
                        model_path = Path(model_script)
                        if model_path.exists():
                            self._model_script_path = model_path
                            logger.info(f"Model config script (pipeline info): {model_path}")
                except Exception as e:
                    logger.debug(
                        f"Manual data path mode: failed to fetch pipeline info for model script: {e}"
                    )
            return self.data_path
        
        # Try to get dataset info from backend
        try:
            dataset_info = self._client.get_dataset_info()
        except Exception as e:
            logger.warning(f"Failed to get dataset info from backend: {e}")
            return None
        self._dataset_info = dataset_info or {}
        
        # Get model script path (for container execution)
        model_script = dataset_info.get("model_script")
        if model_script:
            model_path = Path(model_script)
            if model_path.exists():
                self._model_script_path = model_path
                logger.info(f"Model config script: {model_path}")
            else:
                logger.warning(f"Model script path not found: {model_script}")

        if not dataset_info.get("available"):
            logger.info("No dataset available from backend")
            return None
        
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
                model_script_minio_key = dataset_info.get("model_script_minio_key")
                model_script_local = download_dir / "config_model.py"

                def _download_model_script_if_needed() -> None:
                    if not model_script_minio_key:
                        return
                    if model_script_local.exists():
                        self._model_script_path = model_script_local
                        return
                    if hasattr(self._two_level_cache, "_minio_store"):
                        ok = self._two_level_cache._minio_store.download_file(
                            model_script_minio_key,
                            model_script_local,
                        )
                        if ok:
                            self._model_script_path = model_script_local
                            logger.info(
                                f"Downloaded model script from MinIO: {model_script_minio_key}"
                            )
                        else:
                            logger.warning(
                                f"Failed to download model script from MinIO: {model_script_minio_key}"
                            )
                
                # Check if already downloaded
                if (download_dir / "data.npy").exists():
                    _download_model_script_if_needed()
                    logger.info(f"Dataset already cached at: {download_dir}")
                    return download_dir
                
                logger.info(f"Downloading dataset from MinIO: {minio_key}")
                
                # Use the store from two-level cache
                if hasattr(self._two_level_cache, '_minio_store'):
                    self._two_level_cache._minio_store.download_directory(
                        minio_key, download_dir
                    )
                    _download_model_script_if_needed()
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

    def _start_task_heartbeat(self) -> None:
        """Start a background heartbeat loop while a task is executing."""
        if self._task_heartbeat_thread and self._task_heartbeat_thread.is_alive():
            return

        self._task_heartbeat_stop.clear()

        def _loop() -> None:
            interval = max(1.0, self.heartbeat_interval / 2.0)
            while not self._task_heartbeat_stop.wait(interval):
                if not self._running or self._current_task is None:
                    continue
                try:
                    self._client.heartbeat(status="busy")
                    self._last_heartbeat = time.time()
                except Exception as e:
                    logger.warning(f"Task heartbeat failed: {e}")

        self._task_heartbeat_thread = threading.Thread(
            target=_loop,
            name=f"heartbeat-{self.worker_id}",
            daemon=True,
        )
        self._task_heartbeat_thread.start()

    def _stop_task_heartbeat(self) -> None:
        """Stop the background task heartbeat loop."""
        self._task_heartbeat_stop.set()
        if self._task_heartbeat_thread and self._task_heartbeat_thread.is_alive():
            self._task_heartbeat_thread.join(timeout=2.0)
        self._task_heartbeat_thread = None

    def _resolve_task_output_path(self, task_info: TaskInfo) -> Optional[Path]:
        """Resolve a dependency output directory from backend metadata."""
        candidates: List[Path] = []
        if task_info.output_path:
            candidates.append(Path(task_info.output_path))
        if task_info.cache_key:
            candidates.append(self.cache_dir / task_info.cache_key / "output")
        if task_info.log_path:
            log_path = Path(task_info.log_path)
            candidates.append(log_path.parent.parent / "output")

        for path in candidates:
            try:
                if path.exists():
                    return path
            except Exception:
                continue

        # Cross-worker fallback: dependency output may exist only on a different
        # worker host path. If we have a cache_key and two-level cache is active,
        # rehydrate from MinIO into this worker's local cache.
        if task_info.cache_key and self._two_level_cache:
            try:
                hydrated = self._two_level_cache.get(task_info.cache_key)
                if hydrated and hydrated.exists():
                    logger.info(
                        "Hydrated dependency artifact %s from MinIO for task %s",
                        task_info.cache_key[:12],
                        task_info.id,
                    )
                    return hydrated
            except Exception as e:
                logger.warning(
                    "Failed to hydrate dependency artifact from MinIO "
                    "(task=%s, cache_key=%s): %s",
                    task_info.id,
                    task_info.cache_key[:12],
                    e,
                )
        return None

    def _collect_remote_ancestry(self, task_id: str, visited: Optional[set[str]] = None) -> tuple[list[Path], Optional[TaskInfo]]:
        """Collect full ancestor outputs for a task by querying backend task metadata."""
        if visited is None:
            visited = set()
        if task_id in visited:
            return [], None
        visited.add(task_id)

        task_info = self._client.get_task(task_id)
        if not task_info:
            return [], None

        ancestry: List[Path] = []
        for dep_id in task_info.dependency_ids:
            dep_ancestry, dep_task = self._collect_remote_ancestry(dep_id, visited)
            ancestry.extend(dep_ancestry)
            if dep_task:
                dep_output = self._resolve_task_output_path(dep_task)
                if dep_output:
                    ancestry.append(dep_output)
                    # Track remote deps locally so future tasks don't need refetch.
                    dep_cache_key = dep_task.cache_key or ""
                    self._task_outputs[dep_id] = (dep_cache_key, dep_output, list(dep_ancestry))

        return ancestry, task_info

    @staticmethod
    def _append_unique_paths(paths: List[Path], additions: List[Path]) -> None:
        """Append paths while preserving order and removing duplicates."""
        seen = {str(p.resolve()) if p.exists() else str(p) for p in paths}
        for p in additions:
            key = str(p.resolve()) if p.exists() else str(p)
            if key not in seen:
                paths.append(p)
                seen.add(key)
    
    def _dataset_fingerprint(self, data_dir: Optional[Path]) -> Dict[str, Any]:
        """Build a lightweight dataset identity fingerprint for cache keys."""
        if not data_dir:
            return {"available": False}
        try:
            resolved = data_dir.resolve()
        except Exception:
            resolved = data_dir
        fp: Dict[str, Any] = {
            "available": True,
            "path": str(resolved),
            "name": self._dataset_info.get("name"),
            "variant": self._dataset_info.get("variant"),
            "minio_key": self._dataset_info.get("minio_key"),
        }
        tracked: Dict[str, Any] = {}
        for name in ("data.npy", "labels.npy", "test_data.npy", "test_labels.npy"):
            p = data_dir / name
            if p.exists():
                st = p.stat()
                tracked[name] = {
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                }
        fp["tracked_files"] = tracked
        return fp

    @staticmethod
    def _model_fingerprint(model_script_path: Optional[Path]) -> Dict[str, Any]:
        """Build model-script identity fingerprint for cache keys."""
        if not model_script_path:
            return {"available": False}
        fp: Dict[str, Any] = {
            "available": False,
            "path": str(model_script_path),
        }
        try:
            if model_script_path.exists():
                st = model_script_path.stat()
                fp.update({
                    "available": True,
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                })
                hasher = hashlib.sha256()
                with model_script_path.open("rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
                fp["sha256"] = hasher.hexdigest()
        except Exception as e:
            fp["error"] = str(e)
        return fp

    def _build_cache_context(self, data_dir: Optional[Path]) -> Dict[str, Any]:
        """Build shared cache context so cache keys include model/dataset identity."""
        return {
            "dataset": self._dataset_fingerprint(data_dir),
            "model": self._model_fingerprint(self._model_script_path),
        }

    def _execute_task(self, task: TaskInfo) -> ExecutionResult:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
        """
        self._current_task = task
        self._start_task_heartbeat()
        logger.info(f"Executing task: {task.id} ({task.tool_name})")
        
        try:
            # Refresh dataset/model context when pipeline run changes.
            # Without this, long-lived workers may keep stale model_script/dataset
            # identity and incorrectly reuse cache entries across runs.
            if task.run_id and task.run_id != self._active_run_id:
                logger.info(f"Run changed ({self._active_run_id} -> {task.run_id}); refreshing dataset/model context")
                refreshed = self._fetch_dataset()
                if refreshed:
                    self._dataset_path = refreshed
                # Task IDs are reused across runs (task_1, task_2, ...). Clear
                # per-task output lineage cache to avoid mixing dependency outputs
                # from prior runs (e.g. CIFAR ancestry leaking into CelebA runs).
                self._task_outputs.clear()
                # region agent log
                _debug_log(
                    run_id=task.id,
                    hypothesis_id="W4",
                    location="src/worker/cli.py:_execute_task",
                    message="cleared per-run task output cache on run switch",
                    data={
                        "previous_run_id": self._active_run_id,
                        "new_run_id": task.run_id,
                    },
                )
                # endregion
                self._active_run_id = task.run_id

            # Build parent_hashes (for cache key) and ancestor_dirs (ordered
            # output dirs from all upstream stages, earliest first) in one pass.
            parent_hashes: List[str] = []
            ancestor_dirs: List[Path] = []
            for dep_id in task.dependency_ids:
                if dep_id in self._task_outputs:
                    dep_key, dep_out, dep_anc = self._task_outputs[dep_id]
                    if dep_key:
                        parent_hashes.append(dep_key)
                    self._append_unique_paths(ancestor_dirs, dep_anc)
                    if dep_out is not None:
                        self._append_unique_paths(ancestor_dirs, [dep_out])
                else:
                    logger.warning(f"Dep {dep_id} not yet in _task_outputs; resolving remotely")
                    dep_anc, dep_task = self._collect_remote_ancestry(dep_id)
                    self._append_unique_paths(ancestor_dirs, dep_anc)
                    if dep_task:
                        if dep_task.cache_key:
                            parent_hashes.append(dep_task.cache_key)
                        dep_output = self._resolve_task_output_path(dep_task)
                        if dep_output:
                            self._append_unique_paths(ancestor_dirs, [dep_output])
                            self._task_outputs[dep_id] = (
                                dep_task.cache_key or "",
                                dep_output,
                                list(dep_anc),
                            )
                        else:
                            logger.warning(f"Could not resolve output path for dependency {dep_id}")
                    else:
                        ws_out = self._runner.workspace_dir / dep_id / "output"
                        if ws_out.exists():
                            self._append_unique_paths(ancestor_dirs, [ws_out])

            data_dir = self.data_path if (self.data_path and self.data_path.exists()) else self._dataset_path
            if data_dir is None:
                # Headless backend may expose dataset only after a run is started.
                # Retry dataset fetch at task execution time instead of only worker startup.
                refreshed = self._fetch_dataset()
                if refreshed:
                    self._dataset_path = refreshed
                    data_dir = refreshed
            cache_context = self._build_cache_context(data_dir)
            cache_key = self._compute_cache_key(task, parent_hashes, cache_context)
            run_use_cache = bool(task.config.get("_run_use_cache", True))
            effective_use_cache = self.use_cache and run_use_cache
            extra_mounts: Optional[Dict[str, str]] = None
            # region agent log
            _debug_log(
                run_id=task.id,
                hypothesis_id="W3",
                location="src/worker/cli.py:_execute_task",
                message="resolved cache policy for task",
                data={
                    "worker_use_cache": self.use_cache,
                    "run_use_cache": run_use_cache,
                    "effective_use_cache": effective_use_cache,
                    "run_id": task.run_id,
                },
            )
            # endregion

            if effective_use_cache:
                cached_path = self._check_cache(cache_key, task, parent_hashes, cache_context)
                if cached_path:
                    logger.info(f"Cache hit for task {task.id}")
                    symlink_target = self._runner.workspace_dir / task.id / "output"
                    if not symlink_target.exists():
                        symlink_target.parent.mkdir(parents=True, exist_ok=True)
                        symlink_target.symlink_to(cached_path.resolve())
                    self._task_outputs[task.id] = (cache_key, cached_path, list(ancestor_dirs))
                    return ExecutionResult(
                        success=True,
                        exit_code=0,
                        execution_time_ms=0,
                        output_path=cached_path,
                        artifacts={"cache_hit": True, "cache_key": cache_key}
                    )

            result = self._runner.run_task(
                task,
                input_path=data_dir,
                extra_mounts=extra_mounts,
                model_script_path=self._model_script_path,
                ancestor_dirs=ancestor_dirs or None,
            )

            # Record always so downstream tasks can extend the ancestry chain.
            self._task_outputs[task.id] = (cache_key, result.output_path, list(ancestor_dirs))

            # Store in cache if successful.
            # For evaluation tasks, only cache when the evaluation itself
            # succeeded — failures (e.g. "model.pt not found") should not be
            # persisted so the evaluator re-runs on the next attempt.
            should_cache = (
                effective_use_cache
                and result.success
                and result.output_path
                and not self._eval_failed(task, result.output_path)
            )
            if should_cache:
                self._store_in_cache(
                    cache_key=cache_key,
                    task=task,
                    output_path=result.output_path,
                    execution_time_ms=result.execution_time_ms,
                    parent_hashes=parent_hashes,
                    cache_context=cache_context
                )
                result.artifacts["cache_key"] = cache_key

            return result
            
        finally:
            self._stop_task_heartbeat()
            self._current_task = None
    
    def _compute_cache_key(
        self,
        task: TaskInfo,
        parent_hashes: List[str],
        cache_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Compute cache key for a task."""
        identity = {
            "tool_name": task.tool_name,
            "tool_image": task.tool_image,
            "tool_command": task.tool_command,
            "config": task.config,
            "parents": sorted(parent_hashes),
            "context": cache_context or {},
        }
        json_str = json.dumps(identity, sort_keys=True, separators=(',', ':'))
        return hashlib.blake2s(json_str.encode()).hexdigest()
    
    def _check_cache(
        self,
        cache_key: str,
        task: TaskInfo,
        parent_hashes: List[str],
        cache_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Check cache for a task output."""
        # Prefer two-level cache
        if self._two_level_cache:
            return self._two_level_cache.get(cache_key)
        
        # Fall back to local-only cache
        if self._cache:
            return self._cache.check_cache(task, parent_hashes, cache_context=cache_context)
        
        return None
    
    def _store_in_cache(
        self,
        cache_key: str,
        task: TaskInfo,
        output_path: Path,
        execution_time_ms: int,
        parent_hashes: List[str],
        cache_context: Optional[Dict[str, Any]] = None,
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
            run_id = getattr(task, 'run_id', None)
            self._cache.store_result(
                task=task,
                output_path=output_path,
                execution_time_ms=execution_time_ms,
                parent_hashes=parent_hashes,
                run_id=run_id,
                cache_context=cache_context,
            )
    
    def _eval_failed(self, task: TaskInfo, output_path: Path) -> bool:
        """Return True when an evaluation task's evaluation_results.json reports failure.

        Prevents caching evaluation failures so the evaluator re-runs on the
        next attempt once the underlying issue (e.g. missing model.pt) is fixed.
        Non-evaluation tasks always return False.
        """
        if task.task_type != "evaluation":
            return False
        results_file = output_path / "evaluation_results.json"
        if results_file.exists():
            try:
                payload = json.loads(results_file.read_text())
                return not payload.get("success", True)
            except Exception:
                pass
        # No results file means the container crashed — treat as failure
        return True

    def _extract_evaluation_result(
        self,
        task: TaskInfo,
        result: ExecutionResult,
        logs_snippet: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract evaluator metrics.

        Preferred source is ``evaluation_results.json`` from evaluator output.
        """
        if not (
            result.success
            and task.task_type == "evaluation"
            and result.output_path
            and result.output_path.exists()
        ):
            return None

        candidate_files = [
            "evaluation_results.json",
            "evaluation_result.json",
            "results.json",
            "metrics.json",
        ]
        expected_metrics = task.config.get("metrics", []) if isinstance(task.config, dict) else []

        for filename in candidate_files:
            candidate = result.output_path / filename
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text())
                if not isinstance(payload, dict):
                    continue

                # Normalize older formats where metrics are top-level keys.
                if "metrics" not in payload:
                    numeric_metrics: Dict[str, float] = {}
                    for key, val in payload.items():
                        try:
                            numeric_metrics[key] = float(val)
                        except (TypeError, ValueError):
                            continue
                    if numeric_metrics:
                        payload = {
                            "success": True,
                            "skipped": False,
                            "metrics": numeric_metrics,
                        }

                if isinstance(payload.get("metrics"), dict):
                    # Keep only declared evaluator metrics so stray keys
                    # do not pollute dashboard aggregation.
                    if expected_metrics:
                        payload["metrics"] = {
                            key: val
                            for key, val in payload["metrics"].items()
                            if key in expected_metrics
                        }
                    return payload
            except Exception as e:
                logger.warning(f"Failed to read {filename} for task {task.id}: {e}")

        # Fallback: keep compatibility with old CSV-style sentinel values.
        skip_reason = None
        if logs_snippet:
            for line in logs_snippet.splitlines():
                if "Skipping:" in line:
                    skip_reason = line.strip()
                    break

        fallback_metrics = {metric_name: -1.0 for metric_name in expected_metrics}
        return {
            "success": True,
            "skipped": True,
            "skip_reason": skip_reason or "evaluation_results.json not produced by evaluator",
            "metrics": fallback_metrics,
            "parameters": {"source": "worker_fallback"},
        }

    def _report_result(self, task: TaskInfo, result: ExecutionResult) -> None:
        """
        Report task result to backend.
        
        Args:
            task: Completed task
            result: Execution result
        """
        try:
            # Derive the per-task log file path written by TaskRunner
            # so we can both (a) keep full logs on the worker filesystem,
            # and (b) send a small snippet + path back to the backend.
            log_path = (
                self.workspace_dir
                / task.id
                / "logs"
                / f"{task.task_type}_{task.tool_name.replace(' ', '_')}.log"
            )

            logs_snippet: Optional[str] = None
            if log_path.exists():
                try:
                    content = log_path.read_text()
                    # Limit size to keep API/database payloads small while
                    # still being useful for debugging.
                    max_chars = 10_000
                    if len(content) > max_chars:
                        logs_snippet = content[-max_chars:]
                    else:
                        logs_snippet = content
                except Exception as e:
                    logger.warning(f"Failed to read log file for task {task.id}: {e}")

            # Attach structured info the backend can surface via /tasks/...:
            # - artifacts: cache keys, etc.
            # - logs: tail snippet of the worker log file
            # - log_path: absolute path on the worker for full inspection
            result_payload = {
                "artifacts": result.artifacts,
                "logs": logs_snippet,
                "log_path": str(log_path) if log_path.exists() else None,
                "output_path": str(result.output_path) if result.output_path else None,
            }

            # For evaluator tasks, include structured evaluation_result so backend
            # can persist metrics and expose them on metrics endpoints.
            eval_data = self._extract_evaluation_result(task, result, logs_snippet)
            if eval_data is not None:
                result_payload["evaluation_result"] = eval_data
                logger.debug(f"Included evaluation_result for task {task.id}")

            if result.success:
                self._client.report_task_completed(
                    task_id=task.id,
                    execution_time_ms=result.execution_time_ms,
                    result=result_payload
                )
                self._tasks_completed += 1
            else:
                self._client.report_task_failed(
                    task_id=task.id,
                    error_message=result.error_message or "Unknown error",
                    execution_time_ms=result.execution_time_ms,
                    result=result_payload
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
                    
                    try:
                        # Execute the task
                        result = self._execute_task(task)
                    except Exception as e:
                        # Ensure scheduler visibility: unexpected worker-side
                        # exceptions must still mark the claimed task failed.
                        logger.exception(f"Unhandled task execution error for {task.id}: {e}")
                        result = ExecutionResult(
                            success=False,
                            exit_code=-1,
                            execution_time_ms=0,
                            error_message=str(e),
                        )
                    
                    # Report result (success/failure)
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
            # Early visibility for manual data path mode: warn before tasks fail.
            expected = ("data.npy", "labels.npy", "test_data.npy", "test_labels.npy")
            missing = [name for name in expected if not (self._dataset_path / name).exists()]
            if missing:
                logger.warning(
                    f"Dataset path {self._dataset_path} is missing files: {missing}"
                )
            if self._model_script_path:
                logger.info(f"Model script resolved to: {self._model_script_path}")
            else:
                logger.warning(
                    "Model script not resolved; tools that import `config_model` may fail. "
                    "Set backend model.script or place config_model.py in manual data path."
                )
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
        default=int(os.environ.get("LANDSEER_TASK_TIMEOUT_SECONDS", "7200")),
        metavar="SECONDS",
        help="Task execution timeout in seconds (default: 7200, env: LANDSEER_TASK_TIMEOUT_SECONDS). Use 0 or a negative value to disable timeout.",
    )
    exec_group.add_argument(
        "--runtime",
        type=str,
        choices=["docker", "apptainer", "singularity", "kubernetes", "auto"],
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
        default=os.environ.get("LANDSEER_CACHE_DIR", "/data/landseer/cache"),
        metavar="DIR",
        help="Artifact cache directory (default: /data/landseer/cache, env: LANDSEER_CACHE_DIR)",
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

    init_sentry("worker")
    
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
