"""
Command-line interface for Landseer worker.

The worker connects to the Landseer backend scheduler, claims tasks,
executes them in containers, and reports results back.
"""

import hashlib
import json
import os
import shutil
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import click
import yaml

from ..common import get_logger, init_sentry
from ..common import dataset_manifest as dsm
from ..common.fsutil import clone_file, clone_tree, same_filesystem
from .client import LandseerClient, TaskInfo
from .runner import TaskRunner, ExecutionResult, ContainerRuntime

try:
    from ..store import TwoLevelCache, CacheConfig
except ImportError as exc:  # pragma: no cover - install deps
    raise ImportError(
        "Worker cache requires src.store TwoLevelCache (MinIO). Install project deps."
    ) from exc

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
        task_timeout: int = 28000,
        heartbeat_interval: float = 30.0,
        use_cache: bool = True,
        runtime: Optional[str] = None,
        data_root: Optional[Path] = None,
    ):
        """
        Initialize the worker.
        
        Args:
            backend_url: URL of the Landseer backend API
            worker_id: Worker identifier (auto-generated if not provided)
            workspace_dir: Directory for task workspaces
            cache_dir: Directory for artifact caching
            data_path: Path to input data directory (dataset, model config);
                used only for runs whose dataset it matches
            gpu_id: GPU ID to use (None for CPU-only)
            poll_interval: Seconds between task polls when idle
            task_timeout: Task execution timeout in seconds
            heartbeat_interval: Seconds between heartbeats
            use_cache: Whether to use artifact caching
            runtime: Container runtime to use (auto-detect if None)
            data_root: Directory with <name>/<variant>/ dataset dirs, preferred
                over a MinIO download when the run's dir matches
                (default: LANDSEER_DATA_ROOT)
        """
        self.backend_url = backend_url
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"

        
        # Directories
        self.workspace_dir = workspace_dir or Path(f"/data/landseer/workers/{self.worker_id}")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(
                f"Timeout is {task_timeout} seconds."
            )
        
        self.cache_dir = cache_dir or Path(os.environ.get("LANDSEER_CACHE_DIR", "/data/landseer/cache"))
        self.data_path = data_path  # Path to input data (manual override)
        if data_root is None and os.environ.get("LANDSEER_DATA_ROOT"):
            data_root = Path(os.environ["LANDSEER_DATA_ROOT"])
        self.data_root = data_root  # <root>/<name>/<variant>/ manual dataset dirs
        self._dataset_path: Optional[Path] = None  # Dataset dir of the active run
        
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
        # task_id -> content_id of the dataset that task's output was produced
        # on (None when unknown, e.g. results reported by older workers). Kept
        # next to _task_outputs so local deps get the same lineage check as
        # remote ones.
        self._task_dataset_id: Dict[str, Optional[str]] = {}
        # Manifest of the dataset the backend serves for the active run
        # (from MinIO, the backend's local dir, or GET /dataset/manifest).
        self._dataset_manifest: Optional[Dict[str, Any]] = None
        # (resolved dataset dir, manifest content_id) -> (content_id, provenance_id)
        self._dataset_identity_memo: Dict[tuple, tuple] = {}
        # (resolved dir, manifest content_id) -> mismatch description (None: matches)
        self._dir_check_memo: Dict[tuple, Optional[str]] = {}
        # (name, variant, content_id) -> (dataset dir, manifest) already resolved
        self._resolved_datasets: Dict[tuple, tuple] = {}
        # run_id -> that run's dataset/model context (see _activate_run)
        self._run_contexts: Dict[str, Dict[str, Any]] = {}
        # Set when no verified copy of the active run's dataset could be found
        # and the manual data path does not match it; every task of that run
        # then fails with it.
        self._dataset_mismatch: Optional[str] = None

        # Components (initialized on start)
        self._client: Optional[LandseerClient] = None
        self._runner: Optional[TaskRunner] = None
        self._two_level_cache: Optional["TwoLevelCache"] = None
        self._model_script_path: Optional[Path] = None  # Path to model config script (e.g., config_model.py)
        self._dataset_info: Dict[str, Any] = {}
        self._active_run_id: Optional[str] = None
        
        # Signal handling
        self._setup_signal_handlers()
    
    def _get_minio_store_for_dataset_downloads(self):
        """
        MinIO client used to download the shared dataset directory and model script.
        Production TwoLevelCache exposes this as ``cache.manager.minio``; tests may
        inject ``_minio_store`` on a mock.
        """
        if not self._two_level_cache:
            return None
        tlc = self._two_level_cache
        mgr = getattr(tlc, "manager", None)
        if mgr is not None:
            m = getattr(mgr, "minio", None)
            if m is not None and getattr(m, "is_available", False):
                return m
        injected = getattr(tlc, "_minio_store", None)
        if injected is not None:
            return injected
        return None
    
    def _model_script_cache_path(self, dataset_info: Dict[str, Any]) -> Path:
        """Stable local path for ``config_model.py`` aligned with dataset cache layout."""
        name = dataset_info.get("name") or "dataset"
        variant = dataset_info.get("variant") or "clean"
        return self.cache_dir / "datasets" / str(name) / str(variant) / "config_model.py"

    @staticmethod
    def _file_sha256(path: Path) -> Optional[str]:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None

    def _model_script_ok(self, path: Optional[Path], sha: Optional[str]) -> bool:
        """path is a file and, when the backend sent a hash, has that content."""
        return bool(path and path.is_file() and (not sha or self._file_sha256(path) == sha))

    def _download_model_script(self, dataset_info: Dict[str, Any]) -> Optional[Path]:
        """
        Fetch ``config_model.py`` from MinIO when the backend published
        ``model_script_minio_key`` (remote workers do not have host-local ``model_script`` paths).

        With ``model_script_sha256`` the copy is stored by content hash, so runs
        with different model scripts on the same dataset never share one.
        """
        key = dataset_info.get("model_script_minio_key")
        if not key or not dataset_info.get("minio_available"):
            return None
        minio = self._get_minio_store_for_dataset_downloads()
        if minio is None:
            logger.warning(
                "Dataset advertises model_script_minio_key but worker has no MinIO client "
                "(enable two-level cache + MinIO)"
            )
            return None
        sha = dataset_info.get("model_script_sha256")
        if not sha:
            # Older backend: one script per dataset, kept next to its cache copy.
            dest = self._model_script_cache_path(dataset_info)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                return dest
            if minio.download_file(key, dest):
                logger.info(f"Downloaded model script from MinIO: {key} -> {dest}")
                return dest
            logger.warning(f"Failed to download model script from MinIO: {key}")
            return None
        dest = self.cache_dir / "model_scripts" / sha[:16] / "config_model.py"
        if self._model_script_ok(dest, sha):
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            if minio.download_file(key, tmp) and self._model_script_ok(tmp, sha):
                os.replace(tmp, dest)
                logger.info(f"Downloaded model script from MinIO: {key} -> {dest}")
                return dest
        finally:
            tmp.unlink(missing_ok=True)
        logger.warning(f"Failed to download model script {key} (sha256 {sha[:12]}) from MinIO")
        return None

    def _model_script_from_pipeline_config(self) -> Optional[Path]:
        """Model script named by LANDSEER_PIPELINE_CONFIG (a pinned pipeline yaml), if set."""
        cfg_path_raw = os.environ.get("LANDSEER_PIPELINE_CONFIG")
        if not cfg_path_raw:
            return None
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
                    return model_path
                logger.warning(f"Model script from LANDSEER_PIPELINE_CONFIG not found: {model_path}")
        except Exception as e:
            logger.warning(
                f"Failed to resolve model script from LANDSEER_PIPELINE_CONFIG={cfg_path_raw}: {e}"
            )
        return None

    def _resolve_model_script(self, dataset_info: Dict[str, Any], run_id: Optional[str]) -> Optional[Path]:
        """
        The model config script of the run described by dataset_info.

        Order: the backend's path if it exists here (and matches
        model_script_sha256), the MinIO copy, then LANDSEER_PIPELINE_CONFIG and,
        without a run id, the backend's pipeline info. A script whose content
        differs from the run's is never used.
        """
        sha = dataset_info.get("model_script_sha256")
        model_script = dataset_info.get("model_script")
        if model_script:
            model_path = Path(model_script)
            if self._model_script_ok(model_path, sha):
                logger.info(f"Model config script: {model_path}")
                return model_path
            if model_path.is_file():
                logger.info(
                    f"Local model script {model_path} differs from the run's (sha256 {sha[:12]}); "
                    "fetching the run's copy"
                )
            else:
                logger.warning(f"Model script path not found: {model_script}")
        downloaded = self._download_model_script(dataset_info)
        if downloaded:
            return downloaded
        pinned = self._model_script_from_pipeline_config()
        if pinned and self._model_script_ok(pinned, sha):
            logger.info(f"Model config script (pipeline yaml): {pinned}")
            return pinned
        if run_id is None and self._client is not None:
            # Pipeline info describes the backend's latest run only.
            try:
                pipeline_info = self._client.get_pipeline_info()
                model = pipeline_info.get("model") if isinstance(pipeline_info, dict) else None
                script = model.get("script") if isinstance(model, dict) else None
                if script and self._model_script_ok(Path(script), sha):
                    logger.info(f"Model config script (pipeline info): {script}")
                    return Path(script)
            except Exception as e:
                logger.debug(f"Failed to fetch pipeline info for model script: {e}")
        return None

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
        
        # Cache: local staging + required MinIO (GCS via ILM); local dropped after upload.
        if self.use_cache:
            self._two_level_cache = TwoLevelCache(
                CacheConfig(local_cache_dir=self.cache_dir)
            )
            minio = getattr(self._two_level_cache.manager, "minio", None)
            if minio is None or not minio.is_available:
                endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
                raise RuntimeError(
                    f"Cache requires MinIO (GCS-backed) but it is unavailable at {endpoint}. "
                    "Run helper_scripts/setup_minio_tiering.sh and check .env.db."
                )
            logger.info("Using two-level cache (local staging + MinIO/GCS)")
            if not same_filesystem(self.cache_dir, self.workspace_dir):
                logger.warning(
                    f"Cache dir {self.cache_dir} and workspace {self.workspace_dir} are on "
                    "different filesystems: cache stores and hits will use full copies"
                )
    
    def _fetch_dataset(self, run_id: Optional[str] = None) -> Optional[Path]:
        """
        Resolve the dataset and model script of run_id (None: the dataset the
        backend prepared last) and return the dataset directory.

        Dataset sources, first match wins:
        1. LANDSEER_DATA_PATH / --data-path, if it matches the run's dataset manifest
        2. LANDSEER_DATA_ROOT/<name>/<variant> (--data-root), if it matches
        3. the backend's own directory (worker on the backend host)
        4. a verified copy synced from MinIO into <cache>/datasets/<name>/<variant>
        A manual dir that does not match is skipped, not fatal: the run uses
        the MinIO copy. Only when no verified copy can be had is
        ``_dataset_mismatch`` set (the run's tasks then fail with it). Without a
        backend manifest manual dirs cannot be verified and are used as is.

        Resolved dirs are memoized per dataset (name, variant, content_id), so
        a worker alternating between runs does not re-verify or re-download.

        Returns:
            Path to dataset directory, or None if not available
        """
        self._dataset_mismatch = None
        try:
            dataset_info = self._client.get_dataset_info(run_id=run_id)
        except Exception as e:
            logger.warning(f"Failed to get dataset info from backend: {e}")
            dataset_info = {}
        if not isinstance(dataset_info, dict):
            dataset_info = {}
        self._dataset_info = dataset_info
        self._model_script_path = self._resolve_model_script(dataset_info, run_id)

        name, variant = dataset_info.get("name"), dataset_info.get("variant")
        content_id = dataset_info.get("content_id")
        identity = (name, variant, content_id)
        known = self._resolved_datasets.get(identity) if content_id else None
        if known and known[0].exists():
            self._dataset_manifest = known[1]
            logger.info(f"Dataset {name}/{variant} ({content_id[:12]}) already resolved at {known[0]}")
            return known[0]

        path = self._locate_dataset(run_id, dataset_info)
        if path is not None and content_id and not self._dataset_mismatch:
            self._resolved_datasets[identity] = (path, self._dataset_manifest)
        return path

    def _manual_dataset_dirs(self, dataset_info: Dict[str, Any]) -> List[tuple]:
        """Existing (label, dir) manual candidates for the dataset, in preference order."""
        dirs = []
        if self.data_path:
            dirs.append(("LANDSEER_DATA_PATH", self.data_path))
        if self.data_root and dataset_info.get("name"):
            dirs.append((
                "LANDSEER_DATA_ROOT",
                self.data_root / str(dataset_info["name"]) / str(dataset_info.get("variant") or "clean"),
            ))
        return [(label, d) for label, d in dirs if d.exists()]

    def _locate_dataset(self, run_id: Optional[str], dataset_info: Dict[str, Any]) -> Optional[Path]:
        """Dataset dir for _fetch_dataset: verified manual dirs first, then the backend/MinIO copy."""
        manual = self._manual_dataset_dirs(dataset_info)
        mismatch: Optional[str] = None
        if manual:
            manifest = self._client.get_dataset_manifest(run_id=run_id) if self._client else None
            if not isinstance(manifest, dict) or not manifest.get("content_id"):
                label, path = manual[0]
                logger.warning(
                    f"Backend has no dataset manifest; cannot verify {label} dir {path}; using it as is"
                )
                return path
            self._dataset_manifest = manifest
            for label, path in manual:
                if label == "LANDSEER_DATA_PATH":
                    problem = self._manual_dataset_mismatch(run_id, manifest=manifest, allow_override=False)
                else:
                    problem = self._dir_mismatch(path, manifest)
                    if problem:
                        problem = f"Dataset at {label} dir {path} does not match ({problem})"
                if problem is None:
                    logger.info(
                        f"Using {label} dir {path} for {dataset_info.get('name')}/"
                        f"{dataset_info.get('variant')} ({manifest['content_id'][:12]})"
                    )
                    return path
                logger.info(f"{problem}; trying the next dataset source")
                mismatch = mismatch or problem

        path = self._fetch_backend_dataset(dataset_info)
        if path is not None:
            if mismatch:
                logger.warning(
                    f"Manual dataset dirs do not hold {dataset_info.get('name')}/"
                    f"{dataset_info.get('variant')}; using the verified copy at {path}"
                )
            return path
        if mismatch is None:
            return None
        if os.environ.get("LANDSEER_ALLOW_DATASET_MISMATCH") == "1" and self.data_path and self.data_path.exists():
            # Last resort only: the override never wins over a verified copy.
            self._manual_dataset_mismatch(run_id, manifest=self._dataset_manifest, allow_override=True)
            return self.data_path
        self._dataset_mismatch = (
            f"{mismatch} No verified copy of this dataset is available from the backend or MinIO either."
        )
        return None

    def _fetch_backend_dataset(self, dataset_info: Dict[str, Any]) -> Optional[Path]:
        """The backend's dataset dir if visible here, else a verified MinIO copy."""
        if not dataset_info.get("available"):
            logger.info("No dataset available from backend")
            return None

        # Check if we can use local path (same machine as backend)
        local_path = dataset_info.get("local_path")
        if local_path:
            local_dir = Path(local_path)
            if local_dir.exists() and (local_dir / "data.npy").exists():
                logger.info(f"Using local dataset path: {local_dir}")
                self._dataset_manifest = dsm.load_manifest(local_dir)
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

                minio = self._get_minio_store_for_dataset_downloads()
                if minio is None:
                    logger.warning(
                        "Dataset is in MinIO but worker has no MinIO client "
                        "(two-level cache / MinIO)"
                    )
                else:
                    synced = self._sync_dataset_from_minio(
                        minio, minio_key, download_dir, dataset_info.get("manifest_minio_key")
                    )
                    if synced is not None:
                        return synced

            except Exception as e:
                logger.warning(f"Failed to download dataset from MinIO: {e}")

        logger.warning("Dataset not available locally or via MinIO")
        return None

    def _activate_run(self, run_id: str) -> None:
        """
        Switch the dataset/model context to run_id's.

        Each run's context is remembered, so a worker alternating between
        concurrent runs (e.g. a CIFAR-10 and a CelebA run) switches without
        refetching or re-verifying anything.
        """
        remembered = self._run_contexts.get(run_id)
        if remembered and remembered["path"].exists():
            logger.info(f"Run changed ({self._active_run_id} -> {run_id}); restoring its dataset/model context")
            self._dataset_info = remembered["info"]
            self._dataset_manifest = remembered["manifest"]
            self._dataset_path = remembered["path"]
            self._model_script_path = remembered["model"]
            self._dataset_mismatch = None
        else:
            logger.info(f"Run changed ({self._active_run_id} -> {run_id}); resolving its dataset/model context")
            self._model_script_path = None
            self._dataset_manifest = None
            self._dataset_mismatch = None
            self._dataset_path = self._fetch_dataset(run_id)
            self._remember_run_context(run_id)
        # Task IDs are run-scoped (src/pipeline/run_identity.py), so the
        # per-task lineage caches (_task_outputs, _task_dataset_id) stay
        # valid across run switches and are not cleared: a worker
        # alternating between concurrent runs keeps its local lineage.
        self._active_run_id = run_id

    def _remember_run_context(self, run_id: Optional[str]) -> None:
        """Remember the active dataset/model context for run_id (only once it resolved)."""
        if run_id and self._dataset_path is not None and not self._dataset_mismatch:
            self._run_contexts[run_id] = {
                "info": self._dataset_info,
                "manifest": self._dataset_manifest,
                "path": self._dataset_path,
                "model": self._model_script_path,
            }

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

    @staticmethod
    def _has_files(path: Optional[Path]) -> bool:
        """True if path is a directory that still holds at least one entry."""
        try:
            return path is not None and path.is_dir() and any(path.iterdir())
        except OSError:
            return False

    def _cached_output(self, cache_key: Optional[str]) -> Optional[Path]:
        """This worker's cache copy of cache_key's output (hydrating from MinIO if needed)."""
        if not cache_key:
            return None
        local = self.cache_dir / cache_key / "output"
        if (self.cache_dir / cache_key / ".success").exists() and self._has_files(local):
            return local
        if self._two_level_cache:
            try:
                hydrated = self._two_level_cache.get(cache_key)
                if hydrated and self._has_files(Path(hydrated)):
                    return Path(hydrated)
            except Exception as e:
                logger.warning(f"Failed to fetch cached output {cache_key[:12]}: {e}")
        return None

    def _usable_output(self, path: Optional[Path], cache_key: Optional[str]) -> Optional[Path]:
        """
        Return a dependency output dir that still has its files.

        Workspaces are pruned by age (LANDSEER_WORKSPACE_RETENTION_HOURS), so a
        remembered workspace path can be gone or empty by the time a downstream
        task runs; fall back to the cache copy of the same output.
        """
        if self._has_files(path):
            return path
        cached = self._cached_output(cache_key)
        if cached is not None:
            logger.info(f"Workspace output {path} is gone or empty; using cache copy {cached}")
        return cached

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
            # An existing but emptied workspace dir (pruned) is not a usable output.
            if self._has_files(path):
                return path

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

    def _collect_remote_ancestry(
        self,
        task_id: str,
        visited: Optional[set[str]] = None,
        missing: Optional[List[str]] = None,
    ) -> tuple[list[Path], Optional[TaskInfo]]:
        """Collect full ancestor outputs for a task by querying backend task metadata.

        Ancestors whose output cannot be found anywhere are appended to ``missing``.
        """
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
            dep_ancestry, dep_task = self._collect_remote_ancestry(dep_id, visited, missing)
            ancestry.extend(dep_ancestry)
            if dep_task:
                dep_output = self._resolve_task_output_path(dep_task)
                if dep_output:
                    ancestry.append(dep_output)
                    # Track remote deps locally so future tasks don't need refetch.
                    dep_cache_key = dep_task.cache_key or ""
                    self._task_outputs[dep_id] = (dep_cache_key, dep_output, list(dep_ancestry))
                    self._task_dataset_id[dep_id] = dep_task.dataset_content_id
                elif missing is not None:
                    missing.append(f"ancestor {dep_id} (no output in workspace, cache or MinIO)")

        return ancestry, task_info

    def _parent_dataset_error(
        self,
        dep_id: str,
        parent_sha: Optional[str],
        current_sha: Optional[str],
        data_dir: Optional[Path],
    ) -> Optional[str]:
        """
        Error message if a parent's output was produced on a different dataset
        than the one this worker has, else None. Unknown hashes only warn.
        """
        if parent_sha and current_sha:
            if parent_sha != current_sha:
                return (
                    f"parent {dep_id} was produced on dataset {parent_sha[:12]} but this "
                    f"worker has {current_sha[:12]} "
                    f"(LANDSEER_DATA_PATH={self.data_path or 'unset'}, dataset dir={data_dir})"
                )
            return None
        if not parent_sha:
            logger.warning(
                f"Dataset of parent {dep_id} is unknown (result predates dataset tracking); "
                "cannot verify lineage"
            )
        else:
            logger.warning(
                f"Parent {dep_id} was produced on dataset {parent_sha[:12]} but this worker "
                "has no dataset content_id; cannot verify lineage"
            )
        return None

    def _dir_mismatch(self, data_dir: Path, manifest: Dict[str, Any]) -> Optional[str]:
        """
        None if data_dir holds exactly the manifest's files, else what differs.

        Stat-only until every size matches (a dir of another dataset is
        rejected without hashing), then sha256 via the sidecar. Memoized per
        (dir, content_id), so switching runs never re-verifies a dir.
        """
        resolved = str(data_dir.resolve())
        key = (resolved, manifest["content_id"])
        if key in self._dir_check_memo:
            return self._dir_check_memo[key]
        res = dsm.verify_dir(data_dir, manifest, quick=True)
        if res.ok:
            prov_id = (manifest.get("provenance") or {}).get("provenance_id")
            self._dataset_identity_memo[key] = (manifest["content_id"], prov_id)
            problem = None
        else:
            problem = f"local copy {(res.content_id or 'none')[:12]}; {res.describe()}"
        self._dir_check_memo[key] = problem
        return problem

    def _manual_dataset_mismatch(
        self,
        run_id: Optional[str] = None,
        manifest: Optional[Dict[str, Any]] = None,
        allow_override: bool = True,
    ) -> Optional[str]:
        """
        In manual LANDSEER_DATA_PATH / --data-path mode, check the local copy
        against the backend's dataset manifest (of run_id's dataset). Returns
        an error message on mismatch, None when it matches or cannot be
        compared. With allow_override, LANDSEER_ALLOW_DATASET_MISMATCH=1 turns
        a mismatch into a warning (the dir's own content_id is then used).
        """
        if not (self.data_path and self.data_path.exists()):
            return None
        if manifest is None:
            manifest = self._client.get_dataset_manifest(run_id=run_id) if self._client else None
        if not isinstance(manifest, dict) or not manifest.get("content_id"):
            logger.warning(
                "Backend has no dataset manifest; cannot verify manual data path "
                f"{self.data_path}"
            )
            return None
        self._dataset_manifest = manifest
        problem = self._dir_mismatch(self.data_path, manifest)
        if problem is None:
            extra = dsm.list_unlisted_files(self.data_path, manifest)
            if extra:
                logger.warning(
                    f"{len(extra)} files in {self.data_path} are not in the dataset manifest "
                    f"and are ignored (e.g. {extra[:3]})"
                )
            logger.info(f"Manual data path matches backend dataset ({manifest['content_id'][:12]})")
            return None
        msg = (
            f"Dataset at LANDSEER_DATA_PATH={self.data_path} does not match the backend's "
            f"{self._dataset_info.get('name')}/{self._dataset_info.get('variant')} "
            f"(content_id {manifest['content_id'][:12]}; {problem}). "
            "Point the worker at the backend's dataset (or a LANDSEER_DATA_ROOT holding it), "
            "or set LANDSEER_ALLOW_DATASET_MISMATCH=1 to run anyway."
        )
        if allow_override and os.environ.get("LANDSEER_ALLOW_DATASET_MISMATCH") == "1":
            # Full verify (hashes every file) for the copy's own content_id.
            res = dsm.verify_dir(self.data_path, manifest)
            self._dataset_identity_memo[(str(self.data_path.resolve()), manifest["content_id"])] = (
                res.content_id, None
            )
            logger.warning(msg + " Continuing because LANDSEER_ALLOW_DATASET_MISMATCH=1.")
            return None
        return msg

    @contextmanager
    def _dataset_lock(self, name: str, variant: str):
        """flock shared by all workers using this cache dir, per dataset/variant."""
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX
            yield
            return
        lock_dir = self.cache_dir / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with open(lock_dir / f"dataset-{name}-{variant}.lock", "a+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _sync_dataset_from_minio(
        self,
        minio,
        minio_key: str,
        download_dir: Path,
        manifest_key: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Make download_dir an exact, verified copy of the dataset in MinIO.

        With a manifest: reuse the local copy if it verifies; otherwise build a
        staging dir (local files that still match are cloned, the rest
        downloaded), verify every file, and swap it into place. Without one
        (uploaded by an older backend): download once, unverified.
        """
        manifest_key = manifest_key or f"{minio_key.rstrip('/')}/{dsm.MANIFEST_NAME}"
        raw = minio.download_bytes(manifest_key) if minio.exists(manifest_key) else None
        manifest = dsm.parse_manifest(raw)
        if manifest is None:
            logger.warning(
                f"No dataset manifest at {manifest_key} (uploaded by an older backend); "
                "the dataset copy cannot be verified"
            )
            if not (download_dir / "data.npy").exists():
                logger.info(f"Downloading dataset from MinIO: {minio_key}")
                minio.download_directory(minio_key, download_dir)
            return download_dir

        self._dataset_manifest = manifest
        prov_id = (manifest.get("provenance") or {}).get("provenance_id")
        name, variant = download_dir.parent.name, download_dir.name
        with self._dataset_lock(name, variant):
            self._remove_stale_dataset_staging(download_dir.parent)
            local = dsm.load_manifest(download_dir)
            if local and local.get("content_id") == manifest["content_id"]:
                res = dsm.verify_dir(download_dir, manifest)
                if res.ok:
                    self._dataset_identity_memo[(str(download_dir.resolve()), manifest["content_id"])] = (
                        manifest["content_id"], prov_id
                    )
                    logger.info(f"Dataset cached and verified at {download_dir} ({manifest['content_id'][:12]})")
                    return download_dir
                logger.warning(f"Cached dataset at {download_dir} is corrupted ({res.describe()}); re-downloading")
            elif download_dir.exists() and any(download_dir.iterdir()):
                logger.warning(
                    f"Cached dataset at {download_dir} is stale "
                    f"({(local or {}).get('content_id', 'no manifest')[:12]} != "
                    f"{manifest['content_id'][:12]}); refreshing"
                )

            staging = download_dir.parent / f".tmp-{variant}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            try:
                self._build_dataset_staging(minio, minio_key, manifest, download_dir, staging)
                res = dsm.verify_dir(staging, manifest)
                if not res.ok:
                    raise RuntimeError(f"downloaded dataset does not match its manifest ({res.describe()})")
                dsm.write_manifest(staging, manifest)
                old = download_dir.parent / f".old-{variant}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                if download_dir.exists():
                    os.replace(download_dir, old)
                os.replace(staging, download_dir)
                shutil.rmtree(old, ignore_errors=True)
            except Exception as e:
                shutil.rmtree(staging, ignore_errors=True)
                logger.warning(f"Failed to sync dataset from MinIO ({minio_key}): {e}")
                return None

        self._dataset_identity_memo[(str(download_dir.resolve()), manifest["content_id"])] = (
            manifest["content_id"], prov_id
        )
        logger.info(f"Dataset downloaded and verified at {download_dir} ({manifest['content_id'][:12]})")
        return download_dir

    @staticmethod
    def _build_dataset_staging(minio, minio_key: str, manifest: Dict[str, Any], current: Path, staging: Path) -> None:
        """Fill staging with the manifest's files: clone matching local ones, download the rest."""
        files = manifest["files"]
        same_size = [
            rel for rel, exp in files.items()
            if (current / rel).is_file() and (current / rel).stat().st_size == int(exp["size"])
        ]
        local_hashes = dsm.hash_files(current, same_size) if same_size else {}
        reuse = {rel for rel, e in local_hashes.items() if e["sha256"] == files[rel]["sha256"]}

        staging.mkdir(parents=True)
        sidecar = current / dsm.HASH_SIDECAR
        if sidecar.exists():
            # Clones keep size and mtime, so their cached digests stay valid.
            shutil.copy2(sidecar, staging / dsm.HASH_SIDECAR)
        prefix = minio_key.rstrip("/")

        def _fetch(rel: str) -> None:
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if rel in reuse:
                clone_file(current / rel, dest)
            elif not minio.download_file(f"{prefix}/{rel}", dest):
                raise RuntimeError(f"failed to download {prefix}/{rel}")

        todo = list(files)
        if len(todo) > len(reuse):
            logger.info(
                f"Downloading {len(todo) - len(reuse)} dataset files from MinIO "
                f"(reusing {len(reuse)} unchanged local files)"
            )
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_fetch, todo))

    @staticmethod
    def _remove_stale_dataset_staging(parent: Path, max_age: float = 24 * 3600) -> None:
        """Drop .tmp-*/.old-* dirs left in datasets/<name>/ by crashed workers."""
        cutoff = time.time() - max_age
        try:
            children = list(parent.iterdir())
        except OSError:
            return
        for p in children:
            if p.name.startswith((".tmp-", ".old-")):
                try:
                    if p.stat().st_mtime < cutoff:
                        shutil.rmtree(p, ignore_errors=True)
                except OSError:
                    continue

    @staticmethod
    def _append_unique_paths(paths: List[Path], additions: List[Path]) -> None:
        """Append paths while preserving order and removing duplicates."""
        seen = {str(p.resolve()) if p.exists() else str(p) for p in paths}
        for p in additions:
            key = str(p.resolve()) if p.exists() else str(p)
            if key not in seen:
                paths.append(p)
                seen.add(key)
    
    def _dataset_identity(self, data_dir: Optional[Path]) -> tuple:
        """
        (content_id, provenance_id) of the dataset in data_dir.

        With the backend's manifest, the local copy is verified against it: a
        match gives the manifest's ids, a mismatch the local copy's own
        content_id. Without a manifest, every file in the dir is hashed.
        Memoized per (dir, manifest content_id), so it holds across run
        switches; the sidecar keeps first checks to a stat walk.
        """
        if not data_dir:
            return (None, None)
        try:
            resolved = str(data_dir.resolve())
        except OSError:
            return (None, None)
        manifest = self._dataset_manifest
        memo_key = (resolved, manifest.get("content_id") if manifest else None)
        if memo_key in self._dataset_identity_memo:
            return self._dataset_identity_memo[memo_key]
        try:
            if manifest:
                res = dsm.verify_dir(data_dir, manifest)
                if res.ok:
                    ident = (manifest["content_id"], (manifest.get("provenance") or {}).get("provenance_id"))
                else:
                    logger.warning(
                        f"Dataset at {data_dir} does not match the backend's manifest "
                        f"({res.describe()}); using its own content_id"
                    )
                    ident = (res.content_id, None)
            else:
                local = dsm.build_manifest(data_dir) if data_dir.is_dir() else None
                # An empty or missing dir is "no dataset", not a dataset with no files.
                ident = (local["content_id"] if local and local["file_count"] else None, None)
        except OSError as e:
            logger.warning(f"Could not hash dataset at {data_dir}: {e}")
            return (None, None)
        self._dataset_identity_memo[memo_key] = ident
        return ident

    def _dataset_fingerprint(self, data_dir: Optional[Path]) -> Dict[str, Any]:
        """Dataset identity for cache keys: manifest content_id, identical on any host."""
        content_id, _ = self._dataset_identity(data_dir)
        if not content_id:
            return {"available": False}
        return {
            "available": True,
            "content_id": content_id,
            "name": self._dataset_info.get("name"),
            "variant": self._dataset_info.get("variant"),
        }

    @staticmethod
    def _model_fingerprint(model_script_path: Optional[Path]) -> Dict[str, Any]:
        """Model-script identity for cache keys: content only, identical on any host."""
        if not model_script_path:
            return {"available": False}
        try:
            if not model_script_path.exists():
                return {"available": False}
            hasher = hashlib.sha256()
            with model_script_path.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return {"available": True, "sha256": hasher.hexdigest()}
        except OSError as e:
            logger.warning(f"Could not hash model script {model_script_path}: {e}")
            return {"available": False}

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
            # Switch dataset/model context when the pipeline run changes: runs
            # can use different datasets and model scripts, and the cache key
            # must reflect the task's own run.
            if task.run_id and task.run_id != self._active_run_id:
                self._activate_run(task.run_id)

            data_dir = self._dataset_path
            if data_dir is None and task.run_id is None and self.data_path and self.data_path.exists():
                # Task without a run id (older backend): the manual dir as before.
                data_dir = self.data_path
            if data_dir is None and not self._dataset_mismatch:
                # Headless backend may expose dataset only after a run is started.
                # Retry dataset fetch at task execution time instead of only worker startup.
                refreshed = self._fetch_dataset(task.run_id)
                if refreshed:
                    self._dataset_path = refreshed
                    data_dir = refreshed
                    self._remember_run_context(task.run_id)

            if self._dataset_mismatch:
                logger.error(f"Refusing to run {task.id}: {self._dataset_mismatch}")
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    execution_time_ms=0,
                    error_message=self._dataset_mismatch,
                )
            dataset_id, dataset_prov_id = self._dataset_identity(data_dir)

            # Build parent_hashes (for cache key) and ancestor_dirs (ordered
            # output dirs from all upstream stages, earliest first) in one pass.
            # Refuse to build on a parent produced from a different dataset.
            parent_hashes: List[str] = []
            ancestor_dirs: List[Path] = []
            # Every upstream output must be present: running on a partial input
            # set (e.g. no model.pt) yields wrong results that then get cached.
            missing_deps: List[str] = []
            key_by_output = {
                str(out): key for key, out, _ in self._task_outputs.values() if out is not None and key
            }
            for dep_id in task.dependency_ids:
                if dep_id in self._task_outputs:
                    dep_key, dep_out, dep_anc = self._task_outputs[dep_id]
                    dep_id_hash = self._task_dataset_id.get(dep_id)
                    if dep_key:
                        parent_hashes.append(dep_key)
                    resolved_anc: List[Path] = []
                    for anc in dep_anc:
                        usable = self._usable_output(anc, key_by_output.get(str(anc)))
                        if usable is None:
                            missing_deps.append(f"ancestor of {dep_id} ({anc})")
                        else:
                            resolved_anc.append(usable)
                    self._append_unique_paths(ancestor_dirs, resolved_anc)
                    usable_out = self._usable_output(dep_out, dep_key)
                    if usable_out is None:
                        missing_deps.append(f"{dep_id} ({dep_out})")
                    else:
                        self._append_unique_paths(ancestor_dirs, [usable_out])
                        self._task_outputs[dep_id] = (dep_key, usable_out, resolved_anc)
                else:
                    logger.warning(f"Dep {dep_id} not yet in _task_outputs; resolving remotely")
                    dep_anc, dep_task = self._collect_remote_ancestry(dep_id, missing=missing_deps)
                    dep_id_hash = dep_task.dataset_content_id if dep_task else None
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
                            self._task_dataset_id[dep_id] = dep_id_hash
                        else:
                            missing_deps.append(f"{dep_id} (no output in workspace, cache or MinIO)")
                    else:
                        ws_out = self._runner.workspace_dir / dep_id / "output"
                        if self._has_files(ws_out):
                            self._append_unique_paths(ancestor_dirs, [ws_out])
                        else:
                            missing_deps.append(f"{dep_id} (unknown to backend, no workspace output)")

                lineage_error = self._parent_dataset_error(dep_id, dep_id_hash, dataset_id, data_dir)
                if lineage_error:
                    logger.error(f"Refusing to run {task.id}: {lineage_error}")
                    return ExecutionResult(
                        success=False,
                        exit_code=-1,
                        execution_time_ms=0,
                        error_message=lineage_error,
                        artifacts={
                            "dataset_content_id": dataset_id,
                            "dataset_provenance_id": dataset_prov_id,
                        },
                    )

            if missing_deps:
                error = "Dependency output missing: " + "; ".join(missing_deps)
                logger.error(f"Refusing to run {task.id}: {error}")
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    execution_time_ms=0,
                    error_message=error,
                    artifacts={
                        "dataset_content_id": dataset_id,
                        "dataset_provenance_id": dataset_prov_id,
                    },
                )

            cache_context = self._build_cache_context(data_dir)
            cache_key = self._compute_cache_key(task, parent_hashes, cache_context)
            run_use_cache = bool(task.config.get("_run_use_cache", True))
            effective_use_cache = self.use_cache and run_use_cache
            extra_mounts: Optional[Dict[str, str]] = None

            if effective_use_cache:
                cached_path = self._check_cache(cache_key, task, parent_hashes, cache_context)
                if cached_path:
                    logger.info(f"Cache hit for task {task.id}")
                    workspace_out = self._runner.workspace_dir / task.id / "output"
                    if workspace_out.exists() or workspace_out.is_symlink():
                        if workspace_out.is_dir() and not workspace_out.is_symlink():
                            shutil.rmtree(workspace_out)
                        else:
                            workspace_out.unlink()
                    workspace_out.parent.mkdir(parents=True, exist_ok=True)
                    self._materialize_cache_hit(cache_key, cached_path, workspace_out)
                    self._task_outputs[task.id] = (cache_key, workspace_out, list(ancestor_dirs))
                    self._task_dataset_id[task.id] = dataset_id
                    return ExecutionResult(
                        success=True,
                        exit_code=0,
                        execution_time_ms=0,
                        output_path=workspace_out,
                        artifacts={
                            "cache_hit": True,
                            "cache_key": cache_key,
                            "dataset_content_id": dataset_id,
                            "dataset_provenance_id": dataset_prov_id,
                        }
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
            self._task_dataset_id[task.id] = dataset_id
            result.artifacts["dataset_content_id"] = dataset_id
            result.artifacts["dataset_provenance_id"] = dataset_prov_id

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
        """Check MinIO-backed cache for a task output."""
        if not self._two_level_cache:
            return None
        return self._two_level_cache.get(cache_key)
    
    def _materialize_cache_hit(self, cache_key: str, cached_path: Path, workspace_out: Path) -> None:
        """
        Give the task workspace the cached files without a second full write.

        Entries the local cache keeps are cloned (reflink, else hard link, else
        copy) and stay available for later hits. Entries it will not keep
        (keep-local off, or larger than the local cap) are moved instead.
        """
        manager = getattr(self._two_level_cache, "manager", None)
        keep = True
        if manager is not None:
            try:
                keep = bool(manager.should_keep_local(cache_key))
            except Exception:
                keep = True
        if keep or manager is None:
            clone_tree(cached_path, workspace_out)
            return
        with manager._entry_lock(cache_key):
            # Drop the completion flag first so no reader sees an entry
            # whose output has been moved away.
            try:
                (manager.get_local_path(cache_key) / ".success").unlink()
            except (OSError, TypeError):
                pass
            try:
                os.replace(cached_path, workspace_out)
            except OSError:
                clone_tree(cached_path, workspace_out)
        manager.delete_local(cache_key)

    def _store_in_cache(
        self,
        cache_key: str,
        task: TaskInfo,
        output_path: Path,
        execution_time_ms: int,
        parent_hashes: List[str],
        cache_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store task output locally and upload to MinIO (GCS via ILM)."""
        if not self._two_level_cache:
            raise RuntimeError("Cache store requested but two-level cache is not initialized")
        self._two_level_cache.put(
            cache_key=cache_key,
            source_dir=output_path,
            task_id=task.id,
            tool_name=task.tool_name,
            parent_hashes=parent_hashes,
            metadata={"execution_time_ms": execution_time_ms},
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
        # The backend's latest dataset; each run's own dataset is resolved when
        # its first task arrives, so a mismatch here is not fatal.
        self._dataset_path = self._fetch_dataset()
        if self._dataset_mismatch:
            logger.warning(f"{self._dataset_mismatch} Runs on other datasets are unaffected.")
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


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Landseer Worker - ML Security Pipeline Task Worker",
    epilog="""\
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
""",
)
@click.version_option(version="0.1.0", prog_name="landseer-worker")
@click.option(
    "--backend-url",
    default=lambda: os.environ.get("LANDSEER_BACKEND_URL", "http://localhost:8000"),
    show_default="http://localhost:8000",
    help="Backend API URL (env: LANDSEER_BACKEND_URL)",
)
@click.option(
    "--worker-id",
    default=lambda: os.environ.get("LANDSEER_WORKER_ID"),
    help="Unique worker identifier (auto-generated if not provided, env: LANDSEER_WORKER_ID)",
)
@click.option(
    "--gpu",
    type=int,
    default=None,
    metavar="ID",
    help="GPU ID to use (default: CPU only)",
)
@click.option(
    "--timeout",
    type=int,
    default=lambda: int(os.environ.get("LANDSEER_TASK_TIMEOUT_SECONDS", "28000")),
    show_default="28000",
    metavar="SECONDS",
    help=(
        "Task execution timeout in seconds (env: LANDSEER_TASK_TIMEOUT_SECONDS). "
        "Use 0 or a negative value to disable timeout."
    ),
)
@click.option(
    "--runtime",
    type=click.Choice(["docker", "apptainer", "singularity", "auto"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Container runtime to use",
)
@click.option(
    "--workspace",
    default=lambda: os.environ.get("LANDSEER_WORKSPACE"),
    type=click.Path(),
    metavar="DIR",
    help="Workspace directory for task execution (env: LANDSEER_WORKSPACE)",
)
@click.option(
    "--cache-dir",
    default=lambda: os.environ.get("LANDSEER_CACHE_DIR", "/data/landseer/cache"),
    show_default="/data/landseer/cache",
    type=click.Path(),
    metavar="DIR",
    help="Artifact cache directory (env: LANDSEER_CACHE_DIR)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable artifact caching",
)
@click.option(
    "--data-path",
    default=lambda: os.environ.get("LANDSEER_DATA_PATH"),
    type=click.Path(),
    metavar="DIR",
    help=(
        "Manual path to one dataset directory, used for runs whose dataset it matches; "
        "other runs fetch theirs from the backend/MinIO (env: LANDSEER_DATA_PATH)"
    ),
)
@click.option(
    "--data-root",
    default=lambda: os.environ.get("LANDSEER_DATA_ROOT"),
    type=click.Path(),
    metavar="DIR",
    help=(
        "Directory holding local dataset copies as <name>/<variant>/ (e.g. cifar10/clean). "
        "A run's dataset is taken from here when it matches the backend's manifest, "
        "else synced from MinIO (env: LANDSEER_DATA_ROOT)"
    ),
)
@click.option(
    "--minio-endpoint",
    default=lambda: os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
    show_default="localhost:9000",
    metavar="HOST:PORT",
    help="MinIO server endpoint (env: MINIO_ENDPOINT)",
)
@click.option(
    "--poll-interval",
    type=float,
    default=5.0,
    show_default=True,
    metavar="SECONDS",
    help="Seconds between task polls when idle",
)
@click.option(
    "--heartbeat-interval",
    type=float,
    default=30.0,
    show_default=True,
    metavar="SECONDS",
    help="Seconds between heartbeats",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        case_sensitive=False,
    ),
    default="INFO",
    show_default=True,
    help="Logging level",
)
def cli(
    backend_url: str,
    worker_id: Optional[str],
    gpu: Optional[int],
    timeout: int,
    runtime: str,
    workspace: Optional[str],
    cache_dir: str,
    no_cache: bool,
    data_path: Optional[str],
    data_root: Optional[str],
    minio_endpoint: str,
    poll_interval: float,
    heartbeat_interval: float,
    debug: bool,
    log_level: str,
) -> int:
    """Run the Landseer worker."""
    init_sentry("worker")

    from ..common.pylogger import set_global_log_level
    import logging

    set_global_log_level(getattr(logging, log_level.upper()))
    if debug:
        set_global_log_level(logging.DEBUG)

    resolved_runtime = None if runtime.lower() == "auto" else runtime
    os.environ["MINIO_ENDPOINT"] = minio_endpoint

    worker = Worker(
        backend_url=backend_url,
        worker_id=worker_id,
        workspace_dir=Path(workspace) if workspace else None,
        cache_dir=Path(cache_dir),
        data_path=Path(data_path) if data_path else None,
        data_root=Path(data_root) if data_root else None,
        gpu_id=gpu,
        poll_interval=poll_interval,
        task_timeout=timeout,
        heartbeat_interval=heartbeat_interval,
        use_cache=not no_cache,
        runtime=resolved_runtime,
    )
    return worker.start()


def parse_args(argv: Optional[list[str]] = None) -> SimpleNamespace:
    """Parse CLI arguments without starting the worker (for tests)."""
    with cli.make_context("landseer-worker", list(argv or [])) as ctx:
        return SimpleNamespace(**ctx.params)


def create_parser():
    """Test helper exposing parse_args(), matching the old argparse API."""

    class _Parser:
        def parse_args(self, argv: Optional[list[str]] = None) -> SimpleNamespace:
            return parse_args(argv)

    return _Parser()


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the worker CLI."""
    try:
        result = cli.main(
            args=argv,
            prog_name="landseer-worker",
            standalone_mode=False,
        )
        return int(result) if result is not None else 0
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1


if __name__ == "__main__":
    sys.exit(main())
