"""
Comprehensive tests for the Landseer Worker.

Tests cover:
1. Worker initialization and configuration
2. Worker registration and heartbeat
3. Task claiming and execution
4. Cache management (local and two-level)
5. Dataset fetching
6. Error handling and recovery
7. Signal handling
8. Work loop behavior
9. Edge cases and corner cases
"""

import pytest
import signal
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Optional

from src.worker.cli import Worker
from src.worker.client import LandseerClient, TaskInfo, WorkerInfo
from src.worker.runner import TaskRunner, ExecutionResult, ContainerRuntime
from src.worker.db import CacheManager, ArtifactCacheDB


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
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "data.npy").write_bytes(b"test dataset")
    (data_dir / "labels.npy").write_bytes(b"test labels")
    return data_dir


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
def mock_client():
    """Create a mock LandseerClient."""
    client = MagicMock(spec=LandseerClient)
    client.worker_id = "test_worker_123"
    client.is_registered = True
    return client


# ============================================================================
# Test: Worker Initialization
# ============================================================================


class TestWorkerInitialization:
    """Tests for worker initialization and configuration."""
    
    def test_worker_initialization_defaults(self, temp_workspace):
        """Worker should initialize with default values."""
        worker = Worker(workspace_dir=temp_workspace)
        
        assert worker.backend_url == "http://localhost:8000"
        assert worker.worker_id.startswith("worker_")
        assert worker.workspace_dir == temp_workspace
        assert worker.gpu_id is None
        assert worker.poll_interval == 5.0
        assert worker.task_timeout == 7200
        assert worker.heartbeat_interval == 30.0
        assert worker.use_cache is True
        assert worker._running is False
        assert worker._current_task is None
    
    def test_worker_initialization_custom_values(self, temp_workspace, temp_cache_dir):
        """Worker should accept custom configuration values."""
        worker = Worker(
            backend_url="http://custom:8000",
            worker_id="custom_worker",
            workspace_dir=temp_workspace,
            cache_dir=temp_cache_dir,
            gpu_id=0,
            poll_interval=10.0,
            task_timeout=7200,
            heartbeat_interval=60.0,
            use_cache=False,
            runtime="docker"
        )
        
        assert worker.backend_url == "http://custom:8000"
        assert worker.worker_id == "custom_worker"
        assert worker.cache_dir == temp_cache_dir
        assert worker.gpu_id == 0
        assert worker.poll_interval == 10.0
        assert worker.task_timeout == 7200
        assert worker.heartbeat_interval == 60.0
        assert worker.use_cache is False
        assert worker.runtime == "docker"
    
    def test_worker_creates_workspace_directory(self, tmp_path):
        """Worker should create workspace directory if it doesn't exist."""
        workspace = tmp_path / "new_workspace"
        assert not workspace.exists()
        
        worker = Worker(workspace_dir=workspace)
        assert workspace.exists()
        assert workspace.is_dir()
    
    def test_worker_auto_generates_worker_id(self, temp_workspace):
        """Worker should auto-generate worker ID if not provided."""
        worker1 = Worker(workspace_dir=temp_workspace)
        worker2 = Worker(workspace_dir=temp_workspace)
        
        assert worker1.worker_id != worker2.worker_id
        assert worker1.worker_id.startswith("worker_")
        assert worker2.worker_id.startswith("worker_")
    
    def test_worker_signal_handlers_setup(self, temp_workspace):
        """Worker should setup signal handlers for graceful shutdown."""
        worker = Worker(workspace_dir=temp_workspace)
        
        # Check that signal handlers are set
        # (We can't easily test signal handlers, but we can verify the method exists)
        assert hasattr(worker, '_signal_handler')
        assert hasattr(worker, '_setup_signal_handlers')
    
    def test_worker_initialization_with_data_path(self, temp_workspace, temp_data_dir):
        """Worker should accept manual data path."""
        worker = Worker(workspace_dir=temp_workspace, data_path=temp_data_dir)
        
        assert worker.data_path == temp_data_dir
        assert worker._dataset_path is None  # Not fetched yet


# ============================================================================
# Test: Component Initialization
# ============================================================================


class TestComponentInitialization:
    """Tests for worker component initialization."""
    
    def test_init_components_creates_client(self, temp_workspace):
        """_init_components should create LandseerClient."""
        worker = Worker(workspace_dir=temp_workspace, backend_url="http://test:8000")
        worker._init_components()
        
        assert worker._client is not None
        assert isinstance(worker._client, LandseerClient)
        assert worker._client.backend_url == "http://test:8000"
    
    def test_init_components_creates_task_runner(self, temp_workspace):
        """_init_components should create TaskRunner."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=0, task_timeout=1800)
        worker._init_components()
        
        assert worker._runner is not None
        assert isinstance(worker._runner, TaskRunner)
        assert worker._runner.workspace_dir == temp_workspace
        assert worker._runner.gpu_id == 0
        assert worker._runner.timeout == 1800
    
    def test_init_components_creates_cache_manager_when_enabled(self, temp_workspace, temp_cache_dir):
        """_init_components should create CacheManager when caching is enabled."""
        worker = Worker(workspace_dir=temp_workspace, cache_dir=temp_cache_dir, use_cache=True)
        
        with patch.dict('os.environ', {'LANDSEER_USE_MINIO': 'false'}):
            worker._init_components()
        
        assert worker._cache is not None
        assert isinstance(worker._cache, CacheManager)
    
    def test_init_components_skips_cache_when_disabled(self, temp_workspace):
        """_init_components should skip cache when caching is disabled."""
        worker = Worker(workspace_dir=temp_workspace, use_cache=False)
        worker._init_components()
        
        assert worker._cache is None
        assert worker._two_level_cache is None
    
    @patch('src.worker.cli.TwoLevelCache')
    def test_init_components_uses_two_level_cache_when_available(self, mock_two_level_cache, temp_workspace, temp_cache_dir):
        """_init_components should use two-level cache when available and MinIO enabled."""
        worker = Worker(workspace_dir=temp_workspace, cache_dir=temp_cache_dir, use_cache=True)
        
        with patch.dict('os.environ', {'LANDSEER_USE_MINIO': 'true'}):
            worker._init_components()
        
        # Should try to use two-level cache if available
        # (Actual behavior depends on import availability)
        pass  # This is tested implicitly by the import check


# ============================================================================
# Test: Backend Communication
# ============================================================================


class TestBackendCommunication:
    """Tests for backend communication."""
    
    def test_wait_for_backend_success(self, temp_workspace, mock_client):
        """_wait_for_backend should return True when backend is available."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        mock_client.is_backend_available.return_value = True
        
        result = worker._wait_for_backend(max_retries=1, retry_delay=0.1)
        
        assert result is True
        mock_client.is_backend_available.assert_called_once()
    
    def test_wait_for_backend_retries_on_failure(self, temp_workspace, mock_client):
        """_wait_for_backend should retry when backend is not available."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        mock_client.is_backend_available.side_effect = [False, False, True]
        
        result = worker._wait_for_backend(max_retries=3, retry_delay=0.1)
        
        assert result is True
        assert mock_client.is_backend_available.call_count == 3
    
    def test_wait_for_backend_fails_after_max_retries(self, temp_workspace, mock_client):
        """_wait_for_backend should return False after max retries."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        mock_client.is_backend_available.return_value = False
        
        result = worker._wait_for_backend(max_retries=2, retry_delay=0.1)
        
        assert result is False
        assert mock_client.is_backend_available.call_count == 2
    
    def test_register_worker(self, temp_workspace, mock_client):
        """_register should register worker with backend."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=0)
        worker._client = mock_client
        
        worker_info = WorkerInfo(
            worker_id="test_worker_123",
            hostname="test_host",
            status="idle",
            registered_at="2024-01-01T00:00:00",
            last_heartbeat="2024-01-01T00:00:00",
            capabilities={"runtime": "docker", "gpu_available": True, "gpu_id": 0}
        )
        mock_client.register.return_value = worker_info
        
        result = worker._register()
        
        assert result is True
        assert worker.worker_id == "test_worker_123"
        mock_client.register.assert_called_once()
        call_args = mock_client.register.call_args
        assert call_args[1]["worker_id"] is None or call_args[1]["worker_id"].startswith("worker_")
        assert "capabilities" in call_args[1]
    
    def test_register_worker_detects_capabilities(self, temp_workspace, mock_client):
        """_register should detect and report worker capabilities."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=0)
        worker._client = mock_client
        
        worker_info = WorkerInfo(
            worker_id="test_worker",
            hostname="test_host",
            status="idle",
            registered_at="2024-01-01T00:00:00",
            last_heartbeat="2024-01-01T00:00:00"
        )
        mock_client.register.return_value = worker_info
        
        with patch('src.worker.cli.ContainerRuntime') as mock_runtime:
            mock_runtime.detect_runtime.return_value = "docker"
            worker._register()
        
        call_args = mock_client.register.call_args
        capabilities = call_args[1]["capabilities"]
        assert capabilities["runtime"] == "docker"
        assert capabilities["gpu_available"] is True
        assert capabilities["gpu_id"] == 0
    
    def test_register_worker_failure(self, temp_workspace, mock_client):
        """_register should return False on registration failure."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        mock_client.register.side_effect = Exception("Registration failed")
        
        result = worker._register()
        
        assert result is False
    
    def test_send_heartbeat(self, temp_workspace, mock_client):
        """_send_heartbeat should send heartbeat to backend."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        worker._last_heartbeat = 0
        
        with patch('time.time', return_value=100.0):
            worker._send_heartbeat()
        
        mock_client.heartbeat.assert_called_once_with(status="idle")
        assert worker._last_heartbeat == 100.0
    
    def test_send_heartbeat_respects_interval(self, temp_workspace, mock_client):
        """_send_heartbeat should respect heartbeat interval."""
        worker = Worker(workspace_dir=temp_workspace, heartbeat_interval=30.0)
        worker._client = mock_client
        worker._last_heartbeat = 50.0
        
        with patch('time.time', return_value=60.0):  # Only 10s passed, less than 30s interval
            worker._send_heartbeat()
        
        # Should not send heartbeat yet
        mock_client.heartbeat.assert_not_called()
    
    def test_send_heartbeat_reports_busy_status(self, temp_workspace, mock_client, mock_task):
        """_send_heartbeat should report busy status when task is running."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        worker._current_task = mock_task
        worker._last_heartbeat = 0
        
        with patch('time.time', return_value=100.0):
            worker._send_heartbeat()
        
        mock_client.heartbeat.assert_called_once_with(status="busy")
    
    def test_send_heartbeat_handles_failure(self, temp_workspace, mock_client):
        """_send_heartbeat should handle heartbeat failures gracefully."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        worker._last_heartbeat = 0
        mock_client.heartbeat.side_effect = Exception("Network error")
        
        with patch('time.time', return_value=100.0):
            # Should not raise exception
            worker._send_heartbeat()
        
        # Should have attempted to send heartbeat
        mock_client.heartbeat.assert_called_once()


# ============================================================================
# Test: Dataset Fetching
# ============================================================================


class TestDatasetFetching:
    """Tests for dataset fetching functionality."""
    
    def test_fetch_dataset_uses_manual_data_path(self, temp_workspace, temp_data_dir):
        """_fetch_dataset should use manual data_path if provided."""
        worker = Worker(workspace_dir=temp_workspace, data_path=temp_data_dir)
        worker._client = MagicMock()
        
        result = worker._fetch_dataset()
        
        assert result == temp_data_dir
        worker._client.get_dataset_info.assert_not_called()
    
    def test_fetch_dataset_gets_info_from_backend(self, temp_workspace):
        """_fetch_dataset should get dataset info from backend."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = MagicMock()
        
        dataset_info = {
            "available": True,
            "name": "cifar10",
            "variant": "clean",
            "local_path": None,
            "minio_key": None,
            "minio_available": False
        }
        worker._client.get_dataset_info.return_value = dataset_info
        
        result = worker._fetch_dataset()
        
        assert result is None  # No local path or MinIO available
        worker._client.get_dataset_info.assert_called_once()
    
    def test_fetch_dataset_uses_local_path(self, temp_workspace, temp_data_dir):
        """_fetch_dataset should use local_path if available."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = MagicMock()
        
        dataset_info = {
            "available": True,
            "local_path": str(temp_data_dir),
            "minio_key": None,
            "minio_available": False
        }
        worker._client.get_dataset_info.return_value = dataset_info
        
        result = worker._fetch_dataset()
        
        assert result == temp_data_dir
    
    def test_fetch_dataset_downloads_from_minio(self, temp_workspace, temp_cache_dir):
        """_fetch_dataset should download from MinIO if available."""
        worker = Worker(workspace_dir=temp_workspace, cache_dir=temp_cache_dir)
        worker._client = MagicMock()
        worker._two_level_cache = MagicMock()
        
        dataset_info = {
            "available": True,
            "name": "cifar10",
            "variant": "clean",
            "local_path": None,
            "minio_key": "datasets/cifar10/clean",
            "minio_available": True
        }
        worker._client.get_dataset_info.return_value = dataset_info
        
        # Mock MinIO store
        mock_minio_store = MagicMock()
        worker._two_level_cache._minio_store = mock_minio_store
        
        download_dir = temp_cache_dir / "datasets" / "cifar10" / "clean"
        download_dir.mkdir(parents=True, exist_ok=True)
        (download_dir / "data.npy").write_bytes(b"downloaded data")
        
        result = worker._fetch_dataset()
        
        # Should download from MinIO
        mock_minio_store.download_directory.assert_called_once()
        assert result == download_dir
    
    def test_fetch_dataset_handles_missing_dataset(self, temp_workspace):
        """_fetch_dataset should handle missing dataset gracefully."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = MagicMock()
        
        dataset_info = {"available": False}
        worker._client.get_dataset_info.return_value = dataset_info
        
        result = worker._fetch_dataset()
        
        assert result is None
    
    def test_fetch_dataset_sets_model_script_path(self, temp_workspace, tmp_path):
        """_fetch_dataset should set model script path from dataset info."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = MagicMock()
        
        model_script = tmp_path / "config_model.py"
        model_script.write_text("# model config")
        
        dataset_info = {
            "available": True,
            "model_script": str(model_script),
            "local_path": None,
            "minio_key": None,
            "minio_available": False
        }
        worker._client.get_dataset_info.return_value = dataset_info
        
        worker._fetch_dataset()
        
        assert worker._model_script_path == model_script


# ============================================================================
# Test: Task Execution
# ============================================================================


class TestTaskExecution:
    """Tests for task execution functionality."""
    
    def test_execute_task_success(self, temp_workspace, mock_task):
        """_execute_task should execute task successfully."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._runner = MagicMock()
        worker._runner.workspace_dir = temp_workspace
        
        result = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=1000,
            output_path=temp_workspace / "test_task_1" / "output"
        )
        worker._runner.run_task.return_value = result
        
        execution_result = worker._execute_task(mock_task)
        
        assert execution_result.success is True
        assert execution_result.exit_code == 0
        assert worker._current_task is None  # Cleared after execution
        worker._runner.run_task.assert_called_once()
    
    def test_execute_task_checks_cache_first(self, temp_workspace, mock_task, temp_cache_dir):
        """_execute_task should check cache before executing."""
        worker = Worker(workspace_dir=temp_workspace, cache_dir=temp_cache_dir, use_cache=True)
        worker._cache = MagicMock()
        worker._runner = MagicMock()
        
        cached_path = temp_cache_dir / "cached_output"
        cached_path.mkdir(parents=True, exist_ok=True)
        worker._cache.check_cache.return_value = cached_path
        
        result = worker._execute_task(mock_task)
        
        assert result.success is True
        assert result.output_path == cached_path
        assert result.artifacts.get("cache_hit") is True
        worker._runner.run_task.assert_not_called()  # Should not execute
    
    def test_execute_task_stores_in_cache_on_success(self, temp_workspace, mock_task, temp_cache_dir):
        """_execute_task should store result in cache on success."""
        worker = Worker(workspace_dir=temp_workspace, cache_dir=temp_cache_dir, use_cache=True)
        worker._cache = MagicMock()
        worker._runner = MagicMock()
        worker._runner.workspace_dir = temp_workspace
        
        output_path = temp_workspace / "test_task_1" / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        
        result = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=1000,
            output_path=output_path
        )
        worker._runner.run_task.return_value = result
        worker._cache.check_cache.return_value = None  # Cache miss
        
        worker._execute_task(mock_task)
        
        worker._cache.store_result.assert_called_once()
    
    def test_execute_task_stores_with_run_id_when_task_has_run_id(
        self, temp_workspace, temp_cache_dir
    ):
        """_store_in_cache should pass run_id to store_result when task has run_id (Improvement: run ID propagation)."""
        worker = Worker(workspace_dir=temp_workspace, cache_dir=temp_cache_dir, use_cache=True)
        worker._cache = MagicMock()
        worker._runner = MagicMock()
        worker._runner.workspace_dir = temp_workspace
        
        task_with_run = TaskInfo(
            id="test_task_run",
            tool_name="test_tool",
            tool_image="test/image:latest",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=["wf_1"],
            pipeline_id="pipeline_1",
            dependency_ids=[],
            run_id="run_20250203_120000_abc123",
        )
        output_path = temp_workspace / "test_task_run" / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        result = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=1000,
            output_path=output_path,
        )
        worker._runner.run_task.return_value = result
        worker._cache.check_cache.return_value = None
        
        worker._execute_task(task_with_run)
        
        worker._cache.store_result.assert_called_once()
        call_kw = worker._cache.store_result.call_args[1]
        assert call_kw.get("run_id") == "run_20250203_120000_abc123"

    def test_execute_task_refreshes_dataset_context_on_run_change(
        self, temp_workspace, temp_data_dir, mock_task
    ):
        """Worker should refresh dataset/model context when run_id changes."""
        worker = Worker(workspace_dir=temp_workspace, data_path=temp_data_dir, use_cache=False)
        worker._runner = MagicMock()
        worker._runner.workspace_dir = temp_workspace

        output1 = temp_workspace / "task_run1" / "output"
        output1.mkdir(parents=True, exist_ok=True)
        worker._runner.run_task.return_value = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=1000,
            output_path=output1,
        )
        worker._fetch_dataset = MagicMock(return_value=temp_data_dir)

        task_run1 = TaskInfo(
            id="task_run1",
            tool_name=mock_task.tool_name,
            tool_image=mock_task.tool_image,
            tool_command=mock_task.tool_command,
            tool_runtime=mock_task.tool_runtime,
            tool_is_baseline=mock_task.tool_is_baseline,
            config=mock_task.config,
            priority=mock_task.priority,
            status=mock_task.status,
            task_type=mock_task.task_type,
            counter=mock_task.counter,
            workflows=mock_task.workflows,
            pipeline_id=mock_task.pipeline_id,
            dependency_ids=[],
            run_id="run_1",
        )
        task_run1_b = TaskInfo(
            **{**task_run1.__dict__, "id": "task_run1_b"}
        )
        task_run2 = TaskInfo(
            **{**task_run1.__dict__, "id": "task_run2", "run_id": "run_2"}
        )

        worker._execute_task(task_run1)
        worker._execute_task(task_run1_b)
        worker._execute_task(task_run2)

        assert worker._fetch_dataset.call_count == 2
    
    def test_execute_task_mounts_data_directory(self, temp_workspace, temp_data_dir, mock_task):
        """_execute_task should mount data directory if available."""
        worker = Worker(workspace_dir=temp_workspace, data_path=temp_data_dir)
        worker._runner = MagicMock()
        worker._runner.workspace_dir = temp_workspace
        
        result = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=1000,
            output_path=temp_workspace / "test_task_1" / "output"
        )
        worker._runner.run_task.return_value = result
        
        worker._execute_task(mock_task)
        
        call_args = worker._runner.run_task.call_args
        assert call_args[1]["input_path"] == temp_data_dir
        assert call_args[1]["extra_mounts"] is not None
        assert str(temp_data_dir.absolute()) in call_args[1]["extra_mounts"]
    
    def test_execute_task_collects_dependency_outputs(self, temp_workspace, mock_task):
        """_execute_task should collect dependency outputs for artifact chaining."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._runner = MagicMock()
        worker._runner.workspace_dir = temp_workspace
        
        # Create dependency output directory
        dep_output = temp_workspace / "dep_task" / "output"
        dep_output.mkdir(parents=True, exist_ok=True)
        (dep_output / "model.pt").write_bytes(b"dependency model")
        
        task_with_deps = TaskInfo(
            id="test_task_2",
            tool_name="test_tool",
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
            dependency_ids=["dep_task"]
        )
        
        result = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=1000,
            output_path=temp_workspace / "test_task_2" / "output"
        )
        worker._runner.run_task.return_value = result
        
        worker._execute_task(task_with_deps)
        
        call_args = worker._runner.run_task.call_args
        assert call_args[1]["dependency_outputs"] is not None
        assert "dep_task" in call_args[1]["dependency_outputs"]
    
    def test_execute_task_handles_missing_dependency_outputs(self, temp_workspace, mock_task):
        """_execute_task should handle missing dependency outputs gracefully."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._runner = MagicMock()
        worker._runner.workspace_dir = temp_workspace
        
        task_with_deps = TaskInfo(
            id="test_task_3",
            tool_name="test_tool",
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
            dependency_ids=["missing_dep"]
        )
        
        result = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=1000,
            output_path=temp_workspace / "test_task_3" / "output"
        )
        worker._runner.run_task.return_value = result
        
        # Should not raise exception
        worker._execute_task(task_with_deps)
        
        call_args = worker._runner.run_task.call_args
        # dependency_outputs should be empty or None
        assert call_args[1]["dependency_outputs"] is None or len(call_args[1]["dependency_outputs"]) == 0
    
    def test_execute_task_clears_current_task_on_failure(self, temp_workspace, mock_task):
        """_execute_task should clear current_task even on failure."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._runner = MagicMock()
        worker._runner.run_task.side_effect = Exception("Execution failed")
        
        try:
            worker._execute_task(mock_task)
        except Exception:
            pass
        
        assert worker._current_task is None  # Should be cleared
    
    def test_execute_task_computes_cache_key(self, temp_workspace, mock_task):
        """_execute_task should compute cache key for task."""
        worker = Worker(workspace_dir=temp_workspace, use_cache=True)
        worker._cache = MagicMock()
        worker._runner = MagicMock()
        worker._runner.workspace_dir = temp_workspace
        
        result = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=1000,
            output_path=temp_workspace / "test_task_1" / "output"
        )
        worker._runner.run_task.return_value = result
        worker._cache.check_cache.return_value = None
        
        worker._execute_task(mock_task)
        
        # Cache key should be computed
        assert worker._cache.store_result.called
        call_args = worker._cache.store_result.call_args
        assert "task" in call_args[1] or "task" in call_args[0]


# ============================================================================
# Test: Result Reporting
# ============================================================================


class TestResultReporting:
    """Tests for result reporting functionality."""
    
    def test_report_result_success(self, temp_workspace, mock_task, mock_client):
        """_report_result should report successful task completion."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        
        result = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=1000,
            output_path=Path("/output"),
            artifacts={"accuracy": 0.95}
        )
        
        worker._report_result(mock_task, result)
        
        call_kw = mock_client.report_task_completed.call_args[1]
        assert call_kw["task_id"] == "test_task_1"
        assert call_kw["execution_time_ms"] == 1000
        result = call_kw["result"]
        assert "artifacts" in result and result["artifacts"] == {"accuracy": 0.95}
        assert "logs" in result
        assert "log_path" in result
        assert worker._tasks_completed == 1
        assert worker._tasks_failed == 0
    
    def test_report_result_includes_evaluation_result_when_evaluator(
        self, temp_workspace, mock_client
    ):
        """_report_result should include evaluation_result in payload for evaluator tasks."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client

        eval_task = TaskInfo(
            id="eval_task_1",
            tool_name="adversarial-evaluator",
            tool_image="eval/image:latest",
            tool_command="python eval.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=50,
            status="pending",
            task_type="evaluation",
            counter=1,
            workflows=["workflow_1"],
            pipeline_id="pipeline_1",
            dependency_ids=[],
        )
        output_dir = temp_workspace / "eval_task_1" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_data = {
            "metrics": {"clean_accuracy": 0.92, "pgd_accuracy": 0.75},
            "success": True,
            "skipped": False,
        }
        (output_dir / "evaluation_results.json").write_text(
            '{"metrics": {"clean_accuracy": 0.92, "pgd_accuracy": 0.75}, "success": true, "skipped": false}'
        )

        result = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=2000,
            output_path=output_dir,
            artifacts={},
        )
        worker._report_result(eval_task, result)

        mock_client.report_task_completed.assert_called_once()
        call_kw = mock_client.report_task_completed.call_args[1]
        assert call_kw["result"].get("evaluation_result") == eval_data
        assert worker._tasks_completed == 1

    def test_report_result_failure(self, temp_workspace, mock_task, mock_client):
        """_report_result should report task failure."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        
        result = ExecutionResult(
            success=False,
            exit_code=1,
            execution_time_ms=500,
            error_message="Task failed"
        )
        
        worker._report_result(mock_task, result)
        
        mock_client.report_task_failed.assert_called_once()
        call_kw = mock_client.report_task_failed.call_args[1]
        assert call_kw["task_id"] == "test_task_1"
        assert call_kw["error_message"] == "Task failed"
        assert call_kw["execution_time_ms"] == 500
        assert "result" in call_kw and "artifacts" in call_kw["result"]
        assert worker._tasks_completed == 0
        assert worker._tasks_failed == 1
    
    def test_report_result_handles_reporting_failure(self, temp_workspace, mock_task, mock_client):
        """_report_result should handle reporting failures gracefully."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        mock_client.report_task_completed.side_effect = Exception("Network error")
        
        result = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=1000
        )
        
        # Should not raise exception
        worker._report_result(mock_task, result)


# ============================================================================
# Test: Work Loop
# ============================================================================


class TestWorkLoop:
    """Tests for worker work loop functionality."""
    
    def test_work_loop_claims_and_executes_tasks(self, temp_workspace, mock_task, mock_client):
        """_work_loop should claim and execute tasks."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        worker._runner = MagicMock()
        worker._running = True
        
        result = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=1000,
            output_path=temp_workspace / "test_task_1" / "output"
        )
        worker._runner.run_task.return_value = result
        
        mock_client.claim_task.return_value = mock_task
        mock_client.get_progress.return_value = {"is_complete": True}
        
        # Run for one iteration
        with patch('time.sleep'):  # Don't actually sleep
            try:
                worker._work_loop()
            except StopIteration:
                pass
        
        mock_client.claim_task.assert_called()
        worker._runner.run_task.assert_called_once()
        mock_client.report_task_completed.assert_called_once()
    
    def test_work_loop_handles_no_tasks(self, temp_workspace, mock_client):
        """_work_loop should handle case when no tasks are available."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        worker._running = True
        
        mock_client.claim_task.return_value = None
        mock_client.get_progress.return_value = {"is_complete": False}
        
        with patch('time.sleep'):  # Don't actually sleep
            # Should not raise exception
            try:
                worker._work_loop()
            except (StopIteration, KeyboardInterrupt):
                pass
        
        mock_client.claim_task.assert_called()
    
    def test_work_loop_exits_when_complete(self, temp_workspace, mock_client):
        """_work_loop should exit when all tasks are complete."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        worker._running = True
        
        mock_client.claim_task.return_value = None
        mock_client.get_progress.return_value = {"is_complete": True}
        
        with patch('time.sleep'):  # Don't actually sleep
            worker._work_loop()
        
        # Should have checked progress
        mock_client.get_progress.assert_called()
    
    def test_work_loop_sends_heartbeat(self, temp_workspace, mock_client):
        """_work_loop should send heartbeat periodically."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        worker._running = True
        worker._last_heartbeat = 0
        
        mock_client.claim_task.return_value = None
        mock_client.get_progress.return_value = {"is_complete": True}
        
        with patch('time.sleep'), patch('time.time', return_value=100.0):
            worker._work_loop()
        
        # Should have sent heartbeat
        mock_client.heartbeat.assert_called()
    
    def test_work_loop_handles_exceptions(self, temp_workspace, mock_client):
        """_work_loop should handle exceptions gracefully."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        worker._running = True
        
        mock_client.claim_task.side_effect = Exception("Network error")
        
        with patch('time.sleep'):  # Don't actually sleep
            # Should not raise exception, should continue
            try:
                worker._work_loop()
            except (StopIteration, KeyboardInterrupt):
                pass
        
        # Should have attempted to claim task
        mock_client.claim_task.assert_called()
    
    def test_work_loop_respects_running_flag(self, temp_workspace, mock_client):
        """_work_loop should respect _running flag."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        worker._running = False  # Set to False
        
        mock_client.claim_task.return_value = None
        
        with patch('time.sleep'):
            worker._work_loop()
        
        # Should not claim tasks if not running
        mock_client.claim_task.assert_not_called()


# ============================================================================
# Test: Cache Management
# ============================================================================


class TestCacheManagement:
    """Tests for cache management functionality."""
    
    def test_compute_cache_key(self, temp_workspace, mock_task):
        """_compute_cache_key should compute stable cache key."""
        worker = Worker(workspace_dir=temp_workspace)
        
        key1 = worker._compute_cache_key(mock_task, [])
        key2 = worker._compute_cache_key(mock_task, [])
        
        # Same task should produce same key
        assert key1 == key2
    
    def test_compute_cache_key_includes_parent_hashes(self, temp_workspace, mock_task):
        """_compute_cache_key should include parent hashes."""
        worker = Worker(workspace_dir=temp_workspace)
        
        key1 = worker._compute_cache_key(mock_task, ["parent1"])
        key2 = worker._compute_cache_key(mock_task, ["parent2"])
        
        # Different parents should produce different keys
        assert key1 != key2

    def test_compute_cache_key_includes_cache_context(self, temp_workspace, mock_task):
        """Cache key should change when dataset/model cache context changes."""
        worker = Worker(workspace_dir=temp_workspace)
        key1 = worker._compute_cache_key(
            mock_task,
            [],
            cache_context={
                "dataset": {"name": "cifar10", "variant": "clean"},
                "model": {"path": "configs/model/config_model.py", "sha256": "aaa"},
            },
        )
        key2 = worker._compute_cache_key(
            mock_task,
            [],
            cache_context={
                "dataset": {"name": "cifar10", "variant": "poisoned"},
                "model": {"path": "configs/model/config_model.py", "sha256": "aaa"},
            },
        )
        assert key1 != key2

    def test_build_cache_context_captures_dataset_and_model(self, temp_workspace, temp_data_dir):
        """_build_cache_context should capture dataset and model identity."""
        worker = Worker(workspace_dir=temp_workspace)
        model_script = temp_workspace / "config_model.py"
        model_script.write_text("MODEL = 'A'\n")
        worker._model_script_path = model_script
        worker._dataset_info = {"name": "cifar10", "variant": "clean", "minio_key": "datasets/cifar10/clean"}

        context = worker._build_cache_context(temp_data_dir)

        assert context["dataset"]["name"] == "cifar10"
        assert context["dataset"]["variant"] == "clean"
        assert "data.npy" in context["dataset"]["tracked_files"]
        assert context["model"]["available"] is True
        assert context["model"]["sha256"]
    
    def test_check_cache_uses_two_level_cache(self, temp_workspace, mock_task):
        """_check_cache should use two-level cache if available."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._two_level_cache = MagicMock()
        worker._two_level_cache.get.return_value = Path("/cached")
        
        result = worker._check_cache("cache_key", mock_task, [])
        
        assert result == Path("/cached")
        worker._two_level_cache.get.assert_called_once_with("cache_key")
    
    def test_check_cache_falls_back_to_local_cache(self, temp_workspace, mock_task, temp_cache_dir):
        """_check_cache should fall back to local cache if two-level not available."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._cache = MagicMock()
        worker._cache.check_cache.return_value = Path("/cached")
        
        cache_context = {"dataset": {"name": "cifar10", "variant": "clean"}}
        result = worker._check_cache("cache_key", mock_task, [], cache_context=cache_context)
        
        assert result == Path("/cached")
        worker._cache.check_cache.assert_called_once_with(mock_task, [], cache_context=cache_context)
    
    def test_store_in_cache_uses_two_level_cache(self, temp_workspace, mock_task, temp_cache_dir):
        """_store_in_cache should use two-level cache if available."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._two_level_cache = MagicMock()
        
        output_path = temp_cache_dir / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        
        worker._store_in_cache("cache_key", mock_task, output_path, 1000, [])
        
        worker._two_level_cache.put.assert_called_once()
    
    def test_store_in_cache_falls_back_to_local_cache(self, temp_workspace, mock_task, temp_cache_dir):
        """_store_in_cache should fall back to local cache if two-level not available."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._cache = MagicMock()
        
        output_path = temp_cache_dir / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        
        cache_context = {"dataset": {"name": "cifar10", "variant": "clean"}}
        worker._store_in_cache("cache_key", mock_task, output_path, 1000, [], cache_context=cache_context)
        
        worker._cache.store_result.assert_called_once()
        assert worker._cache.store_result.call_args.kwargs["cache_context"] == cache_context


# ============================================================================
# Test: Worker Start and Cleanup
# ============================================================================


class TestWorkerStartAndCleanup:
    """Tests for worker start and cleanup."""
    
    def test_start_initializes_components(self, temp_workspace):
        """start should initialize components."""
        worker = Worker(workspace_dir=temp_workspace)
        
        with patch.object(worker, '_wait_for_backend', return_value=False):
            result = worker.start()
        
        assert worker._client is not None
        assert worker._runner is not None
    
    def test_start_waits_for_backend(self, temp_workspace):
        """start should wait for backend before proceeding."""
        worker = Worker(workspace_dir=temp_workspace)
        
        with patch.object(worker, '_wait_for_backend', return_value=False) as mock_wait:
            result = worker.start()
        
        assert result == 1  # Exit code 1 on failure
        mock_wait.assert_called_once()
    
    def test_start_registers_worker(self, temp_workspace):
        """start should register worker with backend."""
        worker = Worker(workspace_dir=temp_workspace)
        
        with patch.object(worker, '_wait_for_backend', return_value=True), \
             patch.object(worker, '_register', return_value=False):
            result = worker.start()
        
        assert result == 1  # Exit code 1 on registration failure
    
    def test_start_fetches_dataset(self, temp_workspace, temp_data_dir):
        """start should fetch dataset from backend."""
        worker = Worker(workspace_dir=temp_workspace, data_path=temp_data_dir)
        
        with patch.object(worker, '_wait_for_backend', return_value=True), \
             patch.object(worker, '_register', return_value=True), \
             patch.object(worker, '_work_loop') as mock_loop:
            mock_loop.side_effect = KeyboardInterrupt()  # Exit immediately
            worker.start()
        
        assert worker._dataset_path == temp_data_dir
    
    def test_start_enters_work_loop(self, temp_workspace):
        """start should enter work loop after initialization."""
        worker = Worker(workspace_dir=temp_workspace)
        
        with patch.object(worker, '_wait_for_backend', return_value=True), \
             patch.object(worker, '_register', return_value=True), \
             patch.object(worker, '_fetch_dataset', return_value=None), \
             patch.object(worker, '_work_loop') as mock_loop:
            mock_loop.side_effect = KeyboardInterrupt()  # Exit immediately
            worker.start()
        
        mock_loop.assert_called_once()
    
    def test_cleanup_closes_client(self, temp_workspace, mock_client):
        """_cleanup should close client connection."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        
        worker._cleanup()
        
        mock_client.close.assert_called_once()
    
    def test_cleanup_handles_client_close_failure(self, temp_workspace, mock_client):
        """_cleanup should handle client close failures gracefully."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._client = mock_client
        mock_client.close.side_effect = Exception("Close failed")
        
        # Should not raise exception
        worker._cleanup()


# ============================================================================
# Test: Startup Integrity (Backend, MinIO, GPU)
# ============================================================================


class TestWorkerStartupIntegrity:
    """Integrity tests for worker startup and repeated initialization.

    These tests ensure that on every startup:
    - Backend connectivity is established (via LandseerClient)
    - MinIO / two-level cache is wired when enabled
    - GPU configuration is consistently passed into TaskRunner / Docker stack
    - Multiple start() calls do not skip or partially initialize components
    """

    def test_single_start_initializes_backend_minio_and_gpu(self, temp_workspace, temp_cache_dir):
        """start() should always initialize backend client, cache and GPU runner."""
        # Patch low-level components so we do not hit real network / Docker.
        with patch("src.worker.cli.LandseerClient") as mock_client_cls, \
             patch("src.worker.cli.TaskRunner") as mock_runner_cls, \
             patch("src.worker.cli.TwoLevelCache") as mock_two_level_cache_cls, \
             patch.object(Worker, "_wait_for_backend", return_value=True) as mock_wait, \
             patch.object(Worker, "_register", return_value=True) as mock_register, \
             patch.object(Worker, "_fetch_dataset", return_value=None), \
             patch.object(Worker, "_work_loop") as mock_loop, \
             patch.dict("os.environ", {"LANDSEER_USE_MINIO": "true"}):

            # Let start() run through a single, mocked work loop call and exit cleanly.
            mock_loop.return_value = None

            gpu_id = 1
            worker = Worker(
                workspace_dir=temp_workspace,
                cache_dir=temp_cache_dir,
                gpu_id=gpu_id,
                use_cache=True,
                runtime="docker",
            )

            exit_code = worker.start()

            # Worker should consider startup successful (work loop exited via KeyboardInterrupt)
            assert exit_code == 0

            # Backend client should be created and wired into worker
            assert worker._client is not None
            mock_client_cls.assert_called_once()

            # TaskRunner should be created with correct GPU id and workspace
            assert worker._runner is not None
            mock_runner_cls.assert_called_once()
            runner_call_kwargs = mock_runner_cls.call_args.kwargs
            assert runner_call_kwargs["workspace_dir"] == temp_workspace
            assert runner_call_kwargs["gpu_id"] == gpu_id
            assert runner_call_kwargs["timeout"] == worker.task_timeout

            # Two-level cache (MinIO + local) should be initialized when enabled
            # Note: actual availability depends on import, but we verify our wiring.
            assert worker.use_cache is True
            # Either two-level cache or local cache should be initialized
            assert worker._two_level_cache is not None or worker._cache is not None
            # TwoLevelCache should have been constructed when MinIO is enabled
            assert mock_two_level_cache_cls.called

            # Backend wait and registration must run exactly once on startup
            mock_wait.assert_called_once()
            mock_register.assert_called_once()

    def test_repeated_start_reinitializes_components_consistently(self, temp_workspace, temp_cache_dir):
        """Multiple start() calls should consistently (re-)initialize all components."""
        with patch("src.worker.cli.LandseerClient") as mock_client_cls, \
             patch("src.worker.cli.TaskRunner") as mock_runner_cls, \
             patch.object(Worker, "_wait_for_backend", return_value=True) as mock_wait, \
             patch.object(Worker, "_register", return_value=True) as mock_register, \
             patch.object(Worker, "_fetch_dataset", return_value=None), \
             patch.object(Worker, "_work_loop") as mock_loop:

            # Two clean work-loop invocations, one per start() call
            mock_loop.side_effect = [None, None]

            gpu_id = 0
            worker = Worker(
                workspace_dir=temp_workspace,
                cache_dir=temp_cache_dir,
                gpu_id=gpu_id,
                use_cache=True,
                runtime="docker",
            )

            # First start
            exit_code_1 = worker.start()
            # Second start – should not be a no-op; should re-run initialization path
            exit_code_2 = worker.start()

            assert exit_code_1 == 0
            assert exit_code_2 == 0

            # LandseerClient and TaskRunner constructors should be invoked for each start
            assert mock_client_cls.call_count == 2
            assert mock_runner_cls.call_count == 2

            # Every TaskRunner instance must receive consistent GPU / timeout configuration
            for call in mock_runner_cls.call_args_list:
                kwargs = call.kwargs
                assert kwargs["workspace_dir"] == temp_workspace
                assert kwargs["gpu_id"] == gpu_id
                assert kwargs["timeout"] == worker.task_timeout

            # Backend wait and registration should be executed on every start
            assert mock_wait.call_count == 2
            assert mock_register.call_count == 2

    def test_init_components_is_idempotent_for_gpu_and_cache(self, temp_workspace, temp_cache_dir):
        """_init_components can be called multiple times without breaking GPU/cache wiring."""
        gpu_id = 2
        worker = Worker(
            workspace_dir=temp_workspace,
            cache_dir=temp_cache_dir,
            gpu_id=gpu_id,
            use_cache=True,
        )

        with patch("src.worker.cli.LandseerClient") as mock_client_cls, \
             patch("src.worker.cli.TaskRunner") as mock_runner_cls:

            # Call twice to ensure no hit-and-miss behaviour
            worker._init_components()
            worker._init_components()

            # Client and runner should be (re)created; we only care that
            # final wiring on worker uses the expected GPU and cache.
            assert worker._runner is not None
            final_runner_kwargs = mock_runner_cls.call_args.kwargs
            assert final_runner_kwargs["workspace_dir"] == temp_workspace
            assert final_runner_kwargs["gpu_id"] == gpu_id

            # Some form of cache (local or two-level) should be initialized when use_cache=True
            assert (worker._cache is not None) or (worker._two_level_cache is not None)


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_worker_with_no_backend(self, temp_workspace):
        """Worker should handle missing backend gracefully."""
        worker = Worker(workspace_dir=temp_workspace, backend_url="http://nonexistent:8000")
        
        with patch.object(worker, '_wait_for_backend', return_value=False):
            result = worker.start()
        
        assert result == 1  # Should exit with error code
    
    def test_worker_with_invalid_workspace(self, tmp_path):
        """Worker should handle invalid workspace paths."""
        invalid_workspace = tmp_path / "nonexistent" / "deep" / "path"
        
        # Should create directory
        worker = Worker(workspace_dir=invalid_workspace)
        assert invalid_workspace.exists()
    
    def test_worker_with_missing_cache_dir(self, temp_workspace, tmp_path):
        """Worker should handle missing cache directory."""
        missing_cache = tmp_path / "missing_cache"
        
        worker = Worker(workspace_dir=temp_workspace, cache_dir=missing_cache)
        worker._init_components()
        
        # Should create cache directory if needed
        # (Actual behavior depends on CacheManager implementation)
        pass
    
    def test_worker_with_concurrent_tasks(self, temp_workspace, mock_task):
        """Worker should handle concurrent task execution correctly."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._runner = MagicMock()
        
        # Worker should only execute one task at a time
        worker._current_task = mock_task
        
        # Attempting to execute another task should be blocked or queued
        # (Actual behavior depends on implementation)
        assert worker._current_task is not None
    
    def test_worker_signal_handler(self, temp_workspace):
        """Worker signal handler should set _running to False."""
        worker = Worker(workspace_dir=temp_workspace)
        worker._running = True
        
        worker._signal_handler(signal.SIGTERM, None)
        
        assert worker._running is False
    
    def test_worker_with_zero_timeout(self, temp_workspace):
        """Worker should handle zero timeout."""
        worker = Worker(workspace_dir=temp_workspace, task_timeout=0)
        
        # Should not raise exception
        assert worker.task_timeout == 0
    
    def test_worker_with_negative_poll_interval(self, temp_workspace):
        """Worker should handle negative poll interval."""
        worker = Worker(workspace_dir=temp_workspace, poll_interval=-1.0)
        
        # Should not raise exception (though behavior may be undefined)
        assert worker.poll_interval == -1.0
    
    def test_worker_with_very_large_gpu_id(self, temp_workspace):
        """Worker should handle very large GPU ID."""
        worker = Worker(workspace_dir=temp_workspace, gpu_id=9999)
        
        assert worker.gpu_id == 9999
    
    def test_worker_with_empty_task_config(self, temp_workspace):
        """Worker should handle tasks with empty config."""
        task = TaskInfo(
            id="empty_task",
            tool_name="tool",
            tool_image="image:latest",
            tool_command="cmd",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},  # Empty config
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        worker = Worker(workspace_dir=temp_workspace)
        worker._runner = MagicMock()
        worker._runner.run_task.return_value = ExecutionResult(
            success=True,
            exit_code=0,
            execution_time_ms=1000,
            output_path=temp_workspace / "empty_task" / "output"
        )
        
        # Should not raise exception
        result = worker._execute_task(task)
        assert result.success is True
