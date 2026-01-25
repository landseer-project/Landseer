"""
HTTP client for communicating with the Landseer backend API.

This module provides a client class that handles:
- Worker registration with the backend
- Task claiming and retrieval
- Task status reporting (completion/failure)
- Worker heartbeat
"""

import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx

from ..common import get_logger

logger = get_logger(__name__)


@dataclass
class TaskInfo:
    """Information about a task to execute."""
    id: str
    tool_name: str
    tool_image: str
    tool_command: str
    tool_runtime: Optional[str]
    tool_is_baseline: bool
    config: Dict[str, Any]
    priority: int
    status: str
    task_type: str
    counter: int
    workflows: list
    pipeline_id: str
    dependency_ids: list

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "TaskInfo":
        """Create TaskInfo from API response data."""
        tool = data.get("tool", {})
        container = tool.get("container", {})
        
        return cls(
            id=data.get("id", ""),
            tool_name=tool.get("name", ""),
            tool_image=container.get("image", ""),
            tool_command=container.get("command", ""),
            tool_runtime=container.get("runtime"),
            tool_is_baseline=tool.get("is_baseline", False),
            config=data.get("config", {}),
            priority=data.get("priority", 0),
            status=data.get("status", ""),
            task_type=data.get("task_type", ""),
            counter=data.get("counter", 0),
            workflows=data.get("workflows", []),
            pipeline_id=data.get("pipeline_id", ""),
            dependency_ids=data.get("dependency_ids", [])
        )


@dataclass
class WorkerInfo:
    """Information about this worker."""
    worker_id: str
    hostname: str
    status: str
    registered_at: str
    last_heartbeat: str
    current_task_id: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    capabilities: Dict[str, Any] = field(default_factory=dict)


class LandseerClient:
    """
    HTTP client for the Landseer backend API.
    
    Handles all communication between the worker and the scheduler,
    including registration, task claiming, and status updates.
    """
    
    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Landseer client.
        
        Args:
            backend_url: Base URL of the backend API
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.backend_url = backend_url.rstrip("/")
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        self._client = httpx.Client(
            base_url=self.backend_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        
        self._worker_id: Optional[str] = None
        self._worker_info: Optional[WorkerInfo] = None
    
    @property
    def worker_id(self) -> Optional[str]:
        """Get the registered worker ID."""
        return self._worker_id
    
    @property
    def is_registered(self) -> bool:
        """Check if this worker is registered with the backend."""
        return self._worker_id is not None
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> httpx.Response:
        """
        Make an HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            endpoint: API endpoint path
            json_data: JSON body data
            params: Query parameters
            
        Returns:
            HTTP response
            
        Raises:
            httpx.HTTPError: If request fails after all retries
        """
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                response = self._client.request(
                    method=method,
                    url=endpoint,
                    json=json_data,
                    params=params
                )
                response.raise_for_status()
                return response
                
            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    raise
                last_error = e
                
            except httpx.HTTPError as e:
                last_error = e
            
            if attempt < self.retry_attempts - 1:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.retry_attempts}): {last_error}"
                )
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        raise last_error or httpx.HTTPError("Request failed after all retries")
    
    # =========================================================================
    # Health & Info
    # =========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check backend health status.
        
        Returns:
            Health status dictionary
        """
        response = self._make_request("GET", "/health")
        return response.json()
    
    def is_backend_available(self) -> bool:
        """
        Check if the backend is available and responding.
        
        Returns:
            True if backend is available
        """
        try:
            health = self.health_check()
            return health.get("status") == "ok"
        except Exception as e:
            logger.debug(f"Backend health check failed: {e}")
            return False
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the current pipeline.
        
        Returns:
            Pipeline information dictionary
        """
        response = self._make_request("GET", "/info/pipeline")
        return response.json()
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current pipeline progress.
        
        Returns:
            Progress statistics dictionary
        """
        response = self._make_request("GET", "/progress")
        return response.json()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information from backend.
        
        The backend prepares the dataset on startup and uploads to MinIO.
        Workers can use this to:
        1. Check if dataset is available
        2. Get MinIO key to download dataset
        3. Get local path if running on same machine
        
        Returns:
            Dataset info dictionary with keys:
            - available: bool
            - name: str
            - variant: str
            - local_path: str (if prepared on same machine)
            - minio_key: str (if uploaded to MinIO)
            - minio_available: bool
            - train_samples: int
            - test_samples: int
        """
        try:
            response = self._make_request("GET", "/dataset")
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get dataset info: {e}")
            return {"available": False}
    
    # =========================================================================
    # Worker Registration
    # =========================================================================
    
    def register(
        self,
        worker_id: Optional[str] = None,
        capabilities: Optional[Dict[str, Any]] = None
    ) -> WorkerInfo:
        """
        Register this worker with the backend.
        
        Args:
            worker_id: Optional worker ID (auto-generated if not provided)
            capabilities: Worker capabilities (GPU, memory, etc.)
            
        Returns:
            WorkerInfo with registration details
        """
        hostname = socket.gethostname()
        
        data = {
            "hostname": hostname,
            "worker_id": worker_id,
            "capabilities": capabilities or {}
        }
        
        response = self._make_request("POST", "/workers/register", json_data=data)
        result = response.json()
        
        self._worker_id = result["worker_id"]
        self._worker_info = WorkerInfo(
            worker_id=result["worker_id"],
            hostname=result["hostname"],
            status=result["status"],
            registered_at=result["registered_at"],
            last_heartbeat=result["last_heartbeat"],
            current_task_id=result.get("current_task_id"),
            tasks_completed=result.get("tasks_completed", 0),
            tasks_failed=result.get("tasks_failed", 0),
            capabilities=result.get("capabilities", {})
        )
        
        logger.info(f"Worker registered: {self._worker_id} ({hostname})")
        return self._worker_info
    
    def heartbeat(self, status: Optional[str] = None) -> bool:
        """
        Send a heartbeat to the backend.
        
        Args:
            status: Optional status update
            
        Returns:
            True if heartbeat was successful
        """
        if not self.is_registered:
            raise RuntimeError("Worker not registered. Call register() first.")
        
        data = {
            "worker_id": self._worker_id,
            "status": status
        }
        
        try:
            response = self._make_request(
                "POST",
                f"/workers/{self._worker_id}/heartbeat",
                json_data=data
            )
            return response.json().get("success", False)
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False
    
    # =========================================================================
    # Task Management
    # =========================================================================
    
    def claim_task(self) -> Optional[TaskInfo]:
        """
        Claim the next available task for this worker.
        
        The claimed task's status is automatically set to RUNNING.
        
        Returns:
            TaskInfo if a task is available, None otherwise
        """
        if not self.is_registered:
            raise RuntimeError("Worker not registered. Call register() first.")
        
        response = self._make_request(
            "POST",
            f"/workers/{self._worker_id}/claim"
        )
        result = response.json()
        
        if result.get("has_task"):
            task_data = result["task"]
            task = TaskInfo.from_api_response(task_data)
            logger.info(f"Claimed task: {task.id} ({task.tool_name})")
            return task
        
        logger.debug(f"No task available: {result.get('message')}")
        return None
    
    def get_next_task(self) -> Optional[TaskInfo]:
        """
        Get the next available task (anonymous, without worker association).
        
        This is useful for non-registered workers or testing.
        
        Returns:
            TaskInfo if a task is available, None otherwise
        """
        response = self._make_request("GET", "/tasks/next")
        result = response.json()
        
        if result.get("has_task"):
            task_data = result["task"]
            return TaskInfo.from_api_response(task_data)
        
        return None
    
    def report_task_completed(
        self,
        task_id: str,
        execution_time_ms: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Report a task as completed.
        
        Args:
            task_id: ID of the completed task
            execution_time_ms: Execution time in milliseconds
            result: Task result data
            
        Returns:
            True if update was successful
        """
        data = {
            "task_id": task_id,
            "status": "completed",
            "execution_time_ms": execution_time_ms,
            "result": result
        }
        
        response = self._make_request("PUT", "/tasks/status", json_data=data)
        result_data = response.json()
        
        success = result_data.get("success", False)
        if success:
            logger.info(f"Task {task_id} marked as completed")
        else:
            logger.warning(f"Failed to mark task {task_id} as completed")
        
        return success
    
    def report_task_failed(
        self,
        task_id: str,
        error_message: str,
        execution_time_ms: Optional[int] = None
    ) -> bool:
        """
        Report a task as failed.
        
        Args:
            task_id: ID of the failed task
            error_message: Error message describing the failure
            execution_time_ms: Execution time in milliseconds
            
        Returns:
            True if update was successful
        """
        data = {
            "task_id": task_id,
            "status": "failed",
            "error_message": error_message,
            "execution_time_ms": execution_time_ms
        }
        
        response = self._make_request("PUT", "/tasks/status", json_data=data)
        result_data = response.json()
        
        success = result_data.get("success", False)
        if success:
            logger.info(f"Task {task_id} marked as failed: {error_message}")
        else:
            logger.warning(f"Failed to mark task {task_id} as failed")
        
        return success
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """
        Get information about a specific task.
        
        Args:
            task_id: Task ID
            
        Returns:
            TaskInfo if found, None otherwise
        """
        try:
            response = self._make_request("GET", f"/tasks/{task_id}")
            return TaskInfo.from_api_response(response.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def get_all_tasks(self, status: Optional[str] = None) -> list[TaskInfo]:
        """
        Get all tasks, optionally filtered by status.
        
        Args:
            status: Filter by status (pending, running, completed, failed)
            
        Returns:
            List of TaskInfo objects
        """
        params = {"status": status} if status else None
        response = self._make_request("GET", "/tasks", params=params)
        result = response.json()
        
        return [TaskInfo.from_api_response(t) for t in result.get("tasks", [])]
    
    # =========================================================================
    # Tool Information
    # =========================================================================
    
    def get_tools(self) -> list[Dict[str, Any]]:
        """
        Get all available tools.
        
        Returns:
            List of tool definitions
        """
        response = self._make_request("GET", "/tools")
        return response.json().get("tools", [])
    
    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific tool.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool information if found, None otherwise
        """
        try:
            response = self._make_request("GET", f"/tools/{tool_name}")
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def close(self) -> None:
        """Close the HTTP client connection."""
        self._client.close()
    
    def __enter__(self) -> "LandseerClient":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
