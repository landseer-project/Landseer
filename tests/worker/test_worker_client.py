"""
Comprehensive tests for LandseerClient.

Tests cover:
1. Client initialization
2. HTTP request handling and retries
3. Worker registration
4. Task claiming and reporting
5. Heartbeat functionality
6. Error handling
7. Edge cases
"""

import pytest
import httpx
from unittest.mock import Mock, patch, MagicMock
from typing import Dict

from src.worker.client import LandseerClient, TaskInfo, WorkerInfo


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.Client."""
    client = MagicMock(spec=httpx.Client)
    return client


@pytest.fixture
def client():
    """Create a LandseerClient instance."""
    with patch('src.worker.client.httpx.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        client = LandseerClient(backend_url="http://test:8000")
        client._client = mock_client
        return client


# ============================================================================
# Test: Client Initialization
# ============================================================================


class TestClientInitialization:
    """Tests for client initialization."""
    
    def test_client_initialization_defaults(self):
        """Client should initialize with default values."""
        with patch('src.worker.client.httpx.Client'):
            client = LandseerClient()
        
        assert client.backend_url == "http://localhost:8000"
        assert client.timeout == 30.0
        assert client.retry_attempts == 3
        assert client.retry_delay == 1.0
        assert client._worker_id is None
    
    def test_client_initialization_custom_values(self):
        """Client should accept custom configuration."""
        with patch('src.worker.client.httpx.Client'):
            client = LandseerClient(
                backend_url="http://custom:9000",
                timeout=60.0,
                retry_attempts=5,
                retry_delay=2.0
            )
        
        assert client.backend_url == "http://custom:9000"
        assert client.timeout == 60.0
        assert client.retry_attempts == 5
        assert client.retry_delay == 2.0
    
    def test_client_strips_trailing_slash(self):
        """Client should strip trailing slash from backend URL."""
        with patch('src.worker.client.httpx.Client'):
            client = LandseerClient(backend_url="http://test:8000/")
        
        assert client.backend_url == "http://test:8000"
    
    def test_client_creates_httpx_client(self):
        """Client should create httpx.Client with correct configuration."""
        with patch('src.worker.client.httpx.Client') as mock_client_class:
            client = LandseerClient(backend_url="http://test:8000")
            
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args
            assert call_args[1]["base_url"] == "http://test:8000"
            assert call_args[1]["timeout"] == 30.0
            assert "headers" in call_args[1]


# ============================================================================
# Test: HTTP Request Handling
# ============================================================================


class TestHTTPRequestHandling:
    """Tests for HTTP request handling and retries."""
    
    def test_make_request_success(self, client):
        """_make_request should return response on success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        client._client.request.return_value = mock_response
        
        response = client._make_request("GET", "/test")
        
        assert response == mock_response
        client._client.request.assert_called_once()
    
    def test_make_request_retries_on_server_error(self, client):
        """_make_request should retry on 5xx server errors."""
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        
        client._client.request.side_effect = [
            httpx.HTTPStatusError("Server error", request=MagicMock(), response=mock_response_500),
            mock_response_200
        ]
        
        with patch('time.sleep'):  # Don't actually sleep
            response = client._make_request("GET", "/test")
        
        assert response == mock_response_200
        assert client._client.request.call_count == 2
    
    def test_make_request_does_not_retry_client_errors(self, client):
        """_make_request should not retry on 4xx client errors."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        
        client._client.request.side_effect = httpx.HTTPStatusError(
            "Not found",
            request=MagicMock(),
            response=mock_response
        )
        
        with pytest.raises(httpx.HTTPStatusError):
            client._make_request("GET", "/test")
        
        # Should not retry
        assert client._client.request.call_count == 1
    
    def test_make_request_retries_on_network_error(self, client):
        """_make_request should retry on network errors."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        client._client.request.side_effect = [
            httpx.NetworkError("Network error"),
            mock_response
        ]
        
        with patch('time.sleep'):  # Don't actually sleep
            response = client._make_request("GET", "/test")
        
        assert response == mock_response
        assert client._client.request.call_count == 2
    
    def test_make_request_exponential_backoff(self, client):
        """_make_request should use exponential backoff for retries."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        client._client.request.side_effect = [
            httpx.NetworkError("Error 1"),
            httpx.NetworkError("Error 2"),
            mock_response
        ]
        
        with patch('time.sleep') as mock_sleep:
            client._make_request("GET", "/test")
        
        # Should sleep with increasing delays
        assert mock_sleep.call_count == 2
        # First sleep: retry_delay * 1, second: retry_delay * 2
        assert mock_sleep.call_args_list[0][0][0] == client.retry_delay * 1
        assert mock_sleep.call_args_list[1][0][0] == client.retry_delay * 2
    
    def test_make_request_fails_after_max_retries(self, client):
        """_make_request should raise exception after max retries."""
        client._client.request.side_effect = httpx.NetworkError("Network error")
        
        with patch('time.sleep'):  # Don't actually sleep
            with pytest.raises(httpx.HTTPError):
                client._make_request("GET", "/test")
        
        assert client._client.request.call_count == client.retry_attempts


# ============================================================================
# Test: Health and Info
# ============================================================================


class TestHealthAndInfo:
    """Tests for health check and info endpoints."""
    
    def test_health_check(self, client):
        """health_check should return health status."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "version": "1.0"}
        client._make_request = MagicMock(return_value=mock_response)
        
        result = client.health_check()
        
        assert result == {"status": "ok", "version": "1.0"}
        client._make_request.assert_called_once_with("GET", "/health")
    
    def test_is_backend_available_true(self, client):
        """is_backend_available should return True when backend is healthy."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        client._make_request = MagicMock(return_value=mock_response)
        
        result = client.is_backend_available()
        
        assert result is True
    
    def test_is_backend_available_false(self, client):
        """is_backend_available should return False when backend is unhealthy."""
        client._make_request = MagicMock(side_effect=Exception("Connection failed"))
        
        result = client.is_backend_available()
        
        assert result is False
    
    def test_get_pipeline_info(self, client):
        """get_pipeline_info should return pipeline information."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"name": "test_pipeline", "workflows": 10}
        client._make_request = MagicMock(return_value=mock_response)
        
        result = client.get_pipeline_info()
        
        assert result == {"name": "test_pipeline", "workflows": 10}
        client._make_request.assert_called_once_with("GET", "/info/pipeline")
    
    def test_get_progress(self, client):
        """get_progress should return progress information."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"completed": 5, "total": 10}
        client._make_request = MagicMock(return_value=mock_response)
        
        result = client.get_progress()
        
        assert result == {"completed": 5, "total": 10}
        client._make_request.assert_called_once_with("GET", "/progress")
    
    def test_get_dataset_info(self, client):
        """get_dataset_info should return dataset information."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "available": True,
            "name": "cifar10",
            "variant": "clean"
        }
        client._make_request = MagicMock(return_value=mock_response)
        
        result = client.get_dataset_info()
        
        assert result["available"] is True
        assert result["name"] == "cifar10"
        client._make_request.assert_called_once_with("GET", "/dataset")
    
    def test_get_dataset_info_handles_error(self, client):
        """get_dataset_info should handle errors gracefully."""
        client._make_request = MagicMock(side_effect=Exception("Network error"))
        
        result = client.get_dataset_info()
        
        assert result == {"available": False}


# ============================================================================
# Test: Worker Registration
# ============================================================================


class TestWorkerRegistration:
    """Tests for worker registration."""
    
    def test_register_worker(self, client):
        """register should register worker with backend."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "worker_id": "worker_123",
            "hostname": "test_host",
            "status": "idle",
            "registered_at": "2024-01-01T00:00:00",
            "last_heartbeat": "2024-01-01T00:00:00"
        }
        client._make_request = MagicMock(return_value=mock_response)
        
        with patch('socket.gethostname', return_value="test_host"):
            result = client.register(worker_id="worker_123", capabilities={"gpu": True})
        
        assert isinstance(result, WorkerInfo)
        assert result.worker_id == "worker_123"
        assert client._worker_id == "worker_123"
        client._make_request.assert_called_once()
    
    def test_register_worker_auto_generates_id(self, client):
        """register should auto-generate worker ID if not provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "worker_id": "auto_generated_123",
            "hostname": "test_host",
            "status": "idle",
            "registered_at": "2024-01-01T00:00:00",
            "last_heartbeat": "2024-01-01T00:00:00"
        }
        client._make_request = MagicMock(return_value=mock_response)
        
        with patch('socket.gethostname', return_value="test_host"):
            result = client.register(capabilities={"gpu": True})
        
        assert result.worker_id == "auto_generated_123"
        call_args = client._make_request.call_args
        assert call_args[1]["json_data"]["worker_id"] is None
    
    def test_register_worker_sends_capabilities(self, client):
        """register should send worker capabilities."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "worker_id": "worker_123",
            "hostname": "test_host",
            "status": "idle",
            "registered_at": "2024-01-01T00:00:00",
            "last_heartbeat": "2024-01-01T00:00:00"
        }
        client._make_request = MagicMock(return_value=mock_response)
        
        capabilities = {"runtime": "docker", "gpu_available": True}
        
        with patch('socket.gethostname', return_value="test_host"):
            client.register(capabilities=capabilities)
        
        call_args = client._make_request.call_args
        assert call_args[1]["json_data"]["capabilities"] == capabilities
    
    def test_is_registered_property(self, client):
        """is_registered should return registration status."""
        assert client.is_registered is False
        
        client._worker_id = "worker_123"
        assert client.is_registered is True


# ============================================================================
# Test: Heartbeat
# ============================================================================


class TestHeartbeat:
    """Tests for heartbeat functionality."""
    
    def test_heartbeat_success(self, client):
        """heartbeat should send heartbeat successfully."""
        client._worker_id = "worker_123"
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        client._make_request = MagicMock(return_value=mock_response)
        
        result = client.heartbeat(status="idle")
        
        assert result is True
        call_args = client._make_request.call_args
        assert call_args[0][0] == "POST"
        assert "/workers/worker_123/heartbeat" in call_args[0][1]
        assert call_args[1]["json_data"]["status"] == "idle"
    
    def test_heartbeat_requires_registration(self, client):
        """heartbeat should require worker registration."""
        client._worker_id = None
        
        with pytest.raises(RuntimeError, match="not registered"):
            client.heartbeat()
    
    def test_heartbeat_handles_failure(self, client):
        """heartbeat should handle failures gracefully."""
        client._worker_id = "worker_123"
        client._make_request = MagicMock(side_effect=Exception("Network error"))
        
        result = client.heartbeat()
        
        assert result is False


# ============================================================================
# Test: Task Management
# ============================================================================


class TestTaskManagement:
    """Tests for task claiming and reporting."""
    
    def test_claim_task_success(self, client):
        """claim_task should claim and return task."""
        client._worker_id = "worker_123"
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "has_task": True,
            "task": {
                "id": "task_1",
                "tool": {
                    "name": "test_tool",
                    "container": {
                        "image": "test/image:latest",
                        "command": "python main.py"
                    }
                },
                "config": {},
                "priority": 100,
                "status": "pending",
                "task_type": "pre_training",
                "counter": 1,
                "workflows": [],
                "pipeline_id": "pipeline_1",
                "dependency_ids": []
            }
        }
        client._make_request = MagicMock(return_value=mock_response)
        
        task = client.claim_task()
        
        assert isinstance(task, TaskInfo)
        assert task.id == "task_1"
        assert task.tool_name == "test_tool"
        client._make_request.assert_called_once()
    
    def test_claim_task_no_task_available(self, client):
        """claim_task should return None when no task available."""
        client._worker_id = "worker_123"
        mock_response = MagicMock()
        mock_response.json.return_value = {"has_task": False, "message": "No tasks"}
        client._make_request = MagicMock(return_value=mock_response)
        
        task = client.claim_task()
        
        assert task is None
    
    def test_claim_task_requires_registration(self, client):
        """claim_task should require worker registration."""
        client._worker_id = None
        
        with pytest.raises(RuntimeError, match="not registered"):
            client.claim_task()
    
    def test_report_task_completed(self, client):
        """report_task_completed should report task completion."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        client._make_request = MagicMock(return_value=mock_response)
        
        result = client.report_task_completed(
            task_id="task_1",
            execution_time_ms=1000,
            result={"accuracy": 0.95}
        )
        
        assert result is True
        call_args = client._make_request.call_args
        assert call_args[0][0] == "PUT"
        assert call_args[0][1] == "/tasks/status"
        assert call_args[1]["json_data"]["task_id"] == "task_1"
        assert call_args[1]["json_data"]["status"] == "completed"
    
    def test_report_task_failed(self, client):
        """report_task_failed should report task failure."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        client._make_request = MagicMock(return_value=mock_response)
        
        result = client.report_task_failed(
            task_id="task_1",
            error_message="Task failed",
            execution_time_ms=500
        )
        
        assert result is True
        call_args = client._make_request.call_args
        assert call_args[1]["json_data"]["status"] == "failed"
        assert call_args[1]["json_data"]["error_message"] == "Task failed"
    
    def test_get_task(self, client):
        """get_task should retrieve task information."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "task_1",
            "tool": {"name": "test_tool", "container": {"image": "test/image:latest", "command": "cmd"}},
            "config": {},
            "priority": 100,
            "status": "pending",
            "task_type": "pre_training",
            "counter": 1,
            "workflows": [],
            "pipeline_id": "pipeline_1",
            "dependency_ids": []
        }
        client._make_request = MagicMock(return_value=mock_response)
        
        task = client.get_task("task_1")
        
        assert isinstance(task, TaskInfo)
        assert task.id == "task_1"
        client._make_request.assert_called_once_with("GET", "/tasks/task_1")
    
    def test_get_task_not_found(self, client):
        """get_task should return None for non-existent task."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        client._make_request = MagicMock(side_effect=httpx.HTTPStatusError(
            "Not found",
            request=MagicMock(),
            response=mock_response
        ))
        
        task = client.get_task("nonexistent")
        
        assert task is None
    
    def test_get_all_tasks(self, client):
        """get_all_tasks should retrieve all tasks."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "tasks": [
                {
                    "id": "task_1",
                    "tool": {"name": "tool1", "container": {"image": "img:latest", "command": "cmd"}},
                    "config": {},
                    "priority": 100,
                    "status": "pending",
                    "task_type": "pre_training",
                    "counter": 1,
                    "workflows": [],
                    "pipeline_id": "pipeline_1",
                    "dependency_ids": []
                }
            ]
        }
        client._make_request = MagicMock(return_value=mock_response)
        
        tasks = client.get_all_tasks()
        
        assert len(tasks) == 1
        assert isinstance(tasks[0], TaskInfo)
        client._make_request.assert_called_once_with("GET", "/tasks", params=None)
    
    def test_get_all_tasks_with_status_filter(self, client):
        """get_all_tasks should filter by status."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"tasks": []}
        client._make_request = MagicMock(return_value=mock_response)
        
        client.get_all_tasks(status="completed")
        
        call_args = client._make_request.call_args
        assert call_args[1]["params"]["status"] == "completed"


# ============================================================================
# Test: Tool Information
# ============================================================================


class TestToolInformation:
    """Tests for tool information endpoints."""
    
    def test_get_tools(self, client):
        """get_tools should retrieve all tools."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "tools": [
                {"name": "tool1", "container": {"image": "img:latest"}},
                {"name": "tool2", "container": {"image": "img2:latest"}}
            ]
        }
        client._make_request = MagicMock(return_value=mock_response)
        
        tools = client.get_tools()
        
        assert len(tools) == 2
        client._make_request.assert_called_once_with("GET", "/tools")
    
    def test_get_tool(self, client):
        """get_tool should retrieve specific tool."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "test_tool",
            "container": {"image": "test/image:latest"}
        }
        client._make_request = MagicMock(return_value=mock_response)
        
        tool = client.get_tool("test_tool")
        
        assert tool["name"] == "test_tool"
        client._make_request.assert_called_once_with("GET", "/tools/test_tool")
    
    def test_get_tool_not_found(self, client):
        """get_tool should return None for non-existent tool."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        client._make_request = MagicMock(side_effect=httpx.HTTPStatusError(
            "Not found",
            request=MagicMock(),
            response=mock_response
        ))
        
        tool = client.get_tool("nonexistent")
        
        assert tool is None


# ============================================================================
# Test: Cleanup
# ============================================================================


class TestCleanup:
    """Tests for client cleanup."""
    
    def test_close(self, client):
        """close should close HTTP client."""
        client.close()
        
        client._client.close.assert_called_once()
    
    def test_context_manager(self, client):
        """Client should work as context manager."""
        with patch('src.worker.client.httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            with LandseerClient() as client:
                pass
            
            mock_client.close.assert_called_once()


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_task_info_from_api_response_minimal(self):
        """TaskInfo.from_api_response should handle minimal data."""
        data = {
            "id": "task_1",
            "tool": {
                "name": "tool",
                "container": {"image": "img:latest", "command": "cmd"}
            },
            "config": {},
            "priority": 100,
            "status": "pending",
            "task_type": "pre_training",
            "counter": 1,
            "workflows": [],
            "pipeline_id": "pipeline_1",
            "dependency_ids": []
        }
        
        task = TaskInfo.from_api_response(data)
        
        assert task.id == "task_1"
        assert task.tool_name == "tool"
    
    def test_task_info_from_api_response_with_dependencies(self):
        """TaskInfo.from_api_response should handle dependencies."""
        data = {
            "id": "task_1",
            "tool": {
                "name": "tool",
                "container": {"image": "img:latest", "command": "cmd"}
            },
            "config": {},
            "priority": 100,
            "status": "pending",
            "task_type": "post_training",
            "counter": 1,
            "workflows": [],
            "pipeline_id": "pipeline_1",
            "dependency_ids": ["dep1", "dep2"]
        }
        
        task = TaskInfo.from_api_response(data)
        
        assert len(task.dependency_ids) == 2
        assert "dep1" in task.dependency_ids
        assert "dep2" in task.dependency_ids
    
    def test_client_with_empty_backend_url(self):
        """Client should handle empty backend URL."""
        with patch('src.worker.client.httpx.Client'):
            client = LandseerClient(backend_url="")
        
        assert client.backend_url == ""
    
    def test_client_with_very_long_timeout(self):
        """Client should handle very long timeout."""
        with patch('src.worker.client.httpx.Client'):
            client = LandseerClient(timeout=3600.0)
        
        assert client.timeout == 3600.0
    
    def test_client_with_zero_retry_attempts(self):
        """Client should handle zero retry attempts."""
        with patch('src.worker.client.httpx.Client'):
            client = LandseerClient(retry_attempts=0)
        
        assert client.retry_attempts == 0
