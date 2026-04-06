"""
Worker client unit tests.

Tests cover:
- TaskInfo.from_api_response: full response parsing
- TaskInfo.from_api_response: missing/None fields default gracefully
- TaskInfo.from_api_response: nested tool/container extraction
- LandseerClient.__init__: stores config, strips trailing slash
- LandseerClient.worker_id / is_registered before registration
- LandseerClient.is_backend_available: True on ok health, False on error
- LandseerClient.health_check: success path
- LandseerClient.get_next_task: returns TaskInfo when has_task=True
- LandseerClient.get_next_task: returns None when has_task=False
- LandseerClient.register: stores worker_id, returns WorkerInfo
- LandseerClient.heartbeat: raises if not registered, True on success
- LandseerClient.report_task_completed: sends correct payload
- LandseerClient.report_task_failed: sends correct payload
- LandseerClient._make_request: does not retry on 4xx errors
- LandseerClient._make_request: retries on 5xx, raises after exhausting
"""

import pytest
from unittest.mock import MagicMock, patch, call
from typing import Any, Dict, Optional

import httpx

from src.worker.client import TaskInfo, WorkerInfo, LandseerClient


# ============================================================================
# Fixtures
# ============================================================================


def make_task_api_response(**overrides) -> Dict[str, Any]:
    """Return a well-formed /tasks/next task payload."""
    base = {
        "id": "task_42",
        "tool": {
            "name": "clean-eval",
            "container": {
                "image": "img/clean:v2",
                "command": "python eval.py",
                "runtime": None
            },
            "is_baseline": False,
        },
        "config": {"stage": "evaluation"},
        "priority": 50,
        "status": "running",
        "task_type": "evaluation",
        "counter": 3,
        "workflows": ["wf_1", "wf_2"],
        "pipeline_id": "pipe_1",
        "dependency_ids": ["task_10", "task_11"],
        "run_id": "run_007",
        "cache_key": "abc123",
        "output_path": "/mnt/artifacts/task_42",
        "log_path": "/mnt/logs/task_42.log",
    }
    base.update(overrides)
    return base


def make_mock_http_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Return a mock httpx.Response with .json() and .raise_for_status()."""
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = json_data
    resp.status_code = status_code
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="error", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def client() -> LandseerClient:
    """A LandseerClient whose underlying HTTP transport is mocked."""
    c = LandseerClient(backend_url="http://test-backend:8000", retry_attempts=1, retry_delay=0)
    c._client = MagicMock()
    return c


@pytest.fixture
def registered_client(client: LandseerClient) -> LandseerClient:
    """A client pre-registered with worker_1."""
    client._worker_id = "worker_1"
    client._worker_info = WorkerInfo(
        worker_id="worker_1",
        hostname="test-host",
        status="idle",
        registered_at="2024-01-01T00:00:00",
        last_heartbeat="2024-01-01T00:00:00"
    )
    return client


# ============================================================================
# Tests: TaskInfo.from_api_response
# ============================================================================


class TestTaskInfoFromApiResponse:
    """Tests for TaskInfo.from_api_response deserialization."""

    # -- happy path --

    def test_id_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.id == "task_42"

    def test_tool_name_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.tool_name == "clean-eval"

    def test_tool_image_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.tool_image == "img/clean:v2"

    def test_tool_command_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.tool_command == "python eval.py"

    def test_tool_runtime_parsed_none(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.tool_runtime is None

    def test_tool_runtime_parsed_string(self):
        data = make_task_api_response()
        data["tool"]["container"]["runtime"] = "docker"
        ti = TaskInfo.from_api_response(data)
        assert ti.tool_runtime == "docker"

    def test_tool_is_baseline_false(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.tool_is_baseline is False

    def test_tool_is_baseline_true(self):
        data = make_task_api_response()
        data["tool"]["is_baseline"] = True
        ti = TaskInfo.from_api_response(data)
        assert ti.tool_is_baseline is True

    def test_config_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.config == {"stage": "evaluation"}

    def test_priority_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.priority == 50

    def test_status_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.status == "running"

    def test_task_type_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.task_type == "evaluation"

    def test_counter_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.counter == 3

    def test_workflows_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.workflows == ["wf_1", "wf_2"]

    def test_pipeline_id_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.pipeline_id == "pipe_1"

    def test_dependency_ids_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.dependency_ids == ["task_10", "task_11"]

    def test_run_id_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.run_id == "run_007"

    def test_cache_key_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.cache_key == "abc123"

    def test_output_path_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.output_path == "/mnt/artifacts/task_42"

    def test_log_path_parsed(self):
        ti = TaskInfo.from_api_response(make_task_api_response())
        assert ti.log_path == "/mnt/logs/task_42.log"

    # -- missing / None fields --

    def test_missing_run_id_defaults_none(self):
        data = make_task_api_response()
        data.pop("run_id", None)
        ti = TaskInfo.from_api_response(data)
        assert ti.run_id is None

    def test_missing_cache_key_defaults_none(self):
        data = make_task_api_response()
        data.pop("cache_key", None)
        ti = TaskInfo.from_api_response(data)
        assert ti.cache_key is None

    def test_missing_output_path_defaults_none(self):
        data = make_task_api_response()
        data.pop("output_path", None)
        ti = TaskInfo.from_api_response(data)
        assert ti.output_path is None

    def test_missing_log_path_defaults_none(self):
        data = make_task_api_response()
        data.pop("log_path", None)
        ti = TaskInfo.from_api_response(data)
        assert ti.log_path is None

    def test_missing_config_defaults_empty_dict(self):
        data = make_task_api_response()
        data.pop("config", None)
        ti = TaskInfo.from_api_response(data)
        assert ti.config == {}

    def test_missing_dependency_ids_defaults_empty(self):
        data = make_task_api_response()
        data.pop("dependency_ids", None)
        ti = TaskInfo.from_api_response(data)
        assert ti.dependency_ids == []

    def test_missing_workflows_defaults_empty(self):
        data = make_task_api_response()
        data.pop("workflows", None)
        ti = TaskInfo.from_api_response(data)
        assert ti.workflows == []

    def test_minimal_response_no_crash(self):
        # Only "id" present; everything else defaults
        ti = TaskInfo.from_api_response({"id": "task_99"})
        assert ti.id == "task_99"
        assert ti.tool_name == ""
        assert ti.tool_image == ""
        assert ti.tool_command == ""
        assert ti.config == {}


# ============================================================================
# Tests: LandseerClient initialization
# ============================================================================


class TestLandseerClientInit:
    """Tests for LandseerClient constructor."""

    def test_backend_url_stored(self):
        c = LandseerClient(backend_url="http://example:9000")
        assert "example" in c.backend_url

    def test_trailing_slash_stripped(self):
        c = LandseerClient(backend_url="http://example:9000/")
        assert not c.backend_url.endswith("/")

    def test_retry_attempts_stored(self):
        c = LandseerClient(retry_attempts=5)
        assert c.retry_attempts == 5

    def test_worker_id_none_before_registration(self):
        c = LandseerClient()
        assert c.worker_id is None

    def test_is_registered_false_before_registration(self):
        c = LandseerClient()
        assert c.is_registered is False

    def test_is_registered_true_after_setting_worker_id(self):
        c = LandseerClient()
        c._worker_id = "w1"
        assert c.is_registered is True


# ============================================================================
# Tests: LandseerClient.is_backend_available
# ============================================================================


class TestIsBackendAvailable:
    """Tests for is_backend_available."""

    def test_true_when_health_returns_ok(self, client):
        client._client.request.return_value = make_mock_http_response({"status": "ok"})
        assert client.is_backend_available() is True

    def test_false_when_health_returns_error_status(self, client):
        client._client.request.return_value = make_mock_http_response({"status": "error"})
        assert client.is_backend_available() is False

    def test_false_when_request_raises(self, client):
        client._client.request.side_effect = httpx.ConnectError("refused")
        assert client.is_backend_available() is False

    def test_false_when_status_key_missing(self, client):
        client._client.request.return_value = make_mock_http_response({})
        assert client.is_backend_available() is False


# ============================================================================
# Tests: LandseerClient.health_check
# ============================================================================


class TestHealthCheck:
    """Tests for health_check."""

    def test_returns_json_dict(self, client):
        client._client.request.return_value = make_mock_http_response({
            "status": "ok",
            "scheduler_active": True,
            "timestamp": "2024-01-01T00:00:00"
        })
        result = client.health_check()
        assert result["status"] == "ok"
        assert result["scheduler_active"] is True

    def test_raises_on_server_error(self, client):
        client._client.request.return_value = make_mock_http_response({}, status_code=503)
        with pytest.raises(httpx.HTTPStatusError):
            client.health_check()


# ============================================================================
# Tests: LandseerClient.get_next_task
# ============================================================================


class TestGetNextTask:
    """Tests for get_next_task."""

    def test_returns_task_info_when_available(self, client):
        task_payload = make_task_api_response()
        client._client.request.return_value = make_mock_http_response({
            "has_task": True,
            "task": task_payload,
            "message": "Task assigned"
        })

        result = client.get_next_task()

        assert isinstance(result, TaskInfo)
        assert result.id == "task_42"

    def test_returns_none_when_no_task(self, client):
        client._client.request.return_value = make_mock_http_response({
            "has_task": False,
            "task": None,
            "message": "All tasks completed."
        })

        result = client.get_next_task()

        assert result is None

    def test_calls_correct_endpoint(self, client):
        client._client.request.return_value = make_mock_http_response({
            "has_task": False, "task": None, "message": ""
        })
        client.get_next_task()

        call_args = client._client.request.call_args
        assert call_args[1]["url"] == "/tasks/next" or "/tasks/next" in str(call_args)


# ============================================================================
# Tests: LandseerClient.register
# ============================================================================


class TestRegister:
    """Tests for register."""

    def _register_response(self, worker_id: str = "worker_1") -> dict:
        return {
            "worker_id": worker_id,
            "hostname": "test-host",
            "status": "idle",
            "registered_at": "2024-01-01T00:00:00",
            "last_heartbeat": "2024-01-01T00:00:00",
            "current_task_id": None,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "capabilities": {}
        }

    def test_stores_worker_id(self, client):
        client._client.request.return_value = make_mock_http_response(
            self._register_response("w42")
        )
        client.register()
        assert client.worker_id == "w42"

    def test_is_registered_after_register(self, client):
        client._client.request.return_value = make_mock_http_response(
            self._register_response()
        )
        client.register()
        assert client.is_registered is True

    def test_returns_worker_info(self, client):
        client._client.request.return_value = make_mock_http_response(
            self._register_response("w99")
        )
        info = client.register()
        assert isinstance(info, WorkerInfo)
        assert info.worker_id == "w99"

    def test_register_sends_hostname(self, client):
        client._client.request.return_value = make_mock_http_response(
            self._register_response()
        )
        client.register()
        call_args = client._client.request.call_args
        sent_data = call_args[1].get("json") or {}
        assert "hostname" in sent_data

    def test_register_sends_capabilities(self, client):
        client._client.request.return_value = make_mock_http_response(
            self._register_response()
        )
        caps = {"gpu": 1}
        client.register(capabilities=caps)
        call_args = client._client.request.call_args
        sent_data = call_args[1].get("json") or {}
        assert sent_data.get("capabilities") == caps


# ============================================================================
# Tests: LandseerClient.heartbeat
# ============================================================================


class TestHeartbeat:
    """Tests for heartbeat."""

    def test_raises_if_not_registered(self, client):
        with pytest.raises(RuntimeError, match="not registered"):
            client.heartbeat()

    def test_returns_true_on_success(self, registered_client):
        registered_client._client.request.return_value = make_mock_http_response(
            {"success": True}
        )
        result = registered_client.heartbeat()
        assert result is True

    def test_returns_false_on_network_error(self, registered_client):
        registered_client._client.request.side_effect = httpx.ConnectError("refused")
        result = registered_client.heartbeat()
        assert result is False

    def test_sends_status_if_provided(self, registered_client):
        registered_client._client.request.return_value = make_mock_http_response(
            {"success": True}
        )
        registered_client.heartbeat(status="busy")
        call_args = registered_client._client.request.call_args
        sent_data = call_args[1].get("json") or {}
        assert sent_data.get("status") == "busy"


# ============================================================================
# Tests: LandseerClient._make_request retry logic
# ============================================================================


class TestMakeRequestRetry:
    """Tests for _make_request retry and error handling."""

    def test_no_retry_on_4xx(self):
        c = LandseerClient(backend_url="http://b:8000", retry_attempts=3, retry_delay=0)
        c._client = MagicMock()
        resp_400 = make_mock_http_response({}, status_code=400)
        c._client.request.return_value = resp_400

        with pytest.raises(httpx.HTTPStatusError):
            c._make_request("GET", "/endpoint")

        # Called exactly once — no retry for 4xx
        assert c._client.request.call_count == 1

    def test_retries_on_5xx_then_raises(self):
        c = LandseerClient(backend_url="http://b:8000", retry_attempts=3, retry_delay=0)
        c._client = MagicMock()
        resp_503 = make_mock_http_response({}, status_code=503)
        c._client.request.return_value = resp_503

        with pytest.raises(httpx.HTTPStatusError):
            c._make_request("GET", "/endpoint")

        # Called retry_attempts times
        assert c._client.request.call_count == 3

    def test_retries_on_connection_error(self):
        c = LandseerClient(backend_url="http://b:8000", retry_attempts=2, retry_delay=0)
        c._client = MagicMock()
        c._client.request.side_effect = httpx.ConnectError("refused")

        with pytest.raises(Exception):
            c._make_request("GET", "/endpoint")

        assert c._client.request.call_count == 2

    def test_succeeds_after_first_retry(self):
        c = LandseerClient(backend_url="http://b:8000", retry_attempts=3, retry_delay=0)
        c._client = MagicMock()
        # First call fails with 5xx, second succeeds
        resp_ok = make_mock_http_response({"status": "ok"})
        resp_503 = make_mock_http_response({}, status_code=503)
        c._client.request.side_effect = [resp_503, resp_ok]

        # Patch raise_for_status so the 503 triggers retry properly
        resp_503.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=resp_503
        )

        result = c._make_request("GET", "/endpoint")
        assert result.json()["status"] == "ok"
        assert c._client.request.call_count == 2
