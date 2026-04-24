"""
FastAPI backend API unit tests.

Tests cover:
- GET / root endpoint
- GET /health: scheduler_active flag, status field
- GET /info/pipeline: 503 when uninitialized, pipeline info when initialized
- GET /info/workflows: workflow list with task_ids
- GET /tasks/next: no task when pipeline empty, dispatches task with correct shape
- GET /tasks/next: all-complete message, running-task message
- PUT /tasks/status: 400 for invalid status, 404 for unknown task_id
- PUT /tasks/status: marks task COMPLETED, marks task FAILED
- GET /tasks: all tasks list, filtered by status, 400 for invalid filter
- POST /workers/register: auto-generates worker_id, stores capabilities
- POST /workers/{worker_id}/heartbeat: updates heartbeat, returns 404 for unknown worker
- GET /workers: lists registered workers
- SchedulerState.initialize: raises for unknown scheduler type
- SchedulerState.register_worker: auto-increments counter
- SchedulerState.update_worker_heartbeat: false for unknown worker
- SchedulerState.assign_task_to_worker / complete_worker_task
- SchedulerState.reclaim_stale_workers: resets RUNNING tasks, marks workers offline
"""

import pytest
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.pipeline.tasks import (
    Task,
    TaskType,
    TaskStatus,
    TaskFactory,
    EvaluationTask,
    clear_task_registry,
)
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import Workflow
from src.pipeline.pipeline import DefenseEvaluationPipeline
from src.backend.api import app, get_scheduler_state, SchedulerState
import src.pipeline.tasks as tasks_module


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    """Reset all global mutable state before every test."""
    from src.backend.api import _scheduler_state
    _scheduler_state.scheduler = None
    _scheduler_state.pipeline = None
    _scheduler_state.started_at = None
    _scheduler_state.task_metadata.clear()
    _scheduler_state.workers.clear()
    _scheduler_state._worker_counter = 0
    _scheduler_state._custom_tools.clear()
    _scheduler_state._db_service = None
    clear_task_registry()
    tasks_module._task_id_counter = 0
    tasks_module._workflow_id_counter = 0
    tasks_module._pipeline_id_counter = 0
    yield
    from src.backend.api import _scheduler_state
    _scheduler_state.scheduler = None
    _scheduler_state.pipeline = None
    _scheduler_state.started_at = None
    _scheduler_state.task_metadata.clear()
    _scheduler_state.workers.clear()
    _scheduler_state._worker_counter = 0
    _scheduler_state._custom_tools.clear()
    _scheduler_state._db_service = None
    clear_task_registry()


@pytest.fixture
def client() -> TestClient:
    """TestClient with no mocked dependencies (uses real global state)."""
    return TestClient(app, raise_server_exceptions=True)


def make_tool(name: str = "tool") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        container=ContainerConfig(image=f"img/{name}:v1", command="run"),
    )


def make_task(tool: ToolDefinition = None, task_type: TaskType = TaskType.PRE_TRAINING,
              config: dict = None, deps: list = None) -> Task:
    t = tool or make_tool()
    return TaskFactory.create_task(
        task_type, tool=t,
        config=config or {},
        dependencies=deps or []
    )


def build_pipeline_with_task(task: Task, pipeline_name: str = "test") -> DefenseEvaluationPipeline:
    wf = Workflow(name="wf1", tasks=[task])
    return DefenseEvaluationPipeline(
        name=pipeline_name,
        workflows=[wf],
        config={},
        dataset={"name": "cifar10"},
        model={"script": "/dev/null"}
    )


@pytest.fixture
def initialized_client(client: TestClient) -> TestClient:
    """Return a TestClient whose SchedulerState is initialized with a one-task pipeline."""
    task = make_task(config={"unit": "test"})
    pipeline = build_pipeline_with_task(task)
    from src.backend.api import _scheduler_state
    _scheduler_state.initialize(pipeline)
    return client


@pytest.fixture
def scheduler_state() -> SchedulerState:
    """Return the live global SchedulerState directly for unit-testing its methods."""
    from src.backend.api import _scheduler_state
    return _scheduler_state


# ============================================================================
# Tests: Root and Health endpoints
# ============================================================================


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_api_name(self, client):
        data = client.get("/").json()
        assert "Landseer" in data.get("name", "")

    def test_root_contains_docs_link(self, client):
        data = client.get("/").json()
        assert "docs" in data


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_scheduler_inactive_when_not_initialized(self, client):
        data = client.get("/health").json()
        assert data["scheduler_active"] is False

    def test_health_scheduler_active_when_initialized(self, initialized_client):
        data = initialized_client.get("/health").json()
        assert data["scheduler_active"] is True

    def test_health_contains_timestamp(self, client):
        data = client.get("/health").json()
        assert "timestamp" in data
        # Validate it's an ISO-format datetime string
        datetime.fromisoformat(data["timestamp"])


# ============================================================================
# Tests: Pipeline info endpoints (require initialized scheduler)
# ============================================================================


class TestPipelineInfoEndpoint:
    """Tests for GET /info/pipeline."""

    def test_returns_503_when_uninitialized(self, client):
        response = client.get("/info/pipeline")
        assert response.status_code == 503

    def test_returns_pipeline_id(self, initialized_client):
        data = initialized_client.get("/info/pipeline").json()
        assert "id" in data
        assert data["id"]  # non-empty

    def test_returns_pipeline_name(self, initialized_client):
        data = initialized_client.get("/info/pipeline").json()
        assert data["name"] == "test"

    def test_returns_workflow_count(self, initialized_client):
        data = initialized_client.get("/info/pipeline").json()
        assert data["workflow_count"] == 1

    def test_returns_task_count(self, initialized_client):
        data = initialized_client.get("/info/pipeline").json()
        assert data["task_count"] == 1

    def test_returns_dataset_info(self, initialized_client):
        data = initialized_client.get("/info/pipeline").json()
        assert data["dataset"]["name"] == "cifar10"


class TestWorkflowListEndpoint:
    """Tests for GET /info/workflows."""

    def test_returns_503_when_uninitialized(self, client):
        response = client.get("/info/workflows")
        assert response.status_code == 503

    def test_returns_workflow_list(self, initialized_client):
        data = initialized_client.get("/info/workflows").json()
        assert data["total"] == 1
        assert len(data["workflows"]) == 1

    def test_workflow_has_id_and_name(self, initialized_client):
        wf = initialized_client.get("/info/workflows").json()["workflows"][0]
        assert "id" in wf
        assert wf["name"] == "wf1"

    def test_workflow_has_task_ids(self, initialized_client):
        wf = initialized_client.get("/info/workflows").json()["workflows"][0]
        assert "task_ids" in wf
        assert len(wf["task_ids"]) == 1


# ============================================================================
# Tests: GET /tasks/next
# ============================================================================


class TestGetNextTask:
    """Tests for GET /tasks/next."""

    def test_returns_503_when_uninitialized(self, client):
        response = client.get("/tasks/next")
        assert response.status_code == 503

    def test_returns_has_task_true_when_ready(self, initialized_client):
        data = initialized_client.get("/tasks/next").json()
        assert data["has_task"] is True
        assert data["task"] is not None

    def test_dispatched_task_has_correct_fields(self, initialized_client):
        task_data = initialized_client.get("/tasks/next").json()["task"]
        assert "id" in task_data
        assert "tool" in task_data
        assert "priority" in task_data
        assert "status" in task_data
        assert "task_type" in task_data

    def test_dispatched_task_status_is_running(self, initialized_client):
        task_data = initialized_client.get("/tasks/next").json()["task"]
        assert task_data["status"] == "running"

    def test_no_task_when_all_running(self, initialized_client):
        # Claim the task first
        initialized_client.get("/tasks/next")
        # Try again — now the task is RUNNING, no more PENDING
        data = initialized_client.get("/tasks/next").json()
        assert data["has_task"] is False

    def test_no_task_message_when_all_complete(self, initialized_client):
        from src.backend.api import _scheduler_state
        task = _scheduler_state.scheduler.get_all_tasks()[0]
        task.status = TaskStatus.COMPLETED

        data = initialized_client.get("/tasks/next").json()
        assert data["has_task"] is False
        assert "completed" in data["message"].lower()


# ============================================================================
# Tests: PUT /tasks/status
# ============================================================================


class TestUpdateTaskStatus:
    """Tests for PUT /tasks/status."""

    def test_returns_503_when_uninitialized(self, client):
        response = client.put("/tasks/status", json={
            "task_id": "task_1",
            "status": "completed"
        })
        assert response.status_code == 503

    def test_marks_task_completed(self, initialized_client):
        # Claim the task first (sets to RUNNING)
        task_data = initialized_client.get("/tasks/next").json()["task"]
        task_id = task_data["id"]

        response = initialized_client.put("/tasks/status", json={
            "task_id": task_id,
            "status": "completed"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["new_status"] == "completed"

    def test_marks_task_failed(self, initialized_client):
        task_data = initialized_client.get("/tasks/next").json()["task"]
        task_id = task_data["id"]

        response = initialized_client.put("/tasks/status", json={
            "task_id": task_id,
            "status": "failed",
            "error_message": "container exited with code 1"
        })
        assert response.status_code == 200
        assert response.json()["new_status"] == "failed"

    def test_returns_400_for_invalid_status(self, initialized_client):
        response = initialized_client.put("/tasks/status", json={
            "task_id": "task_1",
            "status": "in_progress"
        })
        assert response.status_code == 400

    def test_returns_404_for_unknown_task_id(self, initialized_client):
        response = initialized_client.put("/tasks/status", json={
            "task_id": "task_nonexistent_xyz",
            "status": "completed"
        })
        assert response.status_code == 404

    def test_400_for_running_status(self, initialized_client):
        response = initialized_client.put("/tasks/status", json={
            "task_id": "task_1",
            "status": "running"
        })
        assert response.status_code == 400


# ============================================================================
# Tests: GET /tasks
# ============================================================================


class TestGetAllTasks:
    """Tests for GET /tasks."""

    def test_returns_503_when_uninitialized(self, client):
        assert client.get("/tasks").status_code == 503

    def test_returns_task_list(self, initialized_client):
        data = initialized_client.get("/tasks").json()
        assert "tasks" in data
        assert data["total"] == 1

    def test_filter_by_pending(self, initialized_client):
        data = initialized_client.get("/tasks?status=pending").json()
        assert all(t["status"] == "pending" for t in data["tasks"])

    def test_filter_by_running(self, initialized_client):
        # Claim the task to make it RUNNING
        initialized_client.get("/tasks/next")
        data = initialized_client.get("/tasks?status=running").json()
        assert data["total"] >= 1
        assert all(t["status"] == "running" for t in data["tasks"])

    def test_filter_by_invalid_status_returns_400(self, initialized_client):
        assert initialized_client.get("/tasks?status=garbage").status_code == 400


# ============================================================================
# Tests: Worker endpoints
# ============================================================================


class TestWorkerRegister:
    """Tests for POST /workers/register."""

    def test_register_returns_worker_id(self, client):
        response = client.post("/workers/register", json={
            "hostname": "worker-host-1"
        })
        assert response.status_code == 200
        data = response.json()
        assert "worker_id" in data
        assert data["worker_id"]

    def test_auto_generated_worker_id_increments(self, client):
        r1 = client.post("/workers/register", json={"hostname": "h1"}).json()
        r2 = client.post("/workers/register", json={"hostname": "h2"}).json()
        assert r1["worker_id"] != r2["worker_id"]

    def test_custom_worker_id_honored(self, client):
        response = client.post("/workers/register", json={
            "worker_id": "my-gpu-node-3",
            "hostname": "gpu3.cluster"
        })
        data = response.json()
        assert data["worker_id"] == "my-gpu-node-3"

    def test_capabilities_stored(self, client):
        caps = {"gpu": 2, "memory_gb": 64}
        client.post("/workers/register", json={
            "hostname": "h1",
            "capabilities": caps
        })
        from src.backend.api import _scheduler_state
        worker = list(_scheduler_state.workers.values())[0]
        assert worker["capabilities"]["gpu"] == 2


class TestWorkerHeartbeat:
    """Tests for POST /workers/{worker_id}/heartbeat."""

    def test_heartbeat_updates_timestamp(self, client):
        # Register worker
        wid = client.post("/workers/register", json={"hostname": "h1"}).json()["worker_id"]
        from src.backend.api import _scheduler_state
        old_hb = _scheduler_state.workers[wid]["last_heartbeat"]

        # Heartbeat
        response = client.post(f"/workers/{wid}/heartbeat", json={"worker_id": wid})
        assert response.status_code == 200

        new_hb = _scheduler_state.workers[wid]["last_heartbeat"]
        assert new_hb >= old_hb

    def test_heartbeat_404_for_unknown_worker(self, client):
        response = client.post("/workers/unknown_worker_xyz/heartbeat",
                               json={"worker_id": "unknown_worker_xyz"})
        assert response.status_code == 404


class TestWorkerList:
    """Tests for GET /workers."""

    def test_returns_empty_list_initially(self, client):
        data = client.get("/workers").json()
        assert data["total"] == 0
        assert data["workers"] == []

    def test_registered_workers_appear(self, client):
        client.post("/workers/register", json={"hostname": "h1"})
        client.post("/workers/register", json={"hostname": "h2"})
        data = client.get("/workers").json()
        assert data["total"] == 2

    def test_worker_has_expected_fields(self, client):
        client.post("/workers/register", json={"hostname": "test-host"})
        worker = client.get("/workers").json()["workers"][0]
        assert "worker_id" in worker
        assert "hostname" in worker
        assert "status" in worker
        assert worker["hostname"] == "test-host"


# ============================================================================
# Tests: SchedulerState methods (unit tests, no HTTP)
# ============================================================================


class TestSchedulerStateInitialize:
    """Tests for SchedulerState.initialize."""

    def test_initialize_sets_pipeline(self, scheduler_state):
        task = make_task()
        pipeline = build_pipeline_with_task(task)
        scheduler_state.initialize(pipeline)
        assert scheduler_state.pipeline is pipeline

    def test_initialize_sets_started_at(self, scheduler_state):
        task = make_task()
        pipeline = build_pipeline_with_task(task)
        before = datetime.now()
        scheduler_state.initialize(pipeline)
        after = datetime.now()
        assert before <= scheduler_state.started_at <= after

    def test_initialize_unknown_scheduler_type_raises(self, scheduler_state):
        task = make_task()
        pipeline = build_pipeline_with_task(task)
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            scheduler_state.initialize(pipeline, scheduler_type="round_robin")

    def test_is_initialized_false_before_init(self, scheduler_state):
        assert scheduler_state.is_initialized() is False

    def test_is_initialized_true_after_init(self, scheduler_state):
        task = make_task()
        pipeline = build_pipeline_with_task(task)
        scheduler_state.initialize(pipeline)
        assert scheduler_state.is_initialized() is True


class TestSchedulerStateWorkerManagement:
    """Tests for SchedulerState worker CRUD methods."""

    def test_register_worker_returns_id(self, scheduler_state):
        wid = scheduler_state.register_worker("host-a")
        assert wid == "worker_1"

    def test_register_worker_counter_increments(self, scheduler_state):
        w1 = scheduler_state.register_worker("host-a")
        w2 = scheduler_state.register_worker("host-b")
        assert w1 != w2

    def test_register_custom_worker_id(self, scheduler_state):
        wid = scheduler_state.register_worker("host-a", worker_id="custom-99")
        assert wid == "custom-99"

    def test_worker_initial_status_idle(self, scheduler_state):
        wid = scheduler_state.register_worker("host-a")
        assert scheduler_state.workers[wid]["status"] == "idle"

    def test_update_heartbeat_returns_true(self, scheduler_state):
        wid = scheduler_state.register_worker("h")
        result = scheduler_state.update_worker_heartbeat(wid)
        assert result is True

    def test_update_heartbeat_false_for_unknown(self, scheduler_state):
        result = scheduler_state.update_worker_heartbeat("no_such_worker")
        assert result is False

    def test_idle_heartbeat_clears_orphan_completed_task_pointer(self, scheduler_state):
        task = make_task()
        pipeline = build_pipeline_with_task(task)
        scheduler_state.initialize(pipeline)
        wid = scheduler_state.register_worker("h")
        scheduler_state.workers[wid]["current_task_id"] = task.id
        task.status = TaskStatus.COMPLETED

        scheduler_state.update_worker_heartbeat(wid, status="idle")

        assert scheduler_state.workers[wid]["current_task_id"] is None

    def test_idle_heartbeat_reclaims_running_task(self, scheduler_state):
        task = make_task()
        pipeline = build_pipeline_with_task(task)
        scheduler_state.initialize(pipeline)
        wid = scheduler_state.register_worker("h")
        scheduler_state.workers[wid]["current_task_id"] = task.id
        task.status = TaskStatus.RUNNING

        scheduler_state.update_worker_heartbeat(wid, status="idle")

        assert task.status == TaskStatus.PENDING
        assert scheduler_state.workers[wid]["current_task_id"] is None

    def test_idle_heartbeat_clears_unknown_task_id(self, scheduler_state):
        task = make_task()
        pipeline = build_pipeline_with_task(task)
        scheduler_state.initialize(pipeline)
        wid = scheduler_state.register_worker("h")
        scheduler_state.workers[wid]["current_task_id"] = "no_such_task_id"

        scheduler_state.update_worker_heartbeat(wid, status="idle")

        assert scheduler_state.workers[wid]["current_task_id"] is None

    def test_assign_task_sets_status_busy(self, scheduler_state):
        wid = scheduler_state.register_worker("h")
        scheduler_state.assign_task_to_worker(wid, "task_1")
        assert scheduler_state.workers[wid]["status"] == "busy"
        assert scheduler_state.workers[wid]["current_task_id"] == "task_1"

    def test_complete_task_sets_status_idle(self, scheduler_state):
        wid = scheduler_state.register_worker("h")
        scheduler_state.assign_task_to_worker(wid, "task_1")
        scheduler_state.complete_worker_task(wid, success=True)
        assert scheduler_state.workers[wid]["status"] == "idle"
        assert scheduler_state.workers[wid]["current_task_id"] is None

    def test_complete_task_success_increments_completed(self, scheduler_state):
        wid = scheduler_state.register_worker("h")
        scheduler_state.assign_task_to_worker(wid, "task_1")
        scheduler_state.complete_worker_task(wid, success=True)
        assert scheduler_state.workers[wid]["tasks_completed"] == 1
        assert scheduler_state.workers[wid]["tasks_failed"] == 0

    def test_complete_task_failure_increments_failed(self, scheduler_state):
        wid = scheduler_state.register_worker("h")
        scheduler_state.assign_task_to_worker(wid, "task_1")
        scheduler_state.complete_worker_task(wid, success=False)
        assert scheduler_state.workers[wid]["tasks_failed"] == 1
        assert scheduler_state.workers[wid]["tasks_completed"] == 0


class TestReclaimeStaleWorkers:
    """Tests for SchedulerState.reclaim_stale_workers."""

    def test_no_stale_workers_returns_empty(self, scheduler_state):
        scheduler_state.register_worker("fresh-host")
        result = scheduler_state.reclaim_stale_workers(stale_timeout_seconds=9999)
        assert result["stale_workers"] == []
        assert result["reclaimed_tasks"] == []

    def test_stale_worker_marked_offline(self, scheduler_state):
        wid = scheduler_state.register_worker("stale-host")
        # Back-date the heartbeat
        stale_time = (datetime.now() - timedelta(seconds=120)).isoformat()
        scheduler_state.workers[wid]["last_heartbeat"] = stale_time

        result = scheduler_state.reclaim_stale_workers(stale_timeout_seconds=90)
        assert wid in result["stale_workers"]
        assert scheduler_state.workers[wid]["status"] == "offline"

    def test_stale_worker_running_task_reclaimed(self, scheduler_state):
        # Set up a pipeline with a RUNNING task assigned to a stale worker
        task = make_task()
        pipeline = build_pipeline_with_task(task)
        scheduler_state.initialize(pipeline)

        wid = scheduler_state.register_worker("stale-host")
        task.status = TaskStatus.RUNNING
        scheduler_state.workers[wid]["current_task_id"] = task.id
        stale_time = (datetime.now() - timedelta(seconds=120)).isoformat()
        scheduler_state.workers[wid]["last_heartbeat"] = stale_time

        result = scheduler_state.reclaim_stale_workers(stale_timeout_seconds=90)

        assert task.id in result["reclaimed_tasks"]
        assert task.status == TaskStatus.PENDING  # reclaimed back to pending

    def test_non_stale_worker_not_affected(self, scheduler_state):
        wid = scheduler_state.register_worker("fresh-host")
        # Fresh heartbeat (no back-dating)
        result = scheduler_state.reclaim_stale_workers(stale_timeout_seconds=60)
        assert wid not in result["stale_workers"]


# ============================================================================
# Tests: task_to_response helper
# ============================================================================


class TestTaskToResponse:
    """Tests for the task_to_response converter."""

    def test_response_contains_tool_info(self, initialized_client):
        task_resp = initialized_client.get("/tasks/next").json()["task"]
        assert "tool" in task_resp
        assert "name" in task_resp["tool"]
        assert "container" in task_resp["tool"]

    def test_response_contains_dependency_ids(self, initialized_client):
        task_resp = initialized_client.get("/tasks/next").json()["task"]
        assert "dependency_ids" in task_resp
        assert isinstance(task_resp["dependency_ids"], list)

    def test_response_dependency_ids_empty_for_root_task(self, initialized_client):
        task_resp = initialized_client.get("/tasks/next").json()["task"]
        assert task_resp["dependency_ids"] == []
