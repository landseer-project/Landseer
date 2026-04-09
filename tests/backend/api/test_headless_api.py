"""
API tests for headless mode (backend started without --config).

Tests cover:
- GET /health returns scheduler_active=False when no pipeline loaded
- GET /info/pipeline returns 503 when scheduler not initialized
- GET /info/workflows returns 503 when scheduler not initialized
- GET /tasks returns 503 when scheduler not initialized
- GET /tasks/next returns 503 when scheduler not initialized
- GET /progress returns 503 when scheduler not initialized
- GET /scheduler/status returns initialized=False
- GET /api/pipeline-configs still works (no scheduler needed)
- POST /workers/register still works (no scheduler needed)
- GET /workers still works (no scheduler needed)
- Transition: start run transitions scheduler from uninitialized to active
- After run started, /health shows scheduler_active=True
- After run started, /progress returns data
"""

import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from src.backend.api import app, _scheduler_state
from src.db import PipelineRunStatus
from src.pipeline.tasks import (
    TaskType,
    TaskStatus,
    TaskFactory,
    clear_task_registry,
)
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline
import src.pipeline.tasks as tasks_module


@pytest.fixture(autouse=True)
def reset_state():
    """Reset scheduler to uninitialized (headless) state before each test."""
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
def client():
    return TestClient(app, raise_server_exceptions=True)


# ============================================================================
# Headless mode: scheduler-dependent endpoints return 503
# ============================================================================


class TestHeadlessEndpoints503:
    """Endpoints that require an initialized scheduler should return 503."""

    def test_health_scheduler_inactive(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["scheduler_active"] is False

    def test_pipeline_info_503(self, client):
        assert client.get("/info/pipeline").status_code == 503

    def test_workflows_503(self, client):
        assert client.get("/info/workflows").status_code == 503

    def test_tasks_503(self, client):
        assert client.get("/tasks").status_code == 503

    def test_tasks_next_503(self, client):
        assert client.get("/tasks/next").status_code == 503

    def test_progress_503(self, client):
        assert client.get("/progress").status_code == 503

    def test_scheduler_status_uninitialized(self, client):
        data = client.get("/scheduler/status").json()
        assert data["initialized"] is False
        assert "pipeline_name" not in data or data.get("pipeline_name") is None


# ============================================================================
# Headless mode: scheduler-independent endpoints still work
# ============================================================================


class TestHeadlessEndpointsWork:
    """Endpoints that do not require an initialized scheduler should still work."""

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_root_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_worker_register_works(self, client):
        resp = client.post("/workers/register", json={"hostname": "gpu-node-1"})
        assert resp.status_code == 200
        assert "worker_id" in resp.json()

    def test_workers_list_works(self, client):
        resp = client.get("/workers")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_pipeline_configs_works(self, client):
        fake_config = SimpleNamespace(
            id="config_mini",
            name="mini",
            description="Mini pipeline",
            config_path="/tmp/mini.yaml",
            attack_config_path=None,
            config_hash="abc",
            created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
            updated_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:01"),
        )
        with patch("src.backend.config_discovery.sync_configs_to_db"), \
             patch("src.backend.config_discovery.get_all_configs", return_value=[fake_config]):
            resp = client.get("/api/pipeline-configs")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1


# ============================================================================
# Headless mode: transition to active by starting a run
# ============================================================================


def _build_minimal_pipeline():
    tool = ToolDefinition(
        name="test_tool",
        container=ContainerConfig(image="test/image:latest", command="run"),
    )
    task = TaskFactory.create_task(task_type=TaskType.PRE_TRAINING, tool=tool, config={})
    workflow = WorkflowFactory.create_workflow(name="wf_001", tasks=[task])
    return DefenseEvaluationPipeline(name="test_pipeline", workflows=[workflow])


class _FakeSessionScope:
    """Context manager that replaces DB session_scope in tests."""
    def __enter__(self):
        return object()
    def __exit__(self, *args):
        pass


class _FakeRunRepo:
    def __init__(self, _session):
        self._run = None

    def get_active_runs_for_config(self, _config_id):
        return []

    def get_next_run_number(self, _config_id):
        return 1

    def create(self, run_data):
        self._run = SimpleNamespace(
            id=run_data["id"],
            pipeline_config_id=run_data["pipeline_config_id"],
            run_number=run_data["run_number"],
            use_cache=run_data["use_cache"],
            status=PipelineRunStatus.PENDING,
            error_message=None,
            created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
            started_at=None,
            completed_at=None,
        )
        return self._run

    def update_status(self, run_id, _status):
        if self._run and self._run.id == run_id:
            self._run.status = PipelineRunStatus.RUNNING
            self._run.started_at = SimpleNamespace(isoformat=lambda: "2026-01-01T00:01:00")
        return self._run


class TestHeadlessToActiveTransition:
    """Starting a run should transition the scheduler from uninitialized to active."""

    def test_start_run_activates_scheduler(self, client):
        assert _scheduler_state.is_initialized() is False

        cfg = SimpleNamespace(
            id="config_mini",
            name="mini",
            config_path="/tmp/mini.yaml",
            attack_config_path=None,
        )
        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _FakeSessionScope), \
             patch("src.db.PipelineRunRepository", _FakeRunRepo), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   return_value=_build_minimal_pipeline()):
            resp = client.post(
                "/api/pipeline-configs/config_mini/runs",
                json={"use_cache": True},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "running"
        assert _scheduler_state.is_initialized() is True

    def test_health_active_after_run_started(self, client):
        cfg = SimpleNamespace(
            id="config_mini", name="mini",
            config_path="/tmp/mini.yaml", attack_config_path=None,
        )
        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _FakeSessionScope), \
             patch("src.db.PipelineRunRepository", _FakeRunRepo), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   return_value=_build_minimal_pipeline()):
            client.post("/api/pipeline-configs/config_mini/runs", json={"use_cache": True})

        health = client.get("/health").json()
        assert health["scheduler_active"] is True

    def test_progress_works_after_run_started(self, client):
        cfg = SimpleNamespace(
            id="config_mini", name="mini",
            config_path="/tmp/mini.yaml", attack_config_path=None,
        )
        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _FakeSessionScope), \
             patch("src.db.PipelineRunRepository", _FakeRunRepo), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   return_value=_build_minimal_pipeline()):
            client.post("/api/pipeline-configs/config_mini/runs", json={"use_cache": True})

        resp = client.get("/progress")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert "pending" in data

    def test_scheduler_status_initialized_after_run(self, client):
        cfg = SimpleNamespace(
            id="config_mini", name="mini",
            config_path="/tmp/mini.yaml", attack_config_path=None,
        )
        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _FakeSessionScope), \
             patch("src.db.PipelineRunRepository", _FakeRunRepo), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   return_value=_build_minimal_pipeline()):
            client.post("/api/pipeline-configs/config_mini/runs", json={"use_cache": True})

        status = client.get("/scheduler/status").json()
        assert status["initialized"] is True
        assert status["pipeline_name"] is not None
