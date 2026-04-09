"""
API tests for pipeline config and per-config run endpoints.

Tests cover:
- GET /api/pipeline-configs: returns discovered configs
- GET /api/pipeline-configs: returns empty list when no configs
- GET /api/pipeline-configs: calls sync_configs_to_db each time
- GET /api/pipeline-configs: response shape matches schema
- GET /api/pipeline-configs: returns 500 on internal error
- GET /api/pipeline-configs/{id}: returns config by ID
- GET /api/pipeline-configs/{id}: returns 404 for unknown ID
- GET /api/pipeline-configs/{config_id}/runs: returns runs for a config
- GET /api/pipeline-configs/{config_id}/runs: returns empty when no runs
- POST /api/pipeline-configs/{config_id}/runs: 404 when config not found
- POST /api/pipeline-configs/{config_id}/runs: 409 when active run exists
- POST /api/pipeline-runs/{run_id}/stop: 404 when run not found
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.backend.api import app, _scheduler_state
from src.db import PipelineRunStatus
from src.pipeline.tasks import (
    TaskType,
    TaskFactory,
    clear_task_registry,
)
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline
import src.pipeline.tasks as tasks_module


@pytest.fixture(autouse=True)
def reset_state():
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
    return TestClient(app, raise_server_exceptions=False)


@contextmanager
def _fake_session_scope():
    yield object()


def _make_config(config_id, name, path="/tmp/test.yaml"):
    return SimpleNamespace(
        id=config_id,
        name=name,
        description=f"Test config {name}",
        config_path=path,
        attack_config_path=None,
        config_hash="abcdef1234567890" * 4,
        created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
        updated_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:01"),
    )


def _make_run(run_id, config_id, run_number, status=PipelineRunStatus.COMPLETED):
    return SimpleNamespace(
        id=run_id,
        pipeline_config_id=config_id,
        run_number=run_number,
        use_cache=True,
        status=status,
        error_message=None,
        created_at=SimpleNamespace(isoformat=lambda: "2026-04-01T10:00:00"),
        started_at=SimpleNamespace(isoformat=lambda: "2026-04-01T10:00:01"),
        completed_at=SimpleNamespace(isoformat=lambda: "2026-04-01T11:00:00")
        if status == PipelineRunStatus.COMPLETED
        else None,
    )


# ============================================================================
# GET /api/pipeline-configs
# ============================================================================


class TestGetPipelineConfigs:
    """Tests for GET /api/pipeline-configs."""

    def test_returns_discovered_configs(self, client):
        configs = [
            _make_config("config_trades", "trades", "/tmp/trades.yaml"),
            _make_config("config_mini", "mini", "/tmp/mini.yaml"),
        ]
        with patch("src.backend.config_discovery.sync_configs_to_db"), \
             patch("src.backend.config_discovery.get_all_configs", return_value=configs):
            resp = client.get("/api/pipeline-configs")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        ids = [c["id"] for c in body["configs"]]
        assert "config_trades" in ids
        assert "config_mini" in ids

    def test_returns_empty_when_no_configs(self, client):
        with patch("src.backend.config_discovery.sync_configs_to_db"), \
             patch("src.backend.config_discovery.get_all_configs", return_value=[]):
            resp = client.get("/api/pipeline-configs")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["configs"] == []

    def test_calls_sync_before_listing(self, client):
        sync_mock = MagicMock()
        with patch("src.backend.config_discovery.sync_configs_to_db", sync_mock), \
             patch("src.backend.config_discovery.get_all_configs", return_value=[]):
            client.get("/api/pipeline-configs")

        sync_mock.assert_called_once()

    def test_response_shape(self, client):
        configs = [_make_config("config_test", "test")]
        with patch("src.backend.config_discovery.sync_configs_to_db"), \
             patch("src.backend.config_discovery.get_all_configs", return_value=configs):
            resp = client.get("/api/pipeline-configs")

        cfg = resp.json()["configs"][0]
        assert "id" in cfg
        assert "name" in cfg
        assert "description" in cfg
        assert "config_path" in cfg
        assert "attack_config_path" in cfg
        assert "config_hash" in cfg
        assert "created_at" in cfg
        assert "updated_at" in cfg

    def test_returns_500_on_sync_error(self, client):
        with patch("src.backend.config_discovery.sync_configs_to_db",
                   side_effect=RuntimeError("DB connection failed")), \
             patch("src.backend.config_discovery.get_all_configs",
                   side_effect=RuntimeError("DB connection failed")):
            resp = client.get("/api/pipeline-configs")

        assert resp.status_code == 500

    def test_works_without_scheduler_initialized(self, client):
        assert _scheduler_state.is_initialized() is False
        with patch("src.backend.config_discovery.sync_configs_to_db"), \
             patch("src.backend.config_discovery.get_all_configs", return_value=[]):
            resp = client.get("/api/pipeline-configs")
        assert resp.status_code == 200


# ============================================================================
# GET /api/pipeline-configs/{config_id}
# ============================================================================


class TestGetPipelineConfigById:
    """Tests for GET /api/pipeline-configs/{config_id}."""

    def test_returns_config(self, client):
        cfg = _make_config("config_trades", "trades", "/tmp/trades.yaml")
        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg):
            resp = client.get("/api/pipeline-configs/config_trades")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "config_trades"
        assert body["name"] == "trades"

    def test_returns_404_for_unknown_id(self, client):
        with patch("src.backend.config_discovery.get_config_by_id", return_value=None):
            resp = client.get("/api/pipeline-configs/config_nonexistent")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_response_has_all_fields(self, client):
        cfg = _make_config("config_mini", "mini")
        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg):
            resp = client.get("/api/pipeline-configs/config_mini")

        body = resp.json()
        for field in ["id", "name", "description", "config_path",
                       "attack_config_path", "config_hash", "created_at", "updated_at"]:
            assert field in body, f"Missing field: {field}"


# ============================================================================
# GET /api/pipeline-configs/{config_id}/runs
# ============================================================================


class TestGetConfigRuns:
    """Tests for GET /api/pipeline-configs/{config_id}/runs."""

    def test_returns_runs_for_config(self, client):
        runs = [
            _make_run("run_1", "config_mini", 1, PipelineRunStatus.COMPLETED),
            _make_run("run_2", "config_mini", 2, PipelineRunStatus.RUNNING),
        ]

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_config_id(self, config_id):
                return [r for r in runs if r.pipeline_config_id == config_id]

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.get("/api/pipeline-configs/config_mini/runs")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2

    def test_returns_empty_for_config_with_no_runs(self, client):
        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_config_id(self, config_id):
                return []

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.get("/api/pipeline-configs/config_trades/runs")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["runs"] == []

    def test_run_response_fields(self, client):
        runs = [_make_run("run_abc", "config_mini", 1)]

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_config_id(self, config_id):
                return runs

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.get("/api/pipeline-configs/config_mini/runs")

        run = resp.json()["runs"][0]
        assert run["id"] == "run_abc"
        assert run["pipeline_config_id"] == "config_mini"
        assert run["run_number"] == 1
        assert run["status"] == "completed"
        assert "created_at" in run
        assert "started_at" in run


# ============================================================================
# POST /api/pipeline-configs/{config_id}/runs — error paths
# ============================================================================


class TestStartRunErrors:
    """Error paths for POST /api/pipeline-configs/{config_id}/runs."""

    def test_404_config_not_found(self, client):
        with patch("src.backend.config_discovery.get_config_by_id", return_value=None):
            resp = client.post(
                "/api/pipeline-configs/config_nonexistent/runs",
                json={"use_cache": True},
            )
        assert resp.status_code == 404

    def test_409_active_run_exists(self, client):
        cfg = _make_config("config_mini", "mini")

        class _Repo:
            def __init__(self, _s):
                pass
            def get_active_runs_for_config(self, config_id):
                return [SimpleNamespace(id="run_existing")]

        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post(
                "/api/pipeline-configs/config_mini/runs",
                json={"use_cache": True},
            )
        assert resp.status_code == 409
        assert "active run" in resp.json()["detail"].lower()


# ============================================================================
# POST /api/pipeline-runs/{run_id}/stop — error paths
# ============================================================================


class TestStopRunErrors:
    """Error paths for POST /api/pipeline-runs/{run_id}/stop."""

    def _init_scheduler(self):
        tool = ToolDefinition(
            name="test_tool",
            container=ContainerConfig(image="test/image:v1", command="run"),
        )
        task = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING, tool=tool, config={}
        )
        wf = WorkflowFactory.create_workflow(name="wf_1", tasks=[task])
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[wf])
        _scheduler_state.initialize(pipeline)
        return pipeline

    def test_stop_404_run_not_found(self, client):
        self._init_scheduler()

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, run_id):
                return None

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post("/api/pipeline-runs/run_nonexistent/stop")

        assert resp.status_code == 404

    def test_stop_400_already_completed(self, client):
        self._init_scheduler()

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, run_id):
                return SimpleNamespace(
                    id=run_id,
                    pipeline_config_id="config_mini",
                    status=PipelineRunStatus.COMPLETED,
                )

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post("/api/pipeline-runs/run_1/stop")

        assert resp.status_code == 400
        assert "cannot stop" in resp.json()["detail"].lower()

    def test_stop_400_already_cancelled(self, client):
        self._init_scheduler()

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, run_id):
                return SimpleNamespace(
                    id=run_id,
                    pipeline_config_id="config_mini",
                    status=PipelineRunStatus.CANCELLED,
                )

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post("/api/pipeline-runs/run_1/stop")

        assert resp.status_code == 400
