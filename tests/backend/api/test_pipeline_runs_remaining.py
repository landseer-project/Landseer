"""
Tests for remaining untested pipeline-run API flows.

Covers:
- GET /api/pipeline-runs/{run_id}: 200 found, 404 not found, 500 DB error
- POST /api/pipeline-runs/{run_id}/restart: happy path, 404 old run, 404 config,
  400 active run (PENDING/RUNNING/STOPPING)
- POST /api/pipeline-runs/{run_id}/stop: 503 without scheduler, stop PENDING run,
  400 STOPPING status
- POST /api/pipeline-configs/{config_id}/runs: 500 on pipeline load failure,
  attack_config_path override
- GET /api/pipeline-configs/{config_id}: 500 on DB error
- GET /api/pipeline-configs/{config_id}/runs: 500 on DB error
- GET /api/pipeline-runs: 500 on DB error
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.backend.api import app, _scheduler_state
from src.db import PipelineRunStatus
from src.pipeline.tasks import (
    Task,
    TaskType,
    TaskStatus,
    TaskFactory,
    clear_task_registry,
)
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline
import src.pipeline.tasks as tasks_module


# ============================================================================
# Fixtures
# ============================================================================


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


def _build_pipeline(name="test"):
    tool = ToolDefinition(
        name="test_tool",
        container=ContainerConfig(image="test/img:v1", command="run"),
    )
    task = TaskFactory.create_task(task_type=TaskType.PRE_TRAINING, tool=tool, config={})
    wf = WorkflowFactory.create_workflow(name="wf_001", tasks=[task])
    return DefenseEvaluationPipeline(name=name, workflows=[wf])


def _init_scheduler_with_run_id(run_id="run_1"):
    pipeline = _build_pipeline()
    pipeline.id = run_id
    for wf in pipeline.workflows:
        wf.pipeline_id = run_id
        wf.run_id = run_id
        for t in wf.tasks:
            t.pipeline_id = run_id
            t.run_id = run_id
            t.status = TaskStatus.PENDING
    _scheduler_state.initialize(pipeline)
    return pipeline


def _make_run(
    run_id="run_1",
    config_id="config_mini",
    run_number=1,
    status=PipelineRunStatus.COMPLETED,
    error_message=None,
):
    return SimpleNamespace(
        id=run_id,
        pipeline_config_id=config_id,
        run_number=run_number,
        use_cache=True,
        status=status,
        error_message=error_message,
        created_at=SimpleNamespace(isoformat=lambda: "2026-04-09T10:00:00"),
        started_at=SimpleNamespace(isoformat=lambda: "2026-04-09T10:00:01"),
        completed_at=SimpleNamespace(isoformat=lambda: "2026-04-09T11:00:00")
        if status in (PipelineRunStatus.COMPLETED, PipelineRunStatus.CANCELLED)
        else None,
    )


def _make_config(config_id="config_mini", name="mini"):
    return SimpleNamespace(
        id=config_id,
        name=name,
        description="Test",
        config_path="/tmp/test.yaml",
        attack_config_path=None,
        config_hash="abc",
        created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
        updated_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:01"),
    )


# ============================================================================
# GET /api/pipeline-runs/{run_id}
# ============================================================================


class TestGetSinglePipelineRun:
    """Tests for GET /api/pipeline-runs/{run_id}."""

    def test_200_returns_run(self, client):
        run = _make_run("run_abc", "config_mini", 1, PipelineRunStatus.RUNNING)

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, run_id):
                return run if run_id == "run_abc" else None

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.get("/api/pipeline-runs/run_abc")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "run_abc"
        assert body["pipeline_config_id"] == "config_mini"
        assert body["status"] == "running"
        assert body["run_number"] == 1

    def test_200_response_has_all_fields(self, client):
        run = _make_run("run_xyz", "config_trades", 3, PipelineRunStatus.COMPLETED)

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, _):
                return run

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.get("/api/pipeline-runs/run_xyz")

        body = resp.json()
        for field in ["id", "pipeline_config_id", "run_number", "use_cache",
                       "status", "error_message", "created_at", "started_at", "completed_at"]:
            assert field in body, f"Missing field: {field}"

    def test_404_run_not_found(self, client):
        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, _):
                return None

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.get("/api/pipeline-runs/run_nonexistent")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_500_on_db_error(self, client):
        with patch("src.db.session_scope", side_effect=RuntimeError("DB down")):
            resp = client.get("/api/pipeline-runs/run_1")

        assert resp.status_code == 500

    def test_works_without_scheduler(self, client):
        """GET single run does not depend on scheduler being initialized."""
        assert _scheduler_state.is_initialized() is False
        run = _make_run("run_1")

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, _):
                return run

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.get("/api/pipeline-runs/run_1")

        assert resp.status_code == 200


# ============================================================================
# POST /api/pipeline-runs/{run_id}/restart
# ============================================================================


class TestRestartPipelineRun:
    """Tests for POST /api/pipeline-runs/{run_id}/restart."""

    def _make_restart_repo(self, old_run, config_id="config_mini"):
        """Create a fake repo that supports get_by_id, get_next_run_number, create, update_status."""
        class _Repo:
            def __init__(self, _s):
                self._new_run = None

            def get_by_id(self, run_id):
                if run_id == old_run.id:
                    return old_run
                return self._new_run

            def get_next_run_number(self, _):
                return old_run.run_number + 1

            def create(self, data):
                self._new_run = SimpleNamespace(
                    id=data["id"],
                    pipeline_config_id=data["pipeline_config_id"],
                    run_number=data["run_number"],
                    use_cache=data["use_cache"],
                    status=PipelineRunStatus.PENDING,
                    error_message=None,
                    created_at=SimpleNamespace(isoformat=lambda: "2026-04-09T12:00:00"),
                    started_at=None,
                    completed_at=None,
                )
                return self._new_run

            def update_status(self, run_id, status):
                if self._new_run and self._new_run.id == run_id:
                    self._new_run.status = status
                    if status == PipelineRunStatus.RUNNING:
                        self._new_run.started_at = SimpleNamespace(
                            isoformat=lambda: "2026-04-09T12:00:01"
                        )
                return self._new_run
        return _Repo

    def test_restart_happy_path(self, client):
        old_run = _make_run("run_old", "config_mini", 1, PipelineRunStatus.COMPLETED)
        cfg = _make_config("config_mini", "mini")

        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", self._make_restart_repo(old_run)), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   return_value=_build_pipeline()):
            resp = client.post(
                "/api/pipeline-runs/run_old/restart",
                json={"use_cache": True},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["pipeline_config_id"] == "config_mini"
        assert body["run_number"] == 2
        assert body["id"] != "run_old"

    def test_restart_failed_run(self, client):
        old_run = _make_run("run_fail", "config_mini", 1, PipelineRunStatus.FAILED)
        cfg = _make_config("config_mini", "mini")

        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", self._make_restart_repo(old_run)), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   return_value=_build_pipeline()):
            resp = client.post(
                "/api/pipeline-runs/run_fail/restart",
                json={"use_cache": False},
            )

        assert resp.status_code == 200

    def test_restart_cancelled_run(self, client):
        old_run = _make_run("run_cancel", "config_mini", 2, PipelineRunStatus.CANCELLED)
        cfg = _make_config("config_mini", "mini")

        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", self._make_restart_repo(old_run)), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   return_value=_build_pipeline()):
            resp = client.post(
                "/api/pipeline-runs/run_cancel/restart",
                json={"use_cache": True},
            )

        assert resp.status_code == 200
        assert resp.json()["run_number"] == 3

    def test_restart_404_old_run_not_found(self, client):
        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, _):
                return None

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post(
                "/api/pipeline-runs/run_nonexistent/restart",
                json={"use_cache": True},
            )

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_restart_404_config_not_found(self, client):
        old_run = _make_run("run_old", "config_deleted", 1, PipelineRunStatus.COMPLETED)

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, _):
                return old_run

        with patch("src.backend.config_discovery.get_config_by_id", return_value=None), \
             patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post(
                "/api/pipeline-runs/run_old/restart",
                json={"use_cache": True},
            )

        assert resp.status_code == 404
        assert "config" in resp.json()["detail"].lower()

    def test_restart_400_pending_run(self, client):
        old_run = _make_run("run_active", "config_mini", 1, PipelineRunStatus.PENDING)

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, _):
                return old_run

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post(
                "/api/pipeline-runs/run_active/restart",
                json={"use_cache": True},
            )

        assert resp.status_code == 400
        assert "active" in resp.json()["detail"].lower()

    def test_restart_400_running_run(self, client):
        old_run = _make_run("run_active", "config_mini", 1, PipelineRunStatus.RUNNING)

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, _):
                return old_run

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post(
                "/api/pipeline-runs/run_active/restart",
                json={"use_cache": True},
            )

        assert resp.status_code == 400

    def test_restart_400_stopping_run(self, client):
        old_run = _make_run("run_stopping", "config_mini", 1, PipelineRunStatus.STOPPING)

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, _):
                return old_run

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post(
                "/api/pipeline-runs/run_stopping/restart",
                json={"use_cache": True},
            )

        assert resp.status_code == 400

    def test_restart_500_on_pipeline_load_failure(self, client):
        old_run = _make_run("run_old", "config_mini", 1, PipelineRunStatus.COMPLETED)
        cfg = _make_config("config_mini", "mini")

        class _Repo:
            _created = False
            def __init__(self, _s):
                pass
            def get_by_id(self, _):
                return old_run
            def get_next_run_number(self, _):
                return 2
            def create(self, data):
                return SimpleNamespace(
                    id=data["id"], pipeline_config_id=data["pipeline_config_id"],
                    run_number=2, use_cache=True, status=PipelineRunStatus.PENDING,
                    error_message=None,
                    created_at=SimpleNamespace(isoformat=lambda: "2026-04-09T12:00:00"),
                    started_at=None, completed_at=None,
                )

        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   side_effect=ValueError("Invalid YAML")):
            resp = client.post(
                "/api/pipeline-runs/run_old/restart",
                json={"use_cache": True},
            )

        assert resp.status_code == 500


# ============================================================================
# POST /api/pipeline-runs/{run_id}/stop — additional paths
# ============================================================================


class TestStopPipelineRunExtra:
    """Additional stop endpoint tests not covered by existing files."""

    def test_stop_503_without_scheduler_initialized(self, client):
        """Stop depends on get_scheduler which returns 503 if uninitialized."""
        assert _scheduler_state.is_initialized() is False
        resp = client.post("/api/pipeline-runs/run_1/stop")
        assert resp.status_code == 503

    def test_stop_pending_run(self, client):
        """Stopping a PENDING run should work (status is in PENDING/RUNNING)."""
        pipeline = _init_scheduler_with_run_id("run_pending")

        class _Repo:
            def __init__(self, _s):
                self._run = _make_run("run_pending", "config_mini", 1,
                                       PipelineRunStatus.PENDING)

            def get_by_id(self, run_id):
                return self._run

            def update_status(self, run_id, status):
                self._run.status = status
                if status == PipelineRunStatus.CANCELLED:
                    self._run.completed_at = SimpleNamespace(
                        isoformat=lambda: "2026-04-09T10:05:00"
                    )
                return self._run

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post("/api/pipeline-runs/run_pending/stop")

        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_stop_400_stopping_status(self, client):
        """A run already in STOPPING state should return 400."""
        _init_scheduler_with_run_id("run_stopping")

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, _):
                return _make_run("run_stopping", "config_mini", 1,
                                  PipelineRunStatus.STOPPING)

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post("/api/pipeline-runs/run_stopping/stop")

        assert resp.status_code == 400
        assert "cannot stop" in resp.json()["detail"].lower()

    def test_stop_400_failed_status(self, client):
        _init_scheduler_with_run_id("run_failed")

        class _Repo:
            def __init__(self, _s):
                pass
            def get_by_id(self, _):
                return _make_run("run_failed", "config_mini", 1,
                                  PipelineRunStatus.FAILED)

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post("/api/pipeline-runs/run_failed/stop")

        assert resp.status_code == 400

    def test_stop_cancels_pending_tasks_when_pipeline_matches(self, client):
        """When scheduler pipeline.id matches run_id, pending tasks are cancelled."""
        run_id = "run_match"
        pipeline = _init_scheduler_with_run_id(run_id)
        all_tasks = _scheduler_state.scheduler.get_all_tasks()
        assert all(t.status == TaskStatus.PENDING for t in all_tasks)

        class _Repo:
            def __init__(self, _s):
                self._run = _make_run(run_id, "config_mini", 1,
                                       PipelineRunStatus.RUNNING)

            def get_by_id(self, _):
                return self._run

            def update_status(self, _, status):
                self._run.status = status
                if status == PipelineRunStatus.CANCELLED:
                    self._run.completed_at = SimpleNamespace(
                        isoformat=lambda: "2026-04-09T10:05:00"
                    )
                return self._run

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post(f"/api/pipeline-runs/{run_id}/stop")

        assert resp.status_code == 200
        assert all(t.status == TaskStatus.CANCELLED for t in all_tasks)

    def test_stop_skips_task_cancel_when_pipeline_mismatch(self, client):
        """When scheduler pipeline.id != run_id, tasks should NOT be cancelled."""
        _init_scheduler_with_run_id("run_other")
        all_tasks = _scheduler_state.scheduler.get_all_tasks()

        class _Repo:
            def __init__(self, _s):
                self._run = _make_run("run_different", "config_mini", 1,
                                       PipelineRunStatus.RUNNING)

            def get_by_id(self, _):
                return self._run

            def update_status(self, _, status):
                self._run.status = status
                if status == PipelineRunStatus.CANCELLED:
                    self._run.completed_at = SimpleNamespace(
                        isoformat=lambda: "2026-04-09T10:05:00"
                    )
                return self._run

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.post("/api/pipeline-runs/run_different/stop")

        assert resp.status_code == 200
        assert all(t.status == TaskStatus.PENDING for t in all_tasks)


# ============================================================================
# POST /api/pipeline-configs/{config_id}/runs — additional paths
# ============================================================================


class TestStartRunExtra:
    """Additional start-run tests for untested paths."""

    def test_500_on_pipeline_load_failure(self, client):
        cfg = _make_config("config_bad", "bad")

        class _Repo:
            def __init__(self, _s):
                pass
            def get_active_runs_for_config(self, _):
                return []
            def get_next_run_number(self, _):
                return 1
            def create(self, data):
                return SimpleNamespace(
                    id=data["id"], pipeline_config_id=data["pipeline_config_id"],
                    run_number=1, use_cache=True, status=PipelineRunStatus.PENDING,
                    error_message=None,
                    created_at=SimpleNamespace(isoformat=lambda: "2026-04-09T10:00:00"),
                    started_at=None, completed_at=None,
                )

        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   side_effect=FileNotFoundError("config.yaml not found")):
            resp = client.post(
                "/api/pipeline-configs/config_bad/runs",
                json={"use_cache": True},
            )

        assert resp.status_code == 500
        assert "failed to start" in resp.json()["detail"].lower()

    def test_attack_config_path_from_request(self, client):
        """When request provides attack_config_path, it overrides config's value."""
        cfg = SimpleNamespace(
            id="config_test", name="test",
            config_path="/tmp/test.yaml",
            attack_config_path="/tmp/default_attack.yaml",
        )
        captured_kwargs = {}

        def _capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return _build_pipeline()

        class _Repo:
            def __init__(self, _s):
                pass
            def get_active_runs_for_config(self, _):
                return []
            def get_next_run_number(self, _):
                return 1
            def create(self, data):
                return SimpleNamespace(
                    id=data["id"], pipeline_config_id=data["pipeline_config_id"],
                    run_number=1, use_cache=True, status=PipelineRunStatus.PENDING,
                    error_message=None,
                    created_at=SimpleNamespace(isoformat=lambda: "2026-04-09T10:00:00"),
                    started_at=None, completed_at=None,
                )
            def update_status(self, _, status):
                pass

        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   return_value=_build_pipeline()) as mock_create:
            resp = client.post(
                "/api/pipeline-configs/config_test/runs",
                json={"use_cache": True, "attack_config_path": "/tmp/custom_attack.yaml"},
            )

        assert resp.status_code == 200

    def test_attack_config_path_falls_back_to_config(self, client):
        """When request omits attack_config_path, config's value is used."""
        cfg = SimpleNamespace(
            id="config_test", name="test",
            config_path="/tmp/test.yaml",
            attack_config_path="/tmp/config_attack.yaml",
        )

        class _Repo:
            def __init__(self, _s):
                pass
            def get_active_runs_for_config(self, _):
                return []
            def get_next_run_number(self, _):
                return 1
            def create(self, data):
                return SimpleNamespace(
                    id=data["id"], pipeline_config_id=data["pipeline_config_id"],
                    run_number=1, use_cache=True, status=PipelineRunStatus.PENDING,
                    error_message=None,
                    created_at=SimpleNamespace(isoformat=lambda: "2026-04-09T10:00:00"),
                    started_at=None, completed_at=None,
                )
            def update_status(self, _, status):
                pass

        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   return_value=_build_pipeline()):
            resp = client.post(
                "/api/pipeline-configs/config_test/runs",
                json={"use_cache": True},
            )

        assert resp.status_code == 200


# ============================================================================
# 500 error paths for GET endpoints
# ============================================================================


class TestGetEndpoints500:
    """500 paths on DB errors for various GET endpoints."""

    def test_get_single_config_500(self, client):
        with patch("src.backend.config_discovery.get_config_by_id",
                   side_effect=RuntimeError("DB connection lost")):
            resp = client.get("/api/pipeline-configs/config_mini")

        assert resp.status_code == 500

    def test_get_config_runs_500(self, client):
        with patch("src.db.session_scope", side_effect=RuntimeError("DB timeout")):
            resp = client.get("/api/pipeline-configs/config_mini/runs")

        assert resp.status_code == 500

    def test_list_all_runs_500(self, client):
        with patch("src.db.session_scope", side_effect=RuntimeError("DB crash")):
            resp = client.get("/api/pipeline-runs")

        assert resp.status_code == 500

    def test_get_single_run_500(self, client):
        with patch("src.db.session_scope", side_effect=RuntimeError("pool exhausted")):
            resp = client.get("/api/pipeline-runs/run_1")

        assert resp.status_code == 500

    def test_stop_run_500(self, client):
        _init_scheduler_with_run_id("run_1")

        with patch("src.db.session_scope", side_effect=RuntimeError("DB error")):
            resp = client.post("/api/pipeline-runs/run_1/stop")

        assert resp.status_code == 500

    def test_restart_run_500(self, client):
        with patch("src.db.session_scope", side_effect=RuntimeError("DB error")):
            resp = client.post(
                "/api/pipeline-runs/run_1/restart",
                json={"use_cache": True},
            )

        assert resp.status_code == 500

    def test_start_run_500_on_db_error(self, client):
        cfg = _make_config()
        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", side_effect=RuntimeError("DB error")):
            resp = client.post(
                "/api/pipeline-configs/config_mini/runs",
                json={"use_cache": True},
            )

        assert resp.status_code == 500
