"""
Tests for multi-run backend API endpoints.

Focus:
- Backend behavior without requiring scheduler pre-initialization
- Multiple-run constraints per config
- Run lifecycle edge cases (start/stop/restart)
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.backend.api import app
from src.db import PipelineRunStatus
from src.pipeline.pipeline import DefenseEvaluationPipeline
from src.pipeline.tasks import TaskType, TaskStatus
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import WorkflowFactory
from src.pipeline.tasks import TaskFactory


@pytest.fixture
def client():
    return TestClient(app)


@contextmanager
def _fake_session_scope():
    """Simple context manager to replace DB session_scope in tests."""
    yield object()


class _FakeRunRepoConflict:
    def __init__(self, _session):
        pass

    def get_active_runs_for_config(self, _config_id):
        return [SimpleNamespace(id="run_existing")]


class _FakeRunRepoEmpty:
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

    def get_by_id(self, _run_id):
        return self._run


class _FakeRunRepoStop:
    def __init__(self, _session):
        self._run = SimpleNamespace(
            id="run_1",
            pipeline_config_id="config_trades",
            run_number=1,
            use_cache=True,
            status=PipelineRunStatus.RUNNING,
            error_message=None,
            created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
            started_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:01:00"),
            completed_at=None,
        )

    def get_by_id(self, _run_id):
        return self._run

    def update_status(self, _run_id, status):
        self._run.status = status
        if self._run.status == PipelineRunStatus.CANCELLED:
            self._run.completed_at = SimpleNamespace(isoformat=lambda: "2026-01-01T00:02:00")
        return self._run


def _build_minimal_pipeline():
    tool = ToolDefinition(
        name="test_tool",
        container=ContainerConfig(image="test/image:latest", command="python main.py"),
    )
    task = TaskFactory.create_task(task_type=TaskType.PRE_TRAINING, tool=tool, config={})
    workflow = WorkflowFactory.create_workflow(name="comb_001", tasks=[task])
    return DefenseEvaluationPipeline(name="test_pipeline", workflows=[workflow])


@pytest.fixture
def initialized_scheduler_for_run():
    """Initialize scheduler so endpoints requiring get_scheduler can execute."""
    pipeline = _build_minimal_pipeline()
    pipeline.id = "run_1"
    for workflow in pipeline.workflows:
        workflow.pipeline_id = "run_1"
        workflow.run_id = "run_1"
        for task in workflow.tasks:
            task.pipeline_id = "run_1"
            task.run_id = "run_1"
            task.status = TaskStatus.PENDING

    from src.backend.api import _scheduler_state

    _scheduler_state.initialize(pipeline)
    return pipeline


class TestPipelineConfigEndpoints:
    def test_get_pipeline_configs_works_without_scheduler_initialized(self, client):
        config = SimpleNamespace(
            id="config_trades",
            name="Trades",
            description="trades config",
            config_path="/tmp/trades.yaml",
            attack_config_path=None,
            config_hash="abc123",
            created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
            updated_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:01"),
        )
        with patch("src.backend.config_discovery.sync_configs_to_db"), patch(
            "src.backend.config_discovery.get_all_configs", return_value=[config]
        ):
            response = client.get("/api/pipeline-configs")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["configs"][0]["id"] == "config_trades"


class TestPipelineRunStart:
    def test_start_run_rejects_if_same_config_has_active_run(self, client):
        cfg = SimpleNamespace(
            id="config_trades",
            name="Trades",
            config_path="/tmp/trades.yaml",
            attack_config_path=None,
        )
        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), patch(
            "src.db.session_scope", _fake_session_scope
        ), patch("src.db.PipelineRunRepository", _FakeRunRepoConflict):
            response = client.post(
                "/api/pipeline-configs/config_trades/runs",
                json={"use_cache": True, "dry_run": False},
            )
        assert response.status_code == 409
        assert "active run" in response.json()["detail"].lower()

    def test_start_run_initializes_scheduler_for_selected_config(self, client):
        cfg = SimpleNamespace(
            id="config_trades",
            name="Trades",
            config_path="/tmp/trades.yaml",
            attack_config_path=None,
        )
        fake_repo = _FakeRunRepoEmpty(None)
        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), patch(
            "src.db.session_scope", _fake_session_scope
        ), patch("src.db.PipelineRunRepository", lambda _s: fake_repo), patch(
            "src.pipeline.config_loader.create_pipeline_from_config",
            return_value=_build_minimal_pipeline(),
        ):
            response = client.post(
                "/api/pipeline-configs/config_trades/runs",
                json={"use_cache": True, "dry_run": False},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["pipeline_config_id"] == "config_trades"
        assert body["status"] == "running"
        assert body["run_number"] == 1


class TestPipelineRunStopAndRestart:
    def test_stop_run_rejects_non_stoppable_status(self, client, initialized_scheduler_for_run):
        class _Repo:
            def __init__(self, _session):
                pass

            def get_by_id(self, _rid):
                return SimpleNamespace(
                    id="run_1",
                    pipeline_config_id="config_trades",
                    run_number=1,
                    use_cache=True,
                    status=PipelineRunStatus.COMPLETED,
                    error_message=None,
                    created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
                    started_at=None,
                    completed_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:02:00"),
                )

        with patch("src.db.session_scope", _fake_session_scope), patch("src.db.PipelineRunRepository", _Repo):
            response = client.post("/api/pipeline-runs/run_1/stop")
        assert response.status_code == 400
        assert "cannot stop run" in response.json()["detail"].lower()

    def test_stop_run_running_transitions_to_cancelled(self, client, initialized_scheduler_for_run):
        with patch("src.db.session_scope", _fake_session_scope), patch("src.db.PipelineRunRepository", _FakeRunRepoStop):
            response = client.post("/api/pipeline-runs/run_1/stop")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == "run_1"
        assert body["status"] == "cancelled"

    def test_restart_rejects_if_old_run_still_active(self, client, initialized_scheduler_for_run):
        class _Repo:
            def __init__(self, _session):
                pass

            def get_by_id(self, _rid):
                return SimpleNamespace(
                    id="run_1",
                    pipeline_config_id="config_trades",
                    status=PipelineRunStatus.RUNNING,
                )

        with patch("src.db.session_scope", _fake_session_scope), patch("src.db.PipelineRunRepository", _Repo):
            response = client.post("/api/pipeline-runs/run_1/restart", json={"use_cache": True})
        assert response.status_code == 400
        assert "active run" in response.json()["detail"].lower()
