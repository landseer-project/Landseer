"""
Integration tests for the headless-to-active pipeline flow.

Tests the full lifecycle:
1. Backend starts in headless mode (no --config)
2. Config discovery finds pipeline YAMLs on disk
3. User triggers a run via POST /api/pipeline-configs/{id}/runs
4. Scheduler becomes active, workers can claim tasks
5. Run appears in GET /api/pipeline-runs
6. Subsequent scheduler-dependent endpoints work

These tests use more realistic mocking than pure unit tests,
simulating the actual flow a user would follow.
"""

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
from src.pipeline.workflow import Workflow, WorkflowFactory
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
    return TestClient(app, raise_server_exceptions=True)


class _FakeSessionScope:
    def __enter__(self):
        return object()
    def __exit__(self, *args):
        pass


def _build_pipeline(name="mini_test"):
    tool = ToolDefinition(
        name="pre_xgbod",
        container=ContainerConfig(image="test/xgbod:v1", command="run"),
    )
    task = TaskFactory.create_task(task_type=TaskType.PRE_TRAINING, tool=tool, config={})
    workflow = WorkflowFactory.create_workflow(name="wf_mini_001", tasks=[task])
    return DefenseEvaluationPipeline(name=name, workflows=[workflow])


class _RunStorage:
    """Shared run storage to simulate DB across multiple endpoint calls."""
    def __init__(self):
        self.runs = []
        self._counter = 0

    def make_repo(self, _session):
        storage = self
        class _Repo:
            def __init__(self):
                pass
            def get_active_runs_for_config(self, config_id):
                return [r for r in storage.runs
                        if r.pipeline_config_id == config_id
                        and r.status in (PipelineRunStatus.PENDING, PipelineRunStatus.RUNNING)]
            def get_next_run_number(self, config_id):
                config_runs = [r for r in storage.runs if r.pipeline_config_id == config_id]
                return len(config_runs) + 1
            def create(self, run_data):
                run = SimpleNamespace(
                    id=run_data["id"],
                    pipeline_config_id=run_data["pipeline_config_id"],
                    run_number=run_data["run_number"],
                    use_cache=run_data["use_cache"],
                    status=PipelineRunStatus.PENDING,
                    error_message=None,
                    created_at=SimpleNamespace(isoformat=lambda: "2026-04-09T10:00:00"),
                    started_at=None,
                    completed_at=None,
                )
                storage.runs.append(run)
                return run
            def update_status(self, run_id, status):
                for r in storage.runs:
                    if r.id == run_id:
                        r.status = status
                        if status == PipelineRunStatus.RUNNING:
                            r.started_at = SimpleNamespace(isoformat=lambda: "2026-04-09T10:00:01")
                        return r
                return None
            def get_all(self, config_id=None):
                if config_id:
                    return [r for r in storage.runs if r.pipeline_config_id == config_id]
                return storage.runs
        return _Repo()


class TestFullHeadlessLifecycle:
    """End-to-end test of the headless -> active lifecycle."""

    def test_full_lifecycle(self, client):
        storage = _RunStorage()

        # ── 1. Verify we start headless ──────────────────────────
        assert _scheduler_state.is_initialized() is False
        health = client.get("/health").json()
        assert health["scheduler_active"] is False

        # ── 2. Register a worker (works in headless mode) ────────
        resp = client.post("/workers/register", json={
            "hostname": "gpu-node",
            "capabilities": {"gpu_id": 0},
        })
        assert resp.status_code == 200
        worker_id = resp.json()["worker_id"]

        # ── 3. Discover configs ──────────────────────────────────
        fake_config = SimpleNamespace(
            id="config_mini", name="mini",
            description="Mini pipeline",
            config_path="/tmp/mini.yaml",
            attack_config_path=None,
            config_hash="abc123",
            created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
            updated_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:01"),
        )
        with patch("src.backend.config_discovery.sync_configs_to_db"), \
             patch("src.backend.config_discovery.get_all_configs", return_value=[fake_config]):
            configs = client.get("/api/pipeline-configs").json()

        assert configs["total"] == 1
        config_id = configs["configs"][0]["id"]
        assert config_id == "config_mini"

        # ── 4. Start a run ───────────────────────────────────────
        with patch("src.backend.config_discovery.get_config_by_id", return_value=fake_config), \
             patch("src.db.session_scope", _FakeSessionScope), \
             patch("src.db.PipelineRunRepository", storage.make_repo), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   return_value=_build_pipeline()):
            run_resp = client.post(
                f"/api/pipeline-configs/{config_id}/runs",
                json={"use_cache": True},
            )

        assert run_resp.status_code == 200
        run_data = run_resp.json()
        assert run_data["pipeline_config_id"] == "config_mini"
        assert run_data["status"] == "running"
        assert run_data["run_number"] == 1

        # ── 5. Scheduler is now active ───────────────────────────
        assert _scheduler_state.is_initialized() is True
        health = client.get("/health").json()
        assert health["scheduler_active"] is True

        # ── 6. Progress endpoint works ───────────────────────────
        progress = client.get("/progress").json()
        assert progress["total"] >= 1
        assert progress["pending"] >= 0

        # ── 7. Tasks endpoint works ──────────────────────────────
        tasks = client.get("/tasks").json()
        assert tasks["total"] >= 1

        # ── 8. Worker can claim a task ───────────────────────────
        next_task = client.get("/tasks/next").json()
        assert next_task["has_task"] is True
        assert next_task["task"]["status"] == "running"

        # ── 9. Run appears in pipeline-runs list ─────────────────
        with patch("src.db.session_scope", _FakeSessionScope), \
             patch("src.db.PipelineRunRepository", storage.make_repo):
            runs = client.get("/api/pipeline-runs").json()

        assert runs["total"] == 1
        assert runs["runs"][0]["id"] == run_data["id"]

    def test_second_run_rejected_while_first_active(self, client):
        """Starting a second run for the same config should be rejected (409)."""
        storage = _RunStorage()
        cfg = SimpleNamespace(
            id="config_mini", name="mini",
            config_path="/tmp/mini.yaml", attack_config_path=None,
        )

        # Start first run
        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _FakeSessionScope), \
             patch("src.db.PipelineRunRepository", storage.make_repo), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   return_value=_build_pipeline()):
            resp1 = client.post("/api/pipeline-configs/config_mini/runs", json={"use_cache": True})
        assert resp1.status_code == 200

        # Try to start second run (first is still active)
        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _FakeSessionScope), \
             patch("src.db.PipelineRunRepository", storage.make_repo):
            resp2 = client.post("/api/pipeline-configs/config_mini/runs", json={"use_cache": True})
        assert resp2.status_code == 409
        assert "active run" in resp2.json()["detail"].lower()

    def test_config_not_found_returns_404(self, client):
        """Starting a run with unknown config_id returns 404."""
        with patch("src.backend.config_discovery.get_config_by_id", return_value=None):
            resp = client.post(
                "/api/pipeline-configs/config_nonexistent/runs",
                json={"use_cache": True},
            )
        assert resp.status_code == 404

    def test_workers_registered_before_run_persist_after_run(self, client):
        """Workers registered in headless mode should remain available after a run starts."""
        # Register workers before any run
        w1 = client.post("/workers/register", json={"hostname": "h1"}).json()["worker_id"]
        w2 = client.post("/workers/register", json={"hostname": "h2"}).json()["worker_id"]

        workers_before = client.get("/workers").json()
        assert workers_before["total"] == 2

        # Start a run
        storage = _RunStorage()
        cfg = SimpleNamespace(
            id="config_mini", name="mini",
            config_path="/tmp/mini.yaml", attack_config_path=None,
        )
        with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
             patch("src.db.session_scope", _FakeSessionScope), \
             patch("src.db.PipelineRunRepository", storage.make_repo), \
             patch("src.pipeline.config_loader.create_pipeline_from_config",
                   return_value=_build_pipeline()):
            client.post("/api/pipeline-configs/config_mini/runs", json={"use_cache": True})

        # Workers should still be there
        workers_after = client.get("/workers").json()
        assert workers_after["total"] == 2
        worker_ids = {w["worker_id"] for w in workers_after["workers"]}
        assert w1 in worker_ids
        assert w2 in worker_ids
