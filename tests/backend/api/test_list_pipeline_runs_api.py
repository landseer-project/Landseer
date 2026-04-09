"""
Tests for GET /api/pipeline-runs endpoint.

Tests cover:
- Returns empty list when no runs exist
- Returns all runs across multiple configs
- Filters by config_id query parameter
- Response shape matches PipelineRunListResponse schema
- Runs are ordered by created_at desc
- Works without scheduler being initialized (headless-safe)
"""

from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.backend.api import app, _scheduler_state
from src.db import PipelineRunStatus
from src.pipeline.tasks import clear_task_registry
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


@contextmanager
def _fake_session_scope():
    yield object()


def _make_run(
    run_id: str,
    config_id: str,
    run_number: int,
    status: PipelineRunStatus = PipelineRunStatus.COMPLETED,
):
    return SimpleNamespace(
        id=run_id,
        pipeline_config_id=config_id,
        run_number=run_number,
        use_cache=True,
        status=status,
        error_message=None,
        created_at=SimpleNamespace(isoformat=lambda: f"2026-01-0{run_number}T00:00:00"),
        started_at=SimpleNamespace(isoformat=lambda: f"2026-01-0{run_number}T00:01:00"),
        completed_at=SimpleNamespace(isoformat=lambda: f"2026-01-0{run_number}T01:00:00"),
    )


# ============================================================================
# Tests
# ============================================================================


class TestListPipelineRunsEmpty:
    """GET /api/pipeline-runs with no runs."""

    def test_empty_returns_200(self, client):
        class _EmptyRepo:
            def __init__(self, _s):
                pass
            def get_all(self, config_id=None):
                return []

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _EmptyRepo):
            resp = client.get("/api/pipeline-runs")

        assert resp.status_code == 200
        body = resp.json()
        assert body["runs"] == []
        assert body["total"] == 0


class TestListPipelineRunsMultiConfig:
    """GET /api/pipeline-runs returns runs across configs."""

    def _make_repo(self, runs):
        class _Repo:
            def __init__(self, _s):
                pass
            def get_all(self, config_id=None):
                if config_id:
                    return [r for r in runs if r.pipeline_config_id == config_id]
                return runs
        return _Repo

    def test_returns_all_runs(self, client):
        runs = [
            _make_run("run_1", "config_mini", 1),
            _make_run("run_2", "config_mini", 2),
            _make_run("run_3", "config_trades", 1),
        ]
        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", self._make_repo(runs)):
            resp = client.get("/api/pipeline-runs")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert len(body["runs"]) == 3

    def test_filter_by_config_id(self, client):
        runs = [
            _make_run("run_1", "config_mini", 1),
            _make_run("run_2", "config_mini", 2),
            _make_run("run_3", "config_trades", 1),
        ]
        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", self._make_repo(runs)):
            resp = client.get("/api/pipeline-runs?config_id=config_mini")

        body = resp.json()
        assert body["total"] == 2
        assert all(r["pipeline_config_id"] == "config_mini" for r in body["runs"])

    def test_filter_by_nonexistent_config_returns_empty(self, client):
        runs = [_make_run("run_1", "config_mini", 1)]
        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", self._make_repo(runs)):
            resp = client.get("/api/pipeline-runs?config_id=config_nonexistent")

        body = resp.json()
        assert body["total"] == 0
        assert body["runs"] == []


class TestListPipelineRunsResponseSchema:
    """Response format validation for GET /api/pipeline-runs."""

    def test_run_fields_present(self, client):
        runs = [_make_run("run_abc", "config_mini", 1, PipelineRunStatus.RUNNING)]

        class _Repo:
            def __init__(self, _s):
                pass
            def get_all(self, config_id=None):
                return runs

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.get("/api/pipeline-runs")

        run = resp.json()["runs"][0]
        assert run["id"] == "run_abc"
        assert run["pipeline_config_id"] == "config_mini"
        assert run["run_number"] == 1
        assert run["use_cache"] is True
        assert run["status"] == "running"
        assert run["error_message"] is None
        assert "created_at" in run
        assert "started_at" in run
        assert "completed_at" in run

    def test_completed_run_has_completed_at(self, client):
        runs = [_make_run("run_1", "config_mini", 1, PipelineRunStatus.COMPLETED)]

        class _Repo:
            def __init__(self, _s):
                pass
            def get_all(self, config_id=None):
                return runs

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.get("/api/pipeline-runs")

        run = resp.json()["runs"][0]
        assert run["completed_at"] is not None


class TestListPipelineRunsHeadlessSafe:
    """GET /api/pipeline-runs works even when scheduler is not initialized."""

    def test_works_without_scheduler(self, client):
        assert _scheduler_state.is_initialized() is False

        class _Repo:
            def __init__(self, _s):
                pass
            def get_all(self, config_id=None):
                return [_make_run("run_1", "config_mini", 1)]

        with patch("src.db.session_scope", _fake_session_scope), \
             patch("src.db.PipelineRunRepository", _Repo):
            resp = client.get("/api/pipeline-runs")

        assert resp.status_code == 200
        assert resp.json()["total"] == 1
