"""
DB-backed integration tests for multi-run backend endpoints.

These tests use a real temporary SQLite database and the FastAPI test client.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.backend.api import app
from src.backend.db_service import init_db_service
from src.db import (
    DatabaseConfig,
    init_database,
    get_database,
    session_scope,
    PipelineConfigRepository,
    PipelineRunRepository,
    PipelineRepository,
    TaskRepository,
    ArtifactRepository,
    PipelineRunStatus,
)
from src.db.models import PipelineRunModel, TaskModel, ArtifactModel, TaskStatus as DBTaskStatus
from src.pipeline.pipeline import DefenseEvaluationPipeline
from src.pipeline.tasks import TaskFactory, TaskType, clear_task_registry
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import WorkflowFactory


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def temp_db_and_state(tmp_path):
    # Isolated DB per test
    db_path = tmp_path / "test_pipeline_runs.db"
    db_cfg = DatabaseConfig(db_type="sqlite", sqlite_path=str(db_path))
    init_database(db_cfg, create_tables=True)
    # Initialize DB service so API start_run path syncs tasks/workflows too
    init_db_service(config=db_cfg, enabled=True)

    # Reset scheduler state as these tests focus on backend/run APIs
    from src.backend.api import _scheduler_state

    _scheduler_state.scheduler = None
    _scheduler_state.pipeline = None
    _scheduler_state.started_at = None
    _scheduler_state.task_metadata.clear()
    _scheduler_state.workers.clear()
    _scheduler_state._worker_counter = 0
    _scheduler_state._custom_tools.clear()
    clear_task_registry()
    yield
    get_database().close()


def _build_pipeline():
    tool = ToolDefinition(
        name="db_test_tool",
        container=ContainerConfig(image="test/image:latest", command="python main.py"),
    )
    task = TaskFactory.create_task(task_type=TaskType.PRE_TRAINING, tool=tool, config={"k": "v"})
    workflow = WorkflowFactory.create_workflow(name="comb_001", tasks=[task])
    return DefenseEvaluationPipeline(name="db_test_pipeline", workflows=[workflow])


def test_start_run_persists_run_and_task_run_ids(client):
    # Insert config row first (pipeline_runs has FK to pipeline_configs)
    with session_scope() as session:
        PipelineConfigRepository(session).create(
            {
                "id": "config_trades",
                "name": "Trades",
                "description": "cfg",
                "config_path": "/tmp/trades.yaml",
                "attack_config_path": None,
                "config_hash": "h1",
            }
        )

    cfg = SimpleNamespace(
        id="config_trades",
        name="Trades",
        config_path="/tmp/trades.yaml",
        attack_config_path=None,
    )

    from unittest.mock import patch

    with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), patch(
        "src.pipeline.config_loader.create_pipeline_from_config", return_value=_build_pipeline()
    ):
        response = client.post(
            "/api/pipeline-configs/config_trades/runs",
            json={"use_cache": True, "dry_run": False},
        )

    assert response.status_code == 200
    body = response.json()
    run_id = body["id"]

    with session_scope() as session:
        run = session.query(PipelineRunModel).filter(PipelineRunModel.id == run_id).first()
        assert run is not None
        assert run.pipeline_config_id == "config_trades"
        assert run.status.value == "running"

        tasks = session.query(TaskModel).filter(TaskModel.pipeline_id == run_id).all()
        assert len(tasks) >= 1
        assert all(t.run_id == run_id for t in tasks)


def test_get_runs_returns_multiple_runs_for_same_config_desc_order(client):
    with session_scope() as session:
        cfg_repo = PipelineConfigRepository(session)
        run_repo = PipelineRunRepository(session)

        cfg_repo.create(
            {
                "id": "config_fairness",
                "name": "Fairness",
                "description": "cfg",
                "config_path": "/tmp/fairness.yaml",
                "attack_config_path": None,
                "config_hash": "h1",
            }
        )
        run_repo.create(
            {
                "id": "run_a",
                "pipeline_config_id": "config_fairness",
                "run_number": 1,
                "use_cache": True,
                "status": PipelineRunStatus.COMPLETED,
            }
        )
        run_repo.create(
            {
                "id": "run_b",
                "pipeline_config_id": "config_fairness",
                "run_number": 2,
                "use_cache": False,
                "status": PipelineRunStatus.FAILED,
            }
        )

    response = client.get("/api/pipeline-configs/config_fairness/runs")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["runs"][0]["run_number"] == 2
    assert body["runs"][1]["run_number"] == 1


def test_delete_cache_only_deletes_artifacts_for_target_run(client):
    with session_scope() as session:
        cfg_repo = PipelineConfigRepository(session)
        run_repo = PipelineRunRepository(session)
        pipeline_repo = PipelineRepository(session)
        task_repo = TaskRepository(session)
        artifact_repo = ArtifactRepository(session)

        cfg_repo.create(
            {
                "id": "config_trades",
                "name": "Trades",
                "description": "cfg",
                "config_path": "/tmp/trades.yaml",
                "attack_config_path": None,
                "config_hash": "h1",
            }
        )
        run_repo.create(
            {
                "id": "run_1",
                "pipeline_config_id": "config_trades",
                "run_number": 1,
                "use_cache": True,
                "status": PipelineRunStatus.CANCELLED,
            }
        )
        run_repo.create(
            {
                "id": "run_2",
                "pipeline_config_id": "config_trades",
                "run_number": 2,
                "use_cache": True,
                "status": PipelineRunStatus.RUNNING,
            }
        )
        pipeline_repo.create(
            {
                "id": "run_1",
                "name": "p1",
                "run_id": "run_1",
                "config": {},
                "dataset_config": None,
                "model_config": None,
                "status": "cancelled",
            }
        )
        pipeline_repo.create(
            {
                "id": "run_2",
                "name": "p2",
                "run_id": "run_2",
                "config": {},
                "dataset_config": None,
                "model_config": None,
                "status": "running",
            }
        )
        task_repo.create(
            {
                "id": "task_r1",
                "tool_name": "t",
                "tool_image": "i",
                "tool_command": "c",
                "tool_is_baseline": False,
                "config": {},
                "priority": 1,
                "status": DBTaskStatus.PENDING,
                "task_type": "pre_training",
                "task_hash": "h_task1",
                "counter": 1,
                "pipeline_id": "run_1",
                "run_id": "run_1",
            }
        )
        task_repo.create(
            {
                "id": "task_r2",
                "tool_name": "t",
                "tool_image": "i",
                "tool_command": "c",
                "tool_is_baseline": False,
                "config": {},
                "priority": 1,
                "status": DBTaskStatus.PENDING,
                "task_type": "pre_training",
                "task_hash": "h_task2",
                "counter": 1,
                "pipeline_id": "run_2",
                "run_id": "run_2",
            }
        )
        artifact_repo.create(
            {
                "id": "art_run1",
                "task_id": "task_r1",
                "storage_type": "local",
                "bucket": "b",
                "object_key": "obj/run1",
                "size_bytes": 10,
                "provenance": {},
                "created_by_run_id": "run_1",
            }
        )
        artifact_repo.create(
            {
                "id": "art_run2",
                "task_id": "task_r2",
                "storage_type": "local",
                "bucket": "b",
                "object_key": "obj/run2",
                "size_bytes": 20,
                "provenance": {},
                "created_by_run_id": "run_2",
            }
        )

    response = client.delete("/api/pipeline-runs/run_1/cache")
    assert response.status_code == 200
    assert response.json()["deleted_artifacts"] == 1

    with session_scope() as session:
        remaining = session.query(ArtifactModel).all()
        assert len(remaining) == 1
        assert remaining[0].id == "art_run2"
