"""
Corner-case DB tests for multi-pipeline runs.

These tests focus on repository/model-level behavior and invariants:
- run-number sequencing
- active run filtering
- status timestamp transitions
- uniqueness and foreign-key constraints
- cascade cleanup behavior
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from src.db import (
    DatabaseConfig,
    get_database,
    init_database,
    session_scope,
    PipelineConfigRepository,
    PipelineRunRepository,
    PipelineRepository,
    TaskRepository,
    WorkflowRepository,
    ArtifactRepository,
    PipelineRunStatus,
)
from src.db.models import (
    TaskStatus as DBTaskStatus,
    PipelineRunModel,
    PipelineConfigModel,
    EvaluationResultModel,
)


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    db_path = tmp_path / "corner_cases.db"
    init_database(DatabaseConfig(db_type="sqlite", sqlite_path=str(db_path)), create_tables=True)
    yield
    get_database().close()


def _create_config(session, config_id: str = "config_trades"):
    return PipelineConfigRepository(session).create(
        {
            "id": config_id,
            "name": config_id,
            "description": "cfg",
            "config_path": f"/tmp/{config_id}.yaml",
            "attack_config_path": None,
            "config_hash": "h1",
        }
    )


def _create_run(session, config_id: str, run_id: str, run_number: int, status=PipelineRunStatus.PENDING):
    return PipelineRunRepository(session).create(
        {
            "id": run_id,
            "pipeline_config_id": config_id,
            "run_number": run_number,
            "use_cache": True,
            "status": status,
        }
    )


def test_next_run_number_is_per_config():
    with session_scope() as session:
        _create_config(session, "config_a")
        _create_config(session, "config_b")
        repo = PipelineRunRepository(session)
        _create_run(session, "config_a", "run_a1", 1)
        _create_run(session, "config_a", "run_a2", 2)
        _create_run(session, "config_b", "run_b1", 1)

        assert repo.get_next_run_number("config_a") == 3
        assert repo.get_next_run_number("config_b") == 2
        assert repo.get_next_run_number("config_missing") == 1


def test_active_run_filter_only_returns_pending_running_stopping():
    with session_scope() as session:
        _create_config(session, "config_x")
        repo = PipelineRunRepository(session)
        _create_run(session, "config_x", "r_pending", 1, PipelineRunStatus.PENDING)
        _create_run(session, "config_x", "r_running", 2, PipelineRunStatus.RUNNING)
        _create_run(session, "config_x", "r_stopping", 3, PipelineRunStatus.STOPPING)
        _create_run(session, "config_x", "r_completed", 4, PipelineRunStatus.COMPLETED)
        _create_run(session, "config_x", "r_failed", 5, PipelineRunStatus.FAILED)
        _create_run(session, "config_x", "r_cancelled", 6, PipelineRunStatus.CANCELLED)

        active_ids = {r.id for r in repo.get_active_runs_for_config("config_x")}
        assert active_ids == {"r_pending", "r_running", "r_stopping"}


def test_update_status_sets_started_and_completed_timestamps():
    with session_scope() as session:
        _create_config(session, "config_s")
        repo = PipelineRunRepository(session)
        _create_run(session, "config_s", "run_s1", 1, PipelineRunStatus.PENDING)

        run = repo.update_status("run_s1", PipelineRunStatus.RUNNING)
        assert run is not None
        assert run.started_at is not None
        assert run.completed_at is None

        run = repo.update_status("run_s1", PipelineRunStatus.COMPLETED)
        assert run is not None
        assert run.completed_at is not None


def test_unique_constraint_on_config_and_run_number_enforced():
    with session_scope() as session:
        _create_config(session, "config_u")
        _create_run(session, "config_u", "run_u1", 1)

    with pytest.raises(IntegrityError):
        with session_scope() as session:
            # same config + same run_number should fail
            _create_run(session, "config_u", "run_u2_dup_num", 1)


def test_foreign_key_blocks_run_for_missing_config():
    with pytest.raises(IntegrityError):
        with session_scope() as session:
            PipelineRunRepository(session).create(
                {
                    "id": "run_orphan",
                    "pipeline_config_id": "does_not_exist",
                    "run_number": 1,
                    "use_cache": True,
                    "status": PipelineRunStatus.PENDING,
                }
            )


def test_delete_pipeline_config_cascades_pipeline_runs():
    with session_scope() as session:
        cfg_repo = PipelineConfigRepository(session)
        run_repo = PipelineRunRepository(session)
        cfg_repo.create(
            {
                "id": "config_cascade",
                "name": "cfg",
                "description": "cfg",
                "config_path": "/tmp/cascade.yaml",
                "attack_config_path": None,
                "config_hash": "h1",
            }
        )
        _create_run(session, "config_cascade", "run_c1", 1)
        _create_run(session, "config_cascade", "run_c2", 2)

        assert len(run_repo.get_by_config_id("config_cascade")) == 2
        assert cfg_repo.delete("config_cascade") is True

    with session_scope() as session:
        assert session.query(PipelineConfigModel).filter_by(id="config_cascade").first() is None
        assert session.query(PipelineRunModel).filter_by(pipeline_config_id="config_cascade").count() == 0


def test_artifact_delete_by_run_isolated_and_idempotent():
    with session_scope() as session:
        _create_config(session, "config_art")
        _create_run(session, "config_art", "run_art_1", 1)
        _create_run(session, "config_art", "run_art_2", 2)

        PipelineRepository(session).create(
            {
                "id": "run_art_1",
                "name": "p1",
                "run_id": "run_art_1",
                "config": {},
                "dataset_config": None,
                "model_config": None,
                "status": "running",
            }
        )
        PipelineRepository(session).create(
            {
                "id": "run_art_2",
                "name": "p2",
                "run_id": "run_art_2",
                "config": {},
                "dataset_config": None,
                "model_config": None,
                "status": "running",
            }
        )

        TaskRepository(session).create(
            {
                "id": "task_art_1",
                "tool_name": "t",
                "tool_image": "i",
                "tool_command": "c",
                "tool_is_baseline": False,
                "config": {},
                "priority": 1,
                "status": DBTaskStatus.PENDING,
                "task_type": "pre_training",
                "task_hash": "h1",
                "counter": 1,
                "pipeline_id": "run_art_1",
                "run_id": "run_art_1",
            }
        )
        TaskRepository(session).create(
            {
                "id": "task_art_2",
                "tool_name": "t",
                "tool_image": "i",
                "tool_command": "c",
                "tool_is_baseline": False,
                "config": {},
                "priority": 1,
                "status": DBTaskStatus.PENDING,
                "task_type": "pre_training",
                "task_hash": "h2",
                "counter": 1,
                "pipeline_id": "run_art_2",
                "run_id": "run_art_2",
            }
        )

        art_repo = ArtifactRepository(session)
        art_repo.create(
            {
                "id": "art1",
                "task_id": "task_art_1",
                "storage_type": "local",
                "bucket": "b",
                "object_key": "k1",
                "size_bytes": 10,
                "provenance": {},
                "created_by_run_id": "run_art_1",
            }
        )
        art_repo.create(
            {
                "id": "art2",
                "task_id": "task_art_2",
                "storage_type": "local",
                "bucket": "b",
                "object_key": "k2",
                "size_bytes": 20,
                "provenance": {},
                "created_by_run_id": "run_art_2",
            }
        )

        assert art_repo.delete_by_run_id("run_art_1") == 1
        # idempotent on second call
        assert art_repo.delete_by_run_id("run_art_1") == 0
        assert [a.id for a in art_repo.get_by_run_id("run_art_2")] == ["art2"]


def test_evaluation_result_uniqueness_per_workflow_evaluator_corner_case():
    """
    Corner case: current schema uniqueness is (workflow_id, evaluator_name),
    not (run_id, workflow_id, evaluator_name). This test documents and guards
    current behavior.
    """
    with session_scope() as session:
        _create_config(session, "config_eval")
        _create_run(session, "config_eval", "run_eval_1", 1)
        _create_run(session, "config_eval", "run_eval_2", 2)
        PipelineRepository(session).create(
            {
                "id": "run_eval_1",
                "name": "p1",
                "run_id": "run_eval_1",
                "config": {},
                "dataset_config": None,
                "model_config": None,
                "status": "running",
            }
        )
        PipelineRepository(session).create(
            {
                "id": "run_eval_2",
                "name": "p2",
                "run_id": "run_eval_2",
                "config": {},
                "dataset_config": None,
                "model_config": None,
                "status": "running",
            }
        )
        WorkflowRepository(session).create(
            {
                "id": "wf_shared",
                "name": "comb_001",
                "pipeline_id": "run_eval_1",
                "run_id": "run_eval_1",
                "status": "completed",
            }
        )

        session.add(
            EvaluationResultModel(
                id="er1",
                workflow_id="wf_shared",
                pipeline_id="run_eval_1",
                run_id="run_eval_1",
                evaluator_name="fairness",
                evaluator_image="img",
                metrics={"acc": 0.9},
                success=True,
                skipped=False,
            )
        )

    with pytest.raises(IntegrityError):
        with session_scope() as session:
            # same workflow + evaluator, different run -> still violates current unique constraint
            session.add(
                EvaluationResultModel(
                    id="er2",
                    workflow_id="wf_shared",
                    pipeline_id="run_eval_2",
                    run_id="run_eval_2",
                    evaluator_name="fairness",
                    evaluator_image="img",
                    metrics={"acc": 0.8},
                    success=True,
                    skipped=False,
                )
            )
