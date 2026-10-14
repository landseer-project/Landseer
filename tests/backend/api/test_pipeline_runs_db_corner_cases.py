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
    ExperimentRepository,
    ExperimentRunRepository,
    TaskRepository,
    ArtifactRepository,
    ExperimentRunStatus,
)
from src.db.models import (
    TaskStatus as DBTaskStatus,
    ExperimentRunModel,
    ExperimentModel,
    EvaluationResultModel,
)


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    db_path = tmp_path / "corner_cases.db"
    init_database(DatabaseConfig(db_type="sqlite", sqlite_path=str(db_path)), create_tables=True)
    yield
    get_database().close()


def _create_config(session, config_id: str = "config_trades"):
    return ExperimentRepository(session).create(
        {
            "id": config_id,
            "name": config_id,
            "description": "cfg",
            "config_path": f"/tmp/{config_id}.yaml",
            "attack_config_path": None,
            "config_hash": "h1",
        }
    )


def _create_run(session, config_id: str, run_id: str, run_number: int, status=ExperimentRunStatus.PENDING):
    return ExperimentRunRepository(session).create(
        {
            "id": run_id,
            "experiment_id": config_id,
            "run_number": run_number,
            "use_cache": True,
            "status": status,
        }
    )


def test_next_run_number_is_per_config():
    with session_scope() as session:
        _create_config(session, "config_a")
        _create_config(session, "config_b")
        repo = ExperimentRunRepository(session)
        _create_run(session, "config_a", "run_a1", 1)
        _create_run(session, "config_a", "run_a2", 2)
        _create_run(session, "config_b", "run_b1", 1)

        assert repo.get_next_run_number("config_a") == 3
        assert repo.get_next_run_number("config_b") == 2
        assert repo.get_next_run_number("config_missing") == 1


def test_active_run_filter_only_returns_pending_running_stopping():
    with session_scope() as session:
        _create_config(session, "config_x")
        repo = ExperimentRunRepository(session)
        _create_run(session, "config_x", "r_pending", 1, ExperimentRunStatus.PENDING)
        _create_run(session, "config_x", "r_running", 2, ExperimentRunStatus.RUNNING)
        _create_run(session, "config_x", "r_stopping", 3, ExperimentRunStatus.STOPPING)
        _create_run(session, "config_x", "r_completed", 4, ExperimentRunStatus.COMPLETED)
        _create_run(session, "config_x", "r_failed", 5, ExperimentRunStatus.FAILED)
        _create_run(session, "config_x", "r_cancelled", 6, ExperimentRunStatus.CANCELLED)

        active_ids = {r.id for r in repo.get_active_runs_for_experiment("config_x")}
        assert active_ids == {"r_pending", "r_running", "r_stopping"}


def test_update_status_sets_started_and_completed_timestamps():
    with session_scope() as session:
        _create_config(session, "config_s")
        repo = ExperimentRunRepository(session)
        _create_run(session, "config_s", "run_s1", 1, ExperimentRunStatus.PENDING)

        run = repo.update_status("run_s1", ExperimentRunStatus.RUNNING)
        assert run is not None
        assert run.started_at is not None
        assert run.completed_at is None

        run = repo.update_status("run_s1", ExperimentRunStatus.COMPLETED)
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
            ExperimentRunRepository(session).create(
                {
                    "id": "run_orphan",
                    "experiment_id": "does_not_exist",
                    "run_number": 1,
                    "use_cache": True,
                    "status": ExperimentRunStatus.PENDING,
                }
            )


def test_delete_pipeline_config_cascades_pipeline_runs():
    with session_scope() as session:
        cfg_repo = ExperimentRepository(session)
        run_repo = ExperimentRunRepository(session)
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

        assert len(run_repo.get_by_experiment_id("config_cascade")) == 2
        assert cfg_repo.delete("config_cascade") is True

    with session_scope() as session:
        assert session.query(ExperimentModel).filter_by(id="config_cascade").first() is None
        assert session.query(ExperimentRunModel).filter_by(experiment_id="config_cascade").count() == 0


def test_artifact_delete_by_run_isolated_and_idempotent():
    from src.db.models import (
        ToolModel,
        ToolType,
        ToolStage,
        ToolDefenseType,
        TaskType,
    )

    with session_scope() as session:
        _create_config(session, "config_art")
        _create_run(session, "config_art", "run_art_1", 1)
        _create_run(session, "config_art", "run_art_2", 2)

        session.add(
            ToolModel(
                id="tool_t",
                tool_name="t",
                tool_type=ToolType.DEFENSE,
                tool_defense_type=ToolDefenseType.NONE,
                tool_stage=ToolStage.PRE_TRAINING,
                tool_image="i",
            )
        )
        session.flush()

        TaskRepository(session).create(
            {
                "id": "task_art_1",
                "tool_id": "tool_t",
                "config": {},
                "priority": 1,
                "status": DBTaskStatus.PENDING,
                "task_type": TaskType.TOOL,
                "task_hash": "h1",
                "counter": 1,
                "experiment_run_id": "run_art_1",
            }
        )
        TaskRepository(session).create(
            {
                "id": "task_art_2",
                "tool_id": "tool_t",
                "config": {},
                "priority": 1,
                "status": DBTaskStatus.PENDING,
                "task_type": TaskType.TOOL,
                "task_hash": "h2",
                "counter": 1,
                "experiment_run_id": "run_art_2",
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
                "artifact_type": "tool_output",
                "experiment_run_id": "run_art_1",
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
                "artifact_type": "tool_output",
                "experiment_run_id": "run_art_2",
            }
        )

        assert art_repo.delete_by_run_id("run_art_1") == 1
        assert art_repo.delete_by_run_id("run_art_1") == 0
        assert [a.id for a in art_repo.get_by_run_id("run_art_2")] == ["art2"]


def test_evaluation_result_uniqueness_per_workflow_task():
    """
    Unique on (workflow_id, evaluation_task_id). Tool name/image come from tools
    via the evaluation task — not duplicated on evaluation_results.
    """
    from src.db.models import (
        ToolModel,
        ToolType,
        ToolStage,
        ToolDefenseType,
        TaskModel,
        TaskType,
        WorkflowModel,
        WorkflowStatus,
        ExperimentModel,
        ExperimentRunModel,
        ExperimentRunStatus,
        ModelModel,
        ModelType,
        DatasetModel,
        DatasetType,
        DatasetVariantModel,
        DatasetVariantType,
    )

    with session_scope() as session:
        session.add(
            ToolModel(
                id="tool_fair",
                tool_name="fairness",
                tool_type=ToolType.EVAL,
                tool_defense_type=ToolDefenseType.FAIRNESS,
                tool_stage=ToolStage.POST_TRAINING,
                tool_image="fairness:latest",
            )
        )
        session.add(DatasetModel(id="ds1", name="cifar", dataset_type=DatasetType.IMAGE))
        session.add(
            DatasetVariantModel(
                id="dsv1",
                dataset_id="ds1",
                name="clean",
                dataset_variant_type=DatasetVariantType.CLEAN,
                dataset_path="/data/cifar",
            )
        )
        session.add(
            ModelModel(id="m1", model_type=ModelType.PYTORCH, model_path="/models/m1")
        )
        session.add(
            ExperimentModel(
                id="exp1",
                experiment_name="exp1",
                model_id="m1",
                dataset_variant_id="dsv1",
            )
        )
        session.add(
            ExperimentRunModel(
                id="run_eval_1",
                experiment_id="exp1",
                run_number=1,
                status=ExperimentRunStatus.RUNNING,
            )
        )
        session.add(
            ExperimentRunModel(
                id="run_eval_2",
                experiment_id="exp1",
                run_number=2,
                status=ExperimentRunStatus.RUNNING,
            )
        )
        session.add(
            WorkflowModel(
                id="wf_shared",
                experiment_run_id="run_eval_1",
                status=WorkflowStatus.COMPLETED,
            )
        )
        session.add(
            TaskModel(
                id="task_eval_1",
                task_type=TaskType.EVALUATION,
                tool_id="tool_fair",
                config={},
                status=DBTaskStatus.COMPLETED,
                experiment_run_id="run_eval_1",
            )
        )
        session.flush()

        # Wire ordered workflow ↔ task association
        wf = session.get(WorkflowModel, "wf_shared")
        task = session.get(TaskModel, "task_eval_1")
        wf.tasks.append(task)
        session.flush()

        session.add(
            EvaluationResultModel(
                id="er1",
                workflow_id="wf_shared",
                evaluation_task_id="task_eval_1",
                metrics={"acc": 0.9},
                success=True,
                skipped=False,
            )
        )

    with pytest.raises(IntegrityError):
        with session_scope() as session:
            # same workflow + same evaluation task → unique violation
            session.add(
                EvaluationResultModel(
                    id="er2",
                    workflow_id="wf_shared",
                    evaluation_task_id="task_eval_1",
                    metrics={"acc": 0.8},
                    success=True,
                    skipped=False,
                )
            )
