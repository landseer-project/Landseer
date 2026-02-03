"""
Tests for evaluation result reporting: backend persistence of evaluator results.

Verifies that when workers report completion of evaluator tasks with
evaluation_result in the payload, the backend persists it and metrics API
can return it.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.backend.api import app, _scheduler_state
from src.pipeline.tasks import TaskType, clear_task_registry
from src.pipeline.tools import ToolDefinition, ContainerConfig
from src.pipeline.workflow import WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline

from .conftest import create_task_with_id


@pytest.fixture
def client():
    """Test client."""
    return TestClient(app)


@pytest.fixture
def pipeline_with_evaluator():
    """Pipeline with one evaluation task for result persistence tests."""
    clear_task_registry()
    pipeline_id = "pipeline_metrics_test"
    tool = ToolDefinition(
        name="adversarial-evaluator",
        container=ContainerConfig(
            image="eval/image:latest",
            command="python eval.py"
        ),
    )
    # Do not set pipeline_id on task so Pipeline.__post_init__ can set it
    eval_task = create_task_with_id(
        task_id="task_eval_1",
        tool=tool,
        task_type=TaskType.EVALUATION,
        priority=50,
        pipeline_id=None,
    )
    eval_task.workflows = {"workflow_1"}
    workflow = WorkflowFactory.create_workflow(
        name="comb_001",
        tasks=[eval_task],
    )
    pipeline = DefenseEvaluationPipeline(
        name="metrics_pipeline",
        workflows=[workflow],
    )
    # Override to a stable id and keep tasks in sync
    pipeline.id = pipeline_id
    for w in pipeline.workflows:
        w.pipeline_id = pipeline_id
        for t in w.tasks:
            t.pipeline_id = pipeline_id
    _scheduler_state.initialize(pipeline)
    return pipeline


def test_update_task_status_persists_evaluation_result(
    client, pipeline_with_evaluator
):
    """PUT /tasks/status with evaluation_result for an evaluator task should persist to DB."""
    mock_db = MagicMock()
    mock_db.is_available.return_value = True
    mock_db.save_evaluation_result.return_value = True
    _scheduler_state._db_service = mock_db

    eval_data = {
        "metrics": {"clean_accuracy": 0.92, "pgd_accuracy": 0.78},
        "success": True,
        "skipped": False,
    }
    response = client.put(
        "/tasks/status",
        json={
            "task_id": "task_eval_1",
            "status": "completed",
            "execution_time_ms": 1500,
            "result": {
                "artifacts": {},
                "logs": None,
                "log_path": None,
                "evaluation_result": eval_data,
            },
        },
    )
    assert response.status_code == 200
    assert response.json().get("success") is True

    mock_db.save_evaluation_result.assert_called()
    calls = mock_db.save_evaluation_result.call_args_list
    assert len(calls) >= 1
    call_kw = calls[0][1]
    assert call_kw["workflow_id"] == "workflow_1"
    assert call_kw["pipeline_id"] == "pipeline_metrics_test"
    assert call_kw["evaluator_name"] == "adversarial-evaluator"
    assert call_kw["result_data"] == eval_data
    assert call_kw["evaluation_task_id"] == "task_eval_1"


def test_update_task_status_ignores_evaluation_result_for_non_evaluator(
    client,
):
    """PUT /tasks/status with evaluation_result for a non-evaluator task should not call save_evaluation_result."""
    clear_task_registry()
    pipeline_id = "pipeline_non_eval_test"
    tool_train = ToolDefinition(
        name="train_tool",
        container=ContainerConfig(image="train/image", command="train"),
    )
    tool_eval = ToolDefinition(
        name="adversarial-evaluator",
        container=ContainerConfig(image="eval/image", command="eval"),
    )
    pre_task = create_task_with_id(
        task_id="task_pre_1",
        tool=tool_train,
        task_type=TaskType.PRE_TRAINING,
        pipeline_id=None,
    )
    pre_task.workflows = {"workflow_1"}
    eval_task = create_task_with_id(
        task_id="task_eval_1",
        tool=tool_eval,
        task_type=TaskType.EVALUATION,
        pipeline_id=None,
    )
    eval_task.workflows = {"workflow_1"}
    eval_task.dependencies = [pre_task]
    workflow = WorkflowFactory.create_workflow(
        name="comb_001",
        tasks=[pre_task, eval_task],
    )
    pipeline = DefenseEvaluationPipeline(
        name="non_eval_pipeline",
        workflows=[workflow],
    )
    pipeline.id = pipeline_id
    for w in pipeline.workflows:
        w.pipeline_id = pipeline_id
        for t in w.tasks:
            t.pipeline_id = pipeline_id
    _scheduler_state.initialize(pipeline)

    mock_db = MagicMock()
    mock_db.is_available.return_value = True
    _scheduler_state._db_service = mock_db

    response = client.put(
        "/tasks/status",
        json={
            "task_id": "task_pre_1",
            "status": "completed",
            "execution_time_ms": 1000,
            "result": {
                "artifacts": {},
                "logs": None,
                "log_path": None,
                "evaluation_result": {"metrics": {"acc": 0.9}, "success": True},
            },
        },
    )
    assert response.status_code == 200
    mock_db.save_evaluation_result.assert_not_called()


def test_update_task_status_without_evaluation_result_does_not_call_save(
    client, pipeline_with_evaluator
):
    """PUT /tasks/status without evaluation_result in payload should not call save_evaluation_result."""
    mock_db = MagicMock()
    mock_db.is_available.return_value = True
    _scheduler_state._db_service = mock_db

    response = client.put(
        "/tasks/status",
        json={
            "task_id": "task_eval_1",
            "status": "completed",
            "execution_time_ms": 1500,
            "result": {"artifacts": {}, "logs": None, "log_path": None},
        },
    )
    assert response.status_code == 200
    mock_db.save_evaluation_result.assert_not_called()
