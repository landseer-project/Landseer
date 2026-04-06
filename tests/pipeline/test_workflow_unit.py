"""
Workflow unit tests.

Tests cover:
- Workflow creation and ID generation
- add_task: appends task, registers with workflow/pipeline
- run: executes tasks in order, passes data through chain
- get_tasks_by_type: filters correctly
- get_task_by_id: found and not-found
- get_task_by_tool_name: found and not-found
- has_task: true and false
- WorkflowFactory.create_workflow: sets pipeline_id, registers tasks
- Workflow with empty task list edge case
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List

from src.pipeline.tasks import (
    Task,
    TaskType,
    TaskStatus,
    TaskFactory,
    PreTrainingTask,
    InTrainingTask,
    DeploymentTask,
    EvaluationTask,
    clear_task_registry,
)
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import Workflow, WorkflowFactory
import src.pipeline.tasks as tasks_module


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    clear_task_registry()
    tasks_module._task_id_counter = 0
    tasks_module._workflow_id_counter = 0
    tasks_module._pipeline_id_counter = 0
    yield
    clear_task_registry()
    tasks_module._task_id_counter = 0
    tasks_module._workflow_id_counter = 0
    tasks_module._pipeline_id_counter = 0


@pytest.fixture
def pre_tool() -> ToolDefinition:
    return ToolDefinition(
        name="pre-tool",
        container=ContainerConfig(image="img/pre:v1", command="run"),
    )


@pytest.fixture
def in_tool() -> ToolDefinition:
    return ToolDefinition(
        name="in-tool",
        container=ContainerConfig(image="img/in:v1", command="run"),
    )


@pytest.fixture
def deploy_tool() -> ToolDefinition:
    return ToolDefinition(
        name="deploy-tool",
        container=ContainerConfig(image="img/deploy:v1", command="run"),
    )


@pytest.fixture
def eval_tool() -> ToolDefinition:
    return ToolDefinition(
        name="eval-clean",
        container=ContainerConfig(image="img/eval:v1", command="run"),
    )


def make_task(tool: ToolDefinition, task_type: TaskType = TaskType.PRE_TRAINING,
              config: dict = None) -> Task:
    return TaskFactory.create_task(task_type, tool=tool, config=config or {})


# ============================================================================
# Tests: Workflow creation
# ============================================================================


class TestWorkflowCreation:
    """Tests for basic Workflow instantiation."""

    def test_id_is_auto_generated(self):
        wf = Workflow(name="comb_001")
        assert wf.id.startswith("workflow_")

    def test_unique_ids_for_different_workflows(self):
        wf1 = Workflow(name="comb_001")
        wf2 = Workflow(name="comb_002")
        assert wf1.id != wf2.id

    def test_default_tasks_empty(self):
        wf = Workflow(name="wf")
        assert wf.tasks == []

    def test_default_metadata_empty(self):
        wf = Workflow(name="wf")
        assert wf.metadata == {}

    def test_default_pipeline_id_empty(self):
        wf = Workflow(name="wf")
        assert wf.pipeline_id == ""

    def test_repr_shows_name_and_task_count(self, pre_tool):
        task = make_task(pre_tool)
        wf = Workflow(name="my_wf", tasks=[task])
        r = repr(wf)
        assert "my_wf" in r
        assert "1" in r


# ============================================================================
# Tests: add_task
# ============================================================================


class TestWorkflowAddTask:
    """Tests for Workflow.add_task."""

    def test_task_appended(self, pre_tool):
        # Arrange
        wf = Workflow(name="wf", pipeline_id="p1")
        task = make_task(pre_tool)

        # Act
        wf.add_task(task)

        # Assert
        assert task in wf.tasks

    def test_task_registered_with_workflow(self, pre_tool):
        # add_task calls task.add_to_workflow when pipeline_id is set
        wf = Workflow(name="wf", pipeline_id="pipe_1")
        task = make_task(pre_tool)
        wf.add_task(task)

        assert wf.id in task.workflows
        assert task.counter == 1

    def test_task_not_registered_when_no_pipeline_id(self, pre_tool):
        # If pipeline_id is empty, add_to_workflow is NOT called
        wf = Workflow(name="wf")
        task = make_task(pre_tool)
        original_counter = task.counter
        wf.add_task(task)

        assert task.counter == original_counter  # unchanged

    def test_multiple_tasks_ordered(self, pre_tool, in_tool):
        wf = Workflow(name="wf", pipeline_id="p1")
        t1 = make_task(pre_tool, TaskType.PRE_TRAINING, {"n": 1})
        t2 = make_task(in_tool, TaskType.IN_TRAINING, {"n": 2})
        wf.add_task(t1)
        wf.add_task(t2)

        assert wf.tasks[0] is t1
        assert wf.tasks[1] is t2


# ============================================================================
# Tests: run
# ============================================================================


class TestWorkflowRun:
    """Tests for Workflow.run data pipeline."""

    def test_empty_workflow_returns_input(self):
        wf = Workflow(name="wf")
        result = wf.run(data={"x": 42})
        assert result == {"x": 42}

    def test_single_task_run_called(self, pre_tool):
        # Arrange: patch task.run to transform data
        wf = Workflow(name="wf")
        task = make_task(pre_tool)
        task.run = lambda d: {**d, "step": 1}
        wf.tasks.append(task)

        # Act
        result = wf.run({"x": 0})

        # Assert
        assert result == {"x": 0, "step": 1}

    def test_chained_tasks_pipe_data(self, pre_tool, in_tool):
        # Arrange
        wf = Workflow(name="wf")
        t1 = make_task(pre_tool, config={"n": 1})
        t2 = make_task(in_tool, config={"n": 2})
        t1.run = lambda d: {**d, "pre": True}
        t2.run = lambda d: {**d, "in": True}
        wf.tasks.extend([t1, t2])

        # Act
        result = wf.run({})

        # Assert
        assert result == {"pre": True, "in": True}

    def test_run_none_data(self, pre_tool):
        # run(None) should not crash, task.run receives None
        wf = Workflow(name="wf")
        task = make_task(pre_tool)
        task.run = lambda d: d  # identity
        wf.tasks.append(task)

        result = wf.run(None)
        assert result is None


# ============================================================================
# Tests: get_tasks_by_type
# ============================================================================


class TestGetTasksByType:
    """Tests for Workflow.get_tasks_by_type."""

    def test_returns_matching_type_only(self, pre_tool, in_tool, deploy_tool, eval_tool):
        # Arrange
        wf = Workflow(name="wf")
        t_pre = make_task(pre_tool, TaskType.PRE_TRAINING)
        t_in = make_task(in_tool, TaskType.IN_TRAINING)
        t_deploy = make_task(deploy_tool, TaskType.DEPLOYMENT)
        t_eval = EvaluationTask(tool=eval_tool)
        wf.tasks.extend([t_pre, t_in, t_deploy, t_eval])

        # Act
        pre_tasks = wf.get_tasks_by_type(TaskType.PRE_TRAINING)
        eval_tasks = wf.get_tasks_by_type(TaskType.EVALUATION)

        # Assert
        assert pre_tasks == [t_pre]
        assert eval_tasks == [t_eval]

    def test_returns_empty_when_no_match(self, pre_tool):
        wf = Workflow(name="wf")
        wf.tasks.append(make_task(pre_tool, TaskType.PRE_TRAINING))

        result = wf.get_tasks_by_type(TaskType.DEPLOYMENT)
        assert result == []

    def test_returns_multiple_tasks_of_same_type(self, pre_tool):
        wf = Workflow(name="wf")
        t1 = make_task(pre_tool, TaskType.PRE_TRAINING, {"n": 1})
        t2 = make_task(pre_tool, TaskType.PRE_TRAINING, {"n": 2})
        wf.tasks.extend([t1, t2])

        result = wf.get_tasks_by_type(TaskType.PRE_TRAINING)
        assert len(result) == 2


# ============================================================================
# Tests: get_task_by_id / get_task_by_tool_name / has_task
# ============================================================================


class TestTaskLookup:
    """Tests for task lookup methods."""

    def test_get_task_by_id_found(self, pre_tool):
        wf = Workflow(name="wf")
        task = make_task(pre_tool)
        wf.tasks.append(task)

        result = wf.get_task_by_id(task.id)
        assert result is task

    def test_get_task_by_id_not_found(self, pre_tool):
        wf = Workflow(name="wf")
        wf.tasks.append(make_task(pre_tool))

        result = wf.get_task_by_id("nonexistent_id")
        assert result is None

    def test_get_task_by_tool_name_found(self, pre_tool):
        wf = Workflow(name="wf")
        task = make_task(pre_tool)
        wf.tasks.append(task)

        result = wf.get_task_by_tool_name("pre-tool")
        assert result is task

    def test_get_task_by_tool_name_not_found(self, pre_tool):
        wf = Workflow(name="wf")
        wf.tasks.append(make_task(pre_tool))

        result = wf.get_task_by_tool_name("no-such-tool")
        assert result is None

    def test_has_task_true(self, pre_tool):
        wf = Workflow(name="wf")
        task = make_task(pre_tool)
        wf.tasks.append(task)

        assert wf.has_task(task.id) is True

    def test_has_task_false(self):
        wf = Workflow(name="wf")
        assert wf.has_task("task_999") is False


# ============================================================================
# Tests: WorkflowFactory
# ============================================================================


class TestWorkflowFactory:
    """Tests for WorkflowFactory.create_workflow."""

    def test_creates_workflow_with_name(self):
        wf = WorkflowFactory.create_workflow(name="comb_007")
        assert wf.name == "comb_007"

    def test_pipeline_id_propagated(self, pre_tool):
        task = make_task(pre_tool)
        wf = WorkflowFactory.create_workflow(
            name="wf", tasks=[task], pipeline_id="pipe_99"
        )
        assert wf.pipeline_id == "pipe_99"
        assert task.pipeline_id == "pipe_99"

    def test_tasks_registered_with_workflow(self, pre_tool):
        task = make_task(pre_tool)
        wf = WorkflowFactory.create_workflow(
            name="wf", tasks=[task], pipeline_id="pipe_1"
        )
        assert wf.id in task.workflows

    def test_empty_tasks_default(self):
        wf = WorkflowFactory.create_workflow(name="wf")
        assert wf.tasks == []

    def test_metadata_passed_through(self):
        meta = {"dataset": "cifar10", "model": "resnet18"}
        wf = WorkflowFactory.create_workflow(name="wf", metadata=meta)
        assert wf.metadata == meta

    def test_no_registration_when_no_pipeline_id(self, pre_tool):
        task = make_task(pre_tool)
        WorkflowFactory.create_workflow(name="wf", tasks=[task])
        # No pipeline_id -> add_to_workflow not called -> counter unchanged
        assert task.counter == 0
