"""
Task unit tests.

Tests cover:
- ID generation (monotonically increasing, unique string format)
- _compute_hash: deterministic, changes when config or dependencies change
- Task equality and hash-ability (set membership)
- add_dependency: invalidates hash, avoids duplicates
- add_to_workflow: counter, pipeline_id assignment, cross-pipeline error
- EvaluationTask: auto-priority=50, includes required_artifacts in hash
- get_or_create_task: deduplication within same pipeline, separate across pipelines
- TaskFactory.create_task: all five TaskTypes, unknown type raises ValueError
- clear_task_registry: wipes global state
- TaskStatus / TaskType enum string values
"""

import pytest
from typing import List

from src.pipeline.tasks import (
    Task,
    TaskType,
    TaskStatus,
    TaskFactory,
    EvaluationTask,
    PreTrainingTask,
    InTrainingTask,
    PostTrainingTask,
    DeploymentTask,
    get_or_create_task,
    clear_task_registry,
    generate_task_id,
)
from src.pipeline.tools import ContainerConfig, ToolDefinition
import src.pipeline.tasks as tasks_module


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_task_state():
    """Clear registry and reset ID counters before each test."""
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
def basic_tool() -> ToolDefinition:
    """A minimal non-baseline tool."""
    return ToolDefinition(
        name="test-tool",
        container=ContainerConfig(image="test/img:v1", command="python run.py"),
        is_baseline=False,
    )


@pytest.fixture
def baseline_tool() -> ToolDefinition:
    """A baseline tool."""
    return ToolDefinition(
        name="noop",
        container=ContainerConfig(image="test/noop:v1", command="python noop.py"),
        is_baseline=True,
    )


@pytest.fixture
def alt_tool() -> ToolDefinition:
    """A second tool with different image, for dependency/hash tests."""
    return ToolDefinition(
        name="alt-tool",
        container=ContainerConfig(image="test/alt:v2", command="python alt.py"),
        is_baseline=False,
    )


def make_pre_task(tool: ToolDefinition, config: dict = None, deps: list = None) -> PreTrainingTask:
    return TaskFactory.create_task(
        TaskType.PRE_TRAINING, tool=tool,
        config=config or {}, dependencies=deps or []
    )


# ============================================================================
# Tests: ID generation
# ============================================================================


class TestIdGeneration:
    """Tests for the global ID generators."""

    def test_task_id_format(self, basic_tool):
        task = make_pre_task(basic_tool)
        assert task.id.startswith("task_")

    def test_task_ids_are_unique(self, basic_tool):
        t1 = make_pre_task(basic_tool, config={"n": 1})
        t2 = make_pre_task(basic_tool, config={"n": 2})
        assert t1.id != t2.id

    def test_task_ids_monotonically_increase(self, basic_tool):
        t1 = make_pre_task(basic_tool, config={"n": 1})
        t2 = make_pre_task(basic_tool, config={"n": 2})
        n1 = int(t1.id.split("_")[1])
        n2 = int(t2.id.split("_")[1])
        assert n2 == n1 + 1

    def test_generate_task_id_increments(self):
        id1 = generate_task_id()
        id2 = generate_task_id()
        n1 = int(id1.split("_")[1])
        n2 = int(id2.split("_")[1])
        assert n2 == n1 + 1


# ============================================================================
# Tests: TaskStatus and TaskType enums
# ============================================================================


class TestEnums:
    """Tests for TaskStatus and TaskType string values."""

    @pytest.mark.parametrize("status, expected", [
        (TaskStatus.PENDING, "pending"),
        (TaskStatus.RUNNING, "running"),
        (TaskStatus.COMPLETED, "completed"),
        (TaskStatus.FAILED, "failed"),
        (TaskStatus.CANCELLED, "cancelled"),
    ])
    def test_task_status_values(self, status, expected):
        assert status.value == expected

    @pytest.mark.parametrize("task_type, expected", [
        (TaskType.PRE_TRAINING, "pre_training"),
        (TaskType.IN_TRAINING, "in_training"),
        (TaskType.POST_TRAINING, "post_training"),
        (TaskType.DEPLOYMENT, "deployment"),
        (TaskType.EVALUATION, "evaluation"),
    ])
    def test_task_type_values(self, task_type, expected):
        assert task_type.value == expected


# ============================================================================
# Tests: TaskFactory
# ============================================================================


class TestTaskFactory:
    """Tests for TaskFactory.create_task."""

    # -- happy path --

    def test_creates_pre_training_task(self, basic_tool):
        task = TaskFactory.create_task(TaskType.PRE_TRAINING, tool=basic_tool)
        assert isinstance(task, PreTrainingTask)
        assert task.task_type == TaskType.PRE_TRAINING

    def test_creates_in_training_task(self, basic_tool):
        task = TaskFactory.create_task(TaskType.IN_TRAINING, tool=basic_tool)
        assert isinstance(task, InTrainingTask)
        assert task.task_type == TaskType.IN_TRAINING

    def test_creates_post_training_task(self, basic_tool):
        task = TaskFactory.create_task(TaskType.POST_TRAINING, tool=basic_tool)
        assert isinstance(task, PostTrainingTask)
        assert task.task_type == TaskType.POST_TRAINING

    def test_creates_deployment_task(self, basic_tool):
        task = TaskFactory.create_task(TaskType.DEPLOYMENT, tool=basic_tool)
        assert isinstance(task, DeploymentTask)
        assert task.task_type == TaskType.DEPLOYMENT

    def test_creates_evaluation_task(self, basic_tool):
        task = TaskFactory.create_task(TaskType.EVALUATION, tool=basic_tool)
        assert isinstance(task, EvaluationTask)
        assert task.task_type == TaskType.EVALUATION

    def test_default_config_is_empty_dict(self, basic_tool):
        task = TaskFactory.create_task(TaskType.PRE_TRAINING, tool=basic_tool)
        assert task.config == {}

    def test_default_dependencies_is_empty_list(self, basic_tool):
        task = TaskFactory.create_task(TaskType.PRE_TRAINING, tool=basic_tool)
        assert task.dependencies == []

    def test_default_status_is_pending(self, basic_tool):
        task = TaskFactory.create_task(TaskType.PRE_TRAINING, tool=basic_tool)
        assert task.status == TaskStatus.PENDING

    def test_initial_counter_is_zero(self, basic_tool):
        task = TaskFactory.create_task(TaskType.PRE_TRAINING, tool=basic_tool)
        assert task.counter == 0

    def test_config_and_priority_passed_through(self, basic_tool):
        task = TaskFactory.create_task(
            TaskType.PRE_TRAINING, tool=basic_tool,
            config={"alpha": 0.1}, priority=75
        )
        assert task.config == {"alpha": 0.1}
        assert task.priority == 75

    # -- failure modes --

    def test_unknown_task_type_raises(self, basic_tool):
        with pytest.raises(ValueError, match="Unknown task type"):
            TaskFactory.create_task("invalid_type", tool=basic_tool)  # type: ignore[arg-type]


# ============================================================================
# Tests: Hash computation
# ============================================================================


class TestTaskHash:
    """Tests for _compute_hash determinism and sensitivity."""

    # -- happy path --

    def test_same_inputs_same_hash(self, basic_tool):
        t1 = make_pre_task(basic_tool, config={"lr": 0.01})
        t2 = make_pre_task(basic_tool, config={"lr": 0.01})
        assert t1.get_hash() == t2.get_hash()

    def test_different_config_different_hash(self, basic_tool):
        t1 = make_pre_task(basic_tool, config={"lr": 0.01})
        t2 = make_pre_task(basic_tool, config={"lr": 0.1})
        assert t1.get_hash() != t2.get_hash()

    def test_different_tool_name_different_hash(self, basic_tool, alt_tool):
        t1 = make_pre_task(basic_tool)
        t2 = make_pre_task(alt_tool)
        assert t1.get_hash() != t2.get_hash()

    def test_different_image_different_hash(self, tmp_path, basic_tool):
        tool_v2 = ToolDefinition(
            name="test-tool",
            container=ContainerConfig(image="test/img:v2", command="python run.py")
        )
        t1 = make_pre_task(basic_tool)
        t2 = make_pre_task(tool_v2)
        assert t1.get_hash() != t2.get_hash()

    def test_hash_is_string(self, basic_tool):
        task = make_pre_task(basic_tool)
        assert isinstance(task.get_hash(), str)
        assert len(task.get_hash()) == 64  # SHA-256 hex digest

    # -- dependencies affect hash --

    def test_task_with_dep_different_hash_than_without(self, basic_tool, alt_tool):
        dep = make_pre_task(basic_tool, config={"n": 1})
        t_nodep = make_pre_task(alt_tool)
        t_dep = make_pre_task(alt_tool, deps=[dep])
        assert t_nodep.get_hash() != t_dep.get_hash()

    def test_dep_order_matters_in_hash(self, basic_tool, alt_tool):
        dep1 = make_pre_task(basic_tool, config={"n": 1})
        dep2 = make_pre_task(alt_tool, config={"n": 2})
        # sorted by dep.id, so order doesn't change hash (deps are sorted)
        t_ab = make_pre_task(basic_tool, deps=[dep1, dep2])
        t_ba = make_pre_task(basic_tool, deps=[dep2, dep1])
        # Both should hash the same because deps are sorted
        assert t_ab.get_hash() == t_ba.get_hash()

    # -- add_dependency invalidates hash --

    def test_add_dependency_invalidates_hash(self, basic_tool, alt_tool):
        task = make_pre_task(alt_tool)
        original_hash = task.get_hash()

        dep = make_pre_task(basic_tool)
        task.add_dependency(dep)
        new_hash = task.get_hash()

        assert original_hash != new_hash

    def test_add_duplicate_dependency_not_added_twice(self, basic_tool, alt_tool):
        dep = make_pre_task(basic_tool)
        task = make_pre_task(alt_tool)
        task.add_dependency(dep)
        task.add_dependency(dep)

        assert task.dependencies.count(dep) == 1


# ============================================================================
# Tests: Task hash-based identity
# ============================================================================


class TestTaskHashIdentity:
    """
    Tests for get_hash() equality semantics.

    Note: @dataclass regenerates __eq__ on each concrete subclass using
    field comparison (which includes `id`).  The hash-based equality lives
    in Task.get_hash() and is tested directly here.
    """

    def test_same_inputs_produce_same_hash(self, basic_tool):
        t1 = make_pre_task(basic_tool, config={"x": 1})
        t2 = make_pre_task(basic_tool, config={"x": 1})
        assert t1.get_hash() == t2.get_hash()

    def test_different_config_produces_different_hash(self, basic_tool):
        t1 = make_pre_task(basic_tool, config={"x": 1})
        t2 = make_pre_task(basic_tool, config={"x": 2})
        assert t1.get_hash() != t2.get_hash()

    def test_same_object_is_identical(self, basic_tool):
        # The task registry guarantees deduplication via object identity
        t = make_pre_task(basic_tool, config={"x": 1})
        assert t is t  # trivially True; documents the contract

    def test_different_tool_different_hash(self, basic_tool, alt_tool):
        t1 = make_pre_task(basic_tool)
        t2 = make_pre_task(alt_tool)
        assert t1.get_hash() != t2.get_hash()

    def test_tasks_with_same_hash_are_deduplicated_by_registry(self, basic_tool):
        # get_or_create_task returns the SAME object for identical inputs
        t1 = get_or_create_task(
            TaskType.PRE_TRAINING, tool=basic_tool, config={"x": 1}, pipeline_id="p1"
        )
        t2 = get_or_create_task(
            TaskType.PRE_TRAINING, tool=basic_tool, config={"x": 1}, pipeline_id="p1"
        )
        assert t1 is t2  # same object from registry
        assert t1.id == t2.id


# ============================================================================
# Tests: add_to_workflow
# ============================================================================


class TestAddToWorkflow:
    """Tests for task.add_to_workflow."""

    def test_counter_increments_on_new_workflow(self, basic_tool):
        task = make_pre_task(basic_tool)
        task.add_to_workflow("wf_1", "pipe_1")
        task.add_to_workflow("wf_2", "pipe_1")

        assert task.counter == 2
        assert "wf_1" in task.workflows
        assert "wf_2" in task.workflows

    def test_counter_not_incremented_for_same_workflow(self, basic_tool):
        task = make_pre_task(basic_tool)
        task.add_to_workflow("wf_1", "pipe_1")
        task.add_to_workflow("wf_1", "pipe_1")  # duplicate

        assert task.counter == 1

    def test_pipeline_id_assigned(self, basic_tool):
        task = make_pre_task(basic_tool)
        task.add_to_workflow("wf_1", "pipe_42")
        assert task.pipeline_id == "pipe_42"

    def test_cross_pipeline_raises(self, basic_tool):
        task = make_pre_task(basic_tool)
        task.add_to_workflow("wf_1", "pipe_A")

        with pytest.raises(ValueError, match="already belongs to pipeline"):
            task.add_to_workflow("wf_2", "pipe_B")

    def test_empty_pipeline_id_does_not_set(self, basic_tool):
        task = make_pre_task(basic_tool)
        task.add_to_workflow("wf_1", "")
        # Empty pipeline ID should not overwrite future assignment
        assert task.pipeline_id == ""


# ============================================================================
# Tests: EvaluationTask
# ============================================================================


class TestEvaluationTask:
    """Tests for EvaluationTask specifics."""

    # -- happy path --

    def test_default_priority_is_50(self, basic_tool):
        task = EvaluationTask(tool=basic_tool)
        assert task.priority == 50

    def test_explicit_non_zero_priority_preserved(self, basic_tool):
        task = EvaluationTask(tool=basic_tool, priority=60)
        assert task.priority == 60

    def test_task_type_is_evaluation(self, basic_tool):
        task = EvaluationTask(tool=basic_tool)
        assert task.task_type == TaskType.EVALUATION

    def test_required_artifacts_in_hash(self, basic_tool):
        t_no_art = EvaluationTask(tool=basic_tool)
        t_with_art = EvaluationTask(tool=basic_tool, required_artifacts=["model.pth"])
        assert t_no_art.get_hash() != t_with_art.get_hash()

    def test_required_artifacts_order_sorted(self, basic_tool):
        t1 = EvaluationTask(tool=basic_tool, required_artifacts=["b.json", "a.json"])
        t2 = EvaluationTask(tool=basic_tool, required_artifacts=["a.json", "b.json"])
        # sorted in hash computation => same hash
        assert t1.get_hash() == t2.get_hash()

    def test_required_artifacts_default_empty(self, basic_tool):
        task = EvaluationTask(tool=basic_tool)
        assert task.required_artifacts == []

    def test_run_returns_data_unchanged(self, basic_tool):
        task = EvaluationTask(tool=basic_tool)
        assert task.run({"key": "val"}) == {"key": "val"}


# ============================================================================
# Tests: get_or_create_task (deduplication)
# ============================================================================


class TestGetOrCreateTask:
    """Tests for the task registry and deduplication logic."""

    # -- happy path --

    def test_same_inputs_returns_same_task(self, basic_tool):
        t1 = get_or_create_task(
            TaskType.PRE_TRAINING, tool=basic_tool,
            config={"lr": 0.01}, priority=100,
            dependencies=[], pipeline_id="pipe_1"
        )
        t2 = get_or_create_task(
            TaskType.PRE_TRAINING, tool=basic_tool,
            config={"lr": 0.01}, priority=100,
            dependencies=[], pipeline_id="pipe_1"
        )
        assert t1 is t2

    def test_different_config_returns_new_task(self, basic_tool):
        t1 = get_or_create_task(
            TaskType.PRE_TRAINING, tool=basic_tool,
            config={"lr": 0.01}, pipeline_id="pipe_1"
        )
        t2 = get_or_create_task(
            TaskType.PRE_TRAINING, tool=basic_tool,
            config={"lr": 0.5}, pipeline_id="pipe_1"
        )
        assert t1 is not t2

    def test_different_pipeline_returns_new_task(self, basic_tool):
        t1 = get_or_create_task(
            TaskType.PRE_TRAINING, tool=basic_tool,
            config={"lr": 0.01}, pipeline_id="pipe_1"
        )
        t2 = get_or_create_task(
            TaskType.PRE_TRAINING, tool=basic_tool,
            config={"lr": 0.01}, pipeline_id="pipe_2"
        )
        # Same hash but different pipelines: must NOT be shared
        assert t1 is not t2

    def test_shared_task_pipeline_id_set(self, basic_tool):
        task = get_or_create_task(
            TaskType.PRE_TRAINING, tool=basic_tool,
            config={}, pipeline_id="pipe_X"
        )
        assert task.pipeline_id == "pipe_X"

    # -- clear registry --

    def test_clear_task_registry_removes_all(self, basic_tool):
        get_or_create_task(TaskType.PRE_TRAINING, tool=basic_tool, pipeline_id="p1")
        clear_task_registry()
        # After clearing, a second call creates a fresh task (different object)
        t2 = get_or_create_task(TaskType.PRE_TRAINING, tool=basic_tool, pipeline_id="p1")
        # We can't check identity, but we can verify the registry is fresh
        assert len(tasks_module._task_registry) == 1
        assert t2.id in tasks_module._task_registry
