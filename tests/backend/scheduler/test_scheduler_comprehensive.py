"""
Comprehensive scheduler tests for critical behavior and corner cases.

This suite complements existing scheduler tests with:
- Base scheduler guardrails
- Ready/blocked state transitions
- Completion semantics with mixed outcomes
- Deterministic ordering on equal priorities
- Corner cases (empty pipeline, circular dependencies)
"""

import sys
from pathlib import Path

import pytest

# Ensure repository root is importable when pytest is invoked
# from subdirectories/environments without project-root on PYTHONPATH.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend.scheduler.priority_scheduler import PriorityScheduler
from src.pipeline.pipeline import DefenseEvaluationPipeline
from src.pipeline.tasks import TaskFactory, TaskStatus, TaskType, clear_task_registry
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import WorkflowFactory


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_task_registry()
    yield
    clear_task_registry()


@pytest.fixture
def tool():
    return ToolDefinition(
        name="sched_test_tool",
        container=ContainerConfig(image="test/image:latest", command="python run.py"),
    )


def _mk_task(tool, *, task_type=TaskType.PRE_TRAINING, config=None, deps=None):
    return TaskFactory.create_task(
        task_type=task_type,
        tool=tool,
        config=config or {},
        dependencies=deps or [],
    )


def test_empty_pipeline_has_no_next_task_and_is_complete():
    pipeline = DefenseEvaluationPipeline(name="empty", workflows=[])
    scheduler = PriorityScheduler(pipeline)

    assert scheduler.get_next_task() is None
    assert scheduler.is_complete() is True
    assert scheduler.get_progress() == {
        "total": 0,
        "pending": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
    }


def test_base_update_status_rejects_non_terminal_updates(tool):
    task = _mk_task(tool)
    wf = WorkflowFactory.create_workflow(name="wf", tasks=[task])
    pipeline = DefenseEvaluationPipeline(name="p", workflows=[wf])
    scheduler = PriorityScheduler(pipeline)

    # Move to running first
    picked = scheduler.get_next_task()
    assert picked is not None
    assert picked.status == TaskStatus.RUNNING

    with pytest.raises(ValueError):
        scheduler.update_task_status(task.id, TaskStatus.PENDING)

    with pytest.raises(ValueError):
        scheduler.update_task_status(task.id, TaskStatus.RUNNING)


def test_get_tasks_by_status_tracks_state_transitions(tool):
    t1 = _mk_task(tool, config={"name": "t1"})
    t2 = _mk_task(tool, config={"name": "t2"})
    wf = WorkflowFactory.create_workflow(name="wf", tasks=[t1, t2])
    scheduler = PriorityScheduler(DefenseEvaluationPipeline(name="p", workflows=[wf]))

    assert len(scheduler.get_tasks_by_status(TaskStatus.PENDING)) == 2
    assert len(scheduler.get_tasks_by_status(TaskStatus.RUNNING)) == 0

    first = scheduler.get_next_task()
    assert first is not None
    assert len(scheduler.get_tasks_by_status(TaskStatus.RUNNING)) == 1
    assert len(scheduler.get_tasks_by_status(TaskStatus.PENDING)) == 1

    scheduler.update_task_status(first.id, TaskStatus.COMPLETED)
    assert len(scheduler.get_tasks_by_status(TaskStatus.COMPLETED)) == 1


def test_blocked_when_dependency_failed(tool):
    root = _mk_task(tool, config={"name": "root"})
    child = _mk_task(tool, config={"name": "child"}, deps=[root])
    wf = WorkflowFactory.create_workflow(name="wf", tasks=[root, child])
    scheduler = PriorityScheduler(DefenseEvaluationPipeline(name="p", workflows=[wf]))

    # root starts, then fails
    first = scheduler.get_next_task()
    assert first.id == root.id
    scheduler.update_task_status(root.id, TaskStatus.FAILED)

    # child must remain blocked forever
    assert scheduler.get_next_task() is None
    progress = scheduler.get_progress()
    assert progress["failed"] == 1
    assert progress["pending"] == 1
    assert scheduler.is_complete() is False


def test_is_complete_true_when_all_terminal_even_with_failures(tool):
    a = _mk_task(tool, config={"name": "a"})
    b = _mk_task(tool, config={"name": "b"})
    wf = WorkflowFactory.create_workflow(name="wf", tasks=[a, b])
    scheduler = PriorityScheduler(DefenseEvaluationPipeline(name="p", workflows=[wf]))

    first = scheduler.get_next_task()
    scheduler.update_task_status(first.id, TaskStatus.FAILED)
    second = scheduler.get_next_task()
    scheduler.update_task_status(second.id, TaskStatus.COMPLETED)

    assert scheduler.is_complete() is True
    progress = scheduler.get_progress()
    assert progress["failed"] == 1
    assert progress["completed"] == 1
    assert progress["pending"] == 0
    assert progress["running"] == 0


def test_get_ready_tasks_by_priority_excludes_running_and_blocked(tool):
    root = _mk_task(tool, config={"name": "root"})
    dep = _mk_task(tool, config={"name": "dep"}, deps=[root])
    indep = _mk_task(tool, config={"name": "indep"})
    wf = WorkflowFactory.create_workflow(name="wf", tasks=[root, dep, indep])
    scheduler = PriorityScheduler(DefenseEvaluationPipeline(name="p", workflows=[wf]))

    # Before dispatch: root + indep should be ready; dep blocked
    ready_ids = {t.id for t in scheduler.get_ready_tasks_by_priority()}
    assert root.id in ready_ids
    assert indep.id in ready_ids
    assert dep.id not in ready_ids

    # Dispatch one ready task -> RUNNING should disappear from ready list
    running = scheduler.get_next_task()
    ready_after = {t.id for t in scheduler.get_ready_tasks_by_priority()}
    assert running.id not in ready_after


def test_equal_priority_order_is_stable_by_insertion(tool):
    # Independent tasks at same depth and same counter can tie on priority.
    # Python sort is stable; scheduler should return in task insertion order.
    t1 = _mk_task(tool, config={"idx": 1})
    t2 = _mk_task(tool, config={"idx": 2})
    t3 = _mk_task(tool, config={"idx": 3})
    wf = WorkflowFactory.create_workflow(name="wf", tasks=[t1, t2, t3])
    scheduler = PriorityScheduler(DefenseEvaluationPipeline(name="p", workflows=[wf]))

    picked = [scheduler.get_next_task().id for _ in range(3)]
    assert picked == [t1.id, t2.id, t3.id]


def test_circular_dependency_raises_recursion_error_on_priority_update(tool):
    # Build tasks first
    a = _mk_task(tool, config={"name": "a"})
    b = _mk_task(tool, config={"name": "b"}, deps=[a])
    # introduce cycle a -> b
    a.add_dependency(b)

    wf = WorkflowFactory.create_workflow(name="wf", tasks=[a, b])

    with pytest.raises(RecursionError):
        PriorityScheduler(DefenseEvaluationPipeline(name="p", workflows=[wf]))

