"""Tests for multi-run scheduler (shared _all_tasks)."""

import pytest

from src.pipeline.tasks import (
    TaskStatus,
    TaskType,
    TaskFactory,
    clear_task_registry,
)
from src.pipeline.tools import ToolDefinition, ContainerConfig
from src.pipeline.workflow import WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline
from src.backend.scheduler.priority_scheduler import PriorityScheduler
from src.backend.api.state import SchedulerState


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_task_registry()
    yield
    clear_task_registry()


def _make_pipeline(name: str, run_id: str, tool_name: str) -> DefenseEvaluationPipeline:
    tool = ToolDefinition(
        name=tool_name,
        container=ContainerConfig(image="test/img:latest", command="python main.py"),
    )
    task = TaskFactory.create_task(
        task_type=TaskType.PRE_TRAINING,
        tool=tool,
        config={"stage": "pre_training", "tool_name": tool_name},
        priority=100,
    )
    task.pipeline_id = run_id
    task.run_id = run_id
    task.workflows = {"workflow_1"}

    workflow = WorkflowFactory.create_workflow(name="workflow_1", tasks=[task])
    workflow.id = f"wf_{run_id}"
    workflow.pipeline_id = run_id
    workflow.run_id = run_id

    pipeline = DefenseEvaluationPipeline(name=name, workflows=[workflow])
    pipeline.id = run_id
    return pipeline


def test_add_pipeline_merges_tasks_without_dropping_first():
    p1 = _make_pipeline("pipe_a", "run_a", "tool_a")
    p2 = _make_pipeline("pipe_b", "run_b", "tool_b")

    sched = PriorityScheduler(p1)
    assert len(sched.get_all_tasks()) == 1

    added = sched.add_pipeline(p2)
    assert added == 1
    assert len(sched.get_all_tasks()) == 2
    assert {t.run_id for t in sched.get_all_tasks()} == {"run_a", "run_b"}
    assert len(sched.get_tasks_for_run("run_a")) == 1
    assert len(sched.get_tasks_for_run("run_b")) == 1


def test_progress_and_complete_are_per_run():
    p1 = _make_pipeline("pipe_a", "run_a", "tool_a")
    p2 = _make_pipeline("pipe_b", "run_b", "tool_b")
    sched = PriorityScheduler(p1)
    sched.add_pipeline(p2)

    t_a = sched.get_tasks_for_run("run_a")[0]
    t_a.status = TaskStatus.COMPLETED

    assert sched.is_complete(run_id="run_a") is True
    assert sched.is_complete(run_id="run_b") is False
    assert sched.is_complete() is False

    prog_a = sched.get_progress(run_id="run_a")
    assert prog_a["completed"] == 1 and prog_a["total"] == 1
    prog_b = sched.get_progress(run_id="run_b")
    assert prog_b["pending"] == 1 and prog_b["total"] == 1


def test_task_id_collision_raises():
    p1 = _make_pipeline("pipe_a", "run_a", "tool_a")
    sched = PriorityScheduler(p1)
    # Force a different task object with the same id
    p2 = _make_pipeline("pipe_b", "run_b", "tool_b")
    p2.workflows[0].tasks[0].id = p1.workflows[0].tasks[0].id

    with pytest.raises(ValueError, match="Task ID collision"):
        sched.add_pipeline(p2)


def test_scheduler_state_add_run_merges():
    state = SchedulerState()
    p1 = _make_pipeline("pipe_a", "run_a", "tool_a")
    p2 = _make_pipeline("pipe_b", "run_b", "tool_b")

    n1 = state.add_run(p1)
    n2 = state.add_run(p2)
    assert n1 == 1
    assert n2 == 1
    assert len(state.scheduler.get_all_tasks()) == 2
    assert set(state.pipelines.keys()) == {"run_a", "run_b"}


def test_claim_can_pick_from_either_run():
    p1 = _make_pipeline("pipe_a", "run_a", "tool_a")
    p2 = _make_pipeline("pipe_b", "run_b", "tool_b")
    sched = PriorityScheduler(p1)
    sched.add_pipeline(p2)

    first = sched.get_next_task()
    second = sched.get_next_task()
    assert first is not None and second is not None
    assert {first.run_id, second.run_id} == {"run_a", "run_b"}
    assert sched.get_next_task() is None
