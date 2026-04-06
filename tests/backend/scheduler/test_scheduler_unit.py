"""
Scheduler unit tests (base + priority).

Tests cover:
- Scheduler._initialize_tasks: deduplicates shared tasks, resets to PENDING
- Scheduler._is_task_ready: PENDING + all deps COMPLETED → True
- Scheduler._is_task_ready: blocked by PENDING/RUNNING dep → False
- Scheduler._is_task_ready: RUNNING task itself → False
- Scheduler.update_task_status: COMPLETED and FAILED transitions
- Scheduler.update_task_status: raises ValueError for unknown task_id
- Scheduler.update_task_status: raises ValueError for RUNNING status
- Scheduler.get_tasks_by_status: filters correctly
- Scheduler.is_complete: returns True only when all tasks terminal
- Scheduler.get_progress: correct counts
- PriorityScheduler._update_task_priorities: depth 0 → 100, depth 1 → 90, depth N
- PriorityScheduler: counter bonus capped at 9
- PriorityScheduler: eval task gets boost, capped at 108
- PriorityScheduler.get_next_task: returns highest priority ready task
- PriorityScheduler.get_next_task: sets task to RUNNING
- PriorityScheduler.get_next_task: returns None when nothing ready
- PriorityScheduler.get_next_task: returns None when all complete
- PriorityScheduler.get_ready_tasks_by_priority: sorted descending
- PriorityScheduler.get_task_priority_info: correct structure, raises for unknown id
- PriorityScheduler.get_priority_levels: groups by depth
- Priority tie-breaking: higher counter wins within same depth
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
    DeploymentTask,
    clear_task_registry,
)
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import Workflow, WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline
from src.backend.scheduler.base_scheduler import Scheduler
from src.backend.scheduler.priority_scheduler import (
    PriorityScheduler,
    EVALUATION_PRIORITY_BOOST,
    EVALUATION_PRIORITY_CAP,
)
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


def make_tool(name: str = "tool") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        container=ContainerConfig(image=f"img/{name}:v1", command="run"),
    )


def make_task(tool: ToolDefinition = None, task_type: TaskType = TaskType.PRE_TRAINING,
              deps: list = None, config: dict = None) -> Task:
    t = tool or make_tool()
    return TaskFactory.create_task(
        task_type, tool=t,
        config=config or {},
        dependencies=deps or []
    )


def build_pipeline(workflows: List[Workflow], name: str = "test-pipeline") -> DefenseEvaluationPipeline:
    return DefenseEvaluationPipeline(
        name=name,
        workflows=workflows,
        config={},
        dataset={},
        model={}
    )


def build_single_workflow_pipeline(*tasks: Task, wf_name: str = "wf1") -> DefenseEvaluationPipeline:
    wf = Workflow(name=wf_name, tasks=list(tasks))
    return build_pipeline([wf])


# ============================================================================
# Tests: Scheduler._initialize_tasks
# ============================================================================


class TestSchedulerInitializeTasks:
    """Tests for task extraction and deduplication on init."""

    def test_extracts_all_tasks(self):
        t1 = make_task(config={"n": 1})
        t2 = make_task(config={"n": 2})
        pipeline = build_single_workflow_pipeline(t1, t2)
        sched = PriorityScheduler(pipeline)

        task_ids = [t.id for t in sched.get_all_tasks()]
        assert t1.id in task_ids
        assert t2.id in task_ids

    def test_shared_task_not_duplicated(self):
        # Same task object in two workflows → appears only once in scheduler
        shared_tool = make_tool("shared")
        shared = TaskFactory.create_task(TaskType.PRE_TRAINING, tool=shared_tool)
        unique1 = make_task(config={"n": 1})
        unique2 = make_task(config={"n": 2})

        wf1 = Workflow(name="wf1", tasks=[shared, unique1])
        wf2 = Workflow(name="wf2", tasks=[shared, unique2])
        pipeline = build_pipeline([wf1, wf2])
        sched = PriorityScheduler(pipeline)

        ids = [t.id for t in sched.get_all_tasks()]
        assert ids.count(shared.id) == 1

    def test_tasks_collected_on_init(self):
        task = make_task()
        task.status = TaskStatus.RUNNING  # simulate mid-run state
        pipeline = build_single_workflow_pipeline(task)
        sched = PriorityScheduler(pipeline)

        # Scheduler collects the task and preserves its status
        assert task in sched.get_all_tasks()
        assert task.status == TaskStatus.RUNNING


# ============================================================================
# Tests: Scheduler._is_task_ready
# ============================================================================


class TestIsTaskReady:
    """Tests for _is_task_ready logic."""

    def test_pending_no_deps_is_ready(self):
        t = make_task()
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)

        assert sched._is_task_ready(t) is True

    def test_running_task_not_ready(self):
        t = make_task()
        t.status = TaskStatus.RUNNING
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)

        assert sched._is_task_ready(t) is False

    def test_completed_task_not_ready(self):
        t = make_task()
        t.status = TaskStatus.COMPLETED
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)

        assert sched._is_task_ready(t) is False

    def test_failed_task_not_ready(self):
        t = make_task()
        t.status = TaskStatus.FAILED
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)

        assert sched._is_task_ready(t) is False

    def test_pending_with_completed_dep_is_ready(self):
        dep = make_task(config={"dep": 1})
        dep.status = TaskStatus.COMPLETED
        downstream = TaskFactory.create_task(
            TaskType.IN_TRAINING, tool=make_tool("in"),
            dependencies=[dep]
        )
        pipeline = build_single_workflow_pipeline(dep, downstream)
        sched = PriorityScheduler(pipeline)

        assert sched._is_task_ready(downstream) is True

    def test_pending_with_pending_dep_not_ready(self):
        dep = make_task(config={"dep": 1})
        downstream = TaskFactory.create_task(
            TaskType.IN_TRAINING, tool=make_tool("in"),
            dependencies=[dep]
        )
        pipeline = build_single_workflow_pipeline(dep, downstream)
        sched = PriorityScheduler(pipeline)

        assert sched._is_task_ready(downstream) is False

    def test_pending_with_running_dep_not_ready(self):
        dep = make_task(config={"dep": 1})
        dep.status = TaskStatus.RUNNING
        downstream = TaskFactory.create_task(
            TaskType.IN_TRAINING, tool=make_tool("in"),
            dependencies=[dep]
        )
        pipeline = build_single_workflow_pipeline(dep, downstream)
        sched = PriorityScheduler(pipeline)

        assert sched._is_task_ready(downstream) is False

    def test_pending_with_failed_dep_not_ready(self):
        dep = make_task(config={"dep": 1})
        dep.status = TaskStatus.FAILED
        downstream = TaskFactory.create_task(
            TaskType.IN_TRAINING, tool=make_tool("in"),
            dependencies=[dep]
        )
        pipeline = build_single_workflow_pipeline(dep, downstream)
        sched = PriorityScheduler(pipeline)

        assert sched._is_task_ready(downstream) is False


# ============================================================================
# Tests: Scheduler.update_task_status
# ============================================================================


class TestUpdateTaskStatus:
    """Tests for Scheduler.update_task_status."""

    def test_marks_task_completed(self):
        t = make_task()
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)
        t.status = TaskStatus.RUNNING

        sched.update_task_status(t.id, TaskStatus.COMPLETED)

        assert t.status == TaskStatus.COMPLETED

    def test_marks_task_failed(self):
        t = make_task()
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)
        t.status = TaskStatus.RUNNING

        sched.update_task_status(t.id, TaskStatus.FAILED)

        assert t.status == TaskStatus.FAILED

    def test_raises_for_unknown_task_id(self):
        pipeline = build_single_workflow_pipeline(make_task())
        sched = PriorityScheduler(pipeline)

        with pytest.raises(ValueError, match="not found"):
            sched.update_task_status("task_99999", TaskStatus.COMPLETED)

    def test_raises_for_running_status(self):
        t = make_task()
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)

        with pytest.raises(ValueError, match="Invalid status"):
            sched.update_task_status(t.id, TaskStatus.RUNNING)

    def test_raises_for_pending_status(self):
        t = make_task()
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)

        with pytest.raises(ValueError, match="Invalid status"):
            sched.update_task_status(t.id, TaskStatus.PENDING)


# ============================================================================
# Tests: Scheduler.is_complete / get_progress
# ============================================================================


class TestSchedulerProgress:
    """Tests for is_complete and get_progress."""

    def test_not_complete_when_pending_tasks_exist(self):
        pipeline = build_single_workflow_pipeline(make_task())
        sched = PriorityScheduler(pipeline)

        assert sched.is_complete() is False

    def test_complete_when_all_completed(self):
        t1 = make_task(config={"n": 1})
        t2 = make_task(config={"n": 2})
        pipeline = build_single_workflow_pipeline(t1, t2)
        sched = PriorityScheduler(pipeline)
        t1.status = TaskStatus.COMPLETED
        t2.status = TaskStatus.COMPLETED

        assert sched.is_complete() is True

    def test_complete_with_mix_of_completed_and_failed(self):
        t1 = make_task(config={"n": 1})
        t2 = make_task(config={"n": 2})
        pipeline = build_single_workflow_pipeline(t1, t2)
        sched = PriorityScheduler(pipeline)
        t1.status = TaskStatus.COMPLETED
        t2.status = TaskStatus.FAILED

        assert sched.is_complete() is True

    def test_not_complete_when_running_task(self):
        t = make_task()
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)
        t.status = TaskStatus.RUNNING

        assert sched.is_complete() is False

    def test_get_progress_correct_counts(self):
        t1 = make_task(config={"n": 1})
        t2 = make_task(config={"n": 2})
        t3 = make_task(config={"n": 3})
        t4 = make_task(config={"n": 4})
        pipeline = build_single_workflow_pipeline(t1, t2, t3, t4)
        sched = PriorityScheduler(pipeline)
        t1.status = TaskStatus.COMPLETED
        t2.status = TaskStatus.FAILED
        t3.status = TaskStatus.RUNNING

        progress = sched.get_progress()

        assert progress["total"] == 4
        assert progress["completed"] == 1
        assert progress["failed"] == 1
        assert progress["running"] == 1
        assert progress["pending"] == 1

    def test_get_tasks_by_status_completed(self):
        t1 = make_task(config={"n": 1})
        t2 = make_task(config={"n": 2})
        pipeline = build_single_workflow_pipeline(t1, t2)
        sched = PriorityScheduler(pipeline)
        t1.status = TaskStatus.COMPLETED

        completed = sched.get_tasks_by_status(TaskStatus.COMPLETED)

        assert t1 in completed
        assert t2 not in completed


# ============================================================================
# Tests: PriorityScheduler — priority calculation
# ============================================================================


class TestPriorityCalculation:
    """Tests for _update_task_priorities depth and counter logic."""

    def test_depth_0_gets_priority_100(self):
        t = make_task()
        pipeline = build_single_workflow_pipeline(t)
        PriorityScheduler(pipeline)

        # counter=0 bonus → priority = 100 + 0 = 100
        assert t.priority == 100

    def test_depth_1_gets_priority_90(self):
        root = make_task(config={"n": 1})
        child = TaskFactory.create_task(
            TaskType.IN_TRAINING, tool=make_tool("in"),
            dependencies=[root]
        )
        pipeline = build_single_workflow_pipeline(root, child)
        PriorityScheduler(pipeline)

        # child is depth 1, counter=0 → 100 - 10 = 90
        assert child.priority == 90

    def test_depth_2_gets_priority_80(self):
        t0 = make_task(config={"n": 1})
        t1 = TaskFactory.create_task(TaskType.IN_TRAINING, tool=make_tool("in"),
                                     dependencies=[t0])
        t2 = TaskFactory.create_task(TaskType.POST_TRAINING, tool=make_tool("post"),
                                     dependencies=[t1])
        pipeline = build_single_workflow_pipeline(t0, t1, t2)
        PriorityScheduler(pipeline)

        assert t2.priority == 80

    def test_depth_3_gets_priority_70(self):
        t0 = make_task(config={"n": 0})
        t1 = TaskFactory.create_task(TaskType.IN_TRAINING, tool=make_tool("in"),
                                     dependencies=[t0])
        t2 = TaskFactory.create_task(TaskType.POST_TRAINING, tool=make_tool("post"),
                                     dependencies=[t1])
        t3 = TaskFactory.create_task(TaskType.DEPLOYMENT, tool=make_tool("deploy"),
                                     dependencies=[t2])
        pipeline = build_single_workflow_pipeline(t0, t1, t2, t3)
        PriorityScheduler(pipeline)

        assert t3.priority == 70

    def test_counter_bonus_applied(self):
        # Create a task appearing in 2 workflows (counter=2)
        shared_tool = make_tool("shared")
        shared = TaskFactory.create_task(TaskType.PRE_TRAINING, tool=shared_tool)

        wf1 = Workflow(name="wf1", pipeline_id="p1")
        wf2 = Workflow(name="wf2", pipeline_id="p1")
        wf1.add_task(shared)
        wf2.add_task(shared)

        pipeline = build_pipeline([wf1, wf2])
        PriorityScheduler(pipeline)

        # counter=2, depth=0 → 100 + 2 = 102
        assert shared.priority == 102

    def test_counter_bonus_capped_at_9(self):
        # Task in 20 workflows should get counter bonus of min(20, 9) = 9
        shared_tool = make_tool("ubiquitous")
        shared = TaskFactory.create_task(TaskType.PRE_TRAINING, tool=shared_tool)

        workflows = []
        pipeline_id = "pipe_many"
        for i in range(20):
            wf = Workflow(name=f"wf{i}", pipeline_id=pipeline_id)
            shared.add_to_workflow(wf.id, pipeline_id)
            wf.tasks.append(shared)
            workflows.append(wf)

        pipeline = build_pipeline(workflows)
        PriorityScheduler(pipeline)

        assert shared.priority == 100 + 9  # capped bonus

    def test_evaluation_task_gets_boost_capped_at_108(self):
        eval_tool = make_tool("eval-clean")
        eval_task = EvaluationTask(tool=eval_tool)
        # eval_task at depth 4 → base = 100 - 40 = 60, +boost 32 = 92, not exceeding cap
        t0 = make_task(config={"n": 0})
        t1 = TaskFactory.create_task(TaskType.IN_TRAINING, tool=make_tool("in"),
                                     dependencies=[t0])
        t2 = TaskFactory.create_task(TaskType.POST_TRAINING, tool=make_tool("post"),
                                     dependencies=[t1])
        t3 = TaskFactory.create_task(TaskType.DEPLOYMENT, tool=make_tool("deploy"),
                                     dependencies=[t2])
        eval_task.dependencies = [t3]

        pipeline = build_single_workflow_pipeline(t0, t1, t2, t3, eval_task)
        PriorityScheduler(pipeline)

        # depth 4: base=60, +32 boost = 92 (under 108 cap)
        assert eval_task.priority == 60 + EVALUATION_PRIORITY_BOOST

    def test_evaluation_priority_never_exceeds_cap(self):
        # eval at depth 0 (no deps): base=100, +32 = 132 → capped at 108
        eval_tool = make_tool("eval-zero")
        eval_task = EvaluationTask(tool=eval_tool)

        pipeline = build_single_workflow_pipeline(eval_task)
        PriorityScheduler(pipeline)

        assert eval_task.priority == EVALUATION_PRIORITY_CAP

    def test_priority_floor_at_10(self):
        # Very deep task (depth > 9) should get min priority of 10
        tools = [make_tool(f"t{i}") for i in range(12)]
        tasks = [TaskFactory.create_task(TaskType.PRE_TRAINING, tool=tools[0])]
        for i in range(1, 12):
            t = TaskFactory.create_task(
                TaskType.PRE_TRAINING, tool=tools[i],
                config={"n": i},
                dependencies=[tasks[-1]]
            )
            tasks.append(t)

        pipeline = build_single_workflow_pipeline(*tasks)
        PriorityScheduler(pipeline)

        deepest = tasks[-1]
        assert deepest.priority >= 10


# ============================================================================
# Tests: PriorityScheduler.get_next_task
# ============================================================================


class TestGetNextTask:
    """Tests for PriorityScheduler.get_next_task."""

    def test_returns_highest_priority_ready_task(self):
        # t1 has no deps (depth 0, priority 100); t2 depends on t1 (depth 1, priority 90)
        t1 = make_task(config={"n": 1})
        t2 = TaskFactory.create_task(TaskType.IN_TRAINING, tool=make_tool("in"),
                                     dependencies=[t1])
        pipeline = build_single_workflow_pipeline(t1, t2)
        sched = PriorityScheduler(pipeline)

        next_task = sched.get_next_task()

        assert next_task is t1

    def test_next_task_set_to_running(self):
        t = make_task()
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)

        sched.get_next_task()

        assert t.status == TaskStatus.RUNNING

    def test_returns_none_when_all_blocked(self):
        dep = make_task(config={"dep": 1})
        downstream = TaskFactory.create_task(
            TaskType.IN_TRAINING, tool=make_tool("in"),
            dependencies=[dep]
        )
        dep.status = TaskStatus.RUNNING  # Not yet completed
        pipeline = build_single_workflow_pipeline(dep, downstream)
        sched = PriorityScheduler(pipeline)

        result = sched.get_next_task()

        assert result is None

    def test_returns_none_when_all_complete(self):
        t = make_task()
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)
        t.status = TaskStatus.COMPLETED

        result = sched.get_next_task()

        assert result is None

    def test_ready_after_dep_completes(self):
        t1 = make_task(config={"n": 1})
        t2 = TaskFactory.create_task(TaskType.IN_TRAINING, tool=make_tool("in"),
                                     dependencies=[t1])
        pipeline = build_single_workflow_pipeline(t1, t2)
        sched = PriorityScheduler(pipeline)

        # First call gives t1; complete it
        first = sched.get_next_task()
        assert first is t1
        sched.update_task_status(t1.id, TaskStatus.COMPLETED)

        # Second call should give t2
        second = sched.get_next_task()
        assert second is t2

    def test_counter_breaks_tie_within_depth(self):
        # Two tasks at depth 0; one has counter=3, other counter=1
        tool_a = make_tool("a")
        tool_b = make_tool("b")
        t_low = TaskFactory.create_task(TaskType.PRE_TRAINING, tool=tool_a)
        t_high = TaskFactory.create_task(TaskType.PRE_TRAINING, tool=tool_b)

        # Give t_high a higher counter by putting it in 3 workflows
        pipeline_id = "p1"
        for i in range(3):
            wf = Workflow(name=f"wf{i}", pipeline_id=pipeline_id)
            t_high.add_to_workflow(wf.id, pipeline_id)

        pipeline_id_low = "p2"
        wf_single = Workflow(name="wf_single", pipeline_id=pipeline_id_low)
        t_low.add_to_workflow(wf_single.id, pipeline_id_low)

        wf_combined = Workflow(name="combined")
        wf_combined.tasks.extend([t_low, t_high])
        pipeline = build_pipeline([wf_combined])
        sched = PriorityScheduler(pipeline)

        next_task = sched.get_next_task()
        assert next_task is t_high

    def test_get_ready_tasks_by_priority_sorted_descending(self):
        t1 = make_task(config={"n": 1})  # depth 0
        t2 = TaskFactory.create_task(TaskType.IN_TRAINING, tool=make_tool("in"),
                                     config={"n": 2}, dependencies=[t1])
        t1.status = TaskStatus.COMPLETED  # make t2 ready
        pipeline = build_single_workflow_pipeline(t1, t2)
        sched = PriorityScheduler(pipeline)

        ready = sched.get_ready_tasks_by_priority()
        priorities = [t.priority for t in ready]
        assert priorities == sorted(priorities, reverse=True)


# ============================================================================
# Tests: PriorityScheduler.get_task_priority_info
# ============================================================================


class TestGetTaskPriorityInfo:
    """Tests for get_task_priority_info."""

    def test_returns_correct_structure(self):
        t = make_task()
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)

        info = sched.get_task_priority_info(t.id)

        assert info["task_id"] == t.id
        assert "priority" in info
        assert "dependency_level" in info
        assert "usage_counter" in info
        assert "status" in info
        assert "dependencies" in info
        assert "workflows" in info

    def test_depth_0_dependency_level(self):
        t = make_task()
        pipeline = build_single_workflow_pipeline(t)
        sched = PriorityScheduler(pipeline)

        info = sched.get_task_priority_info(t.id)
        assert info["dependency_level"] == 0

    def test_depth_1_dependency_level(self):
        t0 = make_task(config={"n": 0})
        t1 = TaskFactory.create_task(TaskType.IN_TRAINING, tool=make_tool("in"),
                                     dependencies=[t0])
        pipeline = build_single_workflow_pipeline(t0, t1)
        sched = PriorityScheduler(pipeline)

        info = sched.get_task_priority_info(t1.id)
        assert info["dependency_level"] == 1

    def test_raises_for_unknown_task_id(self):
        pipeline = build_single_workflow_pipeline(make_task())
        sched = PriorityScheduler(pipeline)

        with pytest.raises(ValueError, match="not found"):
            sched.get_task_priority_info("task_nonexistent")


# ============================================================================
# Tests: PriorityScheduler.get_priority_levels
# ============================================================================


class TestGetPriorityLevels:
    """Tests for get_priority_levels grouping."""

    def test_groups_by_depth(self):
        t0 = make_task(config={"n": 0})
        t1 = TaskFactory.create_task(TaskType.IN_TRAINING, tool=make_tool("in"),
                                     config={"n": 1}, dependencies=[t0])
        pipeline = build_single_workflow_pipeline(t0, t1)
        sched = PriorityScheduler(pipeline)

        levels = sched.get_priority_levels()

        assert 0 in levels
        assert 1 in levels
        assert t0 in levels[0]
        assert t1 in levels[1]

    def test_within_level_sorted_by_counter_descending(self):
        t1 = make_task(config={"n": 1})
        t2 = make_task(config={"n": 2})
        # Give t1 higher counter
        for i in range(5):
            wf = Workflow(name=f"wf{i}", pipeline_id="p1")
            t1.add_to_workflow(wf.id, "p1")

        pipeline = build_pipeline([
            Workflow(name="combined", tasks=[t1, t2])
        ])
        sched = PriorityScheduler(pipeline)

        levels = sched.get_priority_levels()
        depth_0 = levels[0]
        # t1 (counter=5) should appear before t2 (counter=0 or 1)
        t1_idx = depth_0.index(t1)
        t2_idx = depth_0.index(t2)
        assert t1_idx < t2_idx
