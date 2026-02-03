"""
Tests for priority calculation in PriorityScheduler.

These tests verify priority calculation as specified in:
- docs/OVERVIEWv1.md: 100 for 0 deps, 90 for 1 dep, 80 for 2 deps, etc.
- docs/SCHED2.md: Tasks with more use counter have higher priority at same level
- docs/Tasks.md: Specific priority examples for workflows
"""

import pytest
from typing import List

from src.pipeline.tasks import (
    Task,
    TaskStatus,
    TaskType,
    TaskFactory,
    clear_task_registry,
)
from src.pipeline.tools import ToolDefinition, ContainerConfig
from src.pipeline.workflow import Workflow, WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline
from src.backend.scheduler.priority_scheduler import PriorityScheduler


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tool_a():
    """Tool A - baseline pre_training."""
    return ToolDefinition(
        name="A",
        container=ContainerConfig(image="test/a:v1", command="python run.py"),
        is_baseline=True
    )


@pytest.fixture
def tool_b():
    """Tool B - actual pre_training."""
    return ToolDefinition(
        name="B",
        container=ContainerConfig(image="test/b:v1", command="python run.py"),
        is_baseline=False
    )


@pytest.fixture
def tool_c():
    """Tool C - during_training."""
    return ToolDefinition(
        name="C",
        container=ContainerConfig(image="test/c:v1", command="python run.py"),
        is_baseline=True
    )


@pytest.fixture
def tool_e():
    """Tool E - post_training."""
    return ToolDefinition(
        name="E",
        container=ContainerConfig(image="test/e:v1", command="python run.py"),
        is_baseline=True
    )


@pytest.fixture
def tool_g():
    """Tool G - deployment."""
    return ToolDefinition(
        name="G",
        container=ContainerConfig(image="test/g:v1", command="python run.py"),
        is_baseline=True
    )


@pytest.fixture(autouse=True)
def cleanup_task_registry():
    """Clear task registry before each test."""
    clear_task_registry()
    yield
    clear_task_registry()


def create_task_with_deps(
    tool: ToolDefinition,
    task_type: TaskType,
    dependencies: List[Task] = None
) -> Task:
    """Helper to create a task with specified dependencies."""
    return TaskFactory.create_task(
        task_type=task_type,
        tool=tool,
        config={"tool_name": tool.name},
        dependencies=dependencies or []
    )


# ============================================================================
# Test: Priority Based on Dependency Depth (OVERVIEWv1.md)
# ============================================================================


class TestPriorityByDependencyDepth:
    """
    Tests for priority calculation based on dependency depth.
    
    From OVERVIEWv1.md:
    - 100 for tools with 0 dependency
    - 90 for tools with 1 dependency
    - 80 for tools with 2 dependencies
    - etc.
    """
    
    def test_depth_0_gets_priority_100(self, tool_a):
        """Tasks with no dependencies should have priority ~100."""
        task_a = create_task_with_deps(tool_a, TaskType.PRE_TRAINING)
        
        workflow = WorkflowFactory.create_workflow(name="test", tasks=[task_a])
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        PriorityScheduler(pipeline)
        
        # Depth 0: priority 100 + counter bonus (0-9)
        assert 100 <= task_a.priority <= 109
    
    def test_depth_1_gets_priority_90(self, tool_a, tool_b):
        """Tasks at depth 1 should have priority ~90."""
        task_a = create_task_with_deps(tool_a, TaskType.PRE_TRAINING)
        task_b = create_task_with_deps(
            tool_b, TaskType.PRE_TRAINING, [task_a]
        )
        
        workflow = WorkflowFactory.create_workflow(
            name="test", tasks=[task_a, task_b]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        PriorityScheduler(pipeline)
        
        # Depth 1: priority 90 + counter bonus
        assert 90 <= task_b.priority <= 99
    
    def test_depth_2_gets_priority_80(self, tool_a, tool_b, tool_c):
        """Tasks at depth 2 should have priority ~80."""
        task_a = create_task_with_deps(tool_a, TaskType.PRE_TRAINING)
        task_b = create_task_with_deps(
            tool_b, TaskType.PRE_TRAINING, [task_a]
        )
        task_c = create_task_with_deps(
            tool_c, TaskType.IN_TRAINING, [task_b]
        )
        
        workflow = WorkflowFactory.create_workflow(
            name="test", tasks=[task_a, task_b, task_c]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        PriorityScheduler(pipeline)
        
        # Depth 2: priority 80 + counter bonus
        assert 80 <= task_c.priority <= 89
    
    def test_priority_decreases_by_10_per_depth(self, tool_a, tool_b, tool_c, tool_e):
        """Priority should decrease by ~10 for each depth level."""
        task_a = create_task_with_deps(tool_a, TaskType.PRE_TRAINING)
        task_b = create_task_with_deps(
            tool_b, TaskType.PRE_TRAINING, [task_a]
        )
        task_c = create_task_with_deps(
            tool_c, TaskType.IN_TRAINING, [task_b]
        )
        task_e = create_task_with_deps(
            tool_e, TaskType.POST_TRAINING, [task_c]
        )
        
        workflow = WorkflowFactory.create_workflow(
            name="test", tasks=[task_a, task_b, task_c, task_e]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        PriorityScheduler(pipeline)
        
        # Each level should be ~10 points apart
        assert task_a.priority > task_b.priority
        assert task_b.priority > task_c.priority
        assert task_c.priority > task_e.priority
        
        # Verify approximate 10-point gaps
        gap_ab = task_a.priority - task_b.priority
        gap_bc = task_b.priority - task_c.priority
        gap_ce = task_c.priority - task_e.priority
        
        # Allow for counter bonus variance (0-9)
        assert 1 <= gap_ab <= 19
        assert 1 <= gap_bc <= 19
        assert 1 <= gap_ce <= 19


# ============================================================================
# Test: Priority Example from Tasks.md
# ============================================================================


class TestTasksMdPriorityExample:
    """
    Tests for the specific priority example in Tasks.md.
    
    From Tasks.md:
    - For workflow A->B->C->E->G: A=100, B=90, C=80, E=70, G=60
    - For workflow B->C->E->G: B=100, C=90, E=80, G=70
    """
    
    def test_priority_for_5_task_workflow(
        self, tool_a, tool_b, tool_c, tool_e, tool_g
    ):
        """
        For A->B->C->E->G:
        A: 100, B: 90, C: 80, E: 70, G: 60
        """
        task_a = create_task_with_deps(tool_a, TaskType.PRE_TRAINING)
        task_b = create_task_with_deps(
            tool_b, TaskType.PRE_TRAINING, [task_a]
        )
        task_c = create_task_with_deps(
            tool_c, TaskType.IN_TRAINING, [task_b]
        )
        task_e = create_task_with_deps(
            tool_e, TaskType.POST_TRAINING, [task_c]
        )
        task_g = create_task_with_deps(
            tool_g, TaskType.DEPLOYMENT, [task_e]
        )
        
        workflow = WorkflowFactory.create_workflow(
            name="A_B_C_E_G",
            tasks=[task_a, task_b, task_c, task_e, task_g]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        PriorityScheduler(pipeline)
        
        # Verify priority levels
        assert 100 <= task_a.priority <= 109  # Depth 0
        assert 90 <= task_b.priority <= 99    # Depth 1
        assert 80 <= task_c.priority <= 89    # Depth 2
        assert 70 <= task_e.priority <= 79    # Depth 3
        assert 60 <= task_g.priority <= 69    # Depth 4
    
    def test_priority_for_4_task_workflow(self, tool_b, tool_c, tool_e, tool_g):
        """
        For B->C->E->G:
        B: 100, C: 90, E: 80, G: 70
        """
        task_b = create_task_with_deps(tool_b, TaskType.PRE_TRAINING)
        task_c = create_task_with_deps(
            tool_c, TaskType.IN_TRAINING, [task_b]
        )
        task_e = create_task_with_deps(
            tool_e, TaskType.POST_TRAINING, [task_c]
        )
        task_g = create_task_with_deps(
            tool_g, TaskType.DEPLOYMENT, [task_e]
        )
        
        workflow = WorkflowFactory.create_workflow(
            name="B_C_E_G",
            tasks=[task_b, task_c, task_e, task_g]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        PriorityScheduler(pipeline)
        
        # Verify priority levels - B is now depth 0
        assert 100 <= task_b.priority <= 109  # Depth 0
        assert 90 <= task_c.priority <= 99    # Depth 1
        assert 80 <= task_e.priority <= 89    # Depth 2
        assert 70 <= task_g.priority <= 79    # Depth 3
    
    def test_same_tool_different_priority_based_on_position(
        self, tool_b, tool_c, tool_e, tool_g, tool_a
    ):
        """
        Same tool can have different priorities based on its position.
        
        In workflow 1 (A->B->C): B has depth 1, priority ~90
        In workflow 2 (B->C): B has depth 0, priority ~100
        
        Since task deduplication is based on tool+deps, these are different tasks.
        """
        # Workflow 1: A->B (B depends on A)
        task_a = create_task_with_deps(tool_a, TaskType.PRE_TRAINING)
        task_b_after_a = create_task_with_deps(
            tool_b, TaskType.PRE_TRAINING, [task_a]
        )
        
        # Workflow 2: B first (no dependencies)
        task_b_first = create_task_with_deps(tool_b, TaskType.PRE_TRAINING)
        
        workflow1 = WorkflowFactory.create_workflow(
            name="wf1", tasks=[task_a, task_b_after_a]
        )
        workflow2 = WorkflowFactory.create_workflow(
            name="wf2", tasks=[task_b_first]
        )
        
        pipeline = DefenseEvaluationPipeline(
            name="test", workflows=[workflow1, workflow2]
        )
        
        PriorityScheduler(pipeline)
        
        # B after A has depth 1
        assert 90 <= task_b_after_a.priority <= 99
        
        # B first has depth 0
        assert 100 <= task_b_first.priority <= 109


# ============================================================================
# Test: Usage Counter Effect on Priority (SCHED2.md)
# ============================================================================


class TestUsageCounterPriority:
    """
    Tests for usage counter effect on priority.
    
    From SCHED2.md:
    Tasks with more use counter should have higher priority at same level.
    """
    
    def test_higher_counter_gives_higher_priority_at_same_depth(self):
        """Tasks with higher counter should have slightly higher priority."""
        tool = ToolDefinition(
            name="shared",
            container=ContainerConfig(image="test:v1", command="run")
        )
        
        # Create two tasks at same depth (no deps)
        task_low = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool,
            config={"name": "low_usage"}
        )
        task_high = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool,
            config={"name": "high_usage"}
        )
        
        # Create workflows to increase counter for high_usage
        workflow1 = WorkflowFactory.create_workflow(
            name="wf1", tasks=[task_low, task_high]
        )
        workflow2 = WorkflowFactory.create_workflow(
            name="wf2", tasks=[task_high]
        )
        workflow3 = WorkflowFactory.create_workflow(
            name="wf3", tasks=[task_high]
        )
        
        pipeline = DefenseEvaluationPipeline(
            name="test",
            workflows=[workflow1, workflow2, workflow3]
        )
        
        PriorityScheduler(pipeline)
        
        # task_high has higher counter
        assert task_high.counter > task_low.counter
        
        # task_high should have higher priority
        assert task_high.priority > task_low.priority
    
    def test_counter_bonus_capped_to_prevent_level_crossing(self):
        """
        Counter bonus should be capped so tasks don't jump priority levels.
        A depth-1 task should never have higher priority than depth-0.
        """
        tool = ToolDefinition(
            name="test",
            container=ContainerConfig(image="test:v1", command="run")
        )
        
        task_depth_0 = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool,
            config={"name": "depth_0"}
        )
        task_depth_1 = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool,
            config={"name": "depth_1"},
            dependencies=[task_depth_0]
        )
        
        # Create many workflows with task_depth_1 to increase its counter
        workflows = [
            WorkflowFactory.create_workflow(
                name=f"wf_{i}",
                tasks=[task_depth_0, task_depth_1]
            )
            for i in range(20)
        ]
        
        pipeline = DefenseEvaluationPipeline(name="test", workflows=workflows)
        
        PriorityScheduler(pipeline)
        
        # Even with high counter, depth-1 should not exceed depth-0 priority
        assert task_depth_0.priority > task_depth_1.priority
    
    def test_depth_0_task_with_20_workflows_has_priority_109(self):
        """
        Specific test for the scenario: depth-0 task with counter=20 should have priority 109.
        
        Priority calculation:
        - Base priority = 100 - (depth * 10) = 100 - 0 = 100
        - Counter bonus = min(counter, 9) = min(20, 9) = 9
        - Total priority = 100 + 9 = 109
        
        This verifies the exact scenario from the UI where a task with Usage Count 20
        and no dependencies (depth 0) has Priority 109.
        """
        tool = ToolDefinition(
            name="pre-xgbod",
            container=ContainerConfig(image="test/pre-xgbod:v1", command="run")
        )
        
        # Create a task with no dependencies (depth 0)
        task = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool,
            config={"stage": "pre_training", "tool_name": "pre-xgbod"}
        )
        
        # Create 20 workflows that all use this task
        workflows = [
            WorkflowFactory.create_workflow(
                name=f"workflow_{i}",
                tasks=[task]
            )
            for i in range(20)
        ]
        
        pipeline = DefenseEvaluationPipeline(name="test", workflows=workflows)
        
        scheduler = PriorityScheduler(pipeline)
        
        # Verify counter is 20
        assert task.counter == 20, f"Expected counter=20, got {task.counter}"
        
        # Verify priority calculation
        # Depth 0: base_priority = 100
        # Counter 20: bonus = min(20, 9) = 9
        # Total = 100 + 9 = 109
        expected_priority = 100 + min(20, 9)  # 100 + 9 = 109
        assert task.priority == expected_priority, \
            f"Expected priority={expected_priority} for depth-0 task with counter=20, " \
            f"got priority={task.priority}"
        
        # Verify it's exactly 109
        assert task.priority == 109, \
            f"Priority should be exactly 109 for depth-0 task with counter=20, " \
            f"got {task.priority}"
    
    def test_counter_bonus_capped_at_9(self):
        """
        Verify that counter bonus is capped at 9, regardless of how many workflows.
        
        A task used in 9, 10, 20, or 100 workflows should all get the same bonus of 9.
        """
        tool = ToolDefinition(
            name="shared_task",
            container=ContainerConfig(image="test/shared:v1", command="run")
        )
        
        # Create tasks with different workflow counts
        task_9 = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool,
            config={"count": 9}
        )
        task_10 = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool,
            config={"count": 10}
        )
        task_20 = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool,
            config={"count": 20}
        )
        task_100 = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool,
            config={"count": 100}
        )
        
        # Create workflows for each task
        workflows = []
        for i in range(9):
            workflows.append(WorkflowFactory.create_workflow(
                name=f"wf_9_{i}", tasks=[task_9]
            ))
        for i in range(10):
            workflows.append(WorkflowFactory.create_workflow(
                name=f"wf_10_{i}", tasks=[task_10]
            ))
        for i in range(20):
            workflows.append(WorkflowFactory.create_workflow(
                name=f"wf_20_{i}", tasks=[task_20]
            ))
        for i in range(100):
            workflows.append(WorkflowFactory.create_workflow(
                name=f"wf_100_{i}", tasks=[task_100]
            ))
        
        pipeline = DefenseEvaluationPipeline(name="test", workflows=workflows)
        
        PriorityScheduler(pipeline)
        
        # All should have same priority (depth 0, counter bonus capped at 9)
        # Priority = 100 + min(counter, 9) = 100 + 9 = 109
        assert task_9.priority == 109
        assert task_10.priority == 109
        assert task_20.priority == 109
        assert task_100.priority == 109
        
        # Verify counters are different
        assert task_9.counter == 9
        assert task_10.counter == 10
        assert task_20.counter == 20
        assert task_100.counter == 100


# ============================================================================
# Test: Priority Levels Grouping
# ============================================================================


class TestPriorityLevels:
    """Tests for get_priority_levels functionality."""
    
    def test_get_priority_levels_groups_by_depth(
        self, tool_a, tool_b, tool_c, tool_e
    ):
        """Tasks should be grouped by their dependency depth."""
        # Create diamond dependency graph
        #     A
        #    / \
        #   B   C
        #    \ /
        #     E
        task_a = create_task_with_deps(tool_a, TaskType.PRE_TRAINING)
        task_b = create_task_with_deps(
            tool_b, TaskType.PRE_TRAINING, [task_a]
        )
        task_c = create_task_with_deps(
            tool_c, TaskType.IN_TRAINING, [task_a]
        )
        task_e = create_task_with_deps(
            tool_e, TaskType.POST_TRAINING, [task_b, task_c]
        )
        
        workflow = WorkflowFactory.create_workflow(
            name="diamond",
            tasks=[task_a, task_b, task_c, task_e]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        levels = scheduler.get_priority_levels()
        
        # Level 0: A
        assert task_a in levels[0]
        
        # Level 1: B and C (both depend on A)
        assert task_b in levels[1]
        assert task_c in levels[1]
        
        # Level 2: E (depends on B and C)
        assert task_e in levels[2]
    
    def test_tasks_within_level_sorted_by_counter(self):
        """Within a level, tasks should be sorted by counter descending."""
        tool = ToolDefinition(
            name="test",
            container=ContainerConfig(image="test:v1", command="run")
        )
        
        task_low = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool,
            config={"name": "low"}
        )
        task_mid = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool,
            config={"name": "mid"}
        )
        task_high = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool,
            config={"name": "high"}
        )
        
        # Different workflow counts
        wf1 = WorkflowFactory.create_workflow(
            name="wf1", tasks=[task_low, task_mid, task_high]
        )
        wf2 = WorkflowFactory.create_workflow(
            name="wf2", tasks=[task_mid, task_high]
        )
        wf3 = WorkflowFactory.create_workflow(
            name="wf3", tasks=[task_high]
        )
        
        pipeline = DefenseEvaluationPipeline(
            name="test", workflows=[wf1, wf2, wf3]
        )
        
        scheduler = PriorityScheduler(pipeline)
        levels = scheduler.get_priority_levels()
        
        # All at level 0, sorted by counter
        level_0 = levels[0]
        counters = [t.counter for t in level_0]
        
        # Should be sorted descending
        assert counters == sorted(counters, reverse=True)


# ============================================================================
# Test: Minimum Priority Floor
# ============================================================================


class TestMinimumPriority:
    """Tests to ensure priority doesn't go below minimum."""
    
    def test_priority_floor_at_10(self):
        """Priority should not go below 10 even for very deep tasks."""
        tool = ToolDefinition(
            name="test",
            container=ContainerConfig(image="test:v1", command="run")
        )
        
        # Create a very deep chain (15 levels = would be priority -50 without floor)
        tasks = []
        for i in range(15):
            deps = [tasks[-1]] if tasks else []
            task = TaskFactory.create_task(
                task_type=TaskType.PRE_TRAINING,
                tool=tool,
                config={"depth": i},
                dependencies=deps
            )
            tasks.append(task)
        
        workflow = WorkflowFactory.create_workflow(name="deep", tasks=tasks)
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        PriorityScheduler(pipeline)
        
        # All tasks should have priority >= 10
        for task in tasks:
            assert task.priority >= 10
        
        # Last task (depth 14) would be 100 - 140 = -40 without floor
        # Should be at least 10
        assert tasks[-1].priority >= 10
