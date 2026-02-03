"""
Tests for the PriorityScheduler implementation.

These tests verify that the priority scheduler correctly:
1. Returns the highest priority task that is ready to execute
2. Updates task status correctly
3. Assigns correct priorities based on dependencies
"""

import pytest
from typing import List

from src.pipeline.tasks import (
    Task,
    TaskStatus,
    TaskType,
    TaskFactory,
    PreTrainingTask,
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
def sample_tool():
    """Create a sample tool for testing."""
    return ToolDefinition(
        name="test_tool",
        container=ContainerConfig(
            image="test/image:latest",
            command="python run.py"
        )
    )


@pytest.fixture
def create_task(sample_tool):
    """Factory fixture to create tasks with unique IDs."""
    task_counter = [0]  # Use list to allow mutation in nested function
    
    def _create_task(
        tool: ToolDefinition = None,
        config: dict = None,
        dependencies: List[Task] = None,
        task_type: TaskType = TaskType.PRE_TRAINING
    ) -> Task:
        task_counter[0] += 1
        tool = tool or sample_tool
        config = config or {"task_num": task_counter[0]}
        return TaskFactory.create_task(
            task_type=task_type,
            tool=tool,
            config=config,
            dependencies=dependencies or []
        )
    
    return _create_task


@pytest.fixture
def simple_pipeline_with_scheduler(create_task):
    """
    Create a simple pipeline with 3 independent tasks.
    
    Task structure:
    - task_a: no dependencies (depth 0, priority ~100)
    - task_b: no dependencies (depth 0, priority ~100)
    - task_c: no dependencies (depth 0, priority ~100)
    """
    task_a = create_task(config={"name": "task_a"})
    task_b = create_task(config={"name": "task_b"})
    task_c = create_task(config={"name": "task_c"})
    
    workflow = WorkflowFactory.create_workflow(
        name="test_workflow",
        tasks=[task_a, task_b, task_c]
    )
    
    pipeline = DefenseEvaluationPipeline(
        name="test_pipeline",
        workflows=[workflow]
    )
    
    scheduler = PriorityScheduler(pipeline)
    
    return {
        "pipeline": pipeline,
        "scheduler": scheduler,
        "tasks": {"a": task_a, "b": task_b, "c": task_c}
    }


@pytest.fixture
def dependency_pipeline_with_scheduler(create_task):
    """
    Create a pipeline with dependency chain.
    
    Task structure:
    - task_root: no dependencies (depth 0, priority 100+)
    - task_mid: depends on task_root (depth 1, priority 90+)
    - task_leaf: depends on task_mid (depth 2, priority 80+)
    """
    task_root = create_task(config={"name": "root"})
    task_mid = create_task(config={"name": "mid"}, dependencies=[task_root])
    task_leaf = create_task(config={"name": "leaf"}, dependencies=[task_mid])
    
    workflow = WorkflowFactory.create_workflow(
        name="dep_workflow",
        tasks=[task_root, task_mid, task_leaf]
    )
    
    pipeline = DefenseEvaluationPipeline(
        name="dep_pipeline",
        workflows=[workflow]
    )
    
    scheduler = PriorityScheduler(pipeline)
    
    return {
        "pipeline": pipeline,
        "scheduler": scheduler,
        "tasks": {"root": task_root, "mid": task_mid, "leaf": task_leaf}
    }


@pytest.fixture(autouse=True)
def cleanup_task_registry():
    """Clear task registry before each test."""
    clear_task_registry()
    yield
    clear_task_registry()


# ============================================================================
# Test 1: get_next_task() returns highest priority ready task
# ============================================================================


class TestGetNextTaskPriority:
    """Tests for get_next_task() returning the highest priority ready task."""
    
    def test_returns_ready_task_when_available(self, simple_pipeline_with_scheduler):
        """get_next_task should return a task when tasks are available."""
        scheduler = simple_pipeline_with_scheduler["scheduler"]
        
        task = scheduler.get_next_task()
        
        assert task is not None
        assert task.status == TaskStatus.RUNNING
    
    def test_returns_none_when_no_ready_tasks(self, dependency_pipeline_with_scheduler):
        """get_next_task should return None when no tasks are ready."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        # Get and complete the root task
        root_task = scheduler.get_next_task()
        assert root_task.id == tasks["root"].id
        scheduler.update_task_status(root_task.id, TaskStatus.COMPLETED)
        
        # Get and "run" the mid task (mark as running but not complete)
        mid_task = scheduler.get_next_task()
        assert mid_task.id == tasks["mid"].id
        # mid_task is now RUNNING, leaf depends on mid
        
        # The leaf task should not be ready since mid is not completed
        # And mid is already running, root is completed
        next_task = scheduler.get_next_task()
        assert next_task is None  # No tasks ready (mid is running, leaf blocked)
    
    def test_returns_highest_priority_task_with_dependencies(
        self, dependency_pipeline_with_scheduler
    ):
        """get_next_task should return task with highest priority (root first)."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        # Root should be returned first (highest priority, no dependencies)
        first_task = scheduler.get_next_task()
        
        assert first_task is not None
        assert first_task.id == tasks["root"].id
        assert first_task.status == TaskStatus.RUNNING
    
    def test_respects_dependency_completion_for_readiness(
        self, dependency_pipeline_with_scheduler
    ):
        """Tasks should only be ready after dependencies complete."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        # Initially only root should be ready
        root = scheduler.get_next_task()
        assert root.id == tasks["root"].id
        
        # Mid should not be returned yet (root not completed)
        next_task = scheduler.get_next_task()
        assert next_task is None
        
        # Complete root
        scheduler.update_task_status(root.id, TaskStatus.COMPLETED)
        
        # Now mid should be ready
        mid = scheduler.get_next_task()
        assert mid is not None
        assert mid.id == tasks["mid"].id
    
    def test_returns_higher_counter_task_at_same_depth(self, create_task):
        """
        Among tasks at the same depth, those with higher counter
        (used in more workflows) should have slightly higher priority.
        """
        # Create two tasks at the same depth
        task_low_usage = create_task(config={"name": "low_usage"})
        task_high_usage = create_task(config={"name": "high_usage"})
        
        # Create multiple workflows to increase counter for high_usage task
        workflow1 = WorkflowFactory.create_workflow(
            name="wf1",
            tasks=[task_low_usage, task_high_usage]
        )
        workflow2 = WorkflowFactory.create_workflow(
            name="wf2",
            tasks=[task_high_usage]  # Only high_usage in this workflow
        )
        workflow3 = WorkflowFactory.create_workflow(
            name="wf3",
            tasks=[task_high_usage]  # Only high_usage in this workflow
        )
        
        pipeline = DefenseEvaluationPipeline(
            name="counter_test_pipeline",
            workflows=[workflow1, workflow2, workflow3]
        )
        
        scheduler = PriorityScheduler(pipeline)
        
        # task_high_usage should have higher counter and thus higher priority
        assert task_high_usage.counter > task_low_usage.counter
        assert task_high_usage.priority > task_low_usage.priority
        
        # First task returned should be the one with higher counter
        first_task = scheduler.get_next_task()
        assert first_task.id == task_high_usage.id


# ============================================================================
# Test 2: update_task_status() updates status correctly
# ============================================================================


class TestUpdateTaskStatus:
    """Tests for update_task_status() correctly updating task status."""
    
    def test_updates_status_to_completed(self, simple_pipeline_with_scheduler):
        """update_task_status should correctly mark a task as COMPLETED."""
        scheduler = simple_pipeline_with_scheduler["scheduler"]
        
        task = scheduler.get_next_task()
        assert task.status == TaskStatus.RUNNING
        
        scheduler.update_task_status(task.id, TaskStatus.COMPLETED)
        
        assert task.status == TaskStatus.COMPLETED
    
    def test_updates_status_to_failed(self, simple_pipeline_with_scheduler):
        """update_task_status should correctly mark a task as FAILED."""
        scheduler = simple_pipeline_with_scheduler["scheduler"]
        
        task = scheduler.get_next_task()
        assert task.status == TaskStatus.RUNNING
        
        scheduler.update_task_status(task.id, TaskStatus.FAILED)
        
        assert task.status == TaskStatus.FAILED
    
    def test_raises_error_for_invalid_task_id(self, simple_pipeline_with_scheduler):
        """update_task_status should raise ValueError for unknown task ID."""
        scheduler = simple_pipeline_with_scheduler["scheduler"]
        
        with pytest.raises(ValueError) as exc_info:
            scheduler.update_task_status("nonexistent_task", TaskStatus.COMPLETED)
        
        assert "not found" in str(exc_info.value)
    
    def test_raises_error_for_invalid_status(self, simple_pipeline_with_scheduler):
        """update_task_status should raise ValueError for invalid status transitions."""
        scheduler = simple_pipeline_with_scheduler["scheduler"]
        
        task = scheduler.get_next_task()
        
        # Should not allow setting status to RUNNING or PENDING via update
        with pytest.raises(ValueError) as exc_info:
            scheduler.update_task_status(task.id, TaskStatus.PENDING)
        
        assert "Invalid status" in str(exc_info.value) or "Only COMPLETED or FAILED" in str(exc_info.value)
    
    def test_completed_task_enables_dependent_tasks(
        self, dependency_pipeline_with_scheduler
    ):
        """Completing a task should make its dependents ready."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        # Get root task
        root = scheduler.get_next_task()
        assert root.id == tasks["root"].id
        
        # Mid should not be ready yet
        ready_tasks = scheduler.get_ready_tasks_by_priority()
        assert not any(t.id == tasks["mid"].id for t in ready_tasks)
        
        # Complete root
        scheduler.update_task_status(root.id, TaskStatus.COMPLETED)
        
        # Now mid should be ready
        ready_tasks = scheduler.get_ready_tasks_by_priority()
        assert any(t.id == tasks["mid"].id for t in ready_tasks)
    
    def test_failed_dependency_blocks_dependent_tasks(
        self, dependency_pipeline_with_scheduler
    ):
        """A failed dependency should keep dependent tasks blocked."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        # Get and fail the root task
        root = scheduler.get_next_task()
        scheduler.update_task_status(root.id, TaskStatus.FAILED)
        
        # Mid should not be ready (root failed, not completed)
        ready_tasks = scheduler.get_ready_tasks_by_priority()
        assert not any(t.id == tasks["mid"].id for t in ready_tasks)
        
        # No tasks should be ready
        next_task = scheduler.get_next_task()
        assert next_task is None


# ============================================================================
# Test 3: Dependency tasks get correct priority
# ============================================================================


class TestDependencyPriority:
    """Tests for correct priority assignment based on dependencies."""
    
    def test_root_tasks_have_highest_priority(self, dependency_pipeline_with_scheduler):
        """Tasks with no dependencies should have priority ~100."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        # Root task (depth 0) should have priority in 100-109 range
        root_priority = tasks["root"].priority
        assert 100 <= root_priority <= 109, f"Root priority {root_priority} should be 100-109"
    
    def test_depth_1_tasks_have_lower_priority(self, dependency_pipeline_with_scheduler):
        """Tasks at depth 1 should have priority ~90."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        # Mid task (depth 1) should have priority in 90-99 range
        mid_priority = tasks["mid"].priority
        assert 90 <= mid_priority <= 99, f"Mid priority {mid_priority} should be 90-99"
    
    def test_depth_2_tasks_have_even_lower_priority(
        self, dependency_pipeline_with_scheduler
    ):
        """Tasks at depth 2 should have priority ~80."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        # Leaf task (depth 2) should have priority in 80-89 range
        leaf_priority = tasks["leaf"].priority
        assert 80 <= leaf_priority <= 89, f"Leaf priority {leaf_priority} should be 80-89"
    
    def test_priority_decreases_with_depth(self, dependency_pipeline_with_scheduler):
        """Priority should decrease as dependency depth increases."""
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        assert tasks["root"].priority > tasks["mid"].priority
        assert tasks["mid"].priority > tasks["leaf"].priority
    
    def test_complex_dependency_graph_priorities(self, create_task):
        r"""
        Test priorities in a more complex dependency graph.
        
        Structure:
            root_a    root_b
               \       /
                mid_ab
                  |
                leaf
        """
        root_a = create_task(config={"name": "root_a"})
        root_b = create_task(config={"name": "root_b"})
        mid_ab = create_task(
            config={"name": "mid_ab"},
            dependencies=[root_a, root_b]
        )
        leaf = create_task(
            config={"name": "leaf"},
            dependencies=[mid_ab]
        )
        
        workflow = WorkflowFactory.create_workflow(
            name="complex_workflow",
            tasks=[root_a, root_b, mid_ab, leaf]
        )
        
        pipeline = DefenseEvaluationPipeline(
            name="complex_pipeline",
            workflows=[workflow]
        )
        
        scheduler = PriorityScheduler(pipeline)
        
        # Both root tasks should have the same priority level (depth 0)
        assert abs(root_a.priority - root_b.priority) <= 9  # Same level, counter diff
        
        # mid_ab has depth 1
        assert root_a.priority > mid_ab.priority
        assert root_b.priority > mid_ab.priority
        
        # leaf has depth 2
        assert mid_ab.priority > leaf.priority
    
    def test_priority_recalculated_after_status_update(
        self, dependency_pipeline_with_scheduler
    ):
        """Priorities should be recalculated after task status updates."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        initial_mid_priority = tasks["mid"].priority
        
        # Complete the root task
        root = scheduler.get_next_task()
        scheduler.update_task_status(root.id, TaskStatus.COMPLETED)
        
        # Priority should remain consistent after recalculation
        assert tasks["mid"].priority == initial_mid_priority
    
    def test_get_priority_levels_groups_tasks_by_depth(
        self, dependency_pipeline_with_scheduler
    ):
        """get_priority_levels should correctly group tasks by dependency depth."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        levels = scheduler.get_priority_levels()
        
        # Should have 3 levels: 0, 1, 2
        assert 0 in levels
        assert 1 in levels
        assert 2 in levels
        
        # Check task assignments
        level_0_ids = [t.id for t in levels[0]]
        level_1_ids = [t.id for t in levels[1]]
        level_2_ids = [t.id for t in levels[2]]
        
        assert tasks["root"].id in level_0_ids
        assert tasks["mid"].id in level_1_ids
        assert tasks["leaf"].id in level_2_ids


# ============================================================================
# Integration Tests
# ============================================================================


class TestSchedulerIntegration:
    """Integration tests for the complete scheduling workflow."""
    
    def test_full_scheduling_cycle(self, dependency_pipeline_with_scheduler):
        """Test a complete scheduling cycle from start to finish."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        # Initial state
        assert not scheduler.is_complete()
        progress = scheduler.get_progress()
        assert progress["pending"] == 3
        assert progress["completed"] == 0
        
        # Execute root
        task1 = scheduler.get_next_task()
        assert task1.id == tasks["root"].id
        scheduler.update_task_status(task1.id, TaskStatus.COMPLETED)
        
        # Execute mid
        task2 = scheduler.get_next_task()
        assert task2.id == tasks["mid"].id
        scheduler.update_task_status(task2.id, TaskStatus.COMPLETED)
        
        # Execute leaf
        task3 = scheduler.get_next_task()
        assert task3.id == tasks["leaf"].id
        scheduler.update_task_status(task3.id, TaskStatus.COMPLETED)
        
        # All done
        assert scheduler.is_complete()
        progress = scheduler.get_progress()
        assert progress["completed"] == 3
        assert progress["pending"] == 0
    
    def test_get_task_priority_info(self, dependency_pipeline_with_scheduler):
        """Test get_task_priority_info returns correct information."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        tasks = dependency_pipeline_with_scheduler["tasks"]
        
        info = scheduler.get_task_priority_info(tasks["mid"].id)
        
        assert info["task_id"] == tasks["mid"].id
        assert info["priority"] == tasks["mid"].priority
        assert info["status"] == TaskStatus.PENDING.value
        assert tasks["root"].id in info["dependencies"]
    
    def test_get_task_priority_info_raises_for_invalid_id(
        self, dependency_pipeline_with_scheduler
    ):
        """get_task_priority_info should raise ValueError for unknown task."""
        scheduler = dependency_pipeline_with_scheduler["scheduler"]
        
        with pytest.raises(ValueError) as exc_info:
            scheduler.get_task_priority_info("invalid_id")
        
        assert "not found" in str(exc_info.value)
