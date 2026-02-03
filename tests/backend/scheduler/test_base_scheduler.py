"""
Tests for the base Scheduler class.

These tests verify the abstract scheduler interface and shared functionality
as specified in docs/SCHED2.md and docs/OVERVIEWv1.md.

From SCHED2.md:
- Scheduler takes Pipeline instance as input
- get_next_task() returns the next task to be executed
- update_task_status(task_id, status) to update task status
- Task statuses: pending, running, completed, failed
- When task is returned by get_next_task(), status updated to running
- When update_task_status() called, status updated accordingly
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
from src.backend.scheduler.base_scheduler import Scheduler


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
    task_counter = [0]
    
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


@pytest.fixture(autouse=True)
def cleanup_task_registry():
    """Clear task registry before each test."""
    clear_task_registry()
    yield
    clear_task_registry()


# ============================================================================
# Test: Scheduler Initialization (SCHED2.md)
# ============================================================================


class TestSchedulerInitialization:
    """Tests for scheduler initialization with Pipeline instance."""
    
    def test_scheduler_takes_pipeline_as_input(self, create_task):
        """
        From SCHED2.md: Scheduler should take Pipeline instance as input.
        """
        task = create_task()
        workflow = WorkflowFactory.create_workflow(name="test", tasks=[task])
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        assert scheduler.pipeline is pipeline
    
    def test_scheduler_extracts_all_tasks_from_pipeline(self, create_task):
        """Scheduler should extract all unique tasks from all workflows."""
        task_a = create_task(config={"name": "a"})
        task_b = create_task(config={"name": "b"})
        task_c = create_task(config={"name": "c"})
        
        workflow1 = WorkflowFactory.create_workflow(
            name="wf1", tasks=[task_a, task_b]
        )
        workflow2 = WorkflowFactory.create_workflow(
            name="wf2", tasks=[task_b, task_c]  # task_b shared
        )
        
        pipeline = DefenseEvaluationPipeline(
            name="test", workflows=[workflow1, workflow2]
        )
        
        scheduler = PriorityScheduler(pipeline)
        all_tasks = scheduler.get_all_tasks()
        
        # Should have 3 unique tasks
        assert len(all_tasks) == 3
        assert task_a in all_tasks
        assert task_b in all_tasks
        assert task_c in all_tasks
    
    def test_all_tasks_start_in_pending_status(self, create_task):
        """
        From SCHED2.md: By default, all tasks will be in pending status.
        """
        task_a = create_task()
        task_b = create_task()
        
        workflow = WorkflowFactory.create_workflow(
            name="test", tasks=[task_a, task_b]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        for task in scheduler.get_all_tasks():
            assert task.status == TaskStatus.PENDING


# ============================================================================
# Test: Task Status Transitions (SCHED2.md)
# ============================================================================


class TestTaskStatusTransitions:
    """Tests for task status management as specified in SCHED2.md."""
    
    def test_get_next_task_sets_status_to_running(self, create_task):
        """
        From SCHED2.md: When task is returned by get_next_task(), 
        status should be updated to running.
        """
        task = create_task()
        workflow = WorkflowFactory.create_workflow(name="test", tasks=[task])
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        assert task.status == TaskStatus.PENDING
        
        returned_task = scheduler.get_next_task()
        
        assert returned_task is task
        assert task.status == TaskStatus.RUNNING
    
    def test_update_task_status_to_completed(self, create_task):
        """
        From SCHED2.md: When update_task_status() called with completed,
        task status should be updated accordingly.
        """
        task = create_task()
        workflow = WorkflowFactory.create_workflow(name="test", tasks=[task])
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        # Get task (sets to RUNNING)
        scheduler.get_next_task()
        assert task.status == TaskStatus.RUNNING
        
        # Update to COMPLETED
        scheduler.update_task_status(task.id, TaskStatus.COMPLETED)
        
        assert task.status == TaskStatus.COMPLETED
    
    def test_update_task_status_to_failed(self, create_task):
        """
        From SCHED2.md: When update_task_status() called with failed,
        task status should be updated accordingly.
        """
        task = create_task()
        workflow = WorkflowFactory.create_workflow(name="test", tasks=[task])
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        scheduler.get_next_task()
        scheduler.update_task_status(task.id, TaskStatus.FAILED)
        
        assert task.status == TaskStatus.FAILED
    
    def test_task_status_enum_values(self):
        """
        From SCHED2.md: Task statuses are pending, running, completed, failed.
        """
        assert hasattr(TaskStatus, 'PENDING')
        assert hasattr(TaskStatus, 'RUNNING')
        assert hasattr(TaskStatus, 'COMPLETED')
        assert hasattr(TaskStatus, 'FAILED')
        
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
    
    def test_cannot_update_to_pending_or_running(self, create_task):
        """
        Only COMPLETED or FAILED are valid status updates via update_task_status.
        """
        task = create_task()
        workflow = WorkflowFactory.create_workflow(name="test", tasks=[task])
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        scheduler.get_next_task()
        
        with pytest.raises(ValueError):
            scheduler.update_task_status(task.id, TaskStatus.PENDING)
        
        with pytest.raises(ValueError):
            scheduler.update_task_status(task.id, TaskStatus.RUNNING)


# ============================================================================
# Test: Task Readiness (Base Scheduler)
# ============================================================================


class TestTaskReadiness:
    """Tests for task readiness checking based on dependencies."""
    
    def test_task_with_no_dependencies_is_ready(self, create_task):
        """A task with no dependencies should be ready immediately."""
        task = create_task()
        workflow = WorkflowFactory.create_workflow(name="test", tasks=[task])
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        # Should return the task since it's ready
        assert scheduler.get_next_task() is task
    
    def test_task_with_pending_dependencies_is_not_ready(self, create_task):
        """A task with pending dependencies should not be ready."""
        task_a = create_task(config={"name": "a"})
        task_b = create_task(config={"name": "b"}, dependencies=[task_a])
        
        workflow = WorkflowFactory.create_workflow(
            name="test", tasks=[task_a, task_b]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        # First call should return task_a (higher priority, no deps)
        first = scheduler.get_next_task()
        assert first is task_a
        
        # Second call should return None (task_b depends on task_a which is RUNNING)
        second = scheduler.get_next_task()
        assert second is None
    
    def test_task_becomes_ready_after_dependencies_complete(self, create_task):
        """A task should become ready after all dependencies complete."""
        task_a = create_task(config={"name": "a"})
        task_b = create_task(config={"name": "b"}, dependencies=[task_a])
        
        workflow = WorkflowFactory.create_workflow(
            name="test", tasks=[task_a, task_b]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        # Get and complete task_a
        scheduler.get_next_task()  # Returns task_a
        scheduler.update_task_status(task_a.id, TaskStatus.COMPLETED)
        
        # Now task_b should be ready
        next_task = scheduler.get_next_task()
        assert next_task is task_b
    
    def test_task_not_ready_if_dependency_failed(self, create_task):
        """A task should not be ready if any dependency failed."""
        task_a = create_task(config={"name": "a"})
        task_b = create_task(config={"name": "b"}, dependencies=[task_a])
        
        workflow = WorkflowFactory.create_workflow(
            name="test", tasks=[task_a, task_b]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        # Get and fail task_a
        scheduler.get_next_task()
        scheduler.update_task_status(task_a.id, TaskStatus.FAILED)
        
        # task_b should not be ready
        next_task = scheduler.get_next_task()
        assert next_task is None


# ============================================================================
# Test: Progress Tracking
# ============================================================================


class TestProgressTracking:
    """Tests for pipeline progress tracking."""
    
    def test_get_progress_returns_correct_counts(self, create_task):
        """get_progress should return correct counts by status."""
        task_a = create_task(config={"name": "a"})
        task_b = create_task(config={"name": "b"})
        task_c = create_task(config={"name": "c"})
        
        workflow = WorkflowFactory.create_workflow(
            name="test", tasks=[task_a, task_b, task_c]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        # Initial state
        progress = scheduler.get_progress()
        assert progress["total"] == 3
        assert progress["pending"] == 3
        assert progress["running"] == 0
        assert progress["completed"] == 0
        assert progress["failed"] == 0
        
        # After getting one task
        scheduler.get_next_task()
        progress = scheduler.get_progress()
        assert progress["pending"] == 2
        assert progress["running"] == 1
    
    def test_is_complete_returns_true_when_all_done(self, create_task):
        """is_complete should return True when all tasks are done."""
        task_a = create_task(config={"name": "a"})
        task_b = create_task(config={"name": "b"})
        
        workflow = WorkflowFactory.create_workflow(
            name="test", tasks=[task_a, task_b]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        assert not scheduler.is_complete()
        
        # Complete all tasks
        t1 = scheduler.get_next_task()
        scheduler.update_task_status(t1.id, TaskStatus.COMPLETED)
        
        t2 = scheduler.get_next_task()
        scheduler.update_task_status(t2.id, TaskStatus.COMPLETED)
        
        assert scheduler.is_complete()
    
    def test_is_complete_true_even_with_failed_tasks(self, create_task):
        """is_complete should return True even if some tasks failed."""
        task_a = create_task(config={"name": "a"})
        task_b = create_task(config={"name": "b"})
        
        workflow = WorkflowFactory.create_workflow(
            name="test", tasks=[task_a, task_b]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        # Complete one, fail one
        t1 = scheduler.get_next_task()
        scheduler.update_task_status(t1.id, TaskStatus.COMPLETED)
        
        t2 = scheduler.get_next_task()
        scheduler.update_task_status(t2.id, TaskStatus.FAILED)
        
        assert scheduler.is_complete()
    
    def test_get_tasks_by_status(self, create_task):
        """get_tasks_by_status should filter tasks correctly."""
        task_a = create_task(config={"name": "a"})
        task_b = create_task(config={"name": "b"})
        task_c = create_task(config={"name": "c"})
        
        workflow = WorkflowFactory.create_workflow(
            name="test", tasks=[task_a, task_b, task_c]
        )
        pipeline = DefenseEvaluationPipeline(name="test", workflows=[workflow])
        
        scheduler = PriorityScheduler(pipeline)
        
        # Initially all pending
        pending = scheduler.get_tasks_by_status(TaskStatus.PENDING)
        assert len(pending) == 3
        
        # After getting one task
        scheduler.get_next_task()
        running = scheduler.get_tasks_by_status(TaskStatus.RUNNING)
        assert len(running) == 1


# ============================================================================
# Test: Multiple Workflows
# ============================================================================


class TestMultipleWorkflows:
    """Tests for scheduling across multiple workflows."""
    
    def test_scheduler_handles_multiple_workflows(self, create_task):
        """Scheduler should handle tasks from multiple workflows."""
        task_a = create_task(config={"name": "a"})
        task_b = create_task(config={"name": "b"})
        task_c = create_task(config={"name": "c"})
        
        workflow1 = WorkflowFactory.create_workflow(
            name="wf1", tasks=[task_a, task_b]
        )
        workflow2 = WorkflowFactory.create_workflow(
            name="wf2", tasks=[task_c]
        )
        
        pipeline = DefenseEvaluationPipeline(
            name="test", workflows=[workflow1, workflow2]
        )
        
        scheduler = PriorityScheduler(pipeline)
        
        # All 3 tasks should be schedulable
        all_tasks = scheduler.get_all_tasks()
        assert len(all_tasks) == 3
    
    def test_shared_task_across_workflows(self, create_task):
        """Shared tasks should only be scheduled once."""
        shared_task = create_task(config={"name": "shared"})
        task_b = create_task(config={"name": "b"}, dependencies=[shared_task])
        task_c = create_task(config={"name": "c"}, dependencies=[shared_task])
        
        workflow1 = WorkflowFactory.create_workflow(
            name="wf1", tasks=[shared_task, task_b]
        )
        workflow2 = WorkflowFactory.create_workflow(
            name="wf2", tasks=[shared_task, task_c]
        )
        
        pipeline = DefenseEvaluationPipeline(
            name="test", workflows=[workflow1, workflow2]
        )
        
        scheduler = PriorityScheduler(pipeline)
        
        # Should have 3 unique tasks (shared_task, task_b, task_c)
        all_tasks = scheduler.get_all_tasks()
        assert len(all_tasks) == 3
        
        # Completing shared_task should unblock both task_b and task_c
        t = scheduler.get_next_task()
        assert t is shared_task
        scheduler.update_task_status(t.id, TaskStatus.COMPLETED)
        
        # Both task_b and task_c should now be ready
        ready = scheduler.get_ready_tasks_by_priority()
        assert len(ready) == 2
