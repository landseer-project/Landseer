"""
Tests for workflow restart and cache recovery functionality.

These tests verify the restart logic from docs/Workflow.md:
- If a tool failed and is present in workflow, rerun it
- If a tool failed and is not present, skip it and continue
"""

import pytest
from typing import Dict, Any, Optional

from src.pipeline.workflow_restart import (
    WorkflowRestartManager,
    FailedTaskInfo,
    WorkflowExecutionState,
)
from src.pipeline.workflow import Workflow, WorkflowFactory
from src.pipeline.tasks import (
    Task,
    TaskType,
    TaskStatus,
    PreTrainingTask,
    InTrainingTask,
    PostTrainingTask,
    TaskFactory,
    clear_task_registry,
)
from src.pipeline.tools import ToolDefinition, ContainerConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tool_a():
    """Tool A."""
    return ToolDefinition(
        name="tool_a",
        container=ContainerConfig(image="test/a:v1", command="python run.py"),
        is_baseline=False
    )


@pytest.fixture
def tool_b():
    """Tool B."""
    return ToolDefinition(
        name="tool_b",
        container=ContainerConfig(image="test/b:v1", command="python run.py"),
        is_baseline=False
    )


@pytest.fixture
def tool_c():
    """Tool C."""
    return ToolDefinition(
        name="tool_c",
        container=ContainerConfig(image="test/c:v1", command="python run.py"),
        is_baseline=False
    )


@pytest.fixture
def simple_workflow(tool_a, tool_b, tool_c):
    """Create a simple workflow A->B->C."""
    task_a = PreTrainingTask(tool=tool_a, config={})
    task_b = InTrainingTask(tool=tool_b, config={})
    task_c = PostTrainingTask(tool=tool_c, config={})
    
    return WorkflowFactory.create_workflow(
        name="test_workflow",
        tasks=[task_a, task_b, task_c],
        pipeline_id="test_pipeline"
    )


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up between tests."""
    clear_task_registry()
    yield
    clear_task_registry()


# ============================================================================
# Test: FailedTaskInfo and WorkflowExecutionState
# ============================================================================


class TestExecutionState:
    """Tests for execution state tracking."""
    
    def test_mark_completed(self):
        """Test marking tasks as completed."""
        state = WorkflowExecutionState(workflow_id="wf_1")
        
        state.mark_completed("task_1")
        
        assert state.is_task_completed("task_1")
        assert state.last_completed_task_id == "task_1"
        assert not state.is_task_failed("task_1")
    
    def test_mark_failed(self):
        """Test marking tasks as failed."""
        state = WorkflowExecutionState(workflow_id="wf_1")
        
        state.mark_failed("task_1")
        
        assert state.is_task_failed("task_1")
        assert not state.is_task_completed("task_1")
    
    def test_mark_skipped(self):
        """Test marking tasks as skipped."""
        state = WorkflowExecutionState(workflow_id="wf_1")
        
        state.mark_skipped("task_1")
        
        assert state.is_task_skipped("task_1")
    
    def test_get_summary(self):
        """Test getting execution state summary."""
        state = WorkflowExecutionState(workflow_id="wf_1")
        state.mark_completed("task_1")
        state.mark_failed("task_2")
        state.mark_skipped("task_3")
        
        summary = state.get_summary()
        
        assert summary["workflow_id"] == "wf_1"
        assert summary["completed_count"] == 1
        assert summary["failed_count"] == 1
        assert summary["skipped_count"] == 1


# ============================================================================
# Test: WorkflowRestartManager
# ============================================================================


class TestWorkflowRestartManager:
    """Tests for workflow restart manager."""
    
    def test_register_failed_task(self):
        """Test registering a failed task."""
        manager = WorkflowRestartManager()
        
        manager.register_failed_task(
            workflow_id="wf_1",
            task_id="task_1",
            task_name="tool_a",
            error_message="Test error"
        )
        
        assert "wf_1" in manager.failed_tasks
        assert len(manager.failed_tasks["wf_1"]) == 1
        assert manager.failed_tasks["wf_1"][0].task_id == "task_1"
        assert manager.failed_tasks["wf_1"][0].task_name == "tool_a"
    
    def test_find_failed_task_by_id(self, simple_workflow):
        """Test finding a failed task by ID."""
        manager = WorkflowRestartManager()
        
        # Register a failed task that exists in workflow
        failed_task = simple_workflow.tasks[0]
        manager.register_failed_task(
            workflow_id=simple_workflow.id,
            task_id=failed_task.id,
            task_name=failed_task.tool.name
        )
        
        # Find it
        found = manager.find_failed_task_in_workflow(
            simple_workflow,
            failed_task.id
        )
        
        assert found is not None
        assert found.id == failed_task.id
    
    def test_find_failed_task_by_tool_name(self, simple_workflow):
        """Test finding a failed task by tool name when ID changed."""
        manager = WorkflowRestartManager()
        
        # Register a failed task with old ID
        old_task_id = "old_task_id_123"
        tool_name = simple_workflow.tasks[0].tool.name
        
        manager.register_failed_task(
            workflow_id=simple_workflow.id,
            task_id=old_task_id,
            task_name=tool_name
        )
        
        # Find it by tool name (simulating workflow regeneration with new IDs)
        found = manager.find_failed_task_in_workflow(
            simple_workflow,
            old_task_id
        )
        
        assert found is not None
        assert found.tool.name == tool_name
    
    def test_find_failed_task_not_found(self, simple_workflow):
        """Test when failed task is not found."""
        manager = WorkflowRestartManager()
        
        manager.register_failed_task(
            workflow_id=simple_workflow.id,
            task_id="nonexistent_task",
            task_name="nonexistent_tool"
        )
        
        found = manager.find_failed_task_in_workflow(
            simple_workflow,
            "nonexistent_task"
        )
        
        assert found is None
    
    def test_should_skip_task_not_in_workflow(self, simple_workflow):
        """Test skipping a task that's not in the workflow."""
        manager = WorkflowRestartManager()
        state = manager.get_execution_state(simple_workflow.id)
        
        # Register a failed task that doesn't exist in workflow
        manager.register_failed_task(
            workflow_id=simple_workflow.id,
            task_id="nonexistent_task",
            task_name="nonexistent_tool"
        )
        
        should_skip = manager.should_skip_task(
            simple_workflow,
            "nonexistent_task",
            state
        )
        
        assert should_skip is True
        assert state.is_task_skipped("nonexistent_task")
    
    def test_should_skip_task_in_workflow(self, simple_workflow):
        """Test not skipping a task that exists in workflow."""
        manager = WorkflowRestartManager()
        state = manager.get_execution_state(simple_workflow.id)
        
        # Task exists in workflow, should not skip
        existing_task = simple_workflow.tasks[0]
        
        should_skip = manager.should_skip_task(
            simple_workflow,
            existing_task.id,
            state
        )
        
        assert should_skip is False
    
    def test_should_rerun_failed_task(self, simple_workflow):
        """Test rerunning a failed task that exists."""
        manager = WorkflowRestartManager()
        state = manager.get_execution_state(simple_workflow.id)
        
        # Register a failed task
        failed_task = simple_workflow.tasks[0]
        manager.register_failed_task(
            workflow_id=simple_workflow.id,
            task_id=failed_task.id,
            task_name=failed_task.tool.name
        )
        
        should_rerun = manager.should_rerun_task(
            simple_workflow,
            failed_task,
            state
        )
        
        assert should_rerun is True
    
    def test_should_not_rerun_completed_task(self, simple_workflow):
        """Test not rerunning a completed task."""
        manager = WorkflowRestartManager()
        state = manager.get_execution_state(simple_workflow.id)
        
        completed_task = simple_workflow.tasks[0]
        state.mark_completed(completed_task.id)
        
        should_rerun = manager.should_rerun_task(
            simple_workflow,
            completed_task,
            state
        )
        
        assert should_rerun is False
    
    def test_prepare_restart_plan(self, simple_workflow):
        """Test preparing a restart plan."""
        manager = WorkflowRestartManager()
        
        # Register a failed task
        failed_task = simple_workflow.tasks[1]
        manager.register_failed_task(
            workflow_id=simple_workflow.id,
            task_id=failed_task.id,
            task_name=failed_task.tool.name
        )
        
        # Mark first task as completed
        state = manager.get_execution_state(simple_workflow.id)
        state.mark_completed(simple_workflow.tasks[0].id)
        
        plan = manager.prepare_restart_plan(simple_workflow)
        
        assert plan["workflow_id"] == simple_workflow.id
        assert len(plan["tasks_to_rerun"]) == 1
        assert len(plan["tasks_to_cache"]) == 1
        assert len(plan["tasks_to_execute"]) >= 1
    
    def test_prepare_restart_plan_with_skipped_task(self, simple_workflow):
        """Test restart plan with a task not in workflow."""
        manager = WorkflowRestartManager()
        
        # Register a failed task that doesn't exist
        manager.register_failed_task(
            workflow_id=simple_workflow.id,
            task_id="nonexistent_task",
            task_name="nonexistent_tool"
        )
        
        plan = manager.prepare_restart_plan(simple_workflow)
        
        assert len(plan["tasks_to_skip"]) == 1
        assert plan["tasks_to_skip"][0]["reason"] == "not_in_workflow"
    
    def test_execute_with_restart(self, simple_workflow):
        """Test executing workflow with restart."""
        manager = WorkflowRestartManager(use_cache=True)
        
        # Register a failed task
        failed_task = simple_workflow.tasks[1]
        manager.register_failed_task(
            workflow_id=simple_workflow.id,
            task_id=failed_task.id,
            task_name=failed_task.tool.name
        )
        
        # Mark first task as completed
        state = manager.get_execution_state(simple_workflow.id)
        state.mark_completed(simple_workflow.tasks[0].id)
        
        # Mock task executor
        execution_order = []
        
        def task_executor(task: Task) -> Dict[str, Any]:
            execution_order.append(task.id)
            # Failed task should be rerun and succeed
            return {"success": True}
        
        def cache_checker(task: Task) -> Optional[str]:
            if state.is_task_completed(task.id):
                return f"/cache/{task.id}"
            return None
        
        results = manager.execute_with_restart(
            simple_workflow,
            task_executor=task_executor,
            cache_checker=cache_checker
        )
        
        # Should have: 1 cached (task 0), 2 completed (task 1 rerun, task 2)
        assert len(results["cached"]) == 1
        assert len(results["completed"]) == 2
        # Failed task should be rerun
        assert failed_task.id in execution_order
    
    def test_execute_with_restart_skip_missing_task(self, simple_workflow):
        """Test executing workflow skipping tasks not in workflow."""
        manager = WorkflowRestartManager(use_cache=True)
        
        # Register a failed task that doesn't exist in the workflow
        manager.register_failed_task(
            workflow_id=simple_workflow.id,
            task_id="nonexistent_task",
            task_name="nonexistent_tool"
        )
        
        def task_executor(task: Task) -> Dict[str, Any]:
            return {"success": True}
        
        results = manager.execute_with_restart(
            simple_workflow,
            task_executor=task_executor
        )
        
        # Should skip the nonexistent task (it's not in workflow.tasks, so it's in plan["tasks_to_skip"])
        # All workflow tasks should complete normally
        assert len(results["skipped"]) >= 1  # At least the nonexistent task
        assert len(results["completed"]) == len(simple_workflow.tasks)


# ============================================================================
# Test: Integration with Workflow
# ============================================================================


class TestWorkflowRestartIntegration:
    """Tests for workflow restart integration."""
    
    def test_workflow_has_task(self, simple_workflow):
        """Test checking if workflow has a task."""
        task = simple_workflow.tasks[0]
        
        assert simple_workflow.has_task(task.id)
        assert not simple_workflow.has_task("nonexistent")
    
    def test_workflow_get_task_by_id(self, simple_workflow):
        """Test getting task by ID."""
        task = simple_workflow.tasks[0]
        
        found = simple_workflow.get_task_by_id(task.id)
        assert found is not None
        assert found.id == task.id
    
    def test_workflow_get_task_by_tool_name(self, simple_workflow):
        """Test getting task by tool name."""
        task = simple_workflow.tasks[0]
        
        found = simple_workflow.get_task_by_tool_name(task.tool.name)
        assert found is not None
        assert found.tool.name == task.tool.name
    
    def test_workflow_prepare_restart(self, simple_workflow):
        """Test workflow prepare_restart method."""
        manager = WorkflowRestartManager()
        
        manager.register_failed_task(
            workflow_id=simple_workflow.id,
            task_id=simple_workflow.tasks[0].id,
            task_name=simple_workflow.tasks[0].tool.name
        )
        
        plan = simple_workflow.prepare_restart(manager)
        
        assert plan["workflow_id"] == simple_workflow.id
        assert len(plan["tasks_to_rerun"]) == 1
