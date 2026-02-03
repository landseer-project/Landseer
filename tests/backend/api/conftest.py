"""
Shared fixtures and utilities for backend API tests.
"""

import pytest
from pathlib import Path

from src.pipeline.tasks import (
    Task,
    TaskStatus,
    TaskType,
    TaskFactory,
    PreTrainingTask,
    PostTrainingTask,
    InTrainingTask,
    DeploymentTask,
    clear_task_registry,
)
from src.pipeline.tools import ToolDefinition, ContainerConfig
from src.pipeline.workflow import Workflow, WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline


@pytest.fixture(autouse=True)
def reset_scheduler_state():
    """Reset scheduler state before each test."""
    from src.backend.api import _scheduler_state
    _scheduler_state.scheduler = None
    _scheduler_state.pipeline = None
    _scheduler_state.started_at = None
    _scheduler_state.task_metadata.clear()
    _scheduler_state.workers.clear()
    _scheduler_state._worker_counter = 0
    _scheduler_state._custom_tools.clear()
    clear_task_registry()
    yield
    _scheduler_state.scheduler = None
    _scheduler_state.pipeline = None
    _scheduler_state.started_at = None
    _scheduler_state.task_metadata.clear()
    _scheduler_state.workers.clear()
    _scheduler_state._worker_counter = 0
    _scheduler_state._custom_tools.clear()
    clear_task_registry()


def create_task_with_id(
    task_id: str,
    tool: ToolDefinition,
    task_type: TaskType = TaskType.PRE_TRAINING,
    config: dict = None,
    dependencies: list = None,
    priority: int = None,
    pipeline_id: str = None
) -> Task:
    """
    Helper to create a task with a specific ID.
    
    This is useful for testing where we need predictable task IDs.
    """
    task = TaskFactory.create_task(
        task_type=task_type,
        tool=tool,
        config=config or {},
        dependencies=dependencies or []
    )
    # Override ID for testing
    task.id = task_id
    if priority is not None:
        task.priority = priority
    if pipeline_id is not None:
        task.pipeline_id = pipeline_id
    return task
