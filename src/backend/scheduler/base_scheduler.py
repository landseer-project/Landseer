"""
Abstract base scheduler for the Landseer pipeline.

This module defines the abstract Scheduler interface that all concrete
scheduler implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pipeline.tasks import Task, TaskStatus
from pipeline.pipeline import Pipeline


class Scheduler(ABC):
    """
    Abstract base class for pipeline schedulers.
    
    A scheduler manages the execution order of tasks in a pipeline,
    taking into account dependencies, priorities, and resource constraints.
    
    Attributes:
        pipeline: The pipeline instance to schedule
    """
    
    def __init__(self, pipeline: Pipeline):
        """
        Initialize the scheduler with a pipeline.
        
        Args:
            pipeline: Pipeline instance containing workflows and tasks
        """
        self.pipeline = pipeline
        self._all_tasks: List[Task] = []
        self._initialize_tasks()
    
    def _initialize_tasks(self) -> None:
        """
        Extract and initialize all tasks from the pipeline's workflows.
        Sets up initial task states and prepares for scheduling.
        """
        self._all_tasks = []
        for workflow in self.pipeline.workflows:
            for task in workflow.tasks:
                if task not in self._all_tasks:
                    self._all_tasks.append(task)
                    # Ensure task starts in pending status
                    if task.status != TaskStatus.PENDING:
                        task.status = TaskStatus.PENDING
    
    @abstractmethod
    def get_next_task(self) -> Optional[Task]:
        """
        Get the next task to execute.
        
        This method should:
        1. Find tasks that are ready to execute (dependencies satisfied)
        2. Select the highest priority task among ready tasks
        3. Update the task's status to RUNNING
        4. Return the task
        
        Returns:
            Next task to execute, or None if no tasks are ready
        """
        pass
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """
        Update the status of a task.
        
        This method should be called when a task completes (successfully or with failure).
        
        Args:
            task_id: ID of the task to update
            status: New status (typically COMPLETED or FAILED)
            
        Raises:
            ValueError: If task_id not found or invalid status transition
        """
        task = self._find_task_by_id(task_id)
        
        if task is None:
            raise ValueError(f"Task with ID '{task_id}' not found in scheduler")
        
        # Validate status transition
        if status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            raise ValueError(
                f"Invalid status update: {status}. "
                "Only COMPLETED or FAILED are allowed for update_task_status."
            )
        
        task.status = status
    
    def _find_task_by_id(self, task_id: str) -> Optional[Task]:
        """
        Find a task by its ID.
        
        Args:
            task_id: ID of the task to find
            
        Returns:
            Task if found, None otherwise
        """
        for task in self._all_tasks:
            if task.id == task_id:
                return task
        return None
    
    def _is_task_ready(self, task: Task) -> bool:
        """
        Check if a task is ready to execute.
        
        A task is ready if:
        - Its status is PENDING
        - All its dependencies have completed successfully
        
        Args:
            task: Task to check
            
        Returns:
            True if task is ready to execute, False otherwise
        """
        if task.status != TaskStatus.PENDING:
            return False
        
        # Check if all dependencies are completed
        for dep in task.dependencies:
            if dep.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def get_all_tasks(self) -> List[Task]:
        """
        Get all tasks managed by this scheduler.
        
        Returns:
            List of all tasks
        """
        return self._all_tasks.copy()
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """
        Get all tasks with a specific status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of tasks with the given status
        """
        return [task for task in self._all_tasks if task.status == status]
    
    def is_complete(self) -> bool:
        """
        Check if all tasks have completed (successfully or failed).
        
        Returns:
            True if all tasks are in COMPLETED or FAILED status
        """
        for task in self._all_tasks:
            if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False
        return True
    
    def get_progress(self) -> dict:
        """
        Get progress statistics for the pipeline.
        
        Returns:
            Dictionary with task counts by status
        """
        stats = {
            "total": len(self._all_tasks),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0
        }
        
        for task in self._all_tasks:
            if task.status == TaskStatus.PENDING:
                stats["pending"] += 1
            elif task.status == TaskStatus.RUNNING:
                stats["running"] += 1
            elif task.status == TaskStatus.COMPLETED:
                stats["completed"] += 1
            elif task.status == TaskStatus.FAILED:
                stats["failed"] += 1
        
        return stats
