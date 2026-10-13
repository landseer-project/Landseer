"""
Abstract base scheduler for the Landseer pipeline.

This module defines the abstract Scheduler interface that all concrete
scheduler implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ...pipeline.tasks import Task, TaskStatus
from ...pipeline.pipeline import Pipeline


class Scheduler(ABC):
    """
    Abstract base class for pipeline schedulers.
    
    A scheduler manages the execution order of tasks in a pipeline,
    taking into account dependencies, priorities, and resource constraints.

    Multi-run: one scheduler may hold tasks from several pipelines/runs in
    ``_all_tasks``. Use ``add_pipeline`` to merge; filter by ``run_id`` for
    progress/completion of a single run.
    
    Attributes:
        pipeline: Most recently added pipeline (backward-compatible pointer)
        pipelines: All active pipelines keyed by run_id
    """
    
    def __init__(self, pipeline: Pipeline):
        """
        Initialize the scheduler with a pipeline.
        
        Args:
            pipeline: Pipeline instance containing workflows and tasks
        """
        self.pipeline = pipeline
        self.pipelines: Dict[str, Pipeline] = {pipeline.id: pipeline}
        self._all_tasks: List[Task] = []
        self._initialize_tasks()
    
    def _initialize_tasks(self) -> None:
        """
        Extract and deduplicate all tasks from the pipeline's workflows.
        Task statuses are preserved as-is; callers are responsible for
        ensuring tasks are in the correct state before creating a scheduler.
        """
        self._all_tasks = []
        seen_task_ids = set()
        for workflow in self.pipeline.workflows:
            for task in workflow.tasks:
                # Dedup by stable task id in O(1). The previous `task in list` check
                # becomes quadratic for large pipelines like trades.
                if task.id not in seen_task_ids:
                    seen_task_ids.add(task.id)
                    self._all_tasks.append(task)

    def add_pipeline(self, pipeline: Pipeline) -> int:
        """
        Merge tasks from another pipeline/run into this scheduler.

        Task IDs must be unique across concurrent runs. Cache keys may still
        be shared across runs (identical work). Returns number of tasks added.
        """
        self.pipelines[pipeline.id] = pipeline
        self.pipeline = pipeline

        seen = {task.id for task in self._all_tasks}
        added = 0
        for workflow in pipeline.workflows:
            for task in workflow.tasks:
                if task.id in seen:
                    existing = self._find_task_by_id(task.id)
                    if existing is not None and existing is not task:
                        raise ValueError(
                            f"Task ID collision when adding pipeline {pipeline.id}: {task.id}"
                        )
                    continue
                seen.add(task.id)
                self._all_tasks.append(task)
                added += 1
        return added

    def get_tasks_for_run(self, run_id: str) -> List[Task]:
        """Return tasks belonging to a run (by run_id or pipeline_id)."""
        return [
            task for task in self._all_tasks
            if getattr(task, "run_id", None) == run_id or task.pipeline_id == run_id
        ]
    
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
    
    def is_complete(self, run_id: Optional[str] = None) -> bool:
        """
        Check if tasks have completed (successfully or failed).

        Args:
            run_id: If set, only tasks for that run; otherwise all tasks.
        """
        tasks = self.get_tasks_for_run(run_id) if run_id else self._all_tasks
        if not tasks:
            return False
        for task in tasks:
            if task.status not in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ]:
                return False
        return True
    
    def get_progress(self, run_id: Optional[str] = None) -> dict:
        """
        Get progress statistics.

        Args:
            run_id: If set, only tasks for that run; otherwise all tasks.
        """
        tasks = self.get_tasks_for_run(run_id) if run_id else self._all_tasks
        stats = {
            "total": len(tasks),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
        }
        
        for task in tasks:
            if task.status == TaskStatus.PENDING:
                stats["pending"] += 1
            elif task.status == TaskStatus.RUNNING:
                stats["running"] += 1
            elif task.status == TaskStatus.COMPLETED:
                stats["completed"] += 1
            elif task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
                stats["failed"] += 1
        
        return stats
