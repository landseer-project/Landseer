"""
Priority-based scheduler for the Landseer pipeline.

This scheduler prioritizes tasks based on:
1. Dependency level (tasks with no dependencies have highest priority)
2. Usage counter (within the same level, tasks used in more workflows have higher priority)
"""

from typing import Optional, List
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pipeline.tasks import Task, TaskStatus
from pipeline.pipeline import Pipeline
from .base_scheduler import Scheduler


class PriorityScheduler(Scheduler):
    """
    Priority-based scheduler that considers task dependencies and usage.
    
    Priority is calculated as:
    - Primary: Dependency level (number of dependencies)
    - Secondary: Usage counter (how many workflows use this task)
    
    Tasks with fewer dependencies and higher usage get scheduled first.
    """
    
    def __init__(self, pipeline: Pipeline):
        """
        Initialize the priority scheduler.
        
        Args:
            pipeline: Pipeline instance to schedule
        """
        super().__init__(pipeline)
        self._update_task_priorities()
    
    def _update_task_priorities(self) -> None:
        """
        Update priority values for all tasks.
        
        Priority calculation:
        - Level priority = length of dependency list (lower is better)
        - Within level, higher counter = higher priority (lower integer value)
        
        Priority formula: (level * 1000) - counter
        This ensures tasks at level 0 always have priority < 1000,
        level 1 tasks have priority 1000-1999, etc.
        """
        for task in self._all_tasks:
            # Primary priority: dependency level
            level = len(task.dependencies)
            
            # Secondary priority: usage counter (inverted, so higher counter = lower number)
            # Multiply level by 1000 to ensure level separation
            # Subtract counter so higher usage = lower (better) priority number
            task.priority = (level * 1000) - task.counter
    
    def get_next_task(self) -> Optional[Task]:
        """
        Get the next task to execute based on priority.
        
        Selects the highest priority (lowest priority number) task
        that is ready to execute (all dependencies completed).
        
        Returns:
            Next task to execute, or None if no tasks are ready
        """
        # Get all ready tasks
        ready_tasks = [task for task in self._all_tasks if self._is_task_ready(task)]
        
        if not ready_tasks:
            return None
        
        # Sort by priority (lower number = higher priority)
        ready_tasks.sort(key=lambda t: t.priority)
        
        # Get the highest priority task
        next_task = ready_tasks[0]
        
        # Update status to RUNNING
        next_task.status = TaskStatus.RUNNING
        
        return next_task
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """
        Update task status and recalculate priorities if needed.
        
        Args:
            task_id: ID of the task to update
            status: New status (COMPLETED or FAILED)
        """
        super().update_task_status(task_id, status)
        
        # Recalculate priorities after status update
        # This ensures dependent tasks get updated priorities
        self._update_task_priorities()
    
    def get_ready_tasks_by_priority(self) -> List[Task]:
        """
        Get all ready tasks sorted by priority.
        
        Useful for debugging or displaying available tasks.
        
        Returns:
            List of ready tasks sorted by priority (highest first)
        """
        ready_tasks = [task for task in self._all_tasks if self._is_task_ready(task)]
        ready_tasks.sort(key=lambda t: t.priority)
        return ready_tasks
    
    def get_task_priority_info(self, task_id: str) -> dict:
        """
        Get detailed priority information for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with priority breakdown
            
        Raises:
            ValueError: If task not found
        """
        task = self._find_task_by_id(task_id)
        
        if task is None:
            raise ValueError(f"Task with ID '{task_id}' not found")
        
        level = len(task.dependencies)
        
        return {
            "task_id": task.id,
            "priority": task.priority,
            "dependency_level": level,
            "usage_counter": task.counter,
            "status": task.status.value,
            "dependencies": [dep.id for dep in task.dependencies],
            "workflows": list(task.workflows)
        }
    
    def get_priority_levels(self) -> dict:
        """
        Get tasks grouped by dependency level.
        
        Returns:
            Dictionary mapping level numbers to lists of tasks
        """
        levels = {}
        
        for task in self._all_tasks:
            level = len(task.dependencies)
            if level not in levels:
                levels[level] = []
            levels[level].append(task)
        
        # Sort tasks within each level by counter (descending)
        for level in levels:
            levels[level].sort(key=lambda t: t.counter, reverse=True)
        
        return levels
