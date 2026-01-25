"""
Priority-based scheduler for the Landseer pipeline.

This scheduler prioritizes tasks based on:
1. Dependency level (tasks with no dependencies have highest priority)
2. Usage counter (within the same level, tasks used in more workflows have higher priority)
"""

from typing import Optional, List, Dict

from ...pipeline.tasks import Task, TaskStatus
from ...pipeline.pipeline import Pipeline
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
        
        Priority calculation per OVERVIEWv1.md:
        - 100 for tools with 0 dependencies (depth 0, highest priority)
        - 90 for tools at depth 1 (depends on depth-0 task)
        - 80 for tools at depth 2, etc.
        
        Depth = longest path from a root task (task with no dependencies)
        
        Higher priority value = runs first (scheduler sorts descending)
        Within same level, tasks with higher usage counter get slight boost.
        """
        # Compute depth for each task using BFS/memoization
        depth_cache: Dict[str, int] = {}
        
        def get_depth(task: Task) -> int:
            if task.id in depth_cache:
                return depth_cache[task.id]
            
            if not task.dependencies:
                depth_cache[task.id] = 0
                return 0
            
            # Depth is 1 + max depth of all dependencies
            max_dep_depth = max(get_depth(dep) for dep in task.dependencies)
            depth = max_dep_depth + 1
            depth_cache[task.id] = depth
            return depth
        
        for task in self._all_tasks:
            # Primary: dependency depth
            # 100 for depth 0, 90 for depth 1, 80 for depth 2, etc.
            depth = get_depth(task)
            base_priority = 100 - (depth * 10)
            
            # Ensure priority doesn't go below 10
            base_priority = max(base_priority, 10)
            
            # Secondary: slight bonus for higher counter (more shared = slightly higher priority)
            # Cap the bonus to avoid crossing level boundaries
            counter_bonus = min(task.counter, 9)
            
            # Higher number = runs first
            task.priority = base_priority + counter_bonus
    
    def get_next_task(self) -> Optional[Task]:
        """
        Get the next task to execute based on priority.
        
        Selects the highest priority task (highest priority number)
        that is ready to execute (all dependencies completed).
        
        Returns:
            Next task to execute, or None if no tasks are ready
        """
        # Get all ready tasks
        ready_tasks = [task for task in self._all_tasks if self._is_task_ready(task)]
        
        if not ready_tasks:
            return None
        
        # Sort by priority descending (higher number = higher priority = runs first)
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        
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
            List of ready tasks sorted by priority (highest priority first)
        """
        ready_tasks = [task for task in self._all_tasks if self._is_task_ready(task)]
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
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
    
    def _get_task_depth(self, task: Task, cache: Dict[str, int] = None) -> int:
        """
        Get the depth of a task in the dependency graph.
        
        Depth is the longest path from a root task (task with no dependencies).
        
        Args:
            task: Task to get depth for
            cache: Optional cache for memoization
            
        Returns:
            Depth of the task (0 for root tasks)
        """
        if cache is None:
            cache = {}
        
        if task.id in cache:
            return cache[task.id]
        
        if not task.dependencies:
            cache[task.id] = 0
            return 0
        
        max_dep_depth = max(self._get_task_depth(dep, cache) for dep in task.dependencies)
        depth = max_dep_depth + 1
        cache[task.id] = depth
        return depth
    
    def get_priority_levels(self) -> dict:
        """
        Get tasks grouped by dependency depth.
        
        Returns:
            Dictionary mapping depth levels to lists of tasks
        """
        levels = {}
        depth_cache: Dict[str, int] = {}
        
        for task in self._all_tasks:
            depth = self._get_task_depth(task, depth_cache)
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(task)
        
        # Sort tasks within each level by counter (descending)
        for level in levels:
            levels[level].sort(key=lambda t: t.counter, reverse=True)
        
        return levels
