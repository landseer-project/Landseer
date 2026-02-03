"""
Workflow restart and cache recovery functionality.

Per Workflow.md:
"If a tool has failed and I am restarting a pipeline run with cache enabled,
try to rerun the tool that failed. If the tool is not present in the workflow,
then skip it and continue with the next tool. If the tool is present in the
workflow, then rerun the tool."
"""

import logging
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, field

from .tasks import Task, TaskStatus, TaskType
from .workflow import Workflow

logger = logging.getLogger(__name__)


@dataclass
class FailedTaskInfo:
    """
    Information about a failed task from a previous run.
    
    Attributes:
        task_id: ID of the failed task
        task_name: Name of the task/tool
        workflow_id: ID of the workflow the task belonged to
        error_message: Optional error message from the failure
        failed_at: Timestamp or run identifier when it failed
    """
    task_id: str
    task_name: str
    workflow_id: str
    error_message: Optional[str] = None
    failed_at: Optional[str] = None


@dataclass
class WorkflowExecutionState:
    """
    Tracks the execution state of a workflow for restart purposes.
    
    Attributes:
        workflow_id: ID of the workflow
        completed_task_ids: Set of task IDs that completed successfully
        failed_task_ids: Set of task IDs that failed
        skipped_task_ids: Set of task IDs that were skipped (not in workflow)
        last_completed_task_id: ID of the last successfully completed task
    """
    workflow_id: str
    completed_task_ids: Set[str] = field(default_factory=set)
    failed_task_ids: Set[str] = field(default_factory=set)
    skipped_task_ids: Set[str] = field(default_factory=set)
    last_completed_task_id: str = ""
    
    def is_task_completed(self, task_id: str) -> bool:
        """Check if a task completed successfully."""
        return task_id in self.completed_task_ids
    
    def is_task_failed(self, task_id: str) -> bool:
        """Check if a task failed."""
        return task_id in self.failed_task_ids
    
    def is_task_skipped(self, task_id: str) -> bool:
        """Check if a task was skipped."""
        return task_id in self.skipped_task_ids
    
    def mark_completed(self, task_id: str) -> None:
        """Mark a task as completed."""
        self.completed_task_ids.add(task_id)
        self.last_completed_task_id = task_id
        # Remove from failed if it was there
        self.failed_task_ids.discard(task_id)
    
    def mark_failed(self, task_id: str) -> None:
        """Mark a task as failed."""
        self.failed_task_ids.add(task_id)
        # Remove from completed if it was there
        self.completed_task_ids.discard(task_id)
    
    def mark_skipped(self, task_id: str) -> None:
        """Mark a task as skipped."""
        self.skipped_task_ids.add(task_id)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution state."""
        return {
            "workflow_id": self.workflow_id,
            "completed_count": len(self.completed_task_ids),
            "failed_count": len(self.failed_task_ids),
            "skipped_count": len(self.skipped_task_ids),
            "last_completed": self.last_completed_task_id,
        }


class WorkflowRestartManager:
    """
    Manages workflow restarts with cache recovery.
    
    This class implements the restart logic from Workflow.md:
    1. Identify failed tasks from previous runs
    2. Check if failed tasks still exist in the current workflow
    3. Rerun tasks that exist, skip tasks that don't
    4. Use cache to avoid re-running completed tasks
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the restart manager.
        
        Args:
            use_cache: Whether to use cache for completed tasks
        """
        self.use_cache = use_cache
        self.execution_states: Dict[str, WorkflowExecutionState] = {}
        self.failed_tasks: Dict[str, List[FailedTaskInfo]] = {}  # workflow_id -> failed tasks
    
    def register_failed_task(
        self,
        workflow_id: str,
        task_id: str,
        task_name: str,
        error_message: Optional[str] = None
    ) -> None:
        """
        Register a failed task from a previous run.
        
        Args:
            workflow_id: ID of the workflow
            task_id: ID of the failed task
            task_name: Name of the task/tool
            error_message: Optional error message
        """
        if workflow_id not in self.failed_tasks:
            self.failed_tasks[workflow_id] = []
        
        failed_info = FailedTaskInfo(
            task_id=task_id,
            task_name=task_name,
            workflow_id=workflow_id,
            error_message=error_message
        )
        
        self.failed_tasks[workflow_id].append(failed_info)
        logger.info(f"Registered failed task {task_id} ({task_name}) in workflow {workflow_id}")
    
    def get_execution_state(self, workflow_id: str) -> WorkflowExecutionState:
        """
        Get or create execution state for a workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            WorkflowExecutionState for the workflow
        """
        if workflow_id not in self.execution_states:
            self.execution_states[workflow_id] = WorkflowExecutionState(workflow_id=workflow_id)
        return self.execution_states[workflow_id]
    
    def find_failed_task_in_workflow(
        self,
        workflow: Workflow,
        failed_task_id: str
    ) -> Optional[Task]:
        """
        Find a failed task in the current workflow.
        
        Checks if the failed task (by ID or by tool name) still exists
        in the workflow.
        
        Args:
            workflow: The current workflow
            failed_task_id: ID of the failed task from previous run
            
        Returns:
            Task if found, None otherwise
        """
        # First try to find by exact task ID
        for task in workflow.tasks:
            if task.id == failed_task_id:
                return task
        
        # If not found by ID, try to find by tool name
        # This handles cases where workflow was regenerated with new task IDs
        # but same tools
        failed_info = None
        for workflow_id, failed_list in self.failed_tasks.items():
            for info in failed_list:
                if info.task_id == failed_task_id:
                    failed_info = info
                    break
            if failed_info:
                break
        
        if failed_info:
            # Try to find task with same tool name
            for task in workflow.tasks:
                if task.tool.name == failed_info.task_name:
                    logger.info(
                        f"Found failed task by tool name: {failed_info.task_name} "
                        f"(old ID: {failed_task_id}, new ID: {task.id})"
                    )
                    return task
        
        return None
    
    def should_rerun_task(
        self,
        workflow: Workflow,
        task: Task,
        execution_state: WorkflowExecutionState
    ) -> bool:
        """
        Determine if a task should be rerun.
        
        A task should be rerun if:
        1. It failed in a previous run AND still exists in the workflow
        2. It hasn't completed yet (not in cache)
        
        Args:
            workflow: The workflow
            task: The task to check
            execution_state: Current execution state
            
        Returns:
            True if task should be rerun, False if it can be skipped
        """
        # If task already completed, don't rerun
        if execution_state.is_task_completed(task.id):
            return False
        
        # If task was skipped before, don't rerun
        if execution_state.is_task_skipped(task.id):
            return False
        
        # Check if this task corresponds to a failed task
        failed_tasks = self.failed_tasks.get(workflow.id, [])
        for failed_info in failed_tasks:
            # Check by task ID
            if task.id == failed_info.task_id:
                logger.info(f"Task {task.id} ({task.tool.name}) failed before, will rerun")
                return True
            
            # Check by tool name (in case task IDs changed)
            if task.tool.name == failed_info.task_name:
                logger.info(f"Task {task.tool.name} failed before, will rerun")
                return True
        
        # If task failed in execution state, rerun it
        if execution_state.is_task_failed(task.id):
            return True
        
        return False
    
    def should_skip_task(
        self,
        workflow: Workflow,
        failed_task_id: str,
        execution_state: WorkflowExecutionState
    ) -> bool:
        """
        Determine if a failed task should be skipped.
        
        A failed task should be skipped if it's not present in the current workflow.
        
        Args:
            workflow: The current workflow
            failed_task_id: ID of the failed task
            execution_state: Current execution state
            
        Returns:
            True if task should be skipped, False otherwise
        """
        # Check if already marked as skipped
        if execution_state.is_task_skipped(failed_task_id):
            return True
        
        # Try to find the task in the workflow
        task = self.find_failed_task_in_workflow(workflow, failed_task_id)
        
        if task is None:
            # Task not found in workflow, should skip
            execution_state.mark_skipped(failed_task_id)
            logger.info(
                f"Failed task {failed_task_id} not found in workflow {workflow.id}, "
                f"will skip and continue"
            )
            return True
        
        # Task found, don't skip
        return False
    
    def prepare_restart_plan(
        self,
        workflow: Workflow
    ) -> Dict[str, Any]:
        """
        Prepare a restart plan for a workflow.
        
        Analyzes the workflow and failed tasks to determine:
        - Which tasks need to be rerun
        - Which tasks can be skipped
        - Which tasks can use cache
        
        Args:
            workflow: The workflow to restart
            
        Returns:
            Dictionary with restart plan
        """
        execution_state = self.get_execution_state(workflow.id)
        failed_tasks = self.failed_tasks.get(workflow.id, [])
        
        plan = {
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "tasks_to_rerun": [],
            "tasks_to_skip": [],
            "tasks_to_cache": [],
            "tasks_to_execute": [],
        }
        
        # Process failed tasks
        for failed_info in failed_tasks:
            task = self.find_failed_task_in_workflow(workflow, failed_info.task_id)
            
            if task is None:
                # Task not in workflow, skip it
                plan["tasks_to_skip"].append({
                    "task_id": failed_info.task_id,
                    "task_name": failed_info.task_name,
                    "reason": "not_in_workflow"
                })
                execution_state.mark_skipped(failed_info.task_id)
            else:
                # Task found, mark for rerun
                plan["tasks_to_rerun"].append({
                    "task_id": task.id,
                    "task_name": task.tool.name,
                    "old_task_id": failed_info.task_id,
                    "reason": "failed_previous_run"
                })
        
        # Process all tasks in workflow
        for task in workflow.tasks:
            if execution_state.is_task_completed(task.id):
                # Task completed, can use cache
                plan["tasks_to_cache"].append({
                    "task_id": task.id,
                    "task_name": task.tool.name
                })
            elif self.should_rerun_task(workflow, task, execution_state):
                # Task needs to be rerun
                plan["tasks_to_execute"].append({
                    "task_id": task.id,
                    "task_name": task.tool.name,
                    "reason": "rerun_failed"
                })
            elif not execution_state.is_task_skipped(task.id):
                # Normal execution
                plan["tasks_to_execute"].append({
                    "task_id": task.id,
                    "task_name": task.tool.name,
                    "reason": "normal"
                })
        
        return plan
    
    def execute_with_restart(
        self,
        workflow: Workflow,
        task_executor: Any,  # Function that executes a task
        cache_checker: Optional[Any] = None  # Function that checks cache
    ) -> Dict[str, Any]:
        """
        Execute a workflow with restart and cache recovery.
        
        This implements the main restart logic from Workflow.md:
        1. Check cache for completed tasks
        2. Rerun failed tasks that still exist
        3. Skip failed tasks that don't exist
        4. Continue with remaining tasks
        
        Args:
            workflow: The workflow to execute
            task_executor: Function to execute a task (task) -> result
            cache_checker: Optional function to check cache (task) -> cached_path or None
            
        Returns:
            Dictionary with execution results
        """
        execution_state = self.get_execution_state(workflow.id)
        plan = self.prepare_restart_plan(workflow)
        
        results = {
            "workflow_id": workflow.id,
            "completed": [],
            "failed": [],
            "skipped": [],
            "cached": [],
        }
        
        logger.info(f"Executing workflow {workflow.name} with restart plan")
        logger.info(f"  Tasks to rerun: {len(plan['tasks_to_rerun'])}")
        logger.info(f"  Tasks to skip: {len(plan['tasks_to_skip'])}")
        logger.info(f"  Tasks to cache: {len(plan['tasks_to_cache'])}")
        logger.info(f"  Tasks to execute: {len(plan['tasks_to_execute'])}")
        
        # Add skipped tasks from plan (tasks not in workflow)
        for skipped_info in plan["tasks_to_skip"]:
            results["skipped"].append(skipped_info)
        
        # Execute tasks in order
        for task in workflow.tasks:
            # Check if this task corresponds to a skipped task (by tool name)
            # This handles cases where task IDs changed but tool name is the same
            is_skipped_by_name = False
            for skipped_info in plan["tasks_to_skip"]:
                if task.tool.name == skipped_info["task_name"]:
                    is_skipped_by_name = True
                    logger.info(f"Skipping task {task.id} ({task.tool.name}) - not in workflow")
                    results["skipped"].append({
                        "task_id": task.id,
                        "task_name": task.tool.name,
                        "old_task_id": skipped_info["task_id"]
                    })
                    break
            
            if is_skipped_by_name:
                continue
            
            # Check cache first
            if self.use_cache and cache_checker:
                cached_path = cache_checker(task)
                if cached_path:
                    logger.info(f"Cache hit for task {task.id} ({task.tool.name})")
                    execution_state.mark_completed(task.id)
                    results["cached"].append({
                        "task_id": task.id,
                        "task_name": task.tool.name,
                        "cache_path": str(cached_path)
                    })
                    continue
            
            # Check if already completed
            if execution_state.is_task_completed(task.id):
                logger.debug(f"Task {task.id} already completed, skipping")
                continue
            
            # Execute the task
            try:
                logger.info(f"Executing task {task.id} ({task.tool.name})")
                result = task_executor(task)
                
                if result and result.get("success", False):
                    execution_state.mark_completed(task.id)
                    results["completed"].append({
                        "task_id": task.id,
                        "task_name": task.tool.name
                    })
                else:
                    execution_state.mark_failed(task.id)
                    self.register_failed_task(
                        workflow.id,
                        task.id,
                        task.tool.name,
                        result.get("error") if result else "Unknown error"
                    )
                    results["failed"].append({
                        "task_id": task.id,
                        "task_name": task.tool.name,
                        "error": result.get("error") if result else "Unknown error"
                    })
            except Exception as e:
                execution_state.mark_failed(task.id)
                self.register_failed_task(
                    workflow.id,
                    task.id,
                    task.tool.name,
                    str(e)
                )
                results["failed"].append({
                    "task_id": task.id,
                    "task_name": task.tool.name,
                    "error": str(e)
                })
                logger.error(f"Task {task.id} failed: {e}")
        
        return results
    
    def get_restart_summary(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get a summary of restart state for a workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Dictionary with restart summary
        """
        execution_state = self.get_execution_state(workflow_id)
        failed_tasks = self.failed_tasks.get(workflow_id, [])
        
        return {
            "workflow_id": workflow_id,
            "execution_state": execution_state.get_summary(),
            "failed_tasks_count": len(failed_tasks),
            "failed_tasks": [
                {
                    "task_id": info.task_id,
                    "task_name": info.task_name,
                    "error": info.error_message
                }
                for info in failed_tasks
            ]
        }
