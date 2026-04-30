"""
Workflow definitions for the Landseer pipeline.

A workflow is a sequence of tasks executed in a specific order to achieve a larger goal.
Each workflow represents one combination of tools from the pipeline configuration.

Per Workflow.md:
- If a tool has failed and restarting with cache enabled, rerun the failed tool
- If the tool is not present in the workflow, skip it and continue
- If the tool is present in the workflow, rerun it
"""

from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional, TYPE_CHECKING
import logging

from .tasks import Task, TaskStatus, generate_workflow_id

if TYPE_CHECKING:
    from .workflow_restart import WorkflowRestartManager

logger = logging.getLogger(__name__)


@dataclass
class Workflow:
    """
    A workflow is a sequence of tasks executed in order.
    
    Workflows represent combinations of tools from different pipeline stages.
    Each workflow has a unique identifier based on the tool combination.
    
    Attributes:
        id: Unique workflow identifier (e.g., "workflow_1")
        name: Human-readable workflow name (e.g., "comb_001")
        tasks: Ordered list of tasks to execute
        metadata: Additional workflow metadata
        pipeline_id: ID of the pipeline this workflow belongs to
    """
    name: str
    tasks: List[Task] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default="", init=False)
    pipeline_id: str = field(default="")
    
    def __post_init__(self):
        """Initialize workflow with unique ID."""
        if not self.id:
            self.id = generate_workflow_id()
    
    def add_task(self, task: Task) -> None:
        """
        Add a task to the workflow.
        
        Args:
            task: Task to add to the workflow
        """
        self.tasks.append(task)
        # Register task with this workflow
        if self.pipeline_id:
            task.add_to_workflow(self.id, self.pipeline_id)
    
    def run(self, data: Any = None) -> Any:
        """
        Execute all tasks in the workflow in order.
        
        Args:
            data: Initial input data
            
        Returns:
            Final output data after all tasks complete
        """
        current_data = data
        for task in self.tasks:
            current_data = task.run(current_data)
        return current_data
    
    def get_tasks_by_type(self, task_type) -> List[Task]:
        """
        Get all tasks of a specific type.
        
        Args:
            task_type: TaskType to filter by
            
        Returns:
            List of tasks matching the type
        """
        return [task for task in self.tasks if task.task_type == task_type]
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """
        Get a task by its ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task if found, None otherwise
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_task_by_tool_name(self, tool_name: str) -> Optional[Task]:
        """
        Get a task by its tool name.
        
        Useful for finding tasks when IDs may have changed between runs.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Task if found, None otherwise
        """
        for task in self.tasks:
            if task.tool.name == tool_name:
                return task
        return None
    
    def has_task(self, task_id: str) -> bool:
        """
        Check if a task exists in this workflow.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            True if task exists, False otherwise
        """
        return self.get_task_by_id(task_id) is not None
    
    def run_with_restart(
        self,
        restart_manager: "WorkflowRestartManager",
        task_executor: Any,
        cache_checker: Optional[Any] = None,
        data: Any = None
    ) -> Dict[str, Any]:
        """
        Execute workflow with restart and cache recovery.
        
        Per Workflow.md:
        - If a tool failed and is present in workflow, rerun it
        - If a tool failed and is not present, skip it and continue
        
        Args:
            restart_manager: WorkflowRestartManager instance
            task_executor: Function to execute a task (task, data) -> result
            cache_checker: Optional function to check cache (task) -> cached_path or None
            data: Initial input data
            
        Returns:
            Dictionary with execution results
        """
        return restart_manager.execute_with_restart(
            workflow=self,
            task_executor=task_executor,
            cache_checker=cache_checker
        )
    
    def prepare_restart(
        self,
        restart_manager: "WorkflowRestartManager"
    ) -> Dict[str, Any]:
        """
        Prepare a restart plan for this workflow.
        
        Args:
            restart_manager: WorkflowRestartManager instance
            
        Returns:
            Dictionary with restart plan
        """
        return restart_manager.prepare_restart_plan(self)
    
    def __repr__(self) -> str:
        """String representation of the workflow."""
        return f"Workflow(name='{self.name}', tasks={len(self.tasks)})"


class WorkflowFactory:
    """Factory for creating workflow instances."""
    
    @classmethod
    def create_workflow(
        cls,
        name: str,
        tasks: Optional[List[Task]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        pipeline_id: str = ""
    ) -> Workflow:
        """
        Create a workflow instance.
        
        Args:
            name: Workflow name/identifier
            tasks: List of tasks for the workflow
            metadata: Optional metadata dictionary
            pipeline_id: ID of the pipeline this workflow belongs to
            
        Returns:
            Created workflow instance
        """
        workflow = Workflow(
            name=name,
            tasks=tasks or [],
            metadata=metadata or {},
            pipeline_id=pipeline_id
        )
        
        # Register tasks with the workflow
        for task in workflow.tasks:
            if pipeline_id:
                task.add_to_workflow(workflow.id, pipeline_id)
        
        return workflow
    
