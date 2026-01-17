"""
Workflow definitions for the Landseer pipeline.

A workflow is a sequence of tasks executed in a specific order to achieve a larger goal.
Each workflow represents one combination of tools from the pipeline configuration.
"""

from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional
from .tasks import Task, generate_workflow_id


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
    
