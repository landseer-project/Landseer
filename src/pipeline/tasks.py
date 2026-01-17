"""
Task definitions for the Landseer pipeline.

Tasks are individual units of work that can be executed independently.
Each task has a priority, tool, config, and dependencies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Dict, Set
from enum import Enum
import hashlib
import json
from collections import Counter

from .tools import ToolDefinition


class TaskType(str, Enum):
    """Types of tasks in the pipeline."""
    PRE_TRAINING = "pre_training"
    IN_TRAINING = "in_training"
    POST_TRAINING = "post_training"
    DEPLOYMENT = "deployment"


class TaskStatus(Enum):
    """Status of a task during execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Global ID counters
_task_id_counter = 0
_tool_id_counter = 0
_workflow_id_counter = 0
_pipeline_id_counter = 0


def generate_task_id() -> str:
    """Generate a unique task ID."""
    global _task_id_counter
    _task_id_counter += 1
    return f"task_{_task_id_counter}"


def generate_tool_id() -> str:
    """Generate a unique tool ID."""
    global _tool_id_counter
    _tool_id_counter += 1
    return f"tool_{_tool_id_counter}"


def generate_workflow_id() -> str:
    """Generate a unique workflow ID."""
    global _workflow_id_counter
    _workflow_id_counter += 1
    return f"workflow_{_workflow_id_counter}"


def generate_pipeline_id() -> str:
    """Generate a unique pipeline ID."""
    global _pipeline_id_counter
    _pipeline_id_counter += 1
    return f"pipeline_{_pipeline_id_counter}"


@dataclass
class Task(ABC):
    """
    Abstract base class for all tasks.
    
    Attributes:
        id: Unique task identifier (e.g., "task_1")
        tool: Tool definition for this task
        config: Configuration parameters for the tool
        priority: Integer priority for scheduling (updated by scheduler)
        dependencies: List of other tasks that must complete before this task
        task_type: Type of task (pre/in/post/deploy)
        status: Current execution status of the task
        counter: Number of workflows this task is part of
        workflows: Set of workflow IDs this task belongs to
        pipeline_id: ID of the pipeline this task belongs to
    """
    tool: ToolDefinition
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    dependencies: List["Task"] = field(default_factory=list)
    id: str = field(default="", init=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, init=False)
    counter: int = field(default=0, init=False)
    workflows: Set[str] = field(default_factory=set, init=False)
    pipeline_id: str = field(default="", init=False)
    _hash: str = field(default="", init=False, repr=False)
    
    def __post_init__(self):
        """Initialize task with unique ID and compute hash."""
        if not self.id:
            self.id = generate_task_id()
        self._compute_hash()
    
    def _compute_hash(self) -> str:
        """
        Compute a hash based on tool, config, and dependencies.
        Tasks with the same hash can be reused.
        """
        # Create a deterministic representation
        hash_data = {
            "tool_name": self.tool.name,
            "tool_image": self.tool.container.image,
            "tool_command": self.tool.container.command,
            "config": self.config,
            "dependencies": sorted([dep.id for dep in self.dependencies])
        }
        
        # Convert to JSON string and hash
        json_str = json.dumps(hash_data, sort_keys=True)
        self._hash = hashlib.sha256(json_str.encode()).hexdigest()
        return self._hash
    
    def get_hash(self) -> str:
        """Get the hash of this task."""
        if not self._hash:
            self._compute_hash()
        return self._hash
    
    def __hash__(self) -> int:
        """Make task hashable for use in sets and dicts."""
        return int(self.get_hash()[:16], 16)  # Use first 16 hex chars as integer
    
    def __eq__(self, other) -> bool:
        """Tasks are equal if they have the same hash."""
        if not isinstance(other, Task):
            return False
        return self.get_hash() == other.get_hash()
    
    def add_to_workflow(self, workflow_id: str, pipeline_id: str) -> None:
        """
        Add this task to a workflow.
        
        Args:
            workflow_id: ID of the workflow
            pipeline_id: ID of the pipeline
        """
        if self.pipeline_id and self.pipeline_id != pipeline_id:
            raise ValueError(f"Task {self.id} already belongs to pipeline {self.pipeline_id}, cannot add to {pipeline_id}")
        
        self.pipeline_id = pipeline_id
        self.workflows.add(workflow_id)
        self.counter += 1
    
    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Return the type of this task."""
        pass
    
    @abstractmethod
    def run(self, data: Any) -> Any:
        """
        Execute the task.
        
        Args:
            data: Input data for the task
            
        Returns:
            Output data from the task
        """
        pass


@dataclass
class PreTrainingTask(Task):
    """Task executed before model training (pre-processing, data preparation)."""
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.PRE_TRAINING
    
    def run(self, data: Any) -> Any:
        """Execute pre-training logic."""
        # Actual implementation will be handled by the tool runner
        return data


@dataclass
class InTrainingTask(Task):
    """Task executed during model training (training defenses, watermarking)."""
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.IN_TRAINING
    
    def run(self, data: Any) -> Any:
        """Execute in-training logic."""
        # Actual implementation will be handled by the tool runner
        return data


@dataclass
class PostTrainingTask(Task):
    """Task executed after model training (post-processing, fine-tuning)."""
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.POST_TRAINING
    
    def run(self, data: Any) -> Any:
        """Execute post-training logic."""
        # Actual implementation will be handled by the tool runner
        return data


@dataclass
class DeploymentTask(Task):
    """Task executed during model deployment (defenses, runtime protections)."""
    
    @property
    def task_type(self) -> TaskType:
        return TaskType.DEPLOYMENT
    
    def run(self, data: Any) -> Any:
        """Execute deployment logic."""
        # Actual implementation will be handled by the tool runner
        return data


# Task registry for deduplication
_task_registry: Dict[str, Task] = {}


def get_or_create_task(
    task_type: TaskType,
    tool: ToolDefinition,
    config: Dict[str, Any] = None,
    priority: int = 0,
    dependencies: List[Task] = None,
    pipeline_id: str = ""
) -> Task:
    """
    Get an existing task with the same hash or create a new one.
    Tasks can only be shared if they belong to the same pipeline.
    
    Args:
        task_type: Type of task to create
        tool: Tool definition
        config: Configuration parameters
        priority: Task priority
        dependencies: List of dependent tasks
        pipeline_id: ID of the pipeline this task belongs to
        
    Returns:
        Existing or newly created task instance
    """
    # Create a temporary task to compute its hash
    temp_task = TaskFactory.create_task(
        task_type=task_type,
        tool=tool,
        config=config,
        priority=priority,
        dependencies=dependencies
    )
    
    task_hash = temp_task.get_hash()
    
    # Check if a task with this hash exists in the same pipeline
    for existing_task in _task_registry.values():
        if (existing_task.get_hash() == task_hash and 
            (not existing_task.pipeline_id or existing_task.pipeline_id == pipeline_id)):
            return existing_task
    
    # No matching task found, register the new one
    temp_task.pipeline_id = pipeline_id
    _task_registry[temp_task.id] = temp_task
    return temp_task


def clear_task_registry():
    """Clear the task registry. Useful for testing."""
    global _task_registry
    _task_registry.clear()


# Task factory for creating tasks by type
class TaskFactory:
    """Factory for creating task instances."""
    
    _task_classes = {
        TaskType.PRE_TRAINING: PreTrainingTask,
        TaskType.IN_TRAINING: InTrainingTask,
        TaskType.POST_TRAINING: PostTrainingTask,
        TaskType.DEPLOYMENT: DeploymentTask,
    }
    
    @classmethod
    def create_task(
        cls,
        task_type: TaskType,
        tool: ToolDefinition,
        config: Dict[str, Any] = None,
        priority: int = 0,
        dependencies: List[Task] = None
    ) -> Task:
        """
        Create a task of the specified type.
        
        Args:
            task_type: Type of task to create
            tool: Tool definition
            config: Configuration parameters
            priority: Task priority
            dependencies: List of dependent tasks
            
        Returns:
            Created task instance
        """
        if task_type not in cls._task_classes:
            raise ValueError(f"Unknown task type: {task_type}")
        
        task_class = cls._task_classes[task_type]
        return task_class(
            tool=tool,
            config=config or {},
            priority=priority,
            dependencies=dependencies or []
        )
    
