"""
SQLAlchemy models for Landseer database.

These models persist task state, worker assignments, and pipeline execution
so the system can recover from crashes and track historical data.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class TaskStatus(str, Enum):
    """Status of a task during execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerStatus(str, Enum):
    """Status of a worker."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


# Association table for task dependencies
task_dependencies = Table(
    'task_dependencies',
    Base.metadata,
    Column('task_id', String(64), ForeignKey('tasks.id'), primary_key=True),
    Column('dependency_id', String(64), ForeignKey('tasks.id'), primary_key=True),
)

# Association table for task-workflow relationships
task_workflows = Table(
    'task_workflows',
    Base.metadata,
    Column('task_id', String(64), ForeignKey('tasks.id'), primary_key=True),
    Column('workflow_id', String(64), ForeignKey('workflows.id'), primary_key=True),
)


class TaskModel(Base):
    """
    Database model for Tasks.
    
    Stores task metadata, status, priority, dependencies, and execution history.
    This enables recovery from crashes and provides audit trail.
    """
    __tablename__ = 'tasks'
    
    # Primary key - matches task.id from pipeline
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # Tool information
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_image: Mapped[str] = mapped_column(String(256), nullable=False)
    tool_command: Mapped[str] = mapped_column(String(512), nullable=True)
    tool_is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Task configuration (JSON)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Scheduling
    priority: Mapped[int] = mapped_column(Integer, default=0, index=True)
    status: Mapped[TaskStatus] = mapped_column(
        SQLEnum(TaskStatus),
        default=TaskStatus.PENDING,
        index=True
    )
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    
    # Task hash for deduplication
    task_hash: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    
    # Execution tracking
    counter: Mapped[int] = mapped_column(Integer, default=0)  # Number of workflows using this task
    attempts: Mapped[int] = mapped_column(Integer, default=0)  # Number of execution attempts
    
    # Worker assignment
    assigned_worker_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey('workers.id'),
        nullable=True,
        index=True
    )
    
    # Pipeline relationship
    pipeline_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey('pipelines.id'),
        nullable=False,
        index=True
    )
    
    # Artifact caching
    cache_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    artifact_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    
    # Execution results
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    logs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    assigned_worker: Mapped[Optional["WorkerModel"]] = relationship(
        "WorkerModel",
        back_populates="assigned_tasks"
    )
    pipeline: Mapped["PipelineModel"] = relationship(
        "PipelineModel",
        back_populates="tasks"
    )
    workflows: Mapped[List["WorkflowModel"]] = relationship(
        "WorkflowModel",
        secondary=task_workflows,
        back_populates="tasks"
    )
    
    # Self-referential many-to-many for dependencies
    dependencies: Mapped[List["TaskModel"]] = relationship(
        "TaskModel",
        secondary=task_dependencies,
        primaryjoin=id == task_dependencies.c.task_id,
        secondaryjoin=id == task_dependencies.c.dependency_id,
        backref="dependents"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "tool_image": self.tool_image,
            "tool_command": self.tool_command,
            "tool_is_baseline": self.tool_is_baseline,
            "config": self.config,
            "priority": self.priority,
            "status": self.status.value if self.status else None,
            "task_type": self.task_type,
            "task_hash": self.task_hash,
            "counter": self.counter,
            "attempts": self.attempts,
            "assigned_worker_id": self.assigned_worker_id,
            "pipeline_id": self.pipeline_id,
            "cache_key": self.cache_key,
            "artifact_path": self.artifact_path,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "dependency_ids": [dep.id for dep in self.dependencies],
            "workflow_ids": [w.id for w in self.workflows],
        }


class WorkerModel(Base):
    """
    Database model for Workers.
    
    Tracks worker registration, heartbeat, capabilities, and task assignments.
    Enables locality-based scheduling and failure detection.
    """
    __tablename__ = 'workers'
    
    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # Worker info
    hostname: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[WorkerStatus] = mapped_column(
        SQLEnum(WorkerStatus),
        default=WorkerStatus.IDLE,
        index=True
    )
    
    # Capabilities (JSON: GPU info, memory, runtime, etc.)
    capabilities: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Current task assignment
    current_task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    
    # Statistics
    tasks_completed: Mapped[int] = mapped_column(Integer, default=0)
    tasks_failed: Mapped[int] = mapped_column(Integer, default=0)
    total_execution_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    registered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    assigned_tasks: Mapped[List["TaskModel"]] = relationship(
        "TaskModel",
        back_populates="assigned_worker",
        foreign_keys=[TaskModel.assigned_worker_id]
    )
    
    # Locality tracking - workflows this worker has handled (for priority boost)
    handled_workflow_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "hostname": self.hostname,
            "status": self.status.value if self.status else None,
            "capabilities": self.capabilities,
            "current_task_id": self.current_task_id,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_execution_time_ms": self.total_execution_time_ms,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "handled_workflow_ids": self.handled_workflow_ids,
        }
    
    def update_heartbeat(self) -> None:
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = datetime.utcnow()
    
    def is_stale(self, timeout_seconds: int = 120) -> bool:
        """Check if worker is stale (no heartbeat within timeout)."""
        if not self.last_heartbeat:
            return True
        elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return elapsed > timeout_seconds


class WorkflowModel(Base):
    """
    Database model for Workflows.
    
    A workflow is a sequence of tasks executed in order.
    Multiple workflows can share tasks.
    """
    __tablename__ = 'workflows'
    
    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # Workflow info
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    
    # Pipeline relationship
    pipeline_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey('pipelines.id'),
        nullable=False,
        index=True
    )
    
    # Status (derived from tasks but cached for performance)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    completed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    failed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    pipeline: Mapped["PipelineModel"] = relationship(
        "PipelineModel",
        back_populates="workflows"
    )
    tasks: Mapped[List["TaskModel"]] = relationship(
        "TaskModel",
        secondary=task_workflows,
        back_populates="workflows"
    )
    # Evaluation results for this workflow (one per evaluator)
    evaluation_results: Mapped[List["EvaluationResultModel"]] = relationship(
        "EvaluationResultModel",
        back_populates="workflow",
        cascade="all, delete-orphan"
    )
    
    def to_dict(self, include_evaluations: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Args:
            include_evaluations: If True, include evaluation results
        """
        result = {
            "id": self.id,
            "name": self.name,
            "pipeline_id": self.pipeline_id,
            "status": self.status,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "task_count": len(self.tasks) if self.tasks else 0,
            "task_ids": [t.id for t in self.tasks] if self.tasks else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
        
        if include_evaluations and self.evaluation_results:
            result["evaluations"] = {
                er.evaluator_name: er.metrics 
                for er in self.evaluation_results
            }
            result["evaluation_count"] = len(self.evaluation_results)
        
        return result
    
    def get_evaluation(self, evaluator_name: str) -> Optional["EvaluationResultModel"]:
        """Get evaluation result for a specific evaluator."""
        if not self.evaluation_results:
            return None
        for er in self.evaluation_results:
            if er.evaluator_name == evaluator_name:
                return er
        return None
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all evaluation metrics grouped by evaluator."""
        if not self.evaluation_results:
            return {}
        return {
            er.evaluator_name: er.metrics 
            for er in self.evaluation_results 
            if er.success and not er.skipped
        }


class PipelineModel(Base):
    """
    Database model for Pipelines.
    
    A pipeline is a collection of workflows used to evaluate ML defenses.
    """
    __tablename__ = 'pipelines'
    
    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # Pipeline info
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    
    # Configuration (JSON)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    dataset_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    model_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(32), default="pending")
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    workflows: Mapped[List["WorkflowModel"]] = relationship(
        "WorkflowModel",
        back_populates="pipeline",
        cascade="all, delete-orphan"
    )
    tasks: Mapped[List["TaskModel"]] = relationship(
        "TaskModel",
        back_populates="pipeline",
        cascade="all, delete-orphan"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "config": self.config,
            "dataset_config": self.dataset_config,
            "model_config": self.model_config,
            "status": self.status,
            "workflow_count": len(self.workflows) if self.workflows else 0,
            "task_count": len(self.tasks) if self.tasks else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class ArtifactModel(Base):
    """
    Database model for Artifacts.
    
    Tracks artifacts stored in the AI store (MinIO) for caching and retrieval.
    """
    __tablename__ = 'artifacts'
    
    # Primary key - the cache key/hash
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # Associated task
    task_id: Mapped[str] = mapped_column(String(64), ForeignKey('tasks.id'), index=True)
    
    # Storage location
    storage_type: Mapped[str] = mapped_column(String(32), default="minio")  # minio, local
    bucket: Mapped[str] = mapped_column(String(128), default="landseer-artifacts")
    object_key: Mapped[str] = mapped_column(String(512), nullable=False)
    
    # Metadata
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    content_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    checksum: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    
    # Provenance (JSON: parent hashes, tool info)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "storage_type": self.storage_type,
            "bucket": self.bucket,
            "object_key": self.object_key,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "checksum": self.checksum,
            "provenance": self.provenance,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
        }


class EvaluationResultModel(Base):
    """
    Database model for Evaluation Results.
    
    Stores evaluation metrics for each WORKFLOW from each evaluator.
    
    IMPORTANT: Evaluations are per-workflow, NOT per-task.
    - Each workflow gets evaluated by multiple evaluators (adversarial, fairness, etc.)
    - Each evaluator produces one result per workflow
    - The unique constraint (workflow_id, evaluator_name) enforces this
    
    The evaluation_task_id field references which EvaluationTask produced this result,
    but the result belongs to the workflow, not the task.
    """
    __tablename__ = 'evaluation_results'
    
    # Primary key
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # --- PRIMARY RELATIONSHIP: Workflow ---
    # Evaluation is per-workflow. This is the main relationship.
    workflow_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey('workflows.id'),
        nullable=False,
        index=True
    )
    
    # Pipeline for denormalization (faster queries)
    pipeline_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey('pipelines.id'),
        nullable=False,
        index=True
    )
    
    # --- OPTIONAL: Which task produced this result ---
    # This references the EvaluationTask that ran the evaluator.
    # It's optional because results can be imported without task reference.
    evaluation_task_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey('tasks.id'),
        nullable=True,
        index=True
    )
    
    # Evaluator info
    evaluator_name: Mapped[str] = mapped_column(String(64), nullable=False)
    evaluator_image: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    
    # Metrics as JSON
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Status
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    skip_reason: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Parameters used for evaluation
    parameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    workflow: Mapped["WorkflowModel"] = relationship(
        "WorkflowModel",
        back_populates="evaluation_results"
    )
    
    # Unique constraint: one result per workflow per evaluator
    # This enforces that evaluations are per-workflow, not per-task
    __table_args__ = (
        UniqueConstraint('workflow_id', 'evaluator_name', name='unique_workflow_evaluator'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "pipeline_id": self.pipeline_id,
            "evaluation_task_id": self.evaluation_task_id,
            "evaluator_name": self.evaluator_name,
            "evaluator_image": self.evaluator_image,
            "metrics": self.metrics,
            "success": self.success,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    def from_evaluation_result(
        cls,
        result_id: str,
        workflow_id: str,
        pipeline_id: str,
        evaluator_name: str,
        result_data: Dict[str, Any],
        evaluation_task_id: Optional[str] = None,
        evaluator_image: Optional[str] = None
    ) -> "EvaluationResultModel":
        """
        Create from evaluation result JSON.
        
        Evaluations are per-workflow, not per-task. Each workflow is evaluated
        by multiple evaluators, and each evaluator produces metrics for that
        workflow's defense effectiveness.
        
        Args:
            result_id: Unique ID for this result
            workflow_id: Workflow that was evaluated (the defense workflow)
            pipeline_id: Pipeline the workflow belongs to
            evaluator_name: Name of the evaluator (e.g., 'adversarial', 'fairness')
            result_data: Parsed evaluation_results.json from container
            evaluation_task_id: Optional ID of the EvaluationTask that ran this
            evaluator_image: Optional container image used
            
        Returns:
            EvaluationResultModel instance
        """
        return cls(
            id=result_id,
            workflow_id=workflow_id,
            pipeline_id=pipeline_id,
            evaluation_task_id=evaluation_task_id,
            evaluator_name=evaluator_name,
            evaluator_image=evaluator_image,
            metrics=result_data.get("metrics", {}),
            success=result_data.get("success", True),
            skipped=result_data.get("skipped", False),
            skip_reason=result_data.get("skip_reason"),
            error=result_data.get("error"),
            parameters=result_data.get("parameters")
        )
