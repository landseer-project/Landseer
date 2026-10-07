"""
SQLAlchemy models for Landseer database.

Normalized experiment-centric schema:

  Dataset → DatasetVariant
  Model
  Tool
  Experiment ──< experiment_tools / experiment_attacks >── Tool
  ExperimentRun → Workflow ──< workflow_tasks >── Task
                → Task → Tool
  Task ──< task_dependencies >── Task
  Artifact ──< artifact_parents >── Artifact
  EvaluationResult → Workflow, Task (tool via task)
  Worker ──< worker_handled_workflows >── Workflow
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
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


# ---------------------------------------------------------------------------
# Association tables
# ---------------------------------------------------------------------------

experiment_tools = Table(
    "experiment_tools",
    Base.metadata,
    Column("experiment_id", String(64), ForeignKey("experiments.id"), primary_key=True),
    Column("tool_id", String(64), ForeignKey("tools.id"), primary_key=True),
)

experiment_attacks = Table(
    "experiment_attacks",
    Base.metadata,
    Column("experiment_id", String(64), ForeignKey("experiments.id"), primary_key=True),
    Column("tool_id", String(64), ForeignKey("tools.id"), primary_key=True),
)

workflow_tasks = Table(
    "workflow_tasks",
    Base.metadata,
    Column("workflow_id", String(64), ForeignKey("workflows.id"), primary_key=True),
    Column("task_id", String(64), ForeignKey("tasks.id"), primary_key=True),
    Column("position", Integer, nullable=False, default=0),
)

task_dependencies = Table(
    "task_dependencies",
    Base.metadata,
    Column("task_id", String(64), ForeignKey("tasks.id"), primary_key=True),
    Column("dependency_id", String(64), ForeignKey("tasks.id"), primary_key=True),
)

artifact_parents = Table(
    "artifact_parents",
    Base.metadata,
    Column("artifact_id", String(64), ForeignKey("artifacts.id"), primary_key=True),
    Column("parent_artifact_id", String(64), ForeignKey("artifacts.id"), primary_key=True),
    Column("position", Integer, nullable=False, default=0),
)

worker_handled_workflows = Table(
    "worker_handled_workflows",
    Base.metadata,
    Column("worker_id", String(64), ForeignKey("workers.id"), primary_key=True),
    Column("workflow_id", String(64), ForeignKey("workflows.id"), primary_key=True),
)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class WorkerStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


class RuntimeType(str, Enum):
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    APPTAINER = "apptainer"
    OTHER = "other"


class WorkerModel(Base):
    __tablename__ = "workers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    hostname: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[WorkerStatus] = mapped_column(
        SQLEnum(WorkerStatus),
        default=WorkerStatus.IDLE,
        index=True,
    )
    has_gpu: Mapped[bool] = mapped_column(Boolean, default=False)
    supported_runtimes: Mapped[List[str]] = mapped_column(JSON, default=list)
    capabilities: Mapped[dict] = mapped_column(JSON, default=dict)

    # use_alter: tasks also FK to workers (assigned_worker_id)
    current_task_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("tasks.id", use_alter=True, name="fk_workers_current_task"),
        nullable=True,
    )

    tasks_completed: Mapped[int] = mapped_column(Integer, default=0)
    tasks_failed: Mapped[int] = mapped_column(Integer, default=0)
    total_execution_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    registered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    assigned_tasks: Mapped[List["TaskModel"]] = relationship(
        "TaskModel",
        back_populates="assigned_worker",
        foreign_keys="[TaskModel.assigned_worker_id]",
    )
    current_task: Mapped[Optional["TaskModel"]] = relationship(
        "TaskModel",
        foreign_keys=[current_task_id],
        post_update=True,
    )
    handled_workflows: Mapped[List["WorkflowModel"]] = relationship(
        "WorkflowModel",
        secondary=worker_handled_workflows,
    )

    @property
    def handled_workflow_ids(self) -> List[str]:
        return [w.id for w in self.handled_workflows]

    def to_dict(self) -> Dict[str, Any]:
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
        self.last_heartbeat = datetime.utcnow()

    def is_stale(self, timeout_seconds: int = 120) -> bool:
        if not self.last_heartbeat:
            return True
        elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return elapsed > timeout_seconds


# ---------------------------------------------------------------------------
# Tools / Datasets / Models
# ---------------------------------------------------------------------------

class ToolDefenseType(str, Enum):
    ADVERSARIAL = "adversarial"
    FAIRNESS = "fairness"
    OTHER = "other"
    NONE = "none"


class ToolStage(str, Enum):
    PRE_TRAINING = "pre_training"
    DURING_TRAINING = "during_training"
    POST_TRAINING = "post_training"
    DEPLOYMENT = "deployment"
    OTHER = "other"


class ToolType(str, Enum):
    DEFENSE = "defense"
    BASELINE = "baseline"
    EVAL = "eval"
    ATTACK = "attack"


class ToolModel(Base):
    __tablename__ = "tools"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_type: Mapped[ToolType] = mapped_column(SQLEnum(ToolType), nullable=False)
    tool_defense_type: Mapped[ToolDefenseType] = mapped_column(
        SQLEnum(ToolDefenseType),
        nullable=False,
        default=ToolDefenseType.NONE,
    )
    tool_stage: Mapped[ToolStage] = mapped_column(SQLEnum(ToolStage), nullable=False)
    tool_image: Mapped[str] = mapped_column(String(512), nullable=False)
    tool_parameters: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)

    @property
    def tool_is_baseline(self) -> bool:
        return self.tool_type == ToolType.BASELINE


class DatasetType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    OTHER = "other"


class DatasetModel(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset_type: Mapped[DatasetType] = mapped_column(SQLEnum(DatasetType), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    variants: Mapped[List["DatasetVariantModel"]] = relationship(
        "DatasetVariantModel",
        back_populates="dataset",
    )


class DatasetVariantType(str, Enum):
    CLEAN = "clean"
    POISONED = "poisoned"
    OTHER = "other"


class DatasetVariantModel(Base):
    __tablename__ = "dataset_variants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(64), ForeignKey("datasets.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset_variant_type: Mapped[DatasetVariantType] = mapped_column(
        SQLEnum(DatasetVariantType),
        nullable=False,
    )
    dataset_path: Mapped[str] = mapped_column(String(512), nullable=False)

    dataset: Mapped["DatasetModel"] = relationship("DatasetModel", back_populates="variants")
    experiments: Mapped[List["ExperimentModel"]] = relationship(
        "ExperimentModel",
        back_populates="dataset_variant",
    )


class ModelType(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    OTHER = "other"


class ModelModel(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_type: Mapped[ModelType] = mapped_column(SQLEnum(ModelType), nullable=False)
    model_path: Mapped[str] = mapped_column(String(512), nullable=False)

    experiments: Mapped[List["ExperimentModel"]] = relationship(
        "ExperimentModel",
        back_populates="model",
    )


# ---------------------------------------------------------------------------
# Experiment hierarchy
# ---------------------------------------------------------------------------

class ExperimentModel(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Optional until catalog rows exist; YAML-driven registration may set paths first.
    model_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("models.id"), nullable=True
    )
    dataset_variant_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("dataset_variants.id"),
        nullable=True,
    )

    config_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    attack_config_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    config_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    model: Mapped[Optional["ModelModel"]] = relationship(
        "ModelModel", back_populates="experiments"
    )
    dataset_variant: Mapped[Optional["DatasetVariantModel"]] = relationship(
        "DatasetVariantModel",
        back_populates="experiments",
    )
    tools: Mapped[List["ToolModel"]] = relationship(
        "ToolModel",
        secondary=experiment_tools,
    )
    attacks: Mapped[List["ToolModel"]] = relationship(
        "ToolModel",
        secondary=experiment_attacks,
    )
    runs: Mapped[List["ExperimentRunModel"]] = relationship(
        "ExperimentRunModel",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )

    # Convenience for callers that used config.name
    @property
    def name(self) -> str:
        return self.experiment_name

    @name.setter
    def name(self, value: str) -> None:
        self.experiment_name = value


class ExperimentRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STOPPING = "stopping"


class ExperimentRunModel(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("experiments.id"),
        nullable=False,
        index=True,
    )
    run_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[ExperimentRunStatus] = mapped_column(
        SQLEnum(ExperimentRunStatus),
        default=ExperimentRunStatus.PENDING,
        index=True,
    )
    use_cache: Mapped[bool] = mapped_column(Boolean, default=True)
    tools_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    experiment: Mapped["ExperimentModel"] = relationship(
        "ExperimentModel",
        back_populates="runs",
    )
    workflows: Mapped[List["WorkflowModel"]] = relationship(
        "WorkflowModel",
        back_populates="experiment_run",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("experiment_id", "run_number", name="unique_experiment_run"),
    )


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowModel(Base):
    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("experiment_runs.id"),
        nullable=False,
        index=True,
    )
    status: Mapped[WorkflowStatus] = mapped_column(
        SQLEnum(WorkflowStatus),
        default=WorkflowStatus.PENDING,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    experiment_run: Mapped["ExperimentRunModel"] = relationship(
        "ExperimentRunModel",
        back_populates="workflows",
    )
    tasks: Mapped[List["TaskModel"]] = relationship(
        "TaskModel",
        secondary=workflow_tasks,
        order_by=workflow_tasks.c.position,
    )
    evaluation_results: Mapped[List["EvaluationResultModel"]] = relationship(
        "EvaluationResultModel",
        back_populates="workflow",
        cascade="all, delete-orphan",
    )
    artifacts: Mapped[List["ArtifactModel"]] = relationship(
        "ArtifactModel",
        back_populates="workflow",
    )

    def set_ordered_tasks(self, session, ordered_tasks: List["TaskModel"]) -> None:
        """Replace membership and write ``position`` 0..n-1."""
        session.execute(
            workflow_tasks.delete().where(workflow_tasks.c.workflow_id == self.id)
        )
        for i, task in enumerate(ordered_tasks):
            session.execute(
                workflow_tasks.insert().values(
                    workflow_id=self.id,
                    task_id=task.id,
                    position=i,
                )
            )
        session.expire(self, ["tasks"])


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """DB task kind. Pipeline stage lives on ToolModel.tool_stage."""

    TOOL = "tool"
    EVALUATION = "evaluation"
    OTHER = "other"


class TaskModel(Base):
    """Executable unit; tool metadata lives on ToolModel via tool_id."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_type: Mapped[TaskType] = mapped_column(SQLEnum(TaskType), nullable=False)
    tool_id: Mapped[str] = mapped_column(String(64), ForeignKey("tools.id"), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, default=dict)

    priority: Mapped[int] = mapped_column(Integer, default=0, index=True)
    status: Mapped[TaskStatus] = mapped_column(
        SQLEnum(TaskStatus),
        default=TaskStatus.PENDING,
        index=True,
    )
    task_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    counter: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)

    assigned_worker_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("workers.id"),
        nullable=True,
        index=True,
    )
    experiment_run_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("experiment_runs.id"),
        nullable=True,
        index=True,
    )

    use_cache: Mapped[bool] = mapped_column(Boolean, default=False)
    artifact_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("artifacts.id", use_alter=True, name="fk_tasks_artifact"),
        nullable=True,
        index=True,
    )

    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    logs: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    assigned_worker: Mapped[Optional["WorkerModel"]] = relationship(
        "WorkerModel",
        back_populates="assigned_tasks",
        foreign_keys=[assigned_worker_id],
    )
    tool: Mapped["ToolModel"] = relationship("ToolModel")
    experiment_run: Mapped[Optional["ExperimentRunModel"]] = relationship("ExperimentRunModel")
    artifact: Mapped[Optional["ArtifactModel"]] = relationship(
        "ArtifactModel",
        foreign_keys=[artifact_id],
        post_update=True,
    )
    dependencies: Mapped[List["TaskModel"]] = relationship(
        "TaskModel",
        secondary=task_dependencies,
        primaryjoin=id == task_dependencies.c.task_id,
        secondaryjoin=id == task_dependencies.c.dependency_id,
        backref="dependents",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_type": self.task_type.value if self.task_type else None,
            "tool_id": self.tool_id,
            "config": self.config,
            "priority": self.priority,
            "status": self.status.value if self.status else None,
            "task_hash": self.task_hash,
            "counter": self.counter,
            "attempts": self.attempts,
            "assigned_worker_id": self.assigned_worker_id,
            "experiment_run_id": self.experiment_run_id,
            "use_cache": self.use_cache,
            "artifact_id": self.artifact_id,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "dependency_ids": [dep.id for dep in self.dependencies],
        }


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

class StorageType(str, Enum):
    MINIO = "minio"
    LOCAL = "local"
    BOTH = "both"


class ArtifactType(str, Enum):
    DATASET = "dataset"
    MODEL = "model"
    TOOL_OUTPUT = "tool_output"
    EVALUATION = "evaluation"
    OTHER = "other"


class ArtifactModel(Base):
    """
    Content-addressed artifact in MinIO/local cache.

    ``id`` is the cache key. Parent lineage is ``parents`` (ordered).
    """

    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    artifact_type: Mapped[ArtifactType] = mapped_column(
        SQLEnum(ArtifactType),
        default=ArtifactType.TOOL_OUTPUT,
        nullable=False,
        index=True,
    )

    task_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("tasks.id", use_alter=True, name="fk_artifacts_task"),
        nullable=True,
        index=True,
    )
    experiment_run_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("experiment_runs.id"),
        nullable=True,
        index=True,
    )
    workflow_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("workflows.id"),
        nullable=True,
        index=True,
    )

    storage_type: Mapped[StorageType] = mapped_column(
        SQLEnum(StorageType),
        default=StorageType.MINIO,
        nullable=False,
    )
    bucket: Mapped[str] = mapped_column(String(128), default="landseer-artifacts")
    object_key: Mapped[str] = mapped_column(String(512), nullable=False)
    local_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    content_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    checksum: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Free-form extras (worker_id, config hash, etc.) — not relational identity
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    task: Mapped[Optional["TaskModel"]] = relationship(
        "TaskModel",
        foreign_keys=[task_id],
        post_update=True,
    )
    workflow: Mapped[Optional["WorkflowModel"]] = relationship(
        "WorkflowModel",
        back_populates="artifacts",
    )
    experiment_run: Mapped[Optional["ExperimentRunModel"]] = relationship(
        "ExperimentRunModel",
    )
    parents: Mapped[List["ArtifactModel"]] = relationship(
        "ArtifactModel",
        secondary=artifact_parents,
        primaryjoin=id == artifact_parents.c.artifact_id,
        secondaryjoin=id == artifact_parents.c.parent_artifact_id,
        order_by=artifact_parents.c.position,
        backref="children",
    )

    @property
    def parent_artifact_ids(self) -> List[str]:
        return [p.id for p in self.parents]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "artifact_type": self.artifact_type.value if self.artifact_type else None,
            "task_id": self.task_id,
            "experiment_run_id": self.experiment_run_id,
            "workflow_id": self.workflow_id,
            "storage_type": self.storage_type.value if self.storage_type else None,
            "bucket": self.bucket,
            "object_key": self.object_key,
            "local_path": self.local_path,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "checksum": self.checksum,
            "parent_artifact_ids": self.parent_artifact_ids,
            "provenance": self.provenance,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed_at": (
                self.last_accessed_at.isoformat() if self.last_accessed_at else None
            ),
        }


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------

class EvaluationResultModel(Base):
    """
    Per-workflow metrics from one evaluation task.

    Run scope is ``workflow.experiment_run_id`` (not duplicated here).
    Tool identity is ``evaluation_task.tool``.
    """

    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    workflow_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("workflows.id"),
        nullable=False,
        index=True,
    )
    evaluation_task_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tasks.id"),
        nullable=False,
        index=True,
    )

    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    skip_reason: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    workflow: Mapped["WorkflowModel"] = relationship(
        "WorkflowModel",
        back_populates="evaluation_results",
    )
    evaluation_task: Mapped["TaskModel"] = relationship(
        "TaskModel",
        foreign_keys=[evaluation_task_id],
    )

    __table_args__ = (
        UniqueConstraint(
            "workflow_id",
            "evaluation_task_id",
            name="unique_workflow_evaluation_task",
        ),
    )

    @property
    def experiment_run_id(self) -> Optional[str]:
        return self.workflow.experiment_run_id if self.workflow else None

    @property
    def evaluator_name(self) -> Optional[str]:
        task = self.evaluation_task
        if task is None or task.tool is None:
            return None
        return task.tool.tool_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "experiment_run_id": self.experiment_run_id,
            "evaluation_task_id": self.evaluation_task_id,
            "evaluator_name": self.evaluator_name,
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
        evaluation_task_id: str,
        result_data: Dict[str, Any],
    ) -> "EvaluationResultModel":
        """Build a row from evaluator ``evaluation_results.json`` payload."""
        return cls(
            id=result_id,
            workflow_id=workflow_id,
            evaluation_task_id=evaluation_task_id,
            metrics=result_data.get("metrics", {}),
            success=result_data.get("success", True),
            skipped=result_data.get("skipped", False),
            skip_reason=result_data.get("skip_reason"),
            error=result_data.get("error"),
            parameters=result_data.get("parameters"),
        )

