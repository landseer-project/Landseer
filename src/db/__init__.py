"""
Database module for Landseer.

Provides SQLAlchemy persistence for experiments, runs, workflows, tasks,
workers, artifacts, and evaluation results.
"""

from .connection import (
    DatabaseConfig,
    Database,
    get_database,
    init_database,
    get_session,
    session_scope,
)
from .models import (
    Base,
    TaskModel,
    WorkerModel,
    WorkflowModel,
    ExperimentModel,
    ExperimentRunModel,
    ArtifactModel,
    EvaluationResultModel,
    ToolModel,
    TaskStatus as DBTaskStatus,
    WorkerStatus,
    ExperimentRunStatus,
    TaskType,
)
from .repository import (
    TaskRepository,
    WorkerRepository,
    WorkflowRepository,
    ExperimentRepository,
    ExperimentRunRepository,
    ArtifactRepository,
)

__all__ = [
    # Connection
    "DatabaseConfig",
    "Database",
    "get_database",
    "init_database",
    "get_session",
    "session_scope",
    # Models
    "Base",
    "TaskModel",
    "WorkerModel",
    "WorkflowModel",
    "ExperimentModel",
    "ExperimentRunModel",
    "ArtifactModel",
    "EvaluationResultModel",
    "ToolModel",
    "DBTaskStatus",
    "WorkerStatus",
    "ExperimentRunStatus",
    "TaskType",
    # Repositories
    "TaskRepository",
    "WorkerRepository",
    "WorkflowRepository",
    "ExperimentRepository",
    "ExperimentRunRepository",
    "ArtifactRepository",
]
