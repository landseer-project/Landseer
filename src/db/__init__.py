"""
Database module for Landseer.

This module provides SQLAlchemy-based database persistence for:
- Tasks (status, priority, dependencies, assigned_worker, attempts, logs)
- Workers (registration, heartbeat, capabilities)
- Workflows (tracking)
- Pipelines (execution state)

The database is designed to persist state so the system can recover from crashes/restarts.
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
    PipelineModel,
    PipelineConfigModel,
    PipelineRunModel,
    ArtifactModel,
    EvaluationResultModel,
    TaskStatus as DBTaskStatus,
    WorkerStatus,
    PipelineRunStatus,
)
from .repository import (
    TaskRepository,
    WorkerRepository,
    WorkflowRepository,
    PipelineRepository,
    PipelineConfigRepository,
    PipelineRunRepository,
    ArtifactRepository,
)

__all__ = [
    # Connection
    'DatabaseConfig',
    'Database',
    'get_database',
    'init_database',
    'get_session',
    'session_scope',
    # Models
    'Base',
    'TaskModel',
    'WorkerModel',
    'WorkflowModel',
    'PipelineModel',
    'PipelineConfigModel',
    'PipelineRunModel',
    'ArtifactModel',
    'EvaluationResultModel',
    'DBTaskStatus',
    'WorkerStatus',
    'PipelineRunStatus',
    # Repositories
    'TaskRepository',
    'WorkerRepository',
    'WorkflowRepository',
    'PipelineRepository',
    'PipelineConfigRepository',
    'PipelineRunRepository',
    'ArtifactRepository',
]
