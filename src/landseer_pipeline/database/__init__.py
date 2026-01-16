"""
Landseer Pipeline Database Module

This module provides database connectivity and ORM models for storing
and querying pipeline run results.
"""

from .db_connection import DatabaseConnection, get_db_connection
from .models import (
    Dataset,
    Model,
    Tool,
    ToolCategory,
    PipelineRun,
    PipelineAttacks,
    Combination,
    CombinationTools,
    CombinationMetrics,
    ToolExecution,
    ArtifactNode,
    ArtifactFile,
    OutputFileProvenance,
)
from .repository import (
    PipelineRunRepository,
    CombinationRepository,
    ToolRepository,
    MetricsRepository,
)

__all__ = [
    # Connection
    "DatabaseConnection",
    "get_db_connection",
    # Models
    "Dataset",
    "Model",
    "Tool",
    "ToolCategory",
    "PipelineRun",
    "PipelineAttacks",
    "Combination",
    "CombinationTools",
    "CombinationMetrics",
    "ToolExecution",
    "ArtifactNode",
    "ArtifactFile",
    "OutputFileProvenance",
    # Repositories
    "PipelineRunRepository",
    "CombinationRepository",
    "ToolRepository",
    "MetricsRepository",
]
