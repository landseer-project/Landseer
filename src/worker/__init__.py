"""
Worker module for Landseer.

This module provides the distributed worker infrastructure for executing
pipeline tasks. Workers connect to the Landseer backend scheduler, claim
tasks, execute them in containers, and report results.

Components:
- client: HTTP client for backend API communication
- runner: Task execution with container support (Docker/Apptainer)
- db: Artifact cache handling for result reuse
- cli: Command-line interface and Worker class

Usage:
    # Start a worker from CLI
    $ landseer-worker --backend-url http://scheduler:8000

    # Or programmatically
    from src.worker import Worker
    
    worker = Worker(backend_url="http://localhost:8000")
    worker.start()
"""

__version__ = "0.1.0"

from .client import (
    LandseerClient,
    TaskInfo,
    WorkerInfo,
)
from .db import (
    ArtifactCacheDB,
    ArtifactMetadata,
    CacheManager,
)
from .runner import (
    ContainerRuntime,
    DockerRunner,
    ApptainerRunner,
    TaskRunner,
    ExecutionResult,
)
from .cli import (
    Worker,
    main,
)

__all__ = [
    # Version
    "__version__",
    # Client
    "LandseerClient",
    "TaskInfo",
    "WorkerInfo",
    # Cache
    "ArtifactCacheDB",
    "ArtifactMetadata",
    "CacheManager",
    # Runner
    "ContainerRuntime",
    "DockerRunner",
    "ApptainerRunner",
    "TaskRunner",
    "ExecutionResult",
    # CLI
    "Worker",
    "main",
]
