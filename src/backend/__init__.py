"""Backend module for Landseer."""

__version__ = "0.1.0"

from .initialization import (
    BackendContext,
    initialize_backend,
    get_backend_context,
    set_backend_context,
)
from .api import app, run_server, get_scheduler_state, SchedulerState
from .db_service import DatabaseService, get_db_service, init_db_service

__all__ = [
    "BackendContext",
    "initialize_backend",
    "get_backend_context",
    "set_backend_context",
    "app",
    "run_server",
    "get_scheduler_state",
    "SchedulerState",
    "DatabaseService",
    "get_db_service",
    "init_db_service",
]
