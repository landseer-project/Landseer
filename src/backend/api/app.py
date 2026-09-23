"""FastAPI application assembly."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.common import get_logger
from src.backend.initialization import get_backend_context
from .state import _scheduler_state, get_scheduler_state, SchedulerState
from .routes import (
    info,
    tasks,
    progress,
    scheduler,
    workers,
    workflows,
    tools,
    pipeline,
    dataset,
    stats,
    registry,
    metrics,
    pipeline_configs,
)

logger = get_logger(__name__)


async def _stale_worker_reclaim_loop(
    state: "SchedulerState",
    check_interval: float = 60.0,
    stale_timeout: float = 90.0,
) -> None:
    """
    Background coroutine: wake every `check_interval` seconds and reclaim tasks
    from workers that have stopped heartbeating for longer than `stale_timeout`.

    Default: check every 60 s, declare a worker stale after 90 s of silence
    (= 3 missed heartbeats at the default 30 s heartbeat interval).
    """
    logger.info(
        f"Stale-worker reclaim loop started "
        f"(check_interval={check_interval}s, stale_timeout={stale_timeout}s)"
    )
    while True:
        await asyncio.sleep(check_interval)
        if not state.is_initialized():
            continue
        try:
            result = state.reclaim_stale_workers(stale_timeout_seconds=stale_timeout)
            if result["stale_workers"]:
                logger.info(
                    f"Auto-reclaim: reset {len(result['reclaimed_tasks'])} task(s) to pending "
                    f"from {len(result['stale_workers'])} stale worker(s)"
                )
        except Exception as exc:
            logger.warning(f"Stale-worker reclaim loop error (will retry): {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Initializes the scheduler from the backend context on startup
    and performs cleanup on shutdown.
    """
    # Startup
    logger.info("Starting Landseer API server...")

    # Try to initialize from backend context if available
    context = get_backend_context()
    if context is not None and context.pipeline is not None:
        _scheduler_state.initialize(context.pipeline)
        logger.info(f"Scheduler auto-initialized with pipeline: {context.pipeline.name}")
    else:
        logger.info("No pipeline loaded at startup — running in headless mode. "
                     "Trigger runs via POST /api/pipeline-configs/<config_id>/runs")

    # Mark any pipeline runs that were RUNNING/STOPPING/PENDING as CANCELLED.
    # These are runs that were interrupted by a previous backend crash or Ctrl+C and
    # will never complete; leaving them as RUNNING would show a stale badge in the UI.
    try:
        from src.db.models import PipelineRunModel, PipelineRunStatus
        from src.db import session_scope
        from datetime import datetime

        with session_scope() as session:
            stale_statuses = [
                PipelineRunStatus.RUNNING,
                PipelineRunStatus.STOPPING,
                PipelineRunStatus.PENDING,
            ]
            stale_runs = session.query(PipelineRunModel).filter(
                PipelineRunModel.status.in_(stale_statuses)
            ).all()
            if stale_runs:
                for run in stale_runs:
                    run.status = PipelineRunStatus.CANCELLED
                    run.error_message = "Backend restarted — run was interrupted"
                    if not run.completed_at:
                        run.completed_at = datetime.utcnow()
                logger.info(
                    f"Startup cleanup: marked {len(stale_runs)} stale run(s) as CANCELLED "
                    f"(ids: {[r.id for r in stale_runs]})"
                )
    except Exception as e:
        logger.warning(f"Startup cleanup of stale pipeline runs failed: {e}", exc_info=True)

    # Pre-load the tool registry so /tools is populated before any pipeline runs.
    # This means Custom Run can show the tool list even with no active run.
    try:
        from src.pipeline.config_loader import init_tool_registry
        init_tool_registry("configs/tools.yaml")
        logger.info("Tool registry pre-loaded from configs/tools.yaml")
    except Exception as e:
        logger.warning(f"Could not pre-load tool registry: {e}")

    # Start the background stale-worker reclaim loop
    reclaim_task = asyncio.create_task(
        _stale_worker_reclaim_loop(_scheduler_state, check_interval=60.0, stale_timeout=90.0)
    )

    yield

    # Shutdown — cancel the background loop cleanly
    reclaim_task.cancel()
    try:
        await reclaim_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down Landseer API server...")



app = FastAPI(
    title="Landseer Scheduler API",
    description="REST API for ML Defense Pipeline Scheduler",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for mod in (
    info,
    tasks,
    progress,
    scheduler,
    workers,
    workflows,
    tools,
    pipeline,
    dataset,
    stats,
    registry,
    metrics,
    pipeline_configs,
):
    app.include_router(mod.router)


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info",
) -> None:
    import uvicorn

    logger.info(f"Starting server at http://{host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")
    uvicorn.run(
        "src.backend.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
