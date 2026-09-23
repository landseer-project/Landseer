"""Info and health endpoints."""
from __future__ import annotations
from src.backend.api.models import HealthResponse, PipelineInfoResponse, WorkflowInfo, WorkflowListResponse

from datetime import datetime

from fastapi import APIRouter, Depends

from src.common import get_logger
from src.backend.scheduler import Scheduler
from src.backend.api.state import SchedulerState, get_scheduler, get_scheduler_state

logger = get_logger(__name__)
router = APIRouter()

@router.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Landseer Scheduler API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }

@router.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check(state: SchedulerState = Depends(get_scheduler_state)):
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        scheduler_active=state.is_initialized()
    )

@router.get("/info/pipeline", response_model=PipelineInfoResponse, tags=["Info"])
async def get_pipeline_info(scheduler: Scheduler = Depends(get_scheduler)):
    """Get information about the current pipeline."""
    pipeline = scheduler.pipeline
    all_tasks = scheduler.get_all_tasks()
    
    return PipelineInfoResponse(
        id=pipeline.id,
        name=pipeline.name,
        workflow_count=len(pipeline.workflows),
        task_count=len(all_tasks),
        dataset=pipeline.dataset,
        model=pipeline.model
    )

@router.get("/info/workflows", response_model=WorkflowListResponse, tags=["Info"])
async def get_workflows(scheduler: Scheduler = Depends(get_scheduler)):
    """Get all workflows in the pipeline."""
    workflows = scheduler.pipeline.workflows
    
    workflow_infos = [
        WorkflowInfo(
            id=w.id,
            name=w.name,
            pipeline_id=w.pipeline_id,
            task_count=len(w.tasks),
            task_ids=[t.id for t in w.tasks]
        )
        for w in workflows
    ]
    
    return WorkflowListResponse(
        workflows=workflow_infos,
        total=len(workflows)
    )
