"""Active pipeline detail endpoint."""
from __future__ import annotations
from src.backend.api.models import PipelineDetailResponse, ProgressResponse

from datetime import datetime

from fastapi import APIRouter, Depends

from src.common import get_logger
from src.backend.scheduler import Scheduler
from src.backend.api.state import SchedulerState, get_scheduler, get_scheduler_state

logger = get_logger(__name__)
router = APIRouter()

@router.get("/pipeline", response_model=PipelineDetailResponse, tags=["Pipeline"])
async def get_pipeline_detail(
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """
    Get detailed pipeline information with runtime statistics.
    
    Includes dataset, model, progress, and timing information.
    """
    pipeline = scheduler.pipeline
    all_tasks = scheduler.get_all_tasks()
    stats = scheduler.get_progress()
    
    # Calculate progress
    total = stats["total"]
    completed = stats["completed"]
    failed = stats["failed"]
    progress_percent = ((completed + failed) / total * 100) if total > 0 else 0.0
    
    progress = ProgressResponse(
        total=total,
        pending=stats["pending"],
        running=stats["running"],
        completed=completed,
        failed=failed,
        progress_percent=round(progress_percent, 2),
        is_complete=scheduler.is_complete()
    )
    
    # Calculate timing
    running_time = None
    estimated_remaining = None
    
    if state.started_at:
        running_time = (datetime.now() - state.started_at).total_seconds()
        
        # Estimate remaining time based on completed tasks
        if completed > 0 and total > completed:
            avg_time_per_task = running_time / completed
            remaining_tasks = total - completed - failed
            estimated_remaining = avg_time_per_task * remaining_tasks
    
    return PipelineDetailResponse(
        id=pipeline.id,
        name=pipeline.name,
        workflow_count=len(pipeline.workflows),
        task_count=len(all_tasks),
        dataset=pipeline.dataset,
        model=pipeline.model,
        started_at=state.started_at.isoformat() if state.started_at else None,
        running_time_seconds=round(running_time, 2) if running_time else None,
        progress=progress,
        estimated_remaining_seconds=round(estimated_remaining, 2) if estimated_remaining else None
    )
