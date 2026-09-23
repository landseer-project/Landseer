"""Scheduler lifecycle endpoints."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from src.common import get_logger
from src.pipeline.tasks import TaskStatus
from src.backend.initialization import get_backend_context
from src.backend.scheduler import Scheduler, PriorityScheduler
from src.backend.api.state import SchedulerState, get_scheduler, get_scheduler_state
from src.backend.api.helpers import task_to_response

logger = get_logger(__name__)
router = APIRouter()

@router.post("/scheduler/initialize", tags=["Scheduler"])
async def initialize_scheduler(
    scheduler_type: str = Query(default="priority", description="Scheduler type to use"),
    state: SchedulerState = Depends(get_scheduler_state)
):
    """
    Initialize or reinitialize the scheduler.
    
    This will use the pipeline from the backend context.
    """
    context = get_backend_context()
    
    if context is None:
        raise HTTPException(
            status_code=503,
            detail="Backend context not available. Start the backend first."
        )
    
    try:
        state.initialize(context.pipeline, scheduler_type)
        return {
            "success": True,
            "message": f"Scheduler initialized with pipeline: {context.pipeline.name}",
            "scheduler_type": scheduler_type,
            "task_count": len(state.scheduler.get_all_tasks())
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/scheduler/reclaim-stale", tags=["Scheduler"])
async def reclaim_stale_tasks(state: SchedulerState = Depends(get_scheduler_state)):
    """
    Reclaim zombie tasks: reset RUNNING tasks whose worker has gone silent (or has no
    worker at all) back to PENDING. Preserves COMPLETED/FAILED task state.

    Combines two passes:
    1. Heartbeat-based: workers with a stale heartbeat are marked offline and their task reclaimed.
    2. Ownership-based: RUNNING tasks with no worker claiming them are also reclaimed
       (handles tasks left over after a backend restart).
    """
    if not state.is_initialized():
        raise HTTPException(status_code=503, detail="Scheduler not initialized.")

    # Pass 1 — heartbeat-based (stale_timeout=0 reclaims any worker with any elapsed time,
    # effectively marking every worker that currently has a task as stale so we get a clean sweep.
    # Use a small positive value to avoid false-positives on a freshly-registered worker.)
    result = state.reclaim_stale_workers(stale_timeout_seconds=1.0)
    reclaimed = list(result["reclaimed_tasks"])

    # Pass 2 — ownership-based: RUNNING tasks with absolutely no worker claiming them
    claimed_task_ids = {
        w["current_task_id"]
        for w in state.workers.values()
        if w.get("current_task_id")
    }
    unclaimed_running: List[str] = []
    for task in state.scheduler.get_all_tasks():
        if task.status == TaskStatus.RUNNING and task.id not in claimed_task_ids and task.id not in reclaimed:
            task.status = TaskStatus.PENDING
            reclaimed.append(task.id)
            unclaimed_running.append(task.id)
            logger.info(f"Reclaimed unclaimed task {task.id} (no worker assigned)")
    logger.info(f"Manual reclaim: {len(reclaimed)} task(s) reset to pending")
    return {
        "success": True,
        "reclaimed_count": len(reclaimed),
        "reclaimed_task_ids": reclaimed,
        "stale_workers": result["stale_workers"],
    }

@router.post("/scheduler/reset", tags=["Scheduler"])
async def reset_scheduler(state: SchedulerState = Depends(get_scheduler_state)):
    """
    Reset the scheduler to initial state.
    
    This reinitializes all tasks to PENDING status and recalculates priorities.
    """
    if not state.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="Scheduler not initialized."
        )
    
    # Reset task statuses on the shared pipeline, then rebuild the scheduler
    pipeline = state.pipeline
    for workflow in pipeline.workflows:
        for task in workflow.tasks:
            task.status = TaskStatus.PENDING
    state.initialize(pipeline)
    state.task_metadata.clear()

    return {
        "success": True,
        "message": "Scheduler reset to initial state",
        "task_count": len(state.scheduler.get_all_tasks()),
    }

@router.get("/scheduler/status", tags=["Scheduler"])
async def get_scheduler_status(state: SchedulerState = Depends(get_scheduler_state)):
    """Get current scheduler status and metadata."""
    if not state.is_initialized():
        return {
            "initialized": False,
            "message": "Scheduler not initialized"
        }
    
    progress = state.scheduler.get_progress()
    
    return {
        "initialized": True,
        "started_at": state.started_at.isoformat() if state.started_at else None,
        "pipeline_name": state.pipeline.name if state.pipeline else None,
        "pipeline_id": state.pipeline.id if state.pipeline else None,
        "progress": progress,
        "is_complete": state.scheduler.is_complete(),
        "task_metadata_count": len(state.task_metadata)
    }

@router.get("/scheduler/next", tags=["Scheduler"])
async def get_scheduler_next_preview(
    scheduler: Scheduler = Depends(get_scheduler),
    state: SchedulerState = Depends(get_scheduler_state)
):
    """
    Preview the next task without assigning it.
    
    Unlike /tasks/next, this does NOT change the task status.
    Useful for monitoring what's coming up next.
    """
    if isinstance(scheduler, PriorityScheduler):
        ready_tasks = scheduler.get_ready_tasks_by_priority()
    else:
        ready_tasks = [t for t in scheduler.get_all_tasks() if scheduler._is_task_ready(t)]
    
    if not ready_tasks:
        return {"has_next": False, "message": "No tasks ready"}
    
    next_task = ready_tasks[0]
    return {
        "has_next": True,
        "next_task": task_to_response(next_task, state),
        "queue_depth": len(ready_tasks)
    }
