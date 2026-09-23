"""Worker registration and claim endpoints."""
from __future__ import annotations
from src.backend.api.models import NextTaskResponse, WorkerHeartbeatRequest, WorkerInfo, WorkerListResponse, WorkerRegisterRequest

from fastapi import APIRouter, Depends, HTTPException

from src.common import get_logger
from src.pipeline.tasks import TaskStatus
from src.backend.scheduler import Scheduler
from src.backend.api.state import SchedulerState, get_scheduler, get_scheduler_state
from src.backend.api.helpers import task_to_response

logger = get_logger(__name__)
router = APIRouter()

@router.post("/workers/register", response_model=WorkerInfo, tags=["Workers"])
async def register_worker(
    request: WorkerRegisterRequest,
    state: SchedulerState = Depends(get_scheduler_state)
):
    """
    Register a new worker with the scheduler.
    
    Workers should register before requesting tasks. Registration provides
    a worker_id that should be included in subsequent requests.
    """
    worker_id = state.register_worker(
        hostname=request.hostname,
        worker_id=request.worker_id,
        capabilities=request.capabilities
    )
    
    worker = state.workers[worker_id]
    return WorkerInfo(**worker)

@router.get("/workers", response_model=WorkerListResponse, tags=["Workers"])
async def list_workers(state: SchedulerState = Depends(get_scheduler_state)):
    """Get all registered workers."""
    workers = [WorkerInfo(**w) for w in state.workers.values()]
    active = sum(1 for w in state.workers.values() if w["status"] != "offline")
    
    return WorkerListResponse(
        workers=workers,
        total=len(workers),
        active=active
    )

@router.get("/workers/{worker_id}", response_model=WorkerInfo, tags=["Workers"])
async def get_worker(
    worker_id: str,
    state: SchedulerState = Depends(get_scheduler_state)
):
    """Get information about a specific worker."""
    if worker_id not in state.workers:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    
    return WorkerInfo(**state.workers[worker_id])

@router.post("/workers/{worker_id}/heartbeat", tags=["Workers"])
async def worker_heartbeat(
    worker_id: str,
    request: WorkerHeartbeatRequest,
    state: SchedulerState = Depends(get_scheduler_state)
):
    """
    Update worker heartbeat.
    
    Workers should call this periodically to indicate they are still alive.
    """
    if not state.update_worker_heartbeat(worker_id, request.status):
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    
    return {"success": True, "worker_id": worker_id}

@router.get("/workers/{worker_id}/task", tags=["Workers"])
async def get_worker_current_task(
    worker_id: str,
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """Get the task currently assigned to a worker."""
    if worker_id not in state.workers:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    
    worker = state.workers[worker_id]
    task_id = worker.get("current_task_id")
    
    if not task_id:
        return {"has_task": False, "message": "Worker has no assigned task"}
    
    task = scheduler._find_task_by_id(task_id)
    if not task:
        return {"has_task": False, "message": "Assigned task not found"}
    
    return {
        "has_task": True,
        "task": task_to_response(task, state)
    }

@router.post("/workers/{worker_id}/claim", response_model=NextTaskResponse, tags=["Workers"])
async def worker_claim_task(
    worker_id: str,
    state: SchedulerState = Depends(get_scheduler_state)
):
    """
    Claim the next available task for a specific worker.
    
    Similar to /tasks/next but associates the task with the worker.
    """
    if worker_id not in state.workers:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    if not state.is_initialized() or state.scheduler is None:
        return NextTaskResponse(
            has_task=False,
            task=None,
            message="Scheduler not initialized yet. Start or resume a pipeline run.",
        )
    
    scheduler = state.scheduler
    ready_count = 0
    blocked_pending_count = 0
    try:
        all_tasks = getattr(scheduler, "_all_tasks", [])
        ready_count = sum(1 for t in all_tasks if scheduler._is_task_ready(t))
        blocked_pending_count = sum(
            1 for t in all_tasks
            if getattr(t, "status", None) == TaskStatus.PENDING and not scheduler._is_task_ready(t)
        )
    except Exception:
        pass
    task = scheduler.get_next_task()
    progress = scheduler.get_progress()
    
    if task is None:
        if progress["running"] > 0:
            message = f"No tasks ready. {progress['running']} task(s) currently running."
        elif scheduler.is_complete():
            message = "All tasks completed."
        else:
            message = "No tasks available."
        return NextTaskResponse(has_task=False, task=None, message=message)
    
    # Associate task with worker
    state.assign_task_to_worker(worker_id, task.id)
    state.store_task_metadata(task_id=task.id, worker_id=worker_id)
    logger.info(f"Task {task.id} claimed by worker {worker_id}")
    return NextTaskResponse(
        has_task=True,
        task=task_to_response(task, state),
        message=f"Task {task.id} assigned to worker {worker_id}"
    )
