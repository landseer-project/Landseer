"""Progress and ready/blocked task endpoints."""
from __future__ import annotations
from src.backend.api.models import PriorityLevelsResponse, ProgressResponse, TaskListResponse

from fastapi import APIRouter, Depends

from src.common import get_logger
from src.pipeline.tasks import TaskStatus
from src.backend.scheduler import Scheduler, PriorityScheduler
from src.backend.api.state import SchedulerState, get_scheduler, get_scheduler_state
from src.backend.api.helpers import task_to_response

logger = get_logger(__name__)
router = APIRouter()

@router.get("/progress", response_model=ProgressResponse, tags=["Progress"])
async def get_progress(scheduler: Scheduler = Depends(get_scheduler)):
    """Get current progress of pipeline execution."""
    stats = scheduler.get_progress()
    
    total = stats["total"]
    completed = stats["completed"]
    failed = stats["failed"]
    
    progress_percent = ((completed + failed) / total * 100) if total > 0 else 0.0
    
    return ProgressResponse(
        total=total,
        pending=stats["pending"],
        running=stats["running"],
        completed=completed,
        failed=failed,
        progress_percent=round(progress_percent, 2),
        is_complete=scheduler.is_complete()
    )

@router.get("/progress/levels", response_model=PriorityLevelsResponse, tags=["Progress"])
async def get_priority_levels(
    scheduler: Scheduler = Depends(get_scheduler),
    state: SchedulerState = Depends(get_scheduler_state)
):
    """
    Get tasks grouped by priority level.
    
    Priority levels are based on dependency depth:
    - Level 0: Tasks with no dependencies
    - Level 1: Tasks that depend on level 0 tasks
    - etc.
    """
    if isinstance(scheduler, PriorityScheduler):
        levels = scheduler.get_priority_levels()
    else:
        # Fall back for base scheduler
        levels = {}
        for task in scheduler.get_all_tasks():
            level = len(task.dependencies)
            if level not in levels:
                levels[level] = []
            levels[level].append(task)
    
    # Convert to response format
    response_levels = {
        level: [task_to_response(t, state) for t in tasks]
        for level, tasks in levels.items()
    }
    
    return PriorityLevelsResponse(levels=response_levels)

@router.get("/progress/ready", response_model=TaskListResponse, tags=["Progress"])
async def get_ready_tasks(
    scheduler: Scheduler = Depends(get_scheduler),
    state: SchedulerState = Depends(get_scheduler_state)
):
    """Get all tasks that are ready to execute (dependencies satisfied)."""
    if isinstance(scheduler, PriorityScheduler):
        ready_tasks = scheduler.get_ready_tasks_by_priority()
    else:
        ready_tasks = [t for t in scheduler.get_all_tasks() if scheduler._is_task_ready(t)]
    
    task_responses = [task_to_response(t, state) for t in ready_tasks]
    
    return TaskListResponse(
        tasks=task_responses,
        total=len(task_responses)
    )

@router.get("/progress/blocked", response_model=TaskListResponse, tags=["Progress"])
async def get_blocked_tasks(
    scheduler: Scheduler = Depends(get_scheduler),
    state: SchedulerState = Depends(get_scheduler_state)
):
    """
    Get all tasks that are blocked due to failed dependencies.
    
    A task is blocked if:
    - Its status is PENDING
    - At least one of its dependencies has FAILED status
    """
    blocked_tasks = []
    all_tasks = scheduler.get_all_tasks()
    
    for task in all_tasks:
        if task.status != TaskStatus.PENDING:
            continue
        
        # Check if any dependency has failed
        has_failed_dep = False
        failed_deps = []
        for dep in task.dependencies:
            if dep.status == TaskStatus.FAILED:
                has_failed_dep = True
                failed_deps.append(dep.id)
        
        if has_failed_dep:
            blocked_tasks.append(task)
    
    task_responses = [task_to_response(t, state) for t in blocked_tasks]
    
    return TaskListResponse(
        tasks=task_responses,
        total=len(task_responses)
    )
