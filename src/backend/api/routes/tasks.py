"""Task listing, claim, and status endpoints."""
from __future__ import annotations
from src.backend.api.models import NextTaskResponse, TaskListResponse, TaskPriorityInfo, TaskResponse, UpdateTaskStatusRequest, UpdateTaskStatusResponse

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.common import get_logger
from src.pipeline.tasks import TaskStatus, TaskType
from src.backend.scheduler import Scheduler, PriorityScheduler
from src.backend.api.state import SchedulerState, get_scheduler, get_scheduler_state
from src.backend.api.helpers import task_to_response

logger = get_logger(__name__)
router = APIRouter()

@router.get("/tasks/next", response_model=NextTaskResponse, tags=["Tasks"])
async def get_next_task(
    scheduler: Scheduler = Depends(get_scheduler),
    state: SchedulerState = Depends(get_scheduler_state)
):
    """
    Get the next task to execute.
    
    Returns the highest-priority task that is ready to execute (all dependencies completed).
    The returned task's status is automatically updated to RUNNING.
    
    Workers should:
    1. Call this endpoint to get a task
    2. Execute the task
    3. Call PUT /tasks/status to report completion or failure
    """
    task = scheduler.get_next_task()
    
    if task is None:
        # Check if all tasks are done or if we're waiting on running tasks
        progress = scheduler.get_progress()
        
        if progress["running"] > 0:
            message = f"No tasks ready. {progress['running']} task(s) currently running."
        elif scheduler.is_complete():
            message = "All tasks completed."
        else:
            message = "No tasks available. Some tasks may be blocked by failed dependencies."
        
        return NextTaskResponse(
            has_task=False,
            task=None,
            message=message
        )
    
    logger.info(f"Dispatching task {task.id} ({task.tool.name}) to worker")
    return NextTaskResponse(
        has_task=True,
        task=task_to_response(task, state),
        message=f"Task {task.id} assigned for execution"
    )

@router.put("/tasks/status", response_model=UpdateTaskStatusResponse, tags=["Tasks"])
async def update_task_status(
    request: UpdateTaskStatusRequest,
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """
    Update the status of a task after execution.
    
    Workers should call this endpoint to report task completion or failure.
    Valid status values: 'completed', 'failed'
    """
    # Validate status
    try:
        new_status = TaskStatus(request.status.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: {request.status}. Must be 'completed' or 'failed'."
        )
    
    if new_status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: {request.status}. Only 'completed' or 'failed' are allowed."
        )
    
    # Update task status
    try:
        scheduler.update_task_status(request.task_id, new_status)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # Store metadata and sync to database
    state.store_task_metadata(
        task_id=request.task_id,
        error_message=request.error_message,
        execution_time_ms=request.execution_time_ms,
        result=request.result,
        status=new_status
    )

    # If this task came from a worker claim, close out worker state/counters.
    assigned_worker_id = state.find_worker_assigned_to_task(request.task_id)
    if assigned_worker_id:
        state.complete_worker_task(
            worker_id=assigned_worker_id,
            success=(new_status == TaskStatus.COMPLETED),
            execution_time_ms=request.execution_time_ms or 0,
        )
    
    # Persist evaluation metrics so the Metrics dashboard can show them
    if (
        new_status == TaskStatus.COMPLETED
        and request.result
        and isinstance(request.result, dict)
        and request.result.get("evaluation_result")
        and state.db_service
        and state.db_service.is_available()
    ):
        task = scheduler._find_task_by_id(request.task_id)
        if task and task.task_type == TaskType.EVALUATION:
            eval_data = request.result["evaluation_result"]
            evaluator_name = task.tool.name
            run_id = getattr(task, 'run_id', None)
            for workflow_id in task.workflows:
                state.db_service.save_evaluation_result(
                    workflow_id=workflow_id,
                    pipeline_id=task.pipeline_id,
                    evaluator_name=evaluator_name,
                    result_data=eval_data,
                    evaluation_task_id=request.task_id,
                    evaluator_image=task.tool.container.image,
                    run_id=run_id
                )

    # If all tasks are terminal, finalize the pipeline-run status in DB.
    # This keeps historical run state in sync with scheduler completion so
    # the UI does not stay stuck on "running".
    if scheduler.is_complete():
        try:
            from src.db import session_scope, PipelineRunRepository, PipelineRunStatus
            run_id = getattr(scheduler.pipeline, "id", None)
            if run_id:
                csv_path: Optional[Path] = None
                with session_scope() as session:
                    run_repo = PipelineRunRepository(session)
                    run = run_repo.get_by_id(run_id)
                    if run and run.status in (PipelineRunStatus.PENDING, PipelineRunStatus.RUNNING, PipelineRunStatus.STOPPING):
                        progress = scheduler.get_progress()
                        final_status = PipelineRunStatus.FAILED if progress.get("failed", 0) > 0 else PipelineRunStatus.COMPLETED
                        run_repo.update_status(run_id, final_status)
                        logger.info(f"Pipeline run {run_id} finalized as {final_status.value}")
                try:
                    csv_path = _export_run_metrics_csv(run_id, scheduler)
                except Exception as export_err:
                    logger.warning(f"Failed to export metrics CSV for run {run_id}: {export_err}", exc_info=True)
                if csv_path:
                    logger.info(f"Exported run metrics CSV: {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to finalize pipeline run status: {e}")
    
    status_str = "completed successfully" if new_status == TaskStatus.COMPLETED else "failed"
    logger.info(f"Task {request.task_id} {status_str}")
    
    return UpdateTaskStatusResponse(
        success=True,
        task_id=request.task_id,
        new_status=new_status.value,
        message=f"Task {request.task_id} marked as {new_status.value}"
    )

@router.get("/tasks", response_model=TaskListResponse, tags=["Tasks"])
async def get_all_tasks(
    status: Optional[str] = Query(default=None, description="Filter by status"),
    limit: int = Query(default=200, ge=1, le=2000, description="Maximum number of tasks to return"),
    offset: int = Query(default=0, ge=0, description="Number of tasks to skip"),
    scheduler: Scheduler = Depends(get_scheduler),
    state: SchedulerState = Depends(get_scheduler_state)
):
    """
    Get all tasks, optionally filtered by status.
    
    Query Parameters:
        status: Filter by task status (pending, running, completed, failed)
    """
    if status:
        try:
            task_status = TaskStatus(status.lower())
            tasks = scheduler.get_tasks_by_status(task_status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Must be one of: pending, running, completed, failed"
            )
    else:
        tasks = scheduler.get_all_tasks()

    total_matching = len(tasks)
    paged_tasks = tasks[offset:offset + limit]
    task_responses = [task_to_response(t, state) for t in paged_tasks]
    return TaskListResponse(
        tasks=task_responses,
        total=total_matching
    )

@router.get("/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
async def get_task(
    task_id: str,
    scheduler: Scheduler = Depends(get_scheduler),
    state: SchedulerState = Depends(get_scheduler_state)
):
    """Get details of a specific task by ID."""
    task = scheduler._find_task_by_id(task_id)
    
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task with ID '{task_id}' not found"
        )
    
    return task_to_response(task, state)

@router.get("/tasks/{task_id}/logs", tags=["Tasks"])
async def get_task_logs(
    task_id: str,
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """Get execution logs for a task."""
    task = scheduler._find_task_by_id(task_id)
    
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task with ID '{task_id}' not found"
        )
    
    metadata = state.task_metadata.get(task_id, {})
    error_message = metadata.get("error_message")
    
    # Try to get logs from result if available
    result = metadata.get("result", {})
    logs = None
    if isinstance(result, dict):
        logs = result.get("logs") or result.get("stdout") or result.get("stderr")
    
    return {
        "task_id": task_id,
        "status": task.status.value,
        "error_message": error_message,
        "logs": logs,
        "worker_id": metadata.get("worker_id"),
        "execution_time_ms": metadata.get("execution_time_ms")
    }

@router.get("/tasks/{task_id}/priority", response_model=TaskPriorityInfo, tags=["Tasks"])
async def get_task_priority(
    task_id: str,
    scheduler: Scheduler = Depends(get_scheduler)
):
    """Get detailed priority information for a specific task."""
    # Check if scheduler has the priority info method
    if isinstance(scheduler, PriorityScheduler):
        try:
            info = scheduler.get_task_priority_info(task_id)
            return TaskPriorityInfo(**info)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        # Fall back for base scheduler
        task = scheduler._find_task_by_id(task_id)
        if task is None:
            raise HTTPException(
                status_code=404,
                detail=f"Task with ID '{task_id}' not found"
            )
        
        return TaskPriorityInfo(
            task_id=task.id,
            priority=task.priority,
            dependency_level=len(task.dependencies),
            usage_counter=task.counter,
            status=task.status.value,
            dependencies=[dep.id for dep in task.dependencies],
            workflows=list(task.workflows)
        )
