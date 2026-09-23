"""Workflow detail and results endpoints."""
from __future__ import annotations
from src.backend.api.models import WorkflowDetailResponse

from fastapi import APIRouter, Depends, HTTPException

from src.common import get_logger
from src.pipeline.tasks import TaskStatus
from src.backend.scheduler import Scheduler
from src.backend.api.state import SchedulerState, get_scheduler, get_scheduler_state
from src.backend.api.helpers import task_to_response

logger = get_logger(__name__)
router = APIRouter()

@router.get("/workflows/{workflow_id}", response_model=WorkflowDetailResponse, tags=["Workflows"])
async def get_workflow_detail(
    workflow_id: str,
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """
    Get detailed information about a specific workflow.
    
    Includes all tasks, their status, and failure reasons.
    """
    # Find workflow by ID or name
    workflow = None
    for w in scheduler.pipeline.workflows:
        if w.id == workflow_id or w.name == workflow_id:
            workflow = w
            break
    
    if workflow is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    
    # Get task details and status
    tasks = [task_to_response(t, state) for t in workflow.tasks]
    completed = sum(1 for t in workflow.tasks if t.status == TaskStatus.COMPLETED)
    failed = sum(1 for t in workflow.tasks if t.status == TaskStatus.FAILED)
    
    # Determine overall workflow status
    if failed > 0:
        status = "failed"
    elif completed == len(workflow.tasks):
        status = "completed"
    elif any(t.status == TaskStatus.RUNNING for t in workflow.tasks):
        status = "running"
    else:
        status = "pending"
    
    # Collect failure reasons
    failure_reasons = []
    for task in workflow.tasks:
        if task.status == TaskStatus.FAILED:
            meta = state.task_metadata.get(task.id, {})
            failure_reasons.append({
                "task_id": task.id,
                "task_name": task.tool.name,
                "error": meta.get("error_message", "Unknown error")
            })
    
    return WorkflowDetailResponse(
        id=workflow.id,
        name=workflow.name,
        pipeline_id=workflow.pipeline_id,
        task_count=len(workflow.tasks),
        tasks=tasks,
        status=status,
        completed_tasks=completed,
        failed_tasks=failed,
        failure_reasons=failure_reasons
    )

@router.get("/workflows/{workflow_id}/results", tags=["Workflows"])
async def get_workflow_results(
    workflow_id: str,
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """Get execution results for all tasks in a workflow."""
    workflow = None
    for w in scheduler.pipeline.workflows:
        if w.id == workflow_id or w.name == workflow_id:
            workflow = w
            break
    
    if workflow is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    
    results = []
    for task in workflow.tasks:
        meta = state.task_metadata.get(task.id, {})
        results.append({
            "task_id": task.id,
            "tool_name": task.tool.name,
            "status": task.status.value,
            "execution_time_ms": meta.get("execution_time_ms"),
            "result": meta.get("result"),
            "error_message": meta.get("error_message"),
            "worker_id": meta.get("worker_id")
        })
    
    return {
        "workflow_id": workflow.id,
        "workflow_name": workflow.name,
        "results": results
    }
