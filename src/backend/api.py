"""
FastAPI-based REST API for the Landseer scheduler.

This module provides HTTP endpoints for workers to:
- Request tasks to execute
- Report task completion/failure
- Query task and pipeline status

Usage:
    The API is initialized with a scheduler instance and exposes endpoints
    for task management. Start with `run_server()` or use the `app` directly.
"""

import asyncio
import csv
from contextlib import asynccontextmanager
from datetime import datetime
import os
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml

from ..common import get_logger
from ..pipeline.tasks import TaskStatus, TaskType
from ..pipeline.pipeline import Pipeline
from .scheduler import Scheduler, PriorityScheduler
from .initialization import get_backend_context, set_backend_context, BackendContext

# Import database service if available
try:
    from .db_service import get_db_service, DatabaseService
    DB_SERVICE_AVAILABLE = True
except ImportError:
    DB_SERVICE_AVAILABLE = False
    DatabaseService = None

logger = get_logger(__name__)


# ==============================================================================
# Pydantic Models for Request/Response
# ==============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="ok", description="Health status")
    timestamp: str = Field(description="Current server timestamp")
    scheduler_active: bool = Field(description="Whether scheduler is initialized")


class ContainerInfo(BaseModel):
    """Container configuration info."""
    image: str = Field(description="Container image name")
    command: str = Field(description="Command to execute")
    runtime: Optional[str] = Field(default=None, description="Container runtime")


class ToolInfo(BaseModel):
    """Tool information for a task."""
    name: str = Field(description="Tool display name")
    key: Optional[str] = Field(default=None, description="Registry key (YAML key) used in tools_override")
    container: ContainerInfo = Field(description="Container configuration")
    is_baseline: bool = Field(default=False, description="Whether this is a baseline tool")
    defense_stage: Optional[str] = Field(default=None, description="Pipeline stage this tool belongs to")


class TaskResponse(BaseModel):
    """Response model for a single task."""
    id: str = Field(description="Unique task identifier")
    tool: ToolInfo = Field(description="Tool definition for this task")
    config: Dict[str, Any] = Field(default_factory=dict, description="Task configuration")
    priority: int = Field(description="Task priority (lower = higher priority)")
    status: str = Field(description="Current task status")
    task_type: str = Field(description="Type of task (pre/in/post/deploy)")
    counter: int = Field(description="Number of workflows using this task")
    workflows: List[str] = Field(default_factory=list, description="Workflow IDs using this task")
    workflow_names: List[str] = Field(default_factory=list, description="Workflow names using this task")
    pipeline_id: str = Field(description="ID of the pipeline this task belongs to")
    run_id: Optional[str] = Field(default=None, description="ID of the pipeline run this task belongs to")
    dependency_ids: List[str] = Field(default_factory=list, description="IDs of dependent tasks")
    cache_hit: Optional[bool] = Field(default=None, description="Whether this task used cache")
    cache_key: Optional[str] = Field(default=None, description="Cache key if cache was used")
    output_path: Optional[str] = Field(default=None, description="Resolved output directory path for task artifacts")
    log_path: Optional[str] = Field(default=None, description="Worker log file path for this task")
    worker_id: Optional[str] = Field(default=None, description="ID of worker that executed this task")
    error_message: Optional[str] = Field(default=None, description="Error message if task failed")
    execution_time_ms: Optional[int] = Field(default=None, description="Execution time in milliseconds")


class TaskListResponse(BaseModel):
    """Response model for list of tasks."""
    tasks: List[TaskResponse] = Field(description="List of tasks")
    total: int = Field(description="Total number of tasks")


class UpdateTaskStatusRequest(BaseModel):
    """Request model for updating task status."""
    task_id: str = Field(description="ID of the task to update")
    status: str = Field(description="New status ('completed' or 'failed')")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_ms: Optional[int] = Field(default=None, description="Execution time in milliseconds")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task execution result/metadata")


class UpdateTaskStatusResponse(BaseModel):
    """Response model for task status update."""
    success: bool = Field(description="Whether update was successful")
    task_id: str = Field(description="ID of the updated task")
    new_status: str = Field(description="New status of the task")
    message: str = Field(description="Status message")


class ProgressResponse(BaseModel):
    """Response model for pipeline progress."""
    total: int = Field(description="Total number of tasks")
    pending: int = Field(description="Number of pending tasks")
    running: int = Field(description="Number of running tasks")
    completed: int = Field(description="Number of completed tasks")
    failed: int = Field(description="Number of failed tasks")
    progress_percent: float = Field(description="Completion percentage")
    is_complete: bool = Field(description="Whether all tasks are done")


class TaskPriorityInfo(BaseModel):
    """Detailed priority information for a task."""
    task_id: str = Field(description="Task ID")
    priority: int = Field(description="Computed priority value")
    dependency_level: int = Field(description="Number of dependencies")
    usage_counter: int = Field(description="Number of workflows using this task")
    status: str = Field(description="Current task status")
    dependencies: List[str] = Field(description="IDs of dependent tasks")
    workflows: List[str] = Field(description="Workflow IDs using this task")


class PriorityLevelsResponse(BaseModel):
    """Response model for tasks grouped by priority level."""
    levels: Dict[int, List[TaskResponse]] = Field(description="Tasks grouped by dependency level")


class PipelineInfoResponse(BaseModel):
    """Response model for pipeline information."""
    id: str = Field(description="Pipeline ID")
    name: str = Field(description="Pipeline name")
    workflow_count: int = Field(description="Number of workflows")
    task_count: int = Field(description="Total number of unique tasks")
    dataset: Optional[Dict[str, Any]] = Field(default=None, description="Dataset configuration")
    model: Optional[Dict[str, Any]] = Field(default=None, description="Model configuration")


class WorkflowInfo(BaseModel):
    """Response model for workflow information."""
    id: str = Field(description="Workflow ID")
    name: str = Field(description="Workflow name")
    pipeline_id: str = Field(description="Pipeline this workflow belongs to")
    task_count: int = Field(description="Number of tasks in workflow")
    task_ids: List[str] = Field(description="IDs of tasks in this workflow")


class WorkflowListResponse(BaseModel):
    """Response model for list of workflows."""
    workflows: List[WorkflowInfo] = Field(description="List of workflows")
    total: int = Field(description="Total number of workflows")


class NextTaskResponse(BaseModel):
    """Response for getting the next task to execute."""
    has_task: bool = Field(description="Whether a task is available")
    task: Optional[TaskResponse] = Field(default=None, description="The next task to execute")
    message: str = Field(description="Status message")


# ------------------------------------------------------------------------------
# Worker Models
# ------------------------------------------------------------------------------

class WorkerRegisterRequest(BaseModel):
    """Request to register a new worker."""
    worker_id: Optional[str] = Field(default=None, description="Optional worker ID (auto-generated if not provided)")
    hostname: str = Field(description="Worker hostname")
    capabilities: Optional[Dict[str, Any]] = Field(default=None, description="Worker capabilities (GPU, memory, etc.)")


class WorkerInfo(BaseModel):
    """Information about a registered worker."""
    worker_id: str = Field(description="Unique worker identifier")
    hostname: str = Field(description="Worker hostname")
    status: str = Field(description="Worker status (idle, busy, offline)")
    registered_at: str = Field(description="Registration timestamp")
    last_heartbeat: str = Field(description="Last heartbeat timestamp")
    current_task_id: Optional[str] = Field(default=None, description="ID of task currently being executed")
    tasks_completed: int = Field(default=0, description="Number of tasks completed by this worker")
    tasks_failed: int = Field(default=0, description="Number of tasks failed by this worker")
    capabilities: Optional[Dict[str, Any]] = Field(default=None, description="Worker capabilities")


class WorkerListResponse(BaseModel):
    """Response for list of workers."""
    workers: List[WorkerInfo] = Field(description="List of registered workers")
    total: int = Field(description="Total number of workers")
    active: int = Field(description="Number of active workers")


class WorkerHeartbeatRequest(BaseModel):
    """Worker heartbeat request."""
    worker_id: str = Field(description="Worker ID")
    status: Optional[str] = Field(default=None, description="Updated status")


# ------------------------------------------------------------------------------
# Workflow Detail Models
# ------------------------------------------------------------------------------

class WorkflowDetailResponse(BaseModel):
    """Detailed workflow information with task results."""
    id: str = Field(description="Workflow ID")
    name: str = Field(description="Workflow name")
    pipeline_id: str = Field(description="Pipeline ID")
    task_count: int = Field(description="Number of tasks")
    tasks: List[TaskResponse] = Field(description="All tasks in this workflow")
    status: str = Field(description="Workflow status (pending, running, completed, failed)")
    completed_tasks: int = Field(description="Number of completed tasks")
    failed_tasks: int = Field(description="Number of failed tasks")
    failure_reasons: List[Dict[str, str]] = Field(default_factory=list, description="Failure reasons for failed tasks")


# ------------------------------------------------------------------------------
# Tool Management Models
# ------------------------------------------------------------------------------

class AddToolRequest(BaseModel):
    """Request to add a new tool."""
    name: str = Field(description="Tool name")
    image: str = Field(description="Container image")
    command: str = Field(description="Command to run")
    runtime: Optional[str] = Field(default=None, description="Container runtime")
    is_baseline: bool = Field(default=False, description="Whether this is a baseline tool")
    defense_stage: Optional[str] = Field(
        default=None,
        description="Pipeline stage this tool belongs to (pre_training/during_training/post_training/deployment)",
    )


class ToolListResponse(BaseModel):
    """Response for list of tools."""
    tools: List[ToolInfo] = Field(description="List of available tools")
    total: int = Field(description="Total number of tools")


# ------------------------------------------------------------------------------
# Extended Pipeline Info
# ------------------------------------------------------------------------------

class PipelineDetailResponse(BaseModel):
    """Extended pipeline information with runtime stats."""
    id: str = Field(description="Pipeline ID")
    name: str = Field(description="Pipeline name")
    workflow_count: int = Field(description="Number of workflows")
    task_count: int = Field(description="Total unique tasks")
    dataset: Optional[Dict[str, Any]] = Field(default=None, description="Dataset configuration")
    model: Optional[Dict[str, Any]] = Field(default=None, description="Model configuration")
    started_at: Optional[str] = Field(default=None, description="Pipeline start time")
    running_time_seconds: Optional[float] = Field(default=None, description="Time elapsed since start")
    progress: ProgressResponse = Field(description="Current progress")
    estimated_remaining_seconds: Optional[float] = Field(default=None, description="Estimated time remaining")


# ==============================================================================
# Scheduler State Management
# ==============================================================================

class SchedulerState:
    """
    Holds the global scheduler state for the API.
    
    This is initialized when the API starts and provides access to:
    - The scheduler instance
    - The pipeline being executed
    - Task execution metadata (timing, errors, etc.)
    - Registered workers
    - Database service for persistence
    """
    
    def __init__(self):
        self.scheduler: Optional[Scheduler] = None
        self.pipeline: Optional[Pipeline] = None
        self.started_at: Optional[datetime] = None
        self.task_metadata: Dict[str, Dict[str, Any]] = {}
        self.workers: Dict[str, Dict[str, Any]] = {}
        self._worker_counter: int = 0
        self._custom_tools: Dict[str, Dict[str, Any]] = {}
        self._db_service: Optional["DatabaseService"] = None
    
    @property
    def db_service(self) -> Optional["DatabaseService"]:
        """Get database service."""
        if self._db_service is None and DB_SERVICE_AVAILABLE:
            try:
                self._db_service = get_db_service()
            except RuntimeError:
                pass
        return self._db_service
    
    def initialize(self, pipeline: Pipeline, scheduler_type: str = "priority") -> None:
        """
        Initialize the scheduler with a pipeline.
        
        Args:
            pipeline: Pipeline instance to schedule
            scheduler_type: Type of scheduler to use ('priority' is default)
        """
        self.pipeline = pipeline
        
        if scheduler_type == "priority":
            t_sched_ctor_start = time.time()
            self.scheduler = PriorityScheduler(pipeline)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        self.started_at = datetime.now()
        logger.info(f"Scheduler initialized with pipeline: {pipeline.name}")
    
    def is_initialized(self) -> bool:
        """Check if scheduler is initialized."""
        return self.scheduler is not None
    
    def store_task_metadata(
        self,
        task_id: str,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        worker_id: Optional[str] = None,
        status: Optional[TaskStatus] = None
    ) -> None:
        """Store execution metadata for a task."""
        self.task_metadata[task_id] = {
            "error_message": error_message,
            "execution_time_ms": execution_time_ms,
            "result": result,
            "worker_id": worker_id,
            "updated_at": datetime.now().isoformat()
        }
        
        # Persist task status to database
        if self.db_service and status:
            self.db_service.sync_task_status(
                task_id=task_id,
                status=status,
                error_message=error_message,
                execution_time_ms=execution_time_ms,
                worker_id=worker_id
            )
    
    def register_worker(
        self,
        hostname: str,
        worker_id: Optional[str] = None,
        capabilities: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a new worker and return its ID."""
        if worker_id is None:
            self._worker_counter += 1
            worker_id = f"worker_{self._worker_counter}"
        
        now = datetime.now().isoformat()
        self.workers[worker_id] = {
            "worker_id": worker_id,
            "hostname": hostname,
            "status": "idle",
            "registered_at": now,
            "last_heartbeat": now,
            "current_task_id": None,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "capabilities": capabilities or {}
        }
        
        # Persist to database
        if self.db_service:
            self.db_service.register_worker(worker_id, hostname, capabilities)
        
        logger.info(f"Worker registered: {worker_id} ({hostname})")
        return worker_id
    
    def update_worker_heartbeat(self, worker_id: str, status: Optional[str] = None) -> bool:
        """Update worker heartbeat timestamp."""
        if worker_id not in self.workers:
            return False
        self.workers[worker_id]["last_heartbeat"] = datetime.now().isoformat()
        if status:
            self.workers[worker_id]["status"] = status
            if status == "idle" and self.workers[worker_id].get("current_task_id"):
                stale_task_id = self.workers[worker_id]["current_task_id"]
                stale_task = (
                    self.scheduler._find_task_by_id(stale_task_id)
                    if self.scheduler is not None
                    else None
                )
                if stale_task and stale_task.status == TaskStatus.RUNNING:
                    # Worker claims to be idle while still owning a running task.
                    # This most commonly happens after claim/report timeouts and can deadlock progress.
                    stale_task.status = TaskStatus.PENDING
                    self.workers[worker_id]["current_task_id"] = None
                    logger.warning(
                        f"Reclaimed stale running task {stale_task_id} from idle heartbeat worker {worker_id}"
                    )
                elif stale_task is None or stale_task.status != TaskStatus.RUNNING:
                    # Orphan pointer: task finished or unknown to scheduler — drop so UI/API stay consistent.
                    self.workers[worker_id]["current_task_id"] = None
                    logger.info(
                        f"Cleared orphan current_task_id {stale_task_id} on idle heartbeat "
                        f"for worker {worker_id}"
                    )
        
        # Persist to database
        if self.db_service:
            self.db_service.update_worker_heartbeat(worker_id, status)
        return True
    
    def assign_task_to_worker(self, worker_id: str, task_id: str) -> None:
        """Mark a worker as busy with a task."""
        if worker_id in self.workers:
            self.workers[worker_id]["status"] = "busy"
            self.workers[worker_id]["current_task_id"] = task_id
            self.workers[worker_id]["last_heartbeat"] = datetime.now().isoformat()
            
            # Persist to database
            if self.db_service:
                self.db_service.assign_task_to_worker(worker_id, task_id)

    def find_worker_assigned_to_task(self, task_id: str) -> Optional[str]:
        """Return the worker currently assigned to task_id, if any."""
        for worker_id, worker in self.workers.items():
            if worker.get("current_task_id") == task_id:
                return worker_id
        return None
    
    def complete_worker_task(self, worker_id: str, success: bool, task_id: Optional[str] = None, execution_time_ms: int = 0) -> None:
        """Mark worker task as complete."""
        if worker_id in self.workers:
            current_task_id = task_id or self.workers[worker_id].get("current_task_id")
            self.workers[worker_id]["status"] = "idle"
            self.workers[worker_id]["current_task_id"] = None
            self.workers[worker_id]["last_heartbeat"] = datetime.now().isoformat()
            if success:
                self.workers[worker_id]["tasks_completed"] += 1
            else:
                self.workers[worker_id]["tasks_failed"] += 1
            
            # Persist to database
            if self.db_service and current_task_id:
                self.db_service.complete_worker_task(
                    worker_id=worker_id,
                    task_id=current_task_id,
                    success=success,
                    execution_time_ms=execution_time_ms
                )
    
    def reclaim_stale_workers(self, stale_timeout_seconds: float = 90.0) -> dict:
        """
        Detect workers whose heartbeat has gone silent, reset their RUNNING tasks
        to PENDING, and mark those workers offline.

        Called automatically by the background reclaim loop every 60 s.
        Also called manually via POST /scheduler/reclaim-stale.

        stale_timeout_seconds: seconds since last heartbeat before a worker is
        considered dead. Default 90 s = 3× the 30 s worker heartbeat interval.
        """
        now = datetime.now()
        reclaimed_tasks: List[str] = []
        stale_worker_ids: List[str] = []

        for worker_id, worker in self.workers.items():
            last_hb_str = worker.get("last_heartbeat")
            if not last_hb_str:
                continue
            elapsed = (now - datetime.fromisoformat(last_hb_str)).total_seconds()
            if elapsed < stale_timeout_seconds:
                continue  # worker is alive

            stale_worker_ids.append(worker_id)
            task_id = worker.get("current_task_id")
            if task_id and self.scheduler:
                task = self.scheduler._find_task_by_id(task_id)
                if task and task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.PENDING
                    reclaimed_tasks.append(task_id)
                    logger.info(f"Reclaimed task {task_id} from stale worker {worker_id} "
                                f"(no heartbeat for {elapsed:.0f}s)")

            worker["status"] = "offline"
            worker["current_task_id"] = None

        if stale_worker_ids:
            logger.warning(f"Stale workers: {stale_worker_ids} | "
                           f"Reclaimed tasks: {reclaimed_tasks}")

        return {"stale_workers": stale_worker_ids, "reclaimed_tasks": reclaimed_tasks}

    def add_tool(
        self,
        name: str,
        image: str,
        command: str,
        runtime: Optional[str] = None,
        is_baseline: bool = False,
        defense_stage: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        """Add a custom tool to the registry."""
        registry_key = key or name
        self._custom_tools[registry_key] = {
            "name": name,
            "container": {
                "image": image,
                "command": command,
                "runtime": runtime
            },
            "is_baseline": is_baseline,
            "defense_stage": defense_stage,
        }
        logger.info(f"Tool added: key={registry_key}, name={name}")
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all tools (from registry + custom)."""
        from ..pipeline.tools import get_all_tools
        tools = get_all_tools()
        # Convert to dict format and merge with custom tools
        result = {}
        for name, tool in tools.items():
            result[name] = {
                "name": tool.name,
                "container": {
                    "image": tool.container.image,
                    "command": tool.container.command,
                    "runtime": tool.container.runtime
                },
                "is_baseline": tool.is_baseline,
                "defense_stage": tool.defense_stage,
            }
        result.update(self._custom_tools)
        return result


# Global scheduler state
_scheduler_state = SchedulerState()


def get_scheduler_state() -> SchedulerState:
    """Dependency injection for scheduler state."""
    return _scheduler_state


def get_scheduler() -> Scheduler:
    """Get the active scheduler, raising an error if not initialized."""
    if not _scheduler_state.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="Scheduler not initialized. Please initialize with a pipeline first."
        )
    return _scheduler_state.scheduler


# ==============================================================================
# Helper Functions
# ==============================================================================

def task_to_response(task, state: Optional["SchedulerState"] = None) -> TaskResponse:
    """Convert a Task object to a TaskResponse model."""
    # Get workflow names from pipeline
    workflow_names = []
    if state and state.pipeline:
        for workflow in state.pipeline.workflows:
            if workflow.id in task.workflows:
                workflow_names.append(workflow.name)
    
    # Get execution metadata from task metadata
    cache_hit = None
    cache_key = None
    output_path = None
    log_path = None
    worker_id = None
    error_message = None
    execution_time_ms = None
    
    if state:
        metadata = state.task_metadata.get(task.id, {})
        worker_id = metadata.get("worker_id")
        error_message = metadata.get("error_message")
        execution_time_ms = metadata.get("execution_time_ms")
        result = metadata.get("result", {})
        if isinstance(result, dict):
            cache_hit = result.get("cache_hit")
            cache_key = result.get("cache_key")
            output_path = result.get("output_path")
            log_path = result.get("log_path")
    
    # Get run_id from task
    run_id = getattr(task, 'run_id', None)
    
    return TaskResponse(
        id=task.id,
        tool=ToolInfo(
            name=task.tool.name,
            container=ContainerInfo(
                image=task.tool.container.image,
                command=task.tool.container.command,
                runtime=task.tool.container.runtime
            ),
            is_baseline=task.tool.is_baseline
        ),
        config=task.config,
        priority=task.priority,
        status=task.status.value,
        task_type=task.task_type.value,
        counter=task.counter,
        workflows=list(task.workflows),
        workflow_names=workflow_names,
        pipeline_id=task.pipeline_id,
        run_id=run_id,
        dependency_ids=[dep.id for dep in task.dependencies],
        cache_hit=cache_hit,
        cache_key=cache_key,
        output_path=output_path,
        log_path=log_path,
        worker_id=worker_id,
        error_message=error_message,
        execution_time_ms=execution_time_ms
    )


def _required_pipeline_keys() -> List[str]:
    """Read configured API keys used to protect mutating endpoints."""
    raw = os.getenv("LANDSEER_PIPELINE_KEYS", "")
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def _require_pipeline_key(x_pipeline_key: Optional[str] = Header(default=None, alias="X-Pipeline-Key")) -> None:
    """
    Enforce the same key gate as run-start when LANDSEER_PIPELINE_KEYS is configured.
    If no keys are configured, endpoint remains open for local/dev workflows.
    """
    allowed = _required_pipeline_keys()
    if not allowed:
        return
    if not x_pipeline_key or x_pipeline_key not in allowed:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Pipeline-Key")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tools_registry_path() -> Path:
    """
    Resolve persistent tools registry YAML path.
    Allows override via LANDSEER_TOOLS_YAML for tests or custom deployments.
    """
    cfg = os.getenv("LANDSEER_TOOLS_YAML", "configs/tools.yaml")
    p = Path(cfg)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    return _repo_root() / p


def _tool_key_from_name(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "custom_tool"


def _persist_tool_to_yaml(tool_key: str, request: AddToolRequest) -> None:
    """Persist a tool entry to tools.yaml so registry survives backend restart."""
    yaml_path = _tools_registry_path()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    tools = data.get("tools")
    if not isinstance(tools, dict):
        tools = {}

    tools[tool_key] = {
        "name": request.name,
        "defense_stage": request.defense_stage,
        "is_baseline": request.is_baseline,
        "container": {
            "image": request.image,
            "command": request.command,
            "runtime": request.runtime,
        },
    }
    data["tools"] = tools

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# ==============================================================================
# FastAPI Application & Lifespan
# ==============================================================================

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
        from ..db.models import PipelineRunModel, PipelineRunStatus
        from ..db import session_scope
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
        from ..pipeline.config_loader import init_tool_registry
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


# Create FastAPI application
app = FastAPI(
    title="Landseer Scheduler API",
    description="REST API for ML Defense Pipeline Scheduler",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Health & Info Endpoints
# ==============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Landseer Scheduler API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check(state: SchedulerState = Depends(get_scheduler_state)):
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        scheduler_active=state.is_initialized()
    )


@app.get("/info/pipeline", response_model=PipelineInfoResponse, tags=["Info"])
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


@app.get("/info/workflows", response_model=WorkflowListResponse, tags=["Info"])
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


# ==============================================================================
# Task Management Endpoints
# ==============================================================================

@app.get("/tasks/next", response_model=NextTaskResponse, tags=["Tasks"])
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


@app.put("/tasks/status", response_model=UpdateTaskStatusResponse, tags=["Tasks"])
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
            from ..db import session_scope, PipelineRunRepository, PipelineRunStatus
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


@app.get("/tasks", response_model=TaskListResponse, tags=["Tasks"])
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


@app.get("/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
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


@app.get("/tasks/{task_id}/logs", tags=["Tasks"])
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


@app.get("/tasks/{task_id}/priority", response_model=TaskPriorityInfo, tags=["Tasks"])
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


# ==============================================================================
# Progress & Statistics Endpoints
# ==============================================================================

@app.get("/progress", response_model=ProgressResponse, tags=["Progress"])
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


@app.get("/progress/levels", response_model=PriorityLevelsResponse, tags=["Progress"])
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


@app.get("/progress/ready", response_model=TaskListResponse, tags=["Progress"])
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


@app.get("/progress/blocked", response_model=TaskListResponse, tags=["Progress"])
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


# ==============================================================================
# Scheduler Management Endpoints
# ==============================================================================

@app.post("/scheduler/initialize", tags=["Scheduler"])
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


@app.post("/scheduler/reclaim-stale", tags=["Scheduler"])
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


@app.post("/scheduler/reset", tags=["Scheduler"])
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
    
    # Reinitialize from the same pipeline
    pipeline = state.pipeline
    state.initialize(pipeline)
    state.task_metadata.clear()
    
    return {
        "success": True,
        "message": "Scheduler reset to initial state",
        "task_count": len(state.scheduler.get_all_tasks())
    }


@app.get("/scheduler/status", tags=["Scheduler"])
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


@app.get("/scheduler/next", tags=["Scheduler"])
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


# ==============================================================================
# Worker Management Endpoints
# ==============================================================================

@app.post("/workers/register", response_model=WorkerInfo, tags=["Workers"])
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


@app.get("/workers", response_model=WorkerListResponse, tags=["Workers"])
async def list_workers(state: SchedulerState = Depends(get_scheduler_state)):
    """Get all registered workers."""
    workers = [WorkerInfo(**w) for w in state.workers.values()]
    active = sum(1 for w in state.workers.values() if w["status"] != "offline")
    
    return WorkerListResponse(
        workers=workers,
        total=len(workers),
        active=active
    )


@app.get("/workers/{worker_id}", response_model=WorkerInfo, tags=["Workers"])
async def get_worker(
    worker_id: str,
    state: SchedulerState = Depends(get_scheduler_state)
):
    """Get information about a specific worker."""
    if worker_id not in state.workers:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    
    return WorkerInfo(**state.workers[worker_id])


@app.post("/workers/{worker_id}/heartbeat", tags=["Workers"])
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


@app.get("/workers/{worker_id}/task", tags=["Workers"])
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


@app.post("/workers/{worker_id}/claim", response_model=NextTaskResponse, tags=["Workers"])
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


# ==============================================================================
# Workflow Detail Endpoints
# ==============================================================================

@app.get("/workflows/{workflow_id}", response_model=WorkflowDetailResponse, tags=["Workflows"])
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


@app.get("/workflows/{workflow_id}/results", tags=["Workflows"])
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


# ==============================================================================
# Tool Management Endpoints
# ==============================================================================

@app.get("/tools", response_model=ToolListResponse, tags=["Tools"])
async def list_tools(state: SchedulerState = Depends(get_scheduler_state)):
    """Get all available tools."""
    tools_dict = state.get_all_tools()

    tools = []
    for name, data in tools_dict.items():
        container = data.get("container", {})
        tools.append(ToolInfo(
            name=data.get("name", name),
            container=ContainerInfo(
                image=container.get("image", ""),
                command=container.get("command", ""),
                runtime=container.get("runtime")
            ),
            is_baseline=data.get("is_baseline", False),
            defense_stage=data.get("defense_stage"),
        ))

    return ToolListResponse(tools=tools, total=len(tools))


@app.get("/tools/{tool_name}", response_model=ToolInfo, tags=["Tools"])
async def get_tool(
    tool_name: str,
    state: SchedulerState = Depends(get_scheduler_state)
):
    """Get information about a specific tool."""
    tools = state.get_all_tools()
    
    if tool_name not in tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    data = tools[tool_name]
    container = data.get("container", {})
    
    return ToolInfo(
        name=data.get("name", tool_name),
        container=ContainerInfo(
            image=container.get("image", ""),
            command=container.get("command", ""),
            runtime=container.get("runtime")
        ),
        is_baseline=data.get("is_baseline", False)
    )


@app.post("/tools", response_model=ToolInfo, tags=["Tools"])
async def add_tool(
    request: AddToolRequest,
    state: SchedulerState = Depends(get_scheduler_state)
):
    """
    Add a new tool to the pipeline.
    
    Note: This adds the tool to the runtime registry only.
    For persistent tools, update configs/tools.yaml.
    """
    state.add_tool(
        name=request.name,
        image=request.image,
        command=request.command,
        runtime=request.runtime,
        is_baseline=request.is_baseline,
        defense_stage=request.defense_stage,
    )
    
    return ToolInfo(
        name=request.name,
        container=ContainerInfo(
            image=request.image,
            command=request.command,
            runtime=request.runtime
        ),
        is_baseline=request.is_baseline,
        defense_stage=request.defense_stage,
    )


# ==============================================================================
# Extended Pipeline Info Endpoints
# ==============================================================================

@app.get("/pipeline", response_model=PipelineDetailResponse, tags=["Pipeline"])
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


# ==============================================================================
# Dataset Endpoints
# ==============================================================================

class DatasetInfoResponse(BaseModel):
    """Response model for dataset information."""
    available: bool = Field(description="Whether dataset is available")
    name: Optional[str] = Field(default=None, description="Dataset name")
    variant: Optional[str] = Field(default=None, description="Dataset variant (clean/poisoned)")
    train_samples: Optional[int] = Field(default=None, description="Number of training samples")
    test_samples: Optional[int] = Field(default=None, description="Number of test samples")
    local_path: Optional[str] = Field(default=None, description="Local path to dataset (if prepared)")
    minio_key: Optional[str] = Field(default=None, description="MinIO object key for dataset")
    minio_available: bool = Field(default=False, description="Whether dataset is in MinIO")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Dataset configuration")
    poisoning: Optional[Dict[str, Any]] = Field(default=None, description="Poisoning configuration if applied")
    model_script: Optional[str] = Field(default=None, description="Path to model config script (e.g., config_model.py)")
    model_script_minio_key: Optional[str] = Field(default=None, description="MinIO object key for model config script")


@app.get("/dataset", response_model=DatasetInfoResponse, tags=["Dataset"])
async def get_dataset_info(state: SchedulerState = Depends(get_scheduler_state)):
    """
    Get dataset information for workers.
    
    Workers use this to:
    1. Check if dataset is available
    2. Get MinIO key to download dataset
    3. Get local path if running on same machine as backend
    
    The backend prepares the dataset on startup and uploads to MinIO.
    Workers should download from MinIO if they don't have local access.
    """
    from .initialization import get_backend_context
    
    context = get_backend_context()
    
    # Check if dataset info is available
    if not context or not context.dataset_info:
        # Fall back to pipeline config
        pipeline_dataset = None
        model_script = None
        if state.pipeline:
            pipeline_dataset = state.pipeline.dataset
            model_cfg = getattr(state.pipeline, "model", None)
            if isinstance(model_cfg, dict):
                model_script = model_cfg.get("script")
        return DatasetInfoResponse(
            available=False,
            minio_available=False,
            config=pipeline_dataset,
            model_script=model_script,
            model_script_minio_key=None,
        )
    
    ds_info = context.dataset_info
    
    # Check if MinIO is available for this dataset
    minio_key = ds_info.get("minio_key")
    minio_available = False
    if minio_key and context.store and context.store.is_available:
        try:
            # Check if the key exists (prefix check)
            minio_available = True  # If we have the key, assume it's there
        except Exception:
            pass
    
    # Get model script path from pipeline config
    model_script = None
    if context.pipeline and context.pipeline.model:
        model_script = context.pipeline.model.get("script")
    
    return DatasetInfoResponse(
        available=True,
        name=ds_info.get("name"),
        variant=ds_info.get("variant"),
        train_samples=ds_info.get("train_samples"),
        test_samples=ds_info.get("test_samples"),
        local_path=ds_info.get("output_dir"),
        minio_key=minio_key,
        minio_available=minio_available,
        config=context.pipeline.dataset if context.pipeline else None,
        poisoning=ds_info.get("poisoning"),
        model_script=model_script,
        model_script_minio_key=ds_info.get("model_script_minio_key"),
    )


@app.get("/dataset/download-url", tags=["Dataset"])
async def get_dataset_download_url(
    state: SchedulerState = Depends(get_scheduler_state),
    expires_in: int = Query(default=3600, description="URL expiration time in seconds")
):
    """
    Get a presigned URL to download the dataset from MinIO.
    
    Workers can use this URL to download the dataset directly.
    """
    from .initialization import get_backend_context
    
    context = get_backend_context()
    
    if not context or not context.dataset_info:
        raise HTTPException(status_code=404, detail="Dataset not available")
    
    minio_key = context.dataset_info.get("minio_key")
    if not minio_key:
        raise HTTPException(status_code=404, detail="Dataset not in MinIO")
    
    if not context.store or not context.store.is_available:
        raise HTTPException(status_code=503, detail="MinIO store not available")
    
    try:
        # Return the key and let workers use their own MinIO connection
        return {
            "minio_key": minio_key,
            "bucket": context.store.config.bucket,
            "endpoint": context.store.config.endpoint,
            "message": "Use MinIO client to download with the provided key"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate download URL: {e}")


# ==============================================================================
# System Statistics Endpoints
# ==============================================================================

@app.get("/stats/database", tags=["Statistics"])
async def get_database_stats(state: SchedulerState = Depends(get_scheduler_state)):
    """
    Get database statistics.
    
    Returns information about database connectivity and stored data.
    """
    if not state.db_service or not state.db_service.is_available():
        return {
            "available": False,
            "message": "Database service not available"
        }
    
    # Get evaluation result counts
    evaluation_count = 0
    try:
        from ..db.models import EvaluationResultModel
        from ..db import session_scope
        
        with session_scope() as session:
            evaluation_count = session.query(EvaluationResultModel).count()
    except Exception as e:
        logger.warning(f"Failed to get evaluation result count: {e}")
    
    stats = {
        "available": True,
        "task_progress": state.db_service.get_task_progress(),
        "worker_stats": state.db_service.get_worker_stats(),
        "evaluation_results": {
            "total_count": evaluation_count,
            "message": f"Found {evaluation_count} evaluation results in database"
        }
    }
    
    # Add pipeline-specific evaluation count if scheduler is initialized
    if state.scheduler and state.scheduler.pipeline:
        try:
            from ..db.models import EvaluationResultModel
            from ..db import session_scope
            
            with session_scope() as session:
                pipeline_eval_count = session.query(EvaluationResultModel).filter(
                    EvaluationResultModel.pipeline_id == state.scheduler.pipeline.id
                ).count()
                stats["evaluation_results"]["pipeline_count"] = pipeline_eval_count
                stats["evaluation_results"]["pipeline_id"] = state.scheduler.pipeline.id
        except Exception as e:
            logger.warning(f"Failed to get pipeline evaluation count: {e}")
    
    return stats


@app.get("/stats/store", tags=["Statistics"])
async def get_store_stats():
    """
    Get artifact store (MinIO) statistics.
    
    Returns information about MinIO connectivity and stored artifacts.
    """
    context = get_backend_context()
    
    if not context or not context.store:
        return {
            "available": False,
            "message": "Artifact store not available"
        }
    
    store = context.store
    if not store.is_available:
        return {
            "available": False,
            "message": "MinIO connection not available"
        }
    
    # Get artifact count and size
    artifact_count = 0
    total_size = 0
    seen_keys = set()
    
    for obj in store.list_objects("artifacts/"):
        total_size += obj.size or 0
        parts = obj.object_name.split("/")
        if len(parts) >= 2:
            cache_key = parts[1]
            if cache_key not in seen_keys:
                seen_keys.add(cache_key)
                artifact_count += 1
    
    return {
        "available": True,
        "endpoint": store.config.endpoint,
        "bucket": store.config.bucket,
        "artifact_count": artifact_count,
        "total_size_bytes": total_size,
        "total_size_human": _format_size(total_size)
    }


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


@app.get("/stats/system", tags=["Statistics"])
async def get_system_stats(state: SchedulerState = Depends(get_scheduler_state)):
    """
    Get overall system statistics.
    
    Combines database, store, and scheduler statistics.
    """
    context = get_backend_context()
    
    # Database status
    db_available = state.db_service and state.db_service.is_available()
    
    # Store status
    store_available = context and context.store and context.store.is_available
    
    # Scheduler status
    scheduler_active = state.is_initialized()
    
    return {
        "scheduler_active": scheduler_active,
        "database_available": db_available,
        "store_available": store_available,
        "workers_registered": len(state.workers),
        "tasks_tracked": len(state.task_metadata),
        "started_at": state.started_at.isoformat() if state.started_at else None
    }


# ==============================================================================
# Registry Endpoints (Tools & Evaluators)
# ==============================================================================

class EvaluatorInfo(BaseModel):
    """Evaluator information."""
    name: str = Field(description="Evaluator name")
    container: ContainerInfo = Field(description="Container configuration")
    required_artifacts: List[str] = Field(default_factory=list, description="Required files")
    metrics: List[str] = Field(default_factory=list, description="Metrics produced")
    defense_types: List[str] = Field(default_factory=list, description="Applicable defense types")


class EvaluatorListResponse(BaseModel):
    """Response for list of evaluators."""
    evaluators: List[EvaluatorInfo] = Field(description="List of evaluators")
    total: int = Field(description="Total number of evaluators")


class AddEvaluatorRequest(BaseModel):
    """Request to add a new evaluator."""
    name: str = Field(description="Evaluator name")
    image: str = Field(description="Container image")
    command: str = Field(description="Command to run")
    runtime: Optional[str] = Field(default=None, description="Container runtime")
    required_artifacts: List[str] = Field(default_factory=list, description="Required artifacts")
    metrics: List[str] = Field(default_factory=list, description="Metrics produced")
    defense_types: List[str] = Field(default_factory=list, description="Applicable defense types")


@app.get("/registry/tools", response_model=ToolListResponse, tags=["Registry"])
async def registry_list_tools(state: SchedulerState = Depends(get_scheduler_state)):
    """Get all registered tools from the registry."""
    tools_dict = state.get_all_tools()

    tools = []
    for name, data in tools_dict.items():
        container = data.get("container", {})
        tools.append(ToolInfo(
            name=data.get("name", name),
            key=name,  # YAML registry key, used for tools_override
            container=ContainerInfo(
                image=container.get("image", ""),
                command=container.get("command", ""),
                runtime=container.get("runtime")
            ),
            is_baseline=data.get("is_baseline", False),
            defense_stage=data.get("defense_stage"),
        ))

    return ToolListResponse(tools=tools, total=len(tools))


@app.post("/registry/tools", response_model=ToolInfo, tags=["Registry"])
async def registry_add_tool(
    request: AddToolRequest,
    state: SchedulerState = Depends(get_scheduler_state),
    _auth: None = Depends(_require_pipeline_key),
):
    """
    Add a new tool to the registry.
    
    Persists to tools registry YAML and also updates runtime registry.
    """
    tool_key = _tool_key_from_name(request.name)
    _persist_tool_to_yaml(tool_key, request)
    try:
        from ..pipeline.tools import init_tool_registry
        init_tool_registry(str(_tools_registry_path()))
    except Exception as exc:
        logger.warning("Failed to reload tool registry after persistence: %s", exc)

    state.add_tool(
        name=request.name,
        image=request.image,
        command=request.command,
        runtime=request.runtime,
        is_baseline=request.is_baseline,
        defense_stage=request.defense_stage,
        key=tool_key,
    )
    
    return ToolInfo(
        name=request.name,
        key=tool_key,
        container=ContainerInfo(
            image=request.image,
            command=request.command,
            runtime=request.runtime
        ),
        is_baseline=request.is_baseline,
        defense_stage=request.defense_stage,
    )


@app.get("/registry/evaluators", response_model=EvaluatorListResponse, tags=["Registry"])
async def registry_list_evaluators():
    """Get all registered evaluators from the registry."""
    from ..pipeline.config_loader import get_all_evaluators, init_evaluator_registry
    
    # Initialize if not already done
    try:
        init_evaluator_registry()
    except Exception:
        pass
    
    evaluators_dict = get_all_evaluators()
    
    evaluators = []
    for name, eval_def in evaluators_dict.items():
        evaluators.append(EvaluatorInfo(
            name=eval_def.name,
            container=ContainerInfo(
                image=eval_def.container.image,
                command=eval_def.container.command,
                runtime=eval_def.container.runtime
            ),
            required_artifacts=eval_def.required_artifacts,
            metrics=eval_def.metrics,
            defense_types=eval_def.defense_types
        ))
    
    return EvaluatorListResponse(evaluators=evaluators, total=len(evaluators))


@app.post("/registry/evaluators", response_model=EvaluatorInfo, tags=["Registry"])
async def registry_add_evaluator(request: AddEvaluatorRequest):
    """
    Add a new evaluator to the registry.
    
    Note: This is runtime only. For persistence, update configs/evaluators.yaml.
    """
    # For now, just return the evaluator info
    # Full persistence would require updating YAML file
    return EvaluatorInfo(
        name=request.name,
        container=ContainerInfo(
            image=request.image,
            command=request.command,
            runtime=request.runtime
        ),
        required_artifacts=request.required_artifacts,
        metrics=request.metrics,
        defense_types=request.defense_types
    )


# ==============================================================================
# Metrics Endpoints
# ==============================================================================

class MetricValue(BaseModel):
    """Single metric value."""
    name: str = Field(description="Metric name")
    value: Optional[float] = Field(default=None, description="Metric value (null if skipped)")


class WorkflowMetrics(BaseModel):
    """Metrics for a single workflow."""
    workflow_id: str = Field(description="Workflow ID")
    workflow_name: str = Field(description="Workflow name")
    workflow_tools: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Non-evaluation tool names grouped by stage"
    )
    workflow_tools_label: str = Field(
        default="",
        description="Human-readable non-evaluation tool sequence"
    )
    metrics: Dict[str, Optional[float]] = Field(description="Metric name to value mapping")
    evaluators_run: List[str] = Field(description="Evaluators that ran")
    evaluators_skipped: List[str] = Field(description="Evaluators that skipped")
    is_baseline: bool = Field(default=False, description="Whether this workflow uses only baseline (noop) tools")


class PipelineMetricsResponse(BaseModel):
    """Metrics for all workflows in a pipeline."""
    pipeline_id: str = Field(description="Pipeline ID")
    pipeline_name: str = Field(description="Pipeline name")
    workflow_count: int = Field(description="Number of workflows")
    metric_names: List[str] = Field(description="All metric names across workflows")
    workflows: List[WorkflowMetrics] = Field(description="Metrics per workflow")
    summary: Dict[str, Dict[str, Optional[float]]] = Field(
        description="Summary stats (min, max, avg) per metric"
    )


def _stage_rank(task_type: Any) -> int:
    """Sort task stages in pipeline order, putting unknowns at the end."""
    key = str(task_type).lower()
    order = {
        TaskType.PRE_TRAINING.value: 0,
        "pre": 0,
        TaskType.IN_TRAINING.value: 1,
        "in": 1,
        "during": 1,
        "during_training": 1,
        TaskType.POST_TRAINING.value: 2,
        "post": 2,
        "post_training": 2,
        TaskType.DEPLOYMENT.value: 3,
        "deploy": 3,
        "deployment": 3,
    }
    return order.get(key, 99)


def _stage_label(task_type: Any) -> str:
    """Map task type to a user-friendly short stage label."""
    key = str(task_type).lower()
    mapping = {
        TaskType.PRE_TRAINING.value: "pre",
        "pre": "pre",
        TaskType.IN_TRAINING.value: "in",
        "in": "in",
        "during": "in",
        "during_training": "in",
        TaskType.POST_TRAINING.value: "post",
        "post": "post",
        "post_training": "post",
        TaskType.DEPLOYMENT.value: "deploy",
        "deploy": "deploy",
        "deployment": "deploy",
    }
    return mapping.get(key, key)


def _workflow_tools_metadata(tasks: List[Any]) -> Dict[str, Any]:
    """Return grouped stage tools and a compact label for workflow display."""
    grouped: Dict[str, List[str]] = {}
    for task in sorted(tasks, key=lambda t: (_stage_rank(getattr(t, "task_type", "")), getattr(t, "tool_name", ""))):
        task_type = getattr(task, "task_type", "")
        if str(task_type).lower() in {TaskType.EVALUATION.value, "evaluation"}:
            continue

        stage = _stage_label(task_type)
        tool_name = getattr(task, "tool_name", None)
        if tool_name is None:
            tool_obj = getattr(task, "tool", None)
            tool_name = getattr(tool_obj, "name", None)
        if not tool_name:
            continue

        grouped.setdefault(stage, [])
        if tool_name not in grouped[stage]:
            grouped[stage].append(str(tool_name))

    label_parts: List[str] = []
    for stage in ("pre", "in", "post", "deploy"):
        tools = grouped.get(stage, [])
        if not tools:
            continue
        label_parts.append(f"{stage}: {', '.join(tools)}")

    return {
        "workflow_tools": grouped,
        "workflow_tools_label": " | ".join(label_parts),
    }


def _allowed_metrics_for_evaluator(evaluator_name: str) -> Optional[set]:
    """
    Return the allowlist of metric names for an evaluator.

    This prevents one evaluator from overwriting another evaluator's metrics
    with undeclared keys (e.g., fingerprinting emitting clean_accuracy).
    """
    try:
        from ..pipeline.config_loader import get_all_evaluators
        evaluators = get_all_evaluators()
        evaluator = evaluators.get(evaluator_name)
        if evaluator and evaluator.metrics:
            return set(evaluator.metrics)
    except Exception:
        pass
    return None


def _should_override_metric(existing_evaluator: Optional[str], new_evaluator: str, metric_name: str) -> bool:
    """
    Decide overwrite behavior when multiple evaluators emit the same metric key.

    clean_accuracy may appear in both clean and adversarial evaluators; prefer
    the clean evaluator value for dashboard consistency.
    """
    if existing_evaluator is None:
        return True
    if metric_name == "clean_accuracy":
        if existing_evaluator == "clean":
            return False
        if new_evaluator == "clean":
            return True
    return False


def _export_run_metrics_csv(run_id: str, scheduler: Scheduler) -> Optional[Path]:
    """
    Export per-workflow evaluation metrics to CSV for a completed run.

    Every expected evaluator is included. If evaluator output is missing, skipped,
    or failed, metric values are written as -1.
    """
    try:
        from ..db import session_scope
        from ..db.models import EvaluationResultModel, PipelineRunModel
    except Exception as e:
        logger.warning(f"Unable to import DB models for metrics CSV export: {e}")
        return None

    pipeline = getattr(scheduler, "pipeline", None)
    if not pipeline:
        return None

    workflow_name_by_id: Dict[str, str] = {wf.id: wf.name for wf in pipeline.workflows}
    workflow_is_baseline: Dict[str, bool] = {}
    expected_evaluators_by_workflow: Dict[str, set] = {}
    expected_metric_names_by_evaluator: Dict[str, set] = {}

    for wf in pipeline.workflows:
        non_eval_tasks = [t for t in wf.tasks if t.task_type != TaskType.EVALUATION]
        workflow_is_baseline[wf.id] = bool(non_eval_tasks) and all(
            getattr(t.tool, "is_baseline", False) for t in non_eval_tasks
        )
        for task in wf.tasks:
            if task.task_type != TaskType.EVALUATION:
                continue
            evaluator_name = str(task.tool.name)
            expected_evaluators_by_workflow.setdefault(wf.id, set()).add(evaluator_name)
            expected_metric_names_by_evaluator.setdefault(evaluator_name, set()).update(
                [str(m) for m in (task.config or {}).get("metrics", []) if str(m).strip()]
            )

    if not expected_evaluators_by_workflow:
        logger.info(f"Run {run_id}: no evaluation tasks found; skipping metrics CSV export")
        return None

    with session_scope() as session:
        run = session.query(PipelineRunModel).filter(PipelineRunModel.id == run_id).first()
        if not run:
            logger.warning(f"Run {run_id} not found while exporting metrics CSV")
            return None

        results = session.query(EvaluationResultModel).filter(
            EvaluationResultModel.run_id == run_id
        ).all()

        # Backward compatibility for rows keyed only by pipeline_id.
        if not results:
            results = session.query(EvaluationResultModel).filter(
                EvaluationResultModel.pipeline_id == run_id
            ).all()

    result_lookup: Dict[tuple, Any] = {}
    observed_metric_names_by_evaluator: Dict[str, set] = {}
    for result in results:
        key = (result.workflow_id, result.evaluator_name)
        result_lookup[key] = result
        observed_metric_names_by_evaluator.setdefault(result.evaluator_name, set()).update(
            list((result.metrics or {}).keys())
        )

    metric_names_by_evaluator: Dict[str, List[str]] = {}
    for evaluator_name in set(expected_metric_names_by_evaluator) | set(observed_metric_names_by_evaluator):
        merged = (
            expected_metric_names_by_evaluator.get(evaluator_name, set())
            | observed_metric_names_by_evaluator.get(evaluator_name, set())
        )
        metric_names_by_evaluator[evaluator_name] = sorted(merged)

    all_evaluators = sorted(
        set(expected_metric_names_by_evaluator.keys())
        | set(observed_metric_names_by_evaluator.keys())
    )
    if not all_evaluators:
        return None

    output_dir = Path("results") / run.pipeline_config_id / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_summary.csv"

    header: List[str] = ["run_id", "workflow_id", "workflow_name", "is_baseline"]
    for evaluator_name in all_evaluators:
        header.append(f"{evaluator_name}.status")
        for metric_name in metric_names_by_evaluator.get(evaluator_name, []):
            header.append(f"{evaluator_name}.{metric_name}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for workflow_id in sorted(expected_evaluators_by_workflow.keys()):
            row: Dict[str, Any] = {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "workflow_name": workflow_name_by_id.get(workflow_id, workflow_id),
                "is_baseline": workflow_is_baseline.get(workflow_id, False),
            }
            expected_evaluators = expected_evaluators_by_workflow.get(workflow_id, set())
            for evaluator_name in all_evaluators:
                result = result_lookup.get((workflow_id, evaluator_name))
                if evaluator_name not in expected_evaluators:
                    status = "not_applicable"
                elif result is None:
                    status = "missing"
                elif result.skipped:
                    status = "skipped"
                elif not result.success:
                    status = "failed"
                else:
                    status = "ok"
                row[f"{evaluator_name}.status"] = status

                for metric_name in metric_names_by_evaluator.get(evaluator_name, []):
                    col = f"{evaluator_name}.{metric_name}"
                    if status != "ok":
                        row[col] = -1
                        continue
                    metric_val = (result.metrics or {}).get(metric_name) if result else None
                    row[col] = metric_val if metric_val is not None else -1

            writer.writerow(row)

    return csv_path


@app.get("/pipelines/{pipeline_id}/metrics", response_model=PipelineMetricsResponse, tags=["Metrics"])
async def get_pipeline_metrics(
    pipeline_id: str,
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """
    Get evaluation metrics for all workflows in a pipeline.
    
    Returns metrics from all evaluators across all workflows,
    including summary statistics.
    """
    pipeline = scheduler.pipeline
    
    if pipeline.id != pipeline_id and pipeline.name != pipeline_id:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    
    # Try to get metrics from database
    all_metrics = []
    metric_names = set()
    db_matched = False  # True only when DB has rows matching current workflow IDs

    if state.db_service and state.db_service.is_available():
        # Get from database
        try:
            from ..db.models import EvaluationResultModel
            from ..db import session_scope

            current_workflow_ids = {wf.id for wf in pipeline.workflows}

            with session_scope() as session:
                results = session.query(EvaluationResultModel).filter(
                    EvaluationResultModel.pipeline_id == pipeline.id,
                    EvaluationResultModel.workflow_id.in_(current_workflow_ids)
                ).all()

                logger.debug(f"Found {len(results)} evaluation results in database for pipeline {pipeline.id}")

                if results:
                    db_matched = True
                    # Group by workflow
                    by_workflow = {}
                    for r in results:
                        if r.workflow_id not in by_workflow:
                            by_workflow[r.workflow_id] = {
                                "metrics": {},
                                "metric_sources": {},
                                "run": [],
                                "skipped": []
                            }

                        if r.skipped:
                            by_workflow[r.workflow_id]["skipped"].append(r.evaluator_name)
                        else:
                            by_workflow[r.workflow_id]["run"].append(r.evaluator_name)
                        allowed_metrics = _allowed_metrics_for_evaluator(r.evaluator_name)
                        for metric_name, value in (r.metrics or {}).items():
                            if allowed_metrics is not None and metric_name not in allowed_metrics:
                                continue
                            wf_bucket = by_workflow[r.workflow_id]
                            existing_source = wf_bucket["metric_sources"].get(metric_name)
                            if _should_override_metric(existing_source, r.evaluator_name, metric_name):
                                wf_bucket["metrics"][metric_name] = value
                                wf_bucket["metric_sources"][metric_name] = r.evaluator_name
                            metric_names.add(metric_name)

                    for wf in pipeline.workflows:
                        wf_data = by_workflow.get(wf.id, {"metrics": {}, "run": [], "skipped": []})
                        non_eval = [t for t in wf.tasks if t.task_type != TaskType.EVALUATION]
                        workflow_tools = _workflow_tools_metadata(wf.tasks)
                        all_metrics.append(WorkflowMetrics(
                            workflow_id=wf.id,
                            workflow_name=wf.name,
                            workflow_tools=workflow_tools["workflow_tools"],
                            workflow_tools_label=workflow_tools["workflow_tools_label"],
                            metrics=wf_data["metrics"],
                            evaluators_run=wf_data["run"],
                            evaluators_skipped=wf_data["skipped"],
                            is_baseline=bool(non_eval) and all(t.tool.is_baseline for t in non_eval),
                        ))

        except Exception as e:
            logger.warning(f"Failed to get metrics from database: {e}", exc_info=True)

    # If database had no rows matching current workflow IDs, fall back to in-memory task metadata.
    # This handles: DB unavailable, fresh backend restart (new sequential IDs), or stale DB rows
    # from a previous run that used different workflow IDs.
    if not db_matched:
        by_workflow = {
            wf.id: {"metrics": {}, "run": [], "skipped": []} for wf in pipeline.workflows
        }
        
        # Walk over all evaluation tasks and extract evaluation_result from task metadata
        for wf in pipeline.workflows:
            for task in wf.tasks:
                if task.task_type != TaskType.EVALUATION:
                    continue
                
                meta = state.task_metadata.get(task.id) or {}
                result = meta.get("result") or {}
                eval_result = result.get("evaluation_result")
                if not isinstance(eval_result, dict):
                    continue
                
                evaluator_name = task.tool.name
                metrics_dict = eval_result.get("metrics") or {}
                skipped = bool(eval_result.get("skipped"))
                
                wf_data = by_workflow[wf.id]
                if skipped:
                    wf_data["skipped"].append(evaluator_name)
                else:
                    wf_data["run"].append(evaluator_name)
                for metric_name, value in metrics_dict.items():
                    try:
                        numeric_val = float(value)
                    except (TypeError, ValueError):
                        continue
                    wf_data["metrics"][metric_name] = numeric_val
                    metric_names.add(metric_name)
        
        for wf in pipeline.workflows:
            wf_data = by_workflow.get(wf.id, {"metrics": {}, "run": [], "skipped": []})
            non_eval = [t for t in wf.tasks if t.task_type != TaskType.EVALUATION]
            workflow_tools = _workflow_tools_metadata(wf.tasks)
            all_metrics.append(WorkflowMetrics(
                workflow_id=wf.id,
                workflow_name=wf.name,
                workflow_tools=workflow_tools["workflow_tools"],
                workflow_tools_label=workflow_tools["workflow_tools_label"],
                metrics=wf_data["metrics"],
                evaluators_run=wf_data["run"],
                evaluators_skipped=wf_data["skipped"],
                is_baseline=bool(non_eval) and all(t.tool.is_baseline for t in non_eval),
            ))

    # If still no results, return empty metrics
    if not all_metrics:
        for wf in pipeline.workflows:
            non_eval = [t for t in wf.tasks if t.task_type != TaskType.EVALUATION]
            workflow_tools = _workflow_tools_metadata(wf.tasks)
            all_metrics.append(WorkflowMetrics(
                workflow_id=wf.id,
                workflow_name=wf.name,
                workflow_tools=workflow_tools["workflow_tools"],
                workflow_tools_label=workflow_tools["workflow_tools_label"],
                metrics={},
                evaluators_run=[],
                evaluators_skipped=[],
                is_baseline=bool(non_eval) and all(t.tool.is_baseline for t in non_eval),
            ))
    
    # Calculate summary statistics
    summary = {}
    for metric_name in metric_names:
        values = [
            wf.metrics.get(metric_name)
            for wf in all_metrics
            if wf.metrics.get(metric_name) is not None
        ]
        
        if values:
            summary[metric_name] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "count": len(values)
            }
        else:
            summary[metric_name] = {"min": None, "max": None, "avg": None, "count": 0}
    
    return PipelineMetricsResponse(
        pipeline_id=pipeline.id,
        pipeline_name=pipeline.name,
        workflow_count=len(pipeline.workflows),
        metric_names=sorted(metric_names),
        workflows=all_metrics,
        summary=summary
    )


@app.get("/workflows/{workflow_id}/metrics", tags=["Metrics"])
async def get_workflow_metrics(
    workflow_id: str,
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """Get evaluation metrics for a specific workflow."""
    # Find workflow
    workflow = None
    for w in scheduler.pipeline.workflows:
        if w.id == workflow_id or w.name == workflow_id:
            workflow = w
            break
    
    if workflow is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    
    metrics = {}
    evaluators_run = []
    evaluators_skipped = []
    
    # Try to get from database
    if state.db_service and state.db_service.is_available():
        try:
            from ..db.models import EvaluationResultModel
            session = state.db_service.get_session()
            results = session.query(EvaluationResultModel).filter(
                EvaluationResultModel.workflow_id == workflow.id
            ).all()
            
            for r in results:
                if r.skipped:
                    evaluators_skipped.append({
                        "evaluator": r.evaluator_name,
                        "reason": r.skip_reason
                    })
                else:
                    evaluators_run.append(r.evaluator_name)
                    metrics.update(r.metrics or {})
                    
        except Exception as e:
            logger.warning(f"Failed to get metrics from database: {e}")
    
    return {
        "workflow_id": workflow.id,
        "workflow_name": workflow.name,
        "pipeline_id": workflow.pipeline_id,
        "metrics": metrics,
        "evaluators_run": evaluators_run,
        "evaluators_skipped": evaluators_skipped
    }


# ==============================================================================
# Pipeline Config and Run Management
# ==============================================================================

class PipelineConfigResponse(BaseModel):
    """Response model for a pipeline config."""
    id: str = Field(description="Config ID")
    name: str = Field(description="Config name")
    description: Optional[str] = Field(default=None, description="Config description")
    config_path: str = Field(description="Path to pipeline config YAML")
    attack_config_path: Optional[str] = Field(default=None, description="Path to attack config YAML")
    config_hash: Optional[str] = Field(default=None, description="Config hash for change detection")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")


class PipelineConfigListResponse(BaseModel):
    """Response model for list of pipeline configs."""
    configs: List[PipelineConfigResponse] = Field(description="List of configs")
    total: int = Field(description="Total number of configs")


class StartPipelineRunRequest(BaseModel):
    """Request to start a new pipeline run."""
    use_cache: bool = Field(default=True, description="Whether to use cache")
    combo_id: Optional[str] = Field(default=None, description="Specific combination ID to run (optional)")
    dry_run: bool = Field(default=False, description="Dry run mode (validate only)")
    attack_config_path: Optional[str] = Field(default=None, description="Override attack config path")
    dataset_name: Optional[str] = Field(default=None, description="Override dataset name (e.g. cifar10, celeba)")
    dataset_variant: Optional[str] = Field(default=None, description="Override dataset variant (clean or poisoned)")
    tools_override: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Override tool lists per stage. Keys are stage names; values are ordered tool name lists. "
                    "Stored as-is and locked for the duration of the run."
    )
    model_script: Optional[str] = Field(
        default=None,
        description="Override model script path (e.g. configs/model/config_model_resnet.py)"
    )


class PipelineRunResponse(BaseModel):
    """Response model for a pipeline run."""
    id: str = Field(description="Run ID")
    pipeline_config_id: str = Field(description="Config ID")
    run_number: int = Field(description="Run number for this config")
    use_cache: bool = Field(description="Whether cache was used")
    tools_config: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Locked tool configuration snapshot (stage → tool names). "
                    "Set at run start and immutable for the lifetime of the run."
    )
    status: str = Field(description="Run status")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: str = Field(description="Creation timestamp")
    started_at: Optional[str] = Field(default=None, description="Start timestamp")
    completed_at: Optional[str] = Field(default=None, description="Completion timestamp")


class PipelineRunListResponse(BaseModel):
    """Response model for list of pipeline runs."""
    runs: List[PipelineRunResponse] = Field(description="List of runs")
    total: int = Field(description="Total number of runs")


class RestartPipelineRunRequest(BaseModel):
    """Request to restart a pipeline run."""
    use_cache: bool = Field(default=True, description="Whether to use cache")
    attack_config_path: Optional[str] = Field(default=None, description="Override attack config path")


@app.get("/api/pipeline-configs", response_model=PipelineConfigListResponse, tags=["Pipeline Configs"])
async def get_pipeline_configs():
    """Get all available pipeline configurations."""
    try:
        from .config_discovery import sync_configs_to_db, get_all_configs
        
        # Sync configs from filesystem to DB
        sync_configs_to_db()
        
        # Get all configs
        configs = get_all_configs()
        
        return PipelineConfigListResponse(
            configs=[
                PipelineConfigResponse(
                    id=config.id,
                    name=config.name,
                    description=config.description,
                    config_path=config.config_path,
                    attack_config_path=config.attack_config_path,
                    config_hash=config.config_hash,
                    created_at=config.created_at.isoformat() if config.created_at else None,
                    updated_at=config.updated_at.isoformat() if config.updated_at else None,
                )
                for config in configs
            ],
            total=len(configs)
        )
    except Exception as e:
        logger.error(f"Failed to get pipeline configs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline configs: {str(e)}")


@app.get("/api/pipeline-configs/{config_id}", response_model=PipelineConfigResponse, tags=["Pipeline Configs"])
async def get_pipeline_config(config_id: str):
    """Get a specific pipeline configuration."""
    try:
        from .config_discovery import get_config_by_id
        
        config = get_config_by_id(config_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"Pipeline config '{config_id}' not found")
        
        return PipelineConfigResponse(
            id=config.id,
            name=config.name,
            description=config.description,
            config_path=config.config_path,
            attack_config_path=config.attack_config_path,
            config_hash=config.config_hash,
            created_at=config.created_at.isoformat() if config.created_at else None,
            updated_at=config.updated_at.isoformat() if config.updated_at else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline config {config_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline config: {str(e)}")


@app.get("/api/model-configs", tags=["Pipeline Configs"])
async def list_model_configs():
    """List available model configuration scripts."""
    import glob as glob_mod
    scripts = sorted(glob_mod.glob("configs/model/*.py"))
    return {
        "models": [
            {
                "path": s,
                "name": Path(s).stem,
            }
            for s in scripts
        ],
        "total": len(scripts),
    }


@app.post("/api/pipeline-configs/{config_id}/runs", response_model=PipelineRunResponse, tags=["Pipeline Runs"])
async def start_pipeline_run(
    config_id: str,
    request: StartPipelineRunRequest,
    state: SchedulerState = Depends(get_scheduler_state),
    _auth: None = Depends(_require_pipeline_key),
):
    """Start a new pipeline run for a configuration."""

    def _start_pipeline_run_background(
        run_id: str,
        config_path: str,
        config_name: str,
        run_number: int,
        request_data: Dict[str, Any],
        state: SchedulerState,
    ) -> None:
        """
        Build pipeline + initialize scheduler in background.

        This keeps the HTTP request fast for large configs (e.g., trades) while
        preserving run state transitions in DB.
        """
        from ..db import session_scope, PipelineRunRepository, PipelineRunStatus
        from ..pipeline.config_loader import create_pipeline_from_config, load_pipeline_config
        from ..data import DatasetManager

        try:
            # Step 1b: transition to RUNNING immediately after background init starts.
            # Pipeline creation for large configs can take minutes; keeping PENDING
            # that whole time makes the UI look stuck even though work is active.
            with session_scope() as session:
                run_repo = PipelineRunRepository(session)
                run_repo.update_status(run_id, PipelineRunStatus.RUNNING)
            # Step 2: Create pipeline (CPU-heavy for large combination counts)
            t_create_start = time.time()
            pipeline = create_pipeline_from_config(
                config_path=config_path,
                tools_yaml_path="configs/tools.yaml",
                evaluators_yaml_path="configs/evaluators.yaml",
                pipeline_name=f"{config_name} (Run {run_number})",
                clear_registry=True,
                include_evaluation=True,
                dataset_name=request_data.get("dataset_name") or None,
                dataset_variant=request_data.get("dataset_variant") or None,
                tools_override=request_data.get("tools_override") or None,
                model_script=request_data.get("model_script") or None,
                attack_config_path=(
                    request_data.get("attack_config_path")
                    or request_data.get("config_attack_config_path")
                    or None
                ),
            )

            # Bind pipeline and all tasks/workflows to this run's ID
            pipeline.id = run_id
            for workflow in pipeline.workflows:
                workflow.pipeline_id = run_id
                workflow.run_id = run_id
                for task in workflow.tasks:
                    task.pipeline_id = run_id
                    task.run_id = run_id
                    # Propagate per-run cache policy to workers via task payload.
                    task.config["_run_use_cache"] = bool(request_data.get("use_cache", True))

            # Step 3: Dataset context (best-effort)
            ctx = get_backend_context()
            if ctx:
                ctx.pipeline = pipeline
                try:
                    cfg = load_pipeline_config(config_path)
                    effective_ds_name = request_data.get("dataset_name") or cfg.dataset.name
                    effective_ds_variant = request_data.get("dataset_variant") or cfg.dataset.variant
                    current_ds = ctx.dataset_info or {}
                    needs_prepare = (
                        not current_ds
                        or current_ds.get("name") != effective_ds_name
                        or current_ds.get("variant") != effective_ds_variant
                    )
                    if needs_prepare:
                        base_dir = Path("./data").resolve()
                        manager = ctx.dataset_manager or DatasetManager(base_dir)
                        poisoning = None
                        if effective_ds_variant == "poisoned":
                            poisoning = cfg.dataset.params.get("poisoning")
                        cfg.dataset.name = effective_ds_name
                        cfg.dataset.variant = effective_ds_variant
                        t_prepare_start = time.time()
                        ds_info = manager.prepare_dataset(
                            name=cfg.dataset.name,
                            variant=cfg.dataset.variant,
                            poisoning=poisoning,
                            **cfg.dataset.params,
                        )
                        if ds_info:
                            ctx.dataset_info = ds_info.to_dict()
                            ctx.dataset_manager = manager
                            if ctx.store and ctx.store.is_available:
                                dir_suffix = Path(ds_info.output_dir).name
                                dataset_key = f"datasets/{cfg.dataset.name}/{dir_suffix}"
                                try:
                                    t_upload_start = time.time()
                                    ctx.store.upload_directory(ds_info.output_dir, dataset_key)
                                    ctx.dataset_info["minio_key"] = dataset_key

                                    model_script = cfg.model.get("script") if cfg and cfg.model else None
                                    if model_script:
                                        model_path = Path(model_script)
                                        if not model_path.is_absolute():
                                            cfg_base = (
                                                Path(ctx.pipeline_config_path).resolve().parent
                                                if ctx.pipeline_config_path
                                                else Path(config_path).resolve().parent
                                            )
                                            model_path = (cfg_base / model_path).resolve()
                                        if model_path.exists() and model_path.is_file():
                                            model_script_key = f"{dataset_key}/config_model.py"
                                            if ctx.store.upload_file(model_path, model_script_key):
                                                ctx.dataset_info["model_script_minio_key"] = model_script_key
                                                logger.info(
                                                    f"Model script uploaded to MinIO: {model_script_key}"
                                                )
                                            else:
                                                logger.warning(
                                                    f"Failed to upload model script to MinIO: {model_path}"
                                                )
                                        else:
                                            logger.warning(
                                                f"Model script path not found for MinIO upload: {model_path}"
                                            )
                                except Exception as e:
                                    logger.warning(f"Failed to upload dataset to MinIO: {e}")
                        else:
                            raise RuntimeError(
                                f"Dataset preparation returned no dataset info for "
                                f"{cfg.dataset.name}/{cfg.dataset.variant}"
                            )
                except Exception as e:
                    logger.warning(f"Dataset context setup failed for run {run_id}: {e}")
                set_backend_context(ctx)

            # Step 4: Initialize scheduler
            state.initialize(pipeline, scheduler_type="priority")

            # Step 5: Mark RUNNING and sync to DB
            with session_scope() as session:
                run_repo = PipelineRunRepository(session)
                run_repo.update_status(run_id, PipelineRunStatus.RUNNING)

            if state.db_service and state.db_service.is_available():
                state.db_service.sync_pipeline_to_db(pipeline)

            logger.info(f"Run {run_id} initialized in background and marked RUNNING")
        except Exception as e:
            logger.error(f"Failed to initialize run {run_id} in background: {e}", exc_info=True)
            # Persist failure state so UI reflects startup errors.
            try:
                with session_scope() as session:
                    run_repo = PipelineRunRepository(session)
                    run_repo.update_status(run_id, PipelineRunStatus.FAILED, error_message=str(e))
            except Exception as db_e:
                logger.error(f"Failed to persist FAILED status for run {run_id}: {db_e}", exc_info=True)

    try:
        from .config_discovery import get_config_by_id
        from ..db import get_session, session_scope, PipelineRunRepository, PipelineRunStatus
        from ..pipeline.config_loader import (
            load_pipeline_config,
            load_attack_config,
            validate_pipeline_tool_dataset_compatibility,
        )
        from ..pipeline.stage_validation import load_tools_and_validate_pipeline_stages
        import uuid
        from datetime import datetime

        # Get config
        config = get_config_by_id(config_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"Pipeline config '{config_id}' not found")

        effective_attack_config_path = request.attack_config_path or config.attack_config_path
        try:
            load_attack_config(effective_attack_config_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        config_path_obj = Path(config.config_path)
        if config_path_obj.exists():
            loaded_cfg = load_pipeline_config(config.config_path)
            effective_dataset_name = request.dataset_name or loaded_cfg.dataset.name
            if request.tools_override:
                for stage_name, tools in request.tools_override.items():
                    stage_cfg = loaded_cfg.pipeline.get(stage_name)
                    if stage_cfg is not None:
                        stage_cfg.tools = list(tools)

            tools_for_validation = load_tools_and_validate_pipeline_stages(
                loaded_cfg.pipeline,
                tools_yaml_path="configs/tools.yaml",
                fetch_remote_labels=True,
            )
            compatibility_issues = validate_pipeline_tool_dataset_compatibility(
                loaded_cfg,
                tools_for_validation,
                effective_dataset_name,
                fetch_remote_labels=True,
            )
            if compatibility_issues:
                first = compatibility_issues[0]
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Tool/dataset compatibility check failed: "
                        f"tool '{first['tool_id']}' ({first['tool_name']}) in stage '{first['stage']}' "
                        f"uses image '{first['image']}' which supports datasets [{first['supported_datasets']}], "
                        f"but requested dataset is '{first['requested_dataset']}'."
                    ),
                )
        else:
            logger.warning(
                "Skipping dataset compatibility check because config path does not exist: %s",
                config.config_path,
            )

        # ── Step 1: DB check + create run record (short-lived session) ──────────
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            active_runs = run_repo.get_active_runs_for_config(config_id)
            if active_runs:
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot start new run: {len(active_runs)} active run(s) already exist for this config"
                )
            run_number = run_repo.get_next_run_number(config_id)
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            run_obj = run_repo.create({
                "id": run_id,
                "pipeline_config_id": config_id,
                "run_number": run_number,
                "use_cache": request.use_cache,
                "tools_config": request.tools_override,
                "status": PipelineRunStatus.PENDING,
            })
            # Capture fields before session closes
            run_id_val = run_obj.id
            run_number_val = run_obj.run_number
            run_use_cache = run_obj.use_cache
            run_tools_config = getattr(run_obj, "tools_config", None)
            run_status = run_obj.status.value
            run_error = run_obj.error_message
            run_created_at = run_obj.created_at.isoformat()
            run_started_at = run_obj.started_at.isoformat() if run_obj.started_at else None
        # ── session closed here; SQLite lock released ──────────────────────────

        # Heavy initialization moved to background task so UI/API call can return fast.
        request_data = request.model_dump()
        request_data["config_attack_config_path"] = config.attack_config_path
        asyncio.create_task(
            asyncio.to_thread(
                _start_pipeline_run_background,
                run_id_val,
                config.config_path,
                config.name,
                run_number_val,
                request_data,
                state,
            )
        )

        # Once initialization is successfully enqueued, report running to clients.
        # The DB row may still be pending for a brief moment until background
        # initialization updates it, but the API contract here is "accepted and active".
        response_status = (
            PipelineRunStatus.RUNNING.value
            if run_status == PipelineRunStatus.PENDING.value
            else run_status
        )

        return PipelineRunResponse(
            id=run_id_val,
            pipeline_config_id=config_id,
            run_number=run_number_val,
            use_cache=run_use_cache,
            tools_config=run_tools_config,
            status=response_status,
            error_message=run_error,
            created_at=run_created_at,
            started_at=run_started_at,
            completed_at=None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start pipeline run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline run: {str(e)}")


@app.get("/api/pipeline-runs/{run_id}", response_model=PipelineRunResponse, tags=["Pipeline Runs"])
async def get_pipeline_run(run_id: str):
    """Get status of a pipeline run."""
    try:
        from ..db import get_session, session_scope, PipelineRunRepository
        
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            run = run_repo.get_by_id(run_id)
            
            if not run:
                raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")
            
            return PipelineRunResponse(
                id=run.id,
                pipeline_config_id=run.pipeline_config_id,
                run_number=run.run_number,
                use_cache=run.use_cache,
                tools_config=getattr(run, "tools_config", None),
                status=run.status.value,
                error_message=run.error_message,
                created_at=run.created_at.isoformat(),
                started_at=run.started_at.isoformat() if run.started_at else None,
                completed_at=run.completed_at.isoformat() if run.completed_at else None,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline run: {str(e)}")


@app.get("/api/pipeline-runs/{run_id}/metrics", response_model=PipelineMetricsResponse, tags=["Metrics"])
async def get_pipeline_run_metrics(run_id: str):
    """
    Get evaluation metrics for all workflows in a historical pipeline run.

    Unlike /pipelines/{pipeline_id}/metrics this endpoint is purely DB-backed
    and does NOT require the scheduler to have this run's pipeline in memory.
    It works for any completed (or partially-completed) run.
    """
    try:
        from ..db import session_scope
        from ..db.models import EvaluationResultModel, PipelineRunModel, WorkflowModel

        with session_scope() as session:
            run = session.query(PipelineRunModel).filter(
                PipelineRunModel.id == run_id
            ).first()
            if not run:
                raise HTTPException(
                    status_code=404,
                    detail=f"Pipeline run '{run_id}' not found"
                )

            workflows = session.query(WorkflowModel).filter(
                WorkflowModel.run_id == run_id
            ).all()

            # Backward-compatibility fallback: older rows may only have pipeline_id set.
            if not workflows:
                workflows = session.query(WorkflowModel).filter(
                    WorkflowModel.pipeline_id == run_id
                ).all()

            results = session.query(EvaluationResultModel).filter(
                EvaluationResultModel.run_id == run_id
            ).all()

            # Backward-compatibility fallback for legacy rows keyed only by pipeline_id.
            if not results:
                results = session.query(EvaluationResultModel).filter(
                    EvaluationResultModel.pipeline_id == run_id
                ).all()

            # Return an empty-but-valid payload so UI can distinguish
            # "run exists, no metrics persisted yet" from true not-found.
            if not workflows and not results:
                return PipelineMetricsResponse(
                    pipeline_id=run_id,
                    pipeline_name=run_id,
                    workflow_count=0,
                    metric_names=[],
                    workflows=[],
                    summary={},
                )

            # Group evaluation results by workflow
            by_workflow: dict = {}
            metric_names: set = set()
            for r in results:
                if r.workflow_id not in by_workflow:
                    by_workflow[r.workflow_id] = {
                        "metrics": {},
                        "metric_sources": {},
                        "run": [],
                        "skipped": []
                    }
                if r.skipped:
                    by_workflow[r.workflow_id]["skipped"].append(r.evaluator_name)
                else:
                    by_workflow[r.workflow_id]["run"].append(r.evaluator_name)
                allowed_metrics = _allowed_metrics_for_evaluator(r.evaluator_name)
                for metric_name, value in (r.metrics or {}).items():
                    if allowed_metrics is not None and metric_name not in allowed_metrics:
                        continue
                    wf_bucket = by_workflow[r.workflow_id]
                    existing_source = wf_bucket["metric_sources"].get(metric_name)
                    if _should_override_metric(existing_source, r.evaluator_name, metric_name):
                        wf_bucket["metrics"][metric_name] = value
                        wf_bucket["metric_sources"][metric_name] = r.evaluator_name
                    metric_names.add(metric_name)

            # Build per-workflow metrics; determine is_baseline from DB task records.
            workflow_map = {wf.id: wf for wf in workflows}
            all_metrics = []
            for wf in workflows:
                non_eval_tasks = [t for t in wf.tasks if t.task_type != "evaluation"]
                is_baseline = bool(non_eval_tasks) and all(
                    t.tool_is_baseline for t in non_eval_tasks
                )
                wf_data = by_workflow.get(wf.id, {"metrics": {}, "run": [], "skipped": []})
                workflow_tools = _workflow_tools_metadata(wf.tasks)
                all_metrics.append(WorkflowMetrics(
                    workflow_id=wf.id,
                    workflow_name=wf.name,
                    workflow_tools=workflow_tools["workflow_tools"],
                    workflow_tools_label=workflow_tools["workflow_tools_label"],
                    metrics=wf_data["metrics"],
                    evaluators_run=wf_data["run"],
                    evaluators_skipped=wf_data["skipped"],
                    is_baseline=is_baseline,
                ))

            # Include result-only workflows that may exist even when workflow rows are missing.
            for workflow_id, wf_data in by_workflow.items():
                if workflow_id in workflow_map:
                    continue
                all_metrics.append(WorkflowMetrics(
                    workflow_id=workflow_id,
                    workflow_name=workflow_id,
                    workflow_tools={},
                    workflow_tools_label="",
                    metrics=wf_data["metrics"],
                    evaluators_run=wf_data["run"],
                    evaluators_skipped=wf_data["skipped"],
                    is_baseline=False,
                ))

            # Compute summary statistics
            summary: dict = {}
            for metric_name in metric_names:
                values = [
                    wf.metrics.get(metric_name)
                    for wf in all_metrics
                    if wf.metrics.get(metric_name) is not None
                ]
                if values:
                    summary[metric_name] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "count": len(values),
                    }
                else:
                    summary[metric_name] = {"min": None, "max": None, "avg": None, "count": 0}

            return PipelineMetricsResponse(
                pipeline_id=run_id,
                pipeline_name=run_id,
                workflow_count=len(workflows),
                metric_names=sorted(metric_names),
                workflows=all_metrics,
                summary=summary,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get run metrics: {str(e)}")


@app.get("/api/pipeline-configs/{config_id}/runs", response_model=PipelineRunListResponse, tags=["Pipeline Runs"])
async def get_pipeline_runs(config_id: str):
    """Get all runs for a pipeline config."""
    try:
        from ..db import get_session, session_scope, PipelineRunRepository
        
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            runs = run_repo.get_by_config_id(config_id)
            
            return PipelineRunListResponse(
                runs=[
                    PipelineRunResponse(
                        id=run.id,
                        pipeline_config_id=run.pipeline_config_id,
                        run_number=run.run_number,
                        use_cache=run.use_cache,
                        tools_config=getattr(run, "tools_config", None),
                        status=run.status.value,
                        error_message=run.error_message,
                        created_at=run.created_at.isoformat(),
                        started_at=run.started_at.isoformat() if run.started_at else None,
                        completed_at=run.completed_at.isoformat() if run.completed_at else None,
                    )
                    for run in runs
                ],
                total=len(runs)
            )
    except Exception as e:
        logger.error(f"Failed to get pipeline runs for {config_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline runs: {str(e)}")


@app.get("/api/pipeline-runs", response_model=PipelineRunListResponse, tags=["Pipeline Runs"])
async def list_all_pipeline_runs(config_id: Optional[str] = Query(default=None)):
    """List all pipeline runs across all configs, optionally filtered by config_id."""
    try:
        from ..db import session_scope, PipelineRunRepository

        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            runs = run_repo.get_all(config_id=config_id)

            return PipelineRunListResponse(
                runs=[
                    PipelineRunResponse(
                        id=run.id,
                        pipeline_config_id=run.pipeline_config_id,
                        run_number=run.run_number,
                        use_cache=run.use_cache,
                        tools_config=getattr(run, "tools_config", None),
                        status=run.status.value,
                        error_message=run.error_message,
                        created_at=run.created_at.isoformat(),
                        started_at=run.started_at.isoformat() if run.started_at else None,
                        completed_at=run.completed_at.isoformat() if run.completed_at else None,
                    )
                    for run in runs
                ],
                total=len(runs)
            )
    except Exception as e:
        logger.error(f"Failed to list pipeline runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list pipeline runs: {str(e)}")


@app.post("/api/pipeline-runs/{run_id}/stop", response_model=PipelineRunResponse, tags=["Pipeline Runs"])
async def stop_pipeline_run(
    run_id: str,
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """Stop a running pipeline."""
    try:
        from ..db import get_session, session_scope, PipelineRunRepository, PipelineRunStatus
        from ..pipeline.tasks import TaskStatus
        
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            run = run_repo.get_by_id(run_id)
            
            if not run:
                raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")
            
            if run.status not in (PipelineRunStatus.PENDING, PipelineRunStatus.RUNNING):
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot stop run with status '{run.status.value}'"
                )
            
            # Update status to stopping
            run_repo.update_status(run_id, PipelineRunStatus.STOPPING)
            
            # Remove pending tasks from scheduler
            if scheduler and scheduler.pipeline and scheduler.pipeline.id == run_id:
                # Cancel all pending tasks for this run
                for task in scheduler._all_tasks:
                    if task.status == TaskStatus.PENDING and task.run_id == run_id:
                        task.status = TaskStatus.CANCELLED
                        if state.db_service:
                            state.db_service.sync_task_status(
                                task.id,
                                TaskStatus.CANCELLED,
                                error_message="Pipeline stopped by user"
                            )
            
            # Update status to cancelled
            run_repo.update_status(run_id, PipelineRunStatus.CANCELLED)
            
            # TODO: Trigger cache cleanup (will implement in cache cleanup task)
            
            run = run_repo.get_by_id(run_id)
            return PipelineRunResponse(
                id=run.id,
                pipeline_config_id=run.pipeline_config_id,
                run_number=run.run_number,
                use_cache=run.use_cache,
                tools_config=getattr(run, "tools_config", None),
                status=run.status.value,
                error_message=run.error_message,
                created_at=run.created_at.isoformat(),
                started_at=run.started_at.isoformat() if run.started_at else None,
                completed_at=run.completed_at.isoformat() if run.completed_at else None,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop pipeline run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stop pipeline run: {str(e)}")


@app.post("/api/pipeline-runs/{run_id}/restart", response_model=PipelineRunResponse, tags=["Pipeline Runs"])
async def restart_pipeline_run(
    run_id: str,
    request: RestartPipelineRunRequest,
    state: SchedulerState = Depends(get_scheduler_state)
):
    """Restart a stopped/failed pipeline run."""
    try:
        from ..db import get_session, session_scope, PipelineRunRepository, PipelineRunStatus
        from .config_discovery import get_config_by_id
        from ..pipeline.config_loader import create_pipeline_from_config, load_attack_config
        import uuid
        from datetime import datetime
        
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            old_run = run_repo.get_by_id(run_id)
            
            if not old_run:
                raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")
            
            # Check if old run is still active
            if old_run.status in (PipelineRunStatus.PENDING, PipelineRunStatus.RUNNING, PipelineRunStatus.STOPPING):
                raise HTTPException(
                    status_code=400,
                    detail="Cannot restart an active run. Stop it first."
                )
            
            # Get config
            config = get_config_by_id(old_run.pipeline_config_id)
            if not config:
                raise HTTPException(status_code=404, detail=f"Pipeline config '{old_run.pipeline_config_id}' not found")

            effective_attack_config_path = request.attack_config_path or config.attack_config_path
            try:
                load_attack_config(effective_attack_config_path)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            # If not using cache, delete cache from previous run
            if not request.use_cache:
                # TODO: Implement cache deletion (will do in cache cleanup task)
                pass
            
            # Create new run
            run_number = run_repo.get_next_run_number(config.id)
            new_run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            new_run = run_repo.create({
                "id": new_run_id,
                "pipeline_config_id": config.id,
                "run_number": run_number,
                "use_cache": request.use_cache,
                "status": PipelineRunStatus.PENDING,
            })
            
            # Load pipeline from config
            pipeline = create_pipeline_from_config(
                config_path=config.config_path,
                tools_yaml_path="configs/tools.yaml",
                evaluators_yaml_path="configs/evaluators.yaml",
                pipeline_name=f"{config.name} (Run {run_number})",
                clear_registry=True,
                include_evaluation=True,
                attack_config_path=effective_attack_config_path,
            )
            
            # Set pipeline ID to new run_id
            pipeline.id = new_run_id
            
            # Set run_id on all tasks and workflows
            for workflow in pipeline.workflows:
                workflow.pipeline_id = new_run_id
                workflow.run_id = new_run_id
                for task in workflow.tasks:
                    task.pipeline_id = new_run_id
                    task.run_id = new_run_id
            
            # Initialize scheduler with this pipeline
            state.initialize(pipeline, scheduler_type="priority")
            
            # Update run status to running
            run_repo.update_status(new_run_id, PipelineRunStatus.RUNNING)
            
            # Sync pipeline to DB
            if state.db_service and state.db_service.is_available():
                state.db_service.sync_pipeline_to_db(pipeline)
            
            return PipelineRunResponse(
                id=new_run.id,
                pipeline_config_id=new_run.pipeline_config_id,
                run_number=new_run.run_number,
                use_cache=new_run.use_cache,
                tools_config=getattr(new_run, "tools_config", None),
                status=new_run.status.value,
                error_message=new_run.error_message,
                created_at=new_run.created_at.isoformat(),
                started_at=new_run.started_at.isoformat() if new_run.started_at else None,
                completed_at=new_run.completed_at.isoformat() if new_run.completed_at else None,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart pipeline run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to restart pipeline run: {str(e)}")


@app.delete("/api/pipeline-runs/{run_id}/cache", tags=["Pipeline Runs"])
async def delete_pipeline_run_cache(run_id: str):
    """Delete cache entries created by a specific run."""
    try:
        from ..db import get_session, session_scope, PipelineRunRepository, ArtifactRepository
        
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            run = run_repo.get_by_id(run_id)
            
            if not run:
                raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")
            
            artifact_repo = ArtifactRepository(session)
            deleted_count = artifact_repo.delete_by_run_id(run_id)
            
            # TODO: Also delete from MinIO/local cache filesystem
            
            return {
                "success": True,
                "run_id": run_id,
                "deleted_artifacts": deleted_count,
                "message": f"Deleted {deleted_count} cache entries for run {run_id}"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete cache for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete cache: {str(e)}")


@app.delete("/api/pipeline-runs/{run_id}/workspace", tags=["Pipeline Runs"])
async def delete_pipeline_run_workspace(run_id: str):
    """Delete workspace files for a specific run."""
    try:
        from pathlib import Path
        import shutil
        
        from ..db import get_session, session_scope, PipelineRunRepository
        
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            run = run_repo.get_by_id(run_id)
            
            if not run:
                raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")
            
            # Delete results directory
            results_dir = Path(f"results/{run.pipeline_config_id}/{run_id}")
            if results_dir.exists():
                shutil.rmtree(results_dir)
                return {
                    "success": True,
                    "run_id": run_id,
                    "deleted_path": str(results_dir),
                    "message": f"Deleted workspace for run {run_id}"
                }
            else:
                return {
                    "success": True,
                    "run_id": run_id,
                    "message": f"No workspace found for run {run_id}"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workspace for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete workspace: {str(e)}")


# ==============================================================================
# Server Startup Function
# ==============================================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info"
) -> None:
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload (development mode)
        log_level: Logging level
    """
    import uvicorn
    
    logger.info(f"Starting server at http://{host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")
    
    uvicorn.run(
        "src.backend.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


if __name__ == "__main__":
    run_server()
