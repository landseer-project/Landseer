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

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..common import get_logger
from ..pipeline.tasks import TaskStatus, TaskType
from ..pipeline.pipeline import Pipeline
from .scheduler import Scheduler, PriorityScheduler
from .initialization import get_backend_context, BackendContext

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
    name: str = Field(description="Tool name")
    container: ContainerInfo = Field(description="Container configuration")
    is_baseline: bool = Field(default=False, description="Whether this is a baseline tool")


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
    pipeline_id: str = Field(description="ID of the pipeline this task belongs to")
    dependency_ids: List[str] = Field(default_factory=list, description="IDs of dependent tasks")


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
    
    def add_tool(
        self,
        name: str,
        image: str,
        command: str,
        runtime: Optional[str] = None,
        is_baseline: bool = False
    ) -> None:
        """Add a custom tool to the registry."""
        self._custom_tools[name] = {
            "name": name,
            "container": {
                "image": image,
                "command": command,
                "runtime": runtime
            },
            "is_baseline": is_baseline
        }
        logger.info(f"Tool added: {name}")
    
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
                "is_baseline": tool.is_baseline
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

def task_to_response(task) -> TaskResponse:
    """Convert a Task object to a TaskResponse model."""
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
        pipeline_id=task.pipeline_id,
        dependency_ids=[dep.id for dep in task.dependencies]
    )


# ==============================================================================
# FastAPI Application & Lifespan
# ==============================================================================

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
    if context is not None:
        _scheduler_state.initialize(context.pipeline)
        logger.info(f"Scheduler auto-initialized with pipeline: {context.pipeline.name}")
    else:
        logger.warning("No backend context found. Scheduler must be initialized via API.")
    
    yield
    
    # Shutdown
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
async def get_next_task(scheduler: Scheduler = Depends(get_scheduler)):
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
        task=task_to_response(task),
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
    scheduler: Scheduler = Depends(get_scheduler)
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
    
    task_responses = [task_to_response(t) for t in tasks]
    
    return TaskListResponse(
        tasks=task_responses,
        total=len(task_responses)
    )


@app.get("/tasks/{task_id}", response_model=TaskResponse, tags=["Tasks"])
async def get_task(
    task_id: str,
    scheduler: Scheduler = Depends(get_scheduler)
):
    """Get details of a specific task by ID."""
    task = scheduler._find_task_by_id(task_id)
    
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task with ID '{task_id}' not found"
        )
    
    return task_to_response(task)


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
async def get_priority_levels(scheduler: Scheduler = Depends(get_scheduler)):
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
        level: [task_to_response(t) for t in tasks]
        for level, tasks in levels.items()
    }
    
    return PriorityLevelsResponse(levels=response_levels)


@app.get("/progress/ready", response_model=TaskListResponse, tags=["Progress"])
async def get_ready_tasks(scheduler: Scheduler = Depends(get_scheduler)):
    """Get all tasks that are ready to execute (dependencies satisfied)."""
    if isinstance(scheduler, PriorityScheduler):
        ready_tasks = scheduler.get_ready_tasks_by_priority()
    else:
        ready_tasks = [t for t in scheduler.get_all_tasks() if scheduler._is_task_ready(t)]
    
    task_responses = [task_to_response(t) for t in ready_tasks]
    
    return TaskListResponse(
        tasks=task_responses,
        total=len(task_responses)
    )


@app.get("/progress/blocked", response_model=TaskListResponse, tags=["Progress"])
async def get_blocked_tasks(scheduler: Scheduler = Depends(get_scheduler)):
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
    
    task_responses = [task_to_response(t) for t in blocked_tasks]
    
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
async def get_scheduler_next_preview(scheduler: Scheduler = Depends(get_scheduler)):
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
        "next_task": task_to_response(next_task),
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
        "task": task_to_response(task)
    }


@app.post("/workers/{worker_id}/claim", response_model=NextTaskResponse, tags=["Workers"])
async def worker_claim_task(
    worker_id: str,
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """
    Claim the next available task for a specific worker.
    
    Similar to /tasks/next but associates the task with the worker.
    """
    if worker_id not in state.workers:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    
    task = scheduler.get_next_task()
    
    if task is None:
        progress = scheduler.get_progress()
        if progress["running"] > 0:
            message = f"No tasks ready. {progress['running']} task(s) currently running."
        elif scheduler.is_complete():
            message = "All tasks completed."
        else:
            message = "No tasks available."
        
        return NextTaskResponse(has_task=False, task=None, message=message)
    
    # Associate task with worker
    state.assign_task_to_worker(worker_id, task.id)
    logger.info(f"Task {task.id} claimed by worker {worker_id}")
    
    return NextTaskResponse(
        has_task=True,
        task=task_to_response(task),
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
    tasks = [task_to_response(t) for t in workflow.tasks]
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
            is_baseline=data.get("is_baseline", False)
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
        is_baseline=request.is_baseline
    )
    
    return ToolInfo(
        name=request.name,
        container=ContainerInfo(
            image=request.image,
            command=request.command,
            runtime=request.runtime
        ),
        is_baseline=request.is_baseline
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
        if state.pipeline:
            pipeline_dataset = state.pipeline.dataset
        return DatasetInfoResponse(
            available=False,
            minio_available=False,
            config=pipeline_dataset
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
        model_script=model_script
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
    
    return {
        "available": True,
        "task_progress": state.db_service.get_task_progress(),
        "worker_stats": state.db_service.get_worker_stats()
    }


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
            container=ContainerInfo(
                image=container.get("image", ""),
                command=container.get("command", ""),
                runtime=container.get("runtime")
            ),
            is_baseline=data.get("is_baseline", False)
        ))
    
    return ToolListResponse(tools=tools, total=len(tools))


@app.post("/registry/tools", response_model=ToolInfo, tags=["Registry"])
async def registry_add_tool(
    request: AddToolRequest,
    state: SchedulerState = Depends(get_scheduler_state)
):
    """
    Add a new tool to the registry.
    
    Note: This adds to runtime registry. For persistence, update configs/tools.yaml.
    """
    state.add_tool(
        name=request.name,
        image=request.image,
        command=request.command,
        runtime=request.runtime,
        is_baseline=request.is_baseline
    )
    
    return ToolInfo(
        name=request.name,
        container=ContainerInfo(
            image=request.image,
            command=request.command,
            runtime=request.runtime
        ),
        is_baseline=request.is_baseline
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
    metrics: Dict[str, Optional[float]] = Field(description="Metric name to value mapping")
    evaluators_run: List[str] = Field(description="Evaluators that ran")
    evaluators_skipped: List[str] = Field(description="Evaluators that skipped")


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
    
    if state.db_service and state.db_service.is_available():
        # Get from database
        try:
            from ..db.models import EvaluationResultModel
            session = state.db_service.get_session()
            results = session.query(EvaluationResultModel).filter(
                EvaluationResultModel.pipeline_id == pipeline.id
            ).all()
            
            # Group by workflow
            by_workflow = {}
            for r in results:
                if r.workflow_id not in by_workflow:
                    by_workflow[r.workflow_id] = {
                        "metrics": {},
                        "run": [],
                        "skipped": []
                    }
                
                if r.skipped:
                    by_workflow[r.workflow_id]["skipped"].append(r.evaluator_name)
                else:
                    by_workflow[r.workflow_id]["run"].append(r.evaluator_name)
                    for metric_name, value in (r.metrics or {}).items():
                        by_workflow[r.workflow_id]["metrics"][metric_name] = value
                        metric_names.add(metric_name)
            
            for wf in pipeline.workflows:
                wf_data = by_workflow.get(wf.id, {"metrics": {}, "run": [], "skipped": []})
                all_metrics.append(WorkflowMetrics(
                    workflow_id=wf.id,
                    workflow_name=wf.name,
                    metrics=wf_data["metrics"],
                    evaluators_run=wf_data["run"],
                    evaluators_skipped=wf_data["skipped"]
                ))
                
        except Exception as e:
            logger.warning(f"Failed to get metrics from database: {e}")
    
    # If no database results, return empty metrics
    if not all_metrics:
        for wf in pipeline.workflows:
            all_metrics.append(WorkflowMetrics(
                workflow_id=wf.id,
                workflow_name=wf.name,
                metrics={},
                evaluators_run=[],
                evaluators_skipped=[]
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
