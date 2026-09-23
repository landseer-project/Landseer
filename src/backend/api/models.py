"""Pydantic request/response models for the scheduler API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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
    model_script: Optional[str] = Field(default=None, description="Path to model config script")
    model_script_minio_key: Optional[str] = Field(default=None, description="MinIO object key for model config script")


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
        description="Non-evaluation tool names grouped by stage",
    )
    workflow_tools_label: str = Field(
        default="",
        description="Human-readable non-evaluation tool sequence",
    )
    metrics: Dict[str, Optional[float]] = Field(description="Metric name to value mapping")
    evaluators_run: List[str] = Field(description="Evaluators that ran")
    evaluators_skipped: List[str] = Field(description="Evaluators that skipped")
    is_baseline: bool = Field(default=False, description="Whether this workflow uses only baseline tools")


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
        description="Override tool lists per stage. Keys are stage names; values are ordered tool name lists.",
    )
    model_script: Optional[str] = Field(
        default=None,
        description="Override model script path (e.g. configs/model/config_model_resnet.py)",
    )


class PipelineRunResponse(BaseModel):
    """Response model for a pipeline run."""
    id: str = Field(description="Run ID")
    pipeline_config_id: str = Field(description="Config ID")
    run_number: int = Field(description="Run number for this config")
    use_cache: bool = Field(description="Whether cache was used")
    tools_config: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Locked tool configuration snapshot (stage → tool names).",
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
