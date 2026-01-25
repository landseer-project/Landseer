// API Types matching backend Pydantic models

export interface HealthResponse {
  status: string;
  timestamp: string;
  scheduler_active: boolean;
}

export interface ContainerInfo {
  image: string;
  command: string;
  runtime: string | null;
}

export interface ToolInfo {
  name: string;
  container: ContainerInfo;
  is_baseline: boolean;
}

export interface TaskResponse {
  id: string;
  tool: ToolInfo;
  config: Record<string, unknown>;
  priority: number;
  status: TaskStatus;
  task_type: TaskType;
  counter: number;
  workflows: string[];
  pipeline_id: string;
  dependency_ids: string[];
}

export interface TaskListResponse {
  tasks: TaskResponse[];
  total: number;
}

export interface UpdateTaskStatusRequest {
  task_id: string;
  status: 'completed' | 'failed';
  error_message?: string;
  execution_time_ms?: number;
  result?: Record<string, unknown>;
}

export interface UpdateTaskStatusResponse {
  success: boolean;
  task_id: string;
  new_status: string;
  message: string;
}

export interface ProgressResponse {
  total: number;
  pending: number;
  running: number;
  completed: number;
  failed: number;
  progress_percent: number;
  is_complete: boolean;
}

export interface TaskPriorityInfo {
  task_id: string;
  priority: number;
  dependency_level: number;
  usage_counter: number;
  status: string;
  dependencies: string[];
  workflows: string[];
}

export interface PriorityLevelsResponse {
  levels: Record<number, TaskResponse[]>;
}

export interface PipelineInfoResponse {
  id: string;
  name: string;
  workflow_count: number;
  task_count: number;
  dataset: Record<string, unknown> | null;
  model: Record<string, unknown> | null;
}

export interface WorkflowInfo {
  id: string;
  name: string;
  pipeline_id: string;
  task_count: number;
  task_ids: string[];
}

export interface WorkflowListResponse {
  workflows: WorkflowInfo[];
  total: number;
}

export interface NextTaskResponse {
  has_task: boolean;
  task: TaskResponse | null;
  message: string;
}

export interface WorkerInfo {
  worker_id: string;
  hostname: string;
  status: WorkerStatus;
  registered_at: string;
  last_heartbeat: string;
  current_task_id: string | null;
  tasks_completed: number;
  tasks_failed: number;
  capabilities: Record<string, unknown> | null;
}

export interface WorkerListResponse {
  workers: WorkerInfo[];
  total: number;
  active: number;
}

export interface WorkflowDetailResponse {
  id: string;
  name: string;
  pipeline_id: string;
  task_count: number;
  tasks: TaskResponse[];
  status: WorkflowStatus;
  completed_tasks: number;
  failed_tasks: number;
  failure_reasons: Array<{
    task_id: string;
    task_name: string;
    error: string;
  }>;
}

export interface ToolListResponse {
  tools: ToolInfo[];
  total: number;
}

export interface AddToolRequest {
  name: string;
  image: string;
  command: string;
  runtime?: string;
  is_baseline?: boolean;
}

export interface PipelineDetailResponse {
  id: string;
  name: string;
  workflow_count: number;
  task_count: number;
  dataset: Record<string, unknown> | null;
  model: Record<string, unknown> | null;
  started_at: string | null;
  running_time_seconds: number | null;
  progress: ProgressResponse;
  estimated_remaining_seconds: number | null;
}

export interface SchedulerStatus {
  initialized: boolean;
  started_at: string | null;
  pipeline_name: string | null;
  pipeline_id: string | null;
  progress: ProgressResponse | null;
  is_complete: boolean;
  task_metadata_count: number;
  message?: string;
}

export interface SchedulerNextPreview {
  has_next: boolean;
  next_task?: TaskResponse;
  queue_depth?: number;
  message?: string;
}

export interface WorkflowResults {
  workflow_id: string;
  workflow_name: string;
  results: Array<{
    task_id: string;
    tool_name: string;
    status: string;
    execution_time_ms: number | null;
    result: Record<string, unknown> | null;
    error_message: string | null;
    worker_id: string | null;
  }>;
}

// Enums
export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed';
export type TaskType = 'pre' | 'in' | 'post' | 'deploy';
export type WorkerStatus = 'idle' | 'busy' | 'offline';
export type WorkflowStatus = 'pending' | 'running' | 'completed' | 'failed';

// Utility type for API errors
export interface ApiError {
  detail: string;
  status_code?: number;
}
