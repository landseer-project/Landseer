import axios, { AxiosInstance } from 'axios';
import type {
  HealthResponse,
  PipelineInfoResponse,
  PipelineDetailResponse,
  WorkflowListResponse,
  WorkflowDetailResponse,
  WorkflowResults,
  TaskListResponse,
  TaskResponse,
  TaskPriorityInfo,
  NextTaskResponse,
  UpdateTaskStatusRequest,
  UpdateTaskStatusResponse,
  ProgressResponse,
  PriorityLevelsResponse,
  WorkerListResponse,
  WorkerInfo,
  ToolListResponse,
  ToolInfo,
  AddToolRequest,
  SchedulerStatus,
  SchedulerNextPreview,
} from '@/types/api';

// Create axios instance with base configuration
const api: AxiosInstance = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail || error.message || 'Unknown error';
    console.error(`[API Error] ${message}`);
    return Promise.reject(error);
  }
);

// ==============================================================================
// Health & Info
// ==============================================================================

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health');
  return data;
}

export async function getPipelineInfo(): Promise<PipelineInfoResponse> {
  const { data } = await api.get<PipelineInfoResponse>('/info/pipeline');
  return data;
}

export async function getPipelineDetail(): Promise<PipelineDetailResponse> {
  const { data } = await api.get<PipelineDetailResponse>('/pipeline');
  return data;
}

// ==============================================================================
// Workflows
// ==============================================================================

export async function getWorkflows(): Promise<WorkflowListResponse> {
  const { data } = await api.get<WorkflowListResponse>('/info/workflows');
  return data;
}

export async function getWorkflowDetail(workflowId: string): Promise<WorkflowDetailResponse> {
  const { data } = await api.get<WorkflowDetailResponse>(`/workflows/${workflowId}`);
  return data;
}

export async function getWorkflowResults(workflowId: string): Promise<WorkflowResults> {
  const { data } = await api.get<WorkflowResults>(`/workflows/${workflowId}/results`);
  return data;
}

// ==============================================================================
// Tasks
// ==============================================================================

export async function getAllTasks(status?: string): Promise<TaskListResponse> {
  const params = status ? { status } : {};
  const { data } = await api.get<TaskListResponse>('/tasks', { params });
  return data;
}

export async function getTask(taskId: string): Promise<TaskResponse> {
  const { data } = await api.get<TaskResponse>(`/tasks/${taskId}`);
  return data;
}

export async function getTaskPriority(taskId: string): Promise<TaskPriorityInfo> {
  const { data } = await api.get<TaskPriorityInfo>(`/tasks/${taskId}/priority`);
  return data;
}

export async function getNextTask(): Promise<NextTaskResponse> {
  const { data } = await api.get<NextTaskResponse>('/tasks/next');
  return data;
}

export async function updateTaskStatus(request: UpdateTaskStatusRequest): Promise<UpdateTaskStatusResponse> {
  const { data } = await api.put<UpdateTaskStatusResponse>('/tasks/status', request);
  return data;
}

// ==============================================================================
// Progress
// ==============================================================================

export async function getProgress(): Promise<ProgressResponse> {
  const { data } = await api.get<ProgressResponse>('/progress');
  return data;
}

export async function getPriorityLevels(): Promise<PriorityLevelsResponse> {
  const { data } = await api.get<PriorityLevelsResponse>('/progress/levels');
  return data;
}

export async function getReadyTasks(): Promise<TaskListResponse> {
  const { data } = await api.get<TaskListResponse>('/progress/ready');
  return data;
}

export async function getBlockedTasks(): Promise<TaskListResponse> {
  const { data } = await api.get<TaskListResponse>('/progress/blocked');
  return data;
}

// ==============================================================================
// Scheduler
// ==============================================================================

export async function getSchedulerStatus(): Promise<SchedulerStatus> {
  const { data } = await api.get<SchedulerStatus>('/scheduler/status');
  return data;
}

export async function getSchedulerNextPreview(): Promise<SchedulerNextPreview> {
  const { data } = await api.get<SchedulerNextPreview>('/scheduler/next');
  return data;
}

export async function initializeScheduler(schedulerType: string = 'priority'): Promise<{ success: boolean; message: string }> {
  const { data } = await api.post('/scheduler/initialize', null, { params: { scheduler_type: schedulerType } });
  return data;
}

export async function resetScheduler(): Promise<{ success: boolean; message: string }> {
  const { data } = await api.post('/scheduler/reset');
  return data;
}

// ==============================================================================
// Workers
// ==============================================================================

export async function getWorkers(): Promise<WorkerListResponse> {
  const { data } = await api.get<WorkerListResponse>('/workers');
  return data;
}

export async function getWorker(workerId: string): Promise<WorkerInfo> {
  const { data } = await api.get<WorkerInfo>(`/workers/${workerId}`);
  return data;
}

export async function registerWorker(hostname: string, workerId?: string, capabilities?: Record<string, unknown>): Promise<WorkerInfo> {
  const { data } = await api.post<WorkerInfo>('/workers/register', {
    hostname,
    worker_id: workerId,
    capabilities,
  });
  return data;
}

export async function workerHeartbeat(workerId: string, status?: string): Promise<{ success: boolean }> {
  const { data } = await api.post(`/workers/${workerId}/heartbeat`, { worker_id: workerId, status });
  return data;
}

export async function workerClaimTask(workerId: string): Promise<NextTaskResponse> {
  const { data } = await api.post<NextTaskResponse>(`/workers/${workerId}/claim`);
  return data;
}

// ==============================================================================
// Tools
// ==============================================================================

export async function getTools(): Promise<ToolListResponse> {
  const { data } = await api.get<ToolListResponse>('/tools');
  return data;
}

export async function getTool(toolName: string): Promise<ToolInfo> {
  const { data } = await api.get<ToolInfo>(`/tools/${toolName}`);
  return data;
}

export async function addTool(request: AddToolRequest): Promise<ToolInfo> {
  const { data } = await api.post<ToolInfo>('/tools', request);
  return data;
}

// ==============================================================================
// Registry (Tools & Evaluators)
// ==============================================================================

export interface EvaluatorInfo {
  name: string;
  container: {
    image: string;
    command: string;
    runtime?: string | null;
  };
  required_artifacts: string[];
  metrics: string[];
  defense_types: string[];
}

export interface EvaluatorListResponse {
  evaluators: EvaluatorInfo[];
  total: number;
}

export async function getRegistryTools(): Promise<ToolListResponse> {
  const { data } = await api.get<ToolListResponse>('/registry/tools');
  return data;
}

export async function getRegistryEvaluators(): Promise<EvaluatorListResponse> {
  const { data } = await api.get<EvaluatorListResponse>('/registry/evaluators');
  return data;
}

export async function addRegistryTool(request: AddToolRequest): Promise<ToolInfo> {
  const { data } = await api.post<ToolInfo>('/registry/tools', request);
  return data;
}

export interface AddEvaluatorRequest {
  name: string;
  image: string;
  command: string;
  runtime?: string;
  required_artifacts?: string[];
  metrics?: string[];
  defense_types?: string[];
}

export async function addRegistryEvaluator(request: AddEvaluatorRequest): Promise<EvaluatorInfo> {
  const { data } = await api.post<EvaluatorInfo>('/registry/evaluators', request);
  return data;
}

// ==============================================================================
// Metrics
// ==============================================================================

export interface WorkflowMetrics {
  workflow_id: string;
  workflow_name: string;
  metrics: Record<string, number | null>;
  evaluators_run: string[];
  evaluators_skipped: string[];
}

export interface PipelineMetricsResponse {
  pipeline_id: string;
  pipeline_name: string;
  workflow_count: number;
  metric_names: string[];
  workflows: WorkflowMetrics[];
  summary: Record<string, { min: number | null; max: number | null; avg: number | null; count: number }>;
}

export async function getPipelineMetrics(pipelineId: string): Promise<PipelineMetricsResponse> {
  const { data } = await api.get<PipelineMetricsResponse>(`/pipelines/${pipelineId}/metrics`);
  return data;
}

export async function getWorkflowMetrics(workflowId: string): Promise<{
  workflow_id: string;
  workflow_name: string;
  pipeline_id: string;
  metrics: Record<string, number | null>;
  evaluators_run: string[];
  evaluators_skipped: { evaluator: string; reason: string }[];
}> {
  const { data } = await api.get(`/workflows/${workflowId}/metrics`);
  return data;
}

export default api;
