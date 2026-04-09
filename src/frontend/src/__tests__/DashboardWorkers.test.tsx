import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { Dashboard } from '../pages/Dashboard';

vi.mock('../lib/api', () => ({
  getPipelineDetail: vi.fn(),
  getProgress: vi.fn(),
  getWorkers: vi.fn(),
  getAllTasks: vi.fn(),
  getRunningTasks: vi.fn(),
  getSchedulerStatus: vi.fn(),
  getReadyTasks: vi.fn(),
  resetScheduler: vi.fn(),
  reclaimStaleTasks: vi.fn(),
}));

import * as api from '../lib/api';

function renderDashboard() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Dashboard />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('Dashboard worker stats/status display', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(api.getPipelineDetail).mockResolvedValue({
      id: 'pipeline-1',
      name: 'Test Pipeline',
      workflow_count: 1,
      task_count: 2,
      dataset: null,
      model: null,
      started_at: '2026-04-09T10:00:00Z',
      running_time_seconds: 120,
      progress: {
        total: 2,
        pending: 0,
        running: 1,
        completed: 1,
        failed: 0,
        progress_percent: 50,
        is_complete: false,
      },
      estimated_remaining_seconds: 120,
    });

    vi.mocked(api.getProgress).mockResolvedValue({
      total: 2,
      pending: 0,
      running: 1,
      completed: 1,
      failed: 0,
      progress_percent: 50,
      is_complete: false,
    });

    vi.mocked(api.getSchedulerStatus).mockResolvedValue({
      initialized: true,
      started_at: '2026-04-09T10:00:00Z',
      pipeline_name: 'Test Pipeline',
      pipeline_id: 'pipeline-1',
      progress: {
        total: 2,
        pending: 0,
        running: 1,
        completed: 1,
        failed: 0,
        progress_percent: 50,
        is_complete: false,
      },
      is_complete: false,
      task_metadata_count: 1,
    });

    vi.mocked(api.getAllTasks).mockResolvedValue({ tasks: [], total: 0 });
    vi.mocked(api.getRunningTasks).mockResolvedValue({
      tasks: [
        {
          id: 'task_1',
          tool: {
            name: 'TRADES',
            container: { image: 'img', command: 'run', runtime: null },
            is_baseline: false,
          },
          config: {},
          priority: 100,
          status: 'running',
          task_type: 'in_training',
          counter: 1,
          workflows: ['wf_1'],
          workflow_names: ['wf_1'],
          pipeline_id: 'pipeline-1',
          dependency_ids: [],
        },
      ],
      total: 1,
    });
    vi.mocked(api.getReadyTasks).mockResolvedValue({ tasks: [], total: 0 });
    vi.mocked(api.resetScheduler).mockResolvedValue({ success: true, message: 'ok' });
    vi.mocked(api.reclaimStaleTasks).mockResolvedValue({
      success: true,
      reclaimed_count: 0,
      reclaimed_task_ids: [],
    });
  });

  it('shows worker done counters from tasks_completed', async () => {
    vi.mocked(api.getWorkers).mockResolvedValue({
      workers: [
        {
          worker_id: 'worker_alpha',
          hostname: 'gpu-a',
          status: 'busy',
          registered_at: '2026-04-09T10:00:00Z',
          last_heartbeat: '2026-04-09T10:00:10Z',
          current_task_id: 'task_1',
          tasks_completed: 7,
          tasks_failed: 1,
          capabilities: { gpu_id: 2 },
        },
      ],
      total: 1,
      active: 1,
    });

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Workers')).toBeInTheDocument();
      expect(screen.getByText('7')).toBeInTheDocument();
      expect(screen.getByText('done')).toBeInTheDocument();
    });
  });

  it('shows busy worker running task and gpu badge', async () => {
    vi.mocked(api.getWorkers).mockResolvedValue({
      workers: [
        {
          worker_id: 'worker_alpha',
          hostname: 'gpu-a',
          status: 'busy',
          registered_at: '2026-04-09T10:00:00Z',
          last_heartbeat: '2026-04-09T10:00:10Z',
          current_task_id: 'task_1',
          tasks_completed: 0,
          tasks_failed: 0,
          capabilities: { gpu_id: 3 },
        },
      ],
      total: 1,
      active: 1,
    });

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/Running:/)).toBeInTheDocument();
      expect(screen.getByText('GPU 3')).toBeInTheDocument();
    });
  });
});

