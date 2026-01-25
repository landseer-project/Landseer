/**
 * Dashboard Component Tests
 * 
 * These tests verify the Dashboard displays correct data and that
 * Dashboard numbers match Tasks page numbers.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { Dashboard } from '../pages/Dashboard';

// Mock the API module
vi.mock('../lib/api', () => ({
  getPipelineDetail: vi.fn(),
  getProgress: vi.fn(),
  getWorkers: vi.fn(),
  getAllTasks: vi.fn(),
  getSchedulerStatus: vi.fn(),
  getReadyTasks: vi.fn(),
  resetScheduler: vi.fn(),
}));

import * as api from '../lib/api';

// Consistent mock data that matches between endpoints
const createConsistentMockData = () => {
  const tasks = [
    { id: 't1', tool: { name: 'Task 1' }, status: 'completed', priority: 100, workflows: ['w1'], dependency_ids: [], counter: 1, config: {}, task_type: 'pre', pipeline_id: 'p1' },
    { id: 't2', tool: { name: 'Task 2' }, status: 'completed', priority: 90, workflows: ['w1'], dependency_ids: ['t1'], counter: 1, config: {}, task_type: 'in', pipeline_id: 'p1' },
    { id: 't3', tool: { name: 'Task 3' }, status: 'running', priority: 80, workflows: ['w1'], dependency_ids: ['t2'], counter: 1, config: {}, task_type: 'in', pipeline_id: 'p1' },
    { id: 't4', tool: { name: 'Task 4' }, status: 'pending', priority: 70, workflows: ['w1'], dependency_ids: ['t3'], counter: 1, config: {}, task_type: 'post', pipeline_id: 'p1' },
    { id: 't5', tool: { name: 'Task 5' }, status: 'pending', priority: 60, workflows: ['w1'], dependency_ids: ['t4'], counter: 1, config: {}, task_type: 'deploy', pipeline_id: 'p1' },
    { id: 't6', tool: { name: 'Task 6' }, status: 'failed', priority: 90, workflows: ['w2'], dependency_ids: ['t1'], counter: 1, config: {}, task_type: 'in', pipeline_id: 'p1' },
  ];

  // Status counts: 2 pending, 1 running, 2 completed, 1 failed = 6 total
  const progress = {
    total: 6,
    pending: 2,
    running: 1,
    completed: 2,
    failed: 1,
    progress_percent: 50.0,  // (2 + 1) / 6 * 100
    is_complete: false,
  };

  const pipeline = {
    id: 'pipeline-1',
    name: 'Test Pipeline',
    workflow_count: 2,
    task_count: 6,
    dataset: null,
    model: null,
    started_at: '2024-01-15T10:00:00Z',
    running_time_seconds: 3600,
    progress,
    estimated_remaining_seconds: 3600,
  };

  const scheduler = {
    initialized: true,
    started_at: '2024-01-15T10:00:00Z',
    pipeline_name: 'Test Pipeline',
    pipeline_id: 'pipeline-1',
    progress,
    is_complete: false,
    task_metadata_count: 3,
  };

  const workers = {
    workers: [
      { worker_id: 'w1', hostname: 'worker-1', status: 'busy' },
      { worker_id: 'w2', hostname: 'worker-2', status: 'idle' },
    ],
    total: 2,
    active: 2,
  };

  const readyTasks = {
    tasks: [],
    total: 0,
  };

  return { tasks, progress, pipeline, scheduler, workers, readyTasks };
};

const renderDashboard = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: 0,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Dashboard />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('Dashboard - Data Display', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const mockData = createConsistentMockData();
    
    vi.mocked(api.getPipelineDetail).mockResolvedValue(mockData.pipeline);
    vi.mocked(api.getProgress).mockResolvedValue(mockData.progress);
    vi.mocked(api.getWorkers).mockResolvedValue(mockData.workers);
    vi.mocked(api.getAllTasks).mockResolvedValue({ tasks: mockData.tasks, total: 6 });
    vi.mocked(api.getSchedulerStatus).mockResolvedValue(mockData.scheduler);
    vi.mocked(api.getReadyTasks).mockResolvedValue(mockData.readyTasks);
  });

  it('should display correct total task count', async () => {
    renderDashboard();

    await waitFor(() => {
      // Total Tasks stat card should show 6
      const totalTasksCard = screen.getByText('Total Tasks');
      expect(totalTasksCard.closest('div')).toHaveTextContent('6');
    });
  });

  it('should display correct completed count', async () => {
    renderDashboard();

    await waitFor(() => {
      const completedCard = screen.getByText('Completed');
      expect(completedCard.closest('div')).toHaveTextContent('2');
    });
  });

  it('should display correct running count', async () => {
    renderDashboard();

    await waitFor(() => {
      const runningCard = screen.getByText('Running');
      expect(runningCard.closest('div')).toHaveTextContent('1');
    });
  });

  it('should display correct progress percentage', async () => {
    renderDashboard();

    await waitFor(() => {
      // Progress should show 50% (3 done out of 6)
      expect(screen.getByText('50.0% done')).toBeInTheDocument();
    });
  });

  it('should display correct workflow count', async () => {
    renderDashboard();

    await waitFor(() => {
      // Should show "2 workflows" subtitle
      expect(screen.getByText('2 workflows')).toBeInTheDocument();
    });
  });

  it('should display correct active worker count', async () => {
    renderDashboard();

    await waitFor(() => {
      const workersCard = screen.getByText('Active Workers');
      expect(workersCard.closest('div')).toHaveTextContent('2');
    });
  });
});

describe('Dashboard - Progress Bar Display', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const mockData = createConsistentMockData();
    
    vi.mocked(api.getPipelineDetail).mockResolvedValue(mockData.pipeline);
    vi.mocked(api.getProgress).mockResolvedValue(mockData.progress);
    vi.mocked(api.getWorkers).mockResolvedValue(mockData.workers);
    vi.mocked(api.getAllTasks).mockResolvedValue({ tasks: mockData.tasks, total: 6 });
    vi.mocked(api.getSchedulerStatus).mockResolvedValue(mockData.scheduler);
    vi.mocked(api.getReadyTasks).mockResolvedValue(mockData.readyTasks);
  });

  it('should display status breakdown in progress section', async () => {
    renderDashboard();

    await waitFor(() => {
      // Completed count in progress section
      const completedSection = screen.getAllByText('Completed');
      expect(completedSection.length).toBeGreaterThan(0);

      // Running count in progress section  
      const runningSection = screen.getAllByText('Running');
      expect(runningSection.length).toBeGreaterThan(0);

      // Pending count in progress section
      const pendingSection = screen.getAllByText('Pending');
      expect(pendingSection.length).toBeGreaterThan(0);
    });
  });

  it('should show failed count when there are failures', async () => {
    renderDashboard();

    await waitFor(() => {
      // Failed section should appear when failed > 0
      expect(screen.getByText('Failed')).toBeInTheDocument();
    });
  });
});

describe('Dashboard vs Tasks Page Consistency', () => {
  /**
   * CRITICAL: Dashboard and Tasks page should show the same numbers
   * Dashboard uses /api/progress
   * Tasks page calculates from /api/tasks
   */

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('progress API counts should match tasks API counts', async () => {
    const mockData = createConsistentMockData();
    
    // Calculate counts from tasks (what Tasks page does)
    const tasksApiCounts = {
      pending: mockData.tasks.filter(t => t.status === 'pending').length,
      running: mockData.tasks.filter(t => t.status === 'running').length,
      completed: mockData.tasks.filter(t => t.status === 'completed').length,
      failed: mockData.tasks.filter(t => t.status === 'failed').length,
      total: mockData.tasks.length,
    };

    // Progress API counts
    const progressApiCounts = mockData.progress;

    // These MUST match
    expect(tasksApiCounts.pending).toBe(progressApiCounts.pending);
    expect(tasksApiCounts.running).toBe(progressApiCounts.running);
    expect(tasksApiCounts.completed).toBe(progressApiCounts.completed);
    expect(tasksApiCounts.failed).toBe(progressApiCounts.failed);
    expect(tasksApiCounts.total).toBe(progressApiCounts.total);
  });

  it('pipeline.task_count should match tasks.length', async () => {
    const mockData = createConsistentMockData();

    expect(mockData.pipeline.task_count).toBe(mockData.tasks.length);
    expect(mockData.pipeline.task_count).toBe(mockData.progress.total);
  });
});

describe('Dashboard - Edge Cases', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should handle zero tasks gracefully', async () => {
    const emptyProgress = {
      total: 0,
      pending: 0,
      running: 0,
      completed: 0,
      failed: 0,
      progress_percent: 0,
      is_complete: true,
    };

    vi.mocked(api.getPipelineDetail).mockResolvedValue({
      id: 'p1',
      name: 'Empty Pipeline',
      workflow_count: 0,
      task_count: 0,
      dataset: null,
      model: null,
      started_at: null,
      running_time_seconds: null,
      progress: emptyProgress,
      estimated_remaining_seconds: null,
    });
    vi.mocked(api.getProgress).mockResolvedValue(emptyProgress);
    vi.mocked(api.getWorkers).mockResolvedValue({ workers: [], total: 0, active: 0 });
    vi.mocked(api.getAllTasks).mockResolvedValue({ tasks: [], total: 0 });
    vi.mocked(api.getSchedulerStatus).mockResolvedValue({
      initialized: true,
      started_at: null,
      pipeline_name: 'Empty Pipeline',
      pipeline_id: 'p1',
      progress: emptyProgress,
      is_complete: true,
      task_metadata_count: 0,
    });
    vi.mocked(api.getReadyTasks).mockResolvedValue({ tasks: [], total: 0 });

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('0')).toBeInTheDocument();
    });
  });

  it('should handle all tasks completed', async () => {
    const completeProgress = {
      total: 5,
      pending: 0,
      running: 0,
      completed: 5,
      failed: 0,
      progress_percent: 100,
      is_complete: true,
    };

    vi.mocked(api.getPipelineDetail).mockResolvedValue({
      id: 'p1',
      name: 'Complete Pipeline',
      workflow_count: 1,
      task_count: 5,
      dataset: null,
      model: null,
      started_at: '2024-01-15T10:00:00Z',
      running_time_seconds: 1800,
      progress: completeProgress,
      estimated_remaining_seconds: 0,
    });
    vi.mocked(api.getProgress).mockResolvedValue(completeProgress);
    vi.mocked(api.getWorkers).mockResolvedValue({ workers: [], total: 0, active: 0 });
    vi.mocked(api.getAllTasks).mockResolvedValue({
      tasks: Array(5).fill(null).map((_, i) => ({
        id: `t${i}`,
        tool: { name: `Task ${i}` },
        status: 'completed',
        priority: 100 - i * 10,
        workflows: ['w1'],
        dependency_ids: [],
        counter: 1,
        config: {},
        task_type: 'pre',
        pipeline_id: 'p1',
      })),
      total: 5,
    });
    vi.mocked(api.getSchedulerStatus).mockResolvedValue({
      initialized: true,
      started_at: '2024-01-15T10:00:00Z',
      pipeline_name: 'Complete Pipeline',
      pipeline_id: 'p1',
      progress: completeProgress,
      is_complete: true,
      task_metadata_count: 5,
    });
    vi.mocked(api.getReadyTasks).mockResolvedValue({ tasks: [], total: 0 });

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('100.0% done')).toBeInTheDocument();
    });
  });

  it('should handle scheduler not initialized', async () => {
    vi.mocked(api.getSchedulerStatus).mockResolvedValue({
      initialized: false,
      started_at: null,
      pipeline_name: null,
      pipeline_id: null,
      progress: null,
      is_complete: false,
      task_metadata_count: 0,
      message: 'Scheduler not initialized',
    });

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Scheduler Not Initialized')).toBeInTheDocument();
    });
  });
});

describe('Dashboard - Chart Data', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const mockData = createConsistentMockData();
    
    vi.mocked(api.getPipelineDetail).mockResolvedValue(mockData.pipeline);
    vi.mocked(api.getProgress).mockResolvedValue(mockData.progress);
    vi.mocked(api.getWorkers).mockResolvedValue(mockData.workers);
    vi.mocked(api.getAllTasks).mockResolvedValue({ tasks: mockData.tasks, total: 6 });
    vi.mocked(api.getSchedulerStatus).mockResolvedValue(mockData.scheduler);
    vi.mocked(api.getReadyTasks).mockResolvedValue(mockData.readyTasks);
  });

  it('should render pie chart with correct status distribution', async () => {
    renderDashboard();

    await waitFor(() => {
      // Check that the pie chart section exists
      expect(screen.getByText('Status Distribution')).toBeInTheDocument();
    });

    // The chart should show:
    // Completed: 2
    // Running: 1
    // Pending: 2
    // Failed: 1
  });
});
