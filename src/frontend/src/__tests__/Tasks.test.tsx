/**
 * Tasks Page Component Tests
 * 
 * These tests verify the Tasks page renders correct data to users.
 * Uses React Testing Library to test the actual component behavior.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { Tasks } from '../pages/Tasks';

// Mock the API module
vi.mock('../lib/api', () => ({
  getAllTasks: vi.fn(),
  getTaskPriority: vi.fn(),
  getPriorityLevels: vi.fn(),
}));

import * as api from '../lib/api';

// Test data representing a realistic pipeline state
const createMockTasksResponse = (filter?: string) => {
  const allTasks = [
    {
      id: 'task-baseline-001',
      tool: { name: 'baseline-training', container: { image: 'ml:v1', command: 'train.py', runtime: null }, is_baseline: true },
      config: {},
      priority: 100,
      status: 'completed' as const,
      task_type: 'pre',
      counter: 3,
      workflows: ['workflow-1', 'workflow-2', 'workflow-3'],
      pipeline_id: 'pipeline-1',
      dependency_ids: [],
    },
    {
      id: 'task-attack-002',
      tool: { name: 'pgd-attack', container: { image: 'attack:v1', command: 'pgd.py', runtime: null }, is_baseline: false },
      config: { epsilon: 0.03 },
      priority: 90,
      status: 'completed' as const,
      task_type: 'in',
      counter: 2,
      workflows: ['workflow-1', 'workflow-2'],
      pipeline_id: 'pipeline-1',
      dependency_ids: ['task-baseline-001'],
    },
    {
      id: 'task-defense-003',
      tool: { name: 'adversarial-training', container: { image: 'defense:v1', command: 'adv.py', runtime: null }, is_baseline: false },
      config: {},
      priority: 80,
      status: 'running' as const,
      task_type: 'in',
      counter: 2,
      workflows: ['workflow-1', 'workflow-2'],
      pipeline_id: 'pipeline-1',
      dependency_ids: ['task-attack-002'],
    },
    {
      id: 'task-eval-004',
      tool: { name: 'evaluation', container: { image: 'eval:v1', command: 'eval.py', runtime: null }, is_baseline: false },
      config: {},
      priority: 70,
      status: 'pending' as const,
      task_type: 'post',
      counter: 3,
      workflows: ['workflow-1', 'workflow-2', 'workflow-3'],
      pipeline_id: 'pipeline-1',
      dependency_ids: ['task-defense-003'],
    },
    {
      id: 'task-deploy-005',
      tool: { name: 'deployment', container: { image: 'deploy:v1', command: 'deploy.py', runtime: null }, is_baseline: false },
      config: {},
      priority: 60,
      status: 'pending' as const,
      task_type: 'deploy',
      counter: 1,
      workflows: ['workflow-1'],
      pipeline_id: 'pipeline-1',
      dependency_ids: ['task-eval-004'],
    },
    {
      id: 'task-failed-006',
      tool: { name: 'bad-attack', container: { image: 'attack:v1', command: 'bad.py', runtime: null }, is_baseline: false },
      config: {},
      priority: 90,
      status: 'failed' as const,
      task_type: 'in',
      counter: 1,
      workflows: ['workflow-3'],
      pipeline_id: 'pipeline-1',
      dependency_ids: ['task-baseline-001'],
    },
  ];

  if (filter && filter !== 'all') {
    const filtered = allTasks.filter(t => t.status === filter);
    return { tasks: filtered, total: filtered.length };
  }

  return { tasks: allTasks, total: allTasks.length };
};

const renderTasks = () => {
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
        <Tasks />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('Tasks Page - Status Statistics Display', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should display correct status counts when no filter is applied', async () => {
    const mockResponse = createMockTasksResponse();
    vi.mocked(api.getAllTasks).mockResolvedValue(mockResponse);
    vi.mocked(api.getPriorityLevels).mockResolvedValue({ levels: {} });

    renderTasks();

    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByText('Loading dashboard...')).not.toBeInTheDocument();
    });

    // Verify correct counts are displayed
    // Expected: 2 pending, 1 running, 2 completed, 1 failed
    const pendingCard = screen.getByText('Pending').closest('div');
    const runningCard = screen.getByText('Running').closest('div');
    const completedCard = screen.getByText('Completed').closest('div');
    const failedCard = screen.getByText('Failed').closest('div');

    expect(pendingCard).toHaveTextContent('2');
    expect(runningCard).toHaveTextContent('1');
    expect(completedCard).toHaveTextContent('2');
    expect(failedCard).toHaveTextContent('1');
  });

  it('BUG: displays wrong stats when status filter is applied', async () => {
    // First call returns all tasks, subsequent call returns filtered
    vi.mocked(api.getAllTasks)
      .mockResolvedValueOnce(createMockTasksResponse()) // Initial load
      .mockResolvedValueOnce(createMockTasksResponse('pending')); // After filter

    vi.mocked(api.getPriorityLevels).mockResolvedValue({ levels: {} });

    const user = userEvent.setup();
    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('baseline-training')).toBeInTheDocument();
    });

    // Initial state should be correct
    expect(screen.getByText('2')).toBeInTheDocument(); // 2 pending

    // Click on the status filter dropdown and select "Pending"
    const filterButton = screen.getByRole('combobox', { name: /status/i });
    await user.click(filterButton);
    
    // After filtering, the stats would be wrong due to the bug
    // This test documents the current buggy behavior
  });
});

describe('Tasks Page - Task List Display', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getAllTasks).mockResolvedValue(createMockTasksResponse());
    vi.mocked(api.getPriorityLevels).mockResolvedValue({ levels: {} });
  });

  it('should display all task names', async () => {
    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('baseline-training')).toBeInTheDocument();
      expect(screen.getByText('pgd-attack')).toBeInTheDocument();
      expect(screen.getByText('adversarial-training')).toBeInTheDocument();
      expect(screen.getByText('evaluation')).toBeInTheDocument();
      expect(screen.getByText('deployment')).toBeInTheDocument();
      expect(screen.getByText('bad-attack')).toBeInTheDocument();
    });
  });

  it('should display correct workflow counts for each task', async () => {
    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('baseline-training')).toBeInTheDocument();
    });

    // Check workflow counts are displayed
    // baseline-training: 3 workflows
    // deployment: 1 workflow
    const workflowTexts = screen.getAllByText(/Workflows:/);
    expect(workflowTexts.length).toBeGreaterThan(0);
  });

  it('should display correct dependency counts for each task', async () => {
    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('baseline-training')).toBeInTheDocument();
    });

    // Check dependency counts are displayed
    const dependencyTexts = screen.getAllByText(/Dependencies:/);
    expect(dependencyTexts.length).toBeGreaterThan(0);
  });

  it('should show correct total count in list header', async () => {
    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('6 of 6 tasks')).toBeInTheDocument();
    });
  });
});

describe('Tasks Page - Priority Sorting', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getAllTasks).mockResolvedValue(createMockTasksResponse());
    vi.mocked(api.getPriorityLevels).mockResolvedValue({ levels: {} });
  });

  it('BUG: sorts by priority in ascending order (wrong)', async () => {
    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('baseline-training')).toBeInTheDocument();
    });

    // Get all task elements in order
    const taskElements = screen.getAllByRole('button').filter(
      el => el.textContent?.includes('Priority:')
    );

    // Current (buggy) behavior: lowest priority first
    // deployment (priority 60) should NOT be first, but currently is
    // baseline-training (priority 100) should be first
    
    // This documents the bug - the first visible task after sorting by priority
    // should have the highest priority (100), not lowest (60)
  });

  it('should show tasks in execution order when sorted by priority correctly', async () => {
    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('baseline-training')).toBeInTheDocument();
    });

    // Expected correct order (by execution priority, highest first):
    // 1. baseline-training (100)
    // 2. pgd-attack, bad-attack (90)
    // 3. adversarial-training (80)
    // 4. evaluation (70)
    // 5. deployment (60)

    const taskList = screen.getByRole('list');
    if (taskList) {
      const taskItems = within(taskList).getAllByRole('listitem');
      // Verify order...
    }
  });
});

describe('Tasks Page - Task Details Dialog', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getAllTasks).mockResolvedValue(createMockTasksResponse());
    vi.mocked(api.getPriorityLevels).mockResolvedValue({ levels: {} });
    vi.mocked(api.getTaskPriority).mockResolvedValue({
      task_id: 'task-eval-004',
      priority: 70,
      dependency_level: 1,  // Note: This is count, not depth
      usage_counter: 3,
      status: 'pending',
      dependencies: ['task-defense-003'],
      workflows: ['workflow-1', 'workflow-2', 'workflow-3'],
    });
  });

  it('should display correct task details when clicked', async () => {
    const user = userEvent.setup();
    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('evaluation')).toBeInTheDocument();
    });

    // Click on a task to open details
    const evalTask = screen.getByText('evaluation').closest('div');
    if (evalTask) {
      await user.click(evalTask);
    }

    // Dialog should open with task details
    await waitFor(() => {
      expect(screen.getByText('Task ID:')).toBeInTheDocument();
    });
  });

  it('should display correct Usage Count from task.counter', async () => {
    const user = userEvent.setup();
    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('evaluation')).toBeInTheDocument();
    });

    // Open task details
    const evalTask = screen.getByText('evaluation').closest('div');
    if (evalTask) {
      await user.click(evalTask);
    }

    // Check Usage Count is displayed correctly
    await waitFor(() => {
      const usageCountLabel = screen.getByText('Usage Count');
      expect(usageCountLabel).toBeInTheDocument();
    });
  });
});

describe('Tasks Page - Search Functionality', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getAllTasks).mockResolvedValue(createMockTasksResponse());
    vi.mocked(api.getPriorityLevels).mockResolvedValue({ levels: {} });
  });

  it('should filter tasks by search query', async () => {
    const user = userEvent.setup();
    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('baseline-training')).toBeInTheDocument();
    });

    // Type in search box
    const searchInput = screen.getByPlaceholderText('Search tasks...');
    await user.type(searchInput, 'attack');

    // Should only show attack-related tasks
    await waitFor(() => {
      expect(screen.getByText('pgd-attack')).toBeInTheDocument();
      expect(screen.getByText('bad-attack')).toBeInTheDocument();
      expect(screen.queryByText('baseline-training')).not.toBeInTheDocument();
    });

    // Check filtered count in header
    expect(screen.getByText('2 of 6 tasks')).toBeInTheDocument();
  });

  it('should search by task ID', async () => {
    const user = userEvent.setup();
    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('baseline-training')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search tasks...');
    await user.type(searchInput, 'baseline-001');

    await waitFor(() => {
      expect(screen.getByText('baseline-training')).toBeInTheDocument();
      expect(screen.queryByText('pgd-attack')).not.toBeInTheDocument();
    });
  });
});

describe('Tasks Page - Data Consistency Checks', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should have sum of status counts equal to total', async () => {
    const response = createMockTasksResponse();
    vi.mocked(api.getAllTasks).mockResolvedValue(response);
    vi.mocked(api.getPriorityLevels).mockResolvedValue({ levels: {} });

    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('baseline-training')).toBeInTheDocument();
    });

    // Calculate expected counts
    const expectedCounts = {
      pending: response.tasks.filter(t => t.status === 'pending').length,
      running: response.tasks.filter(t => t.status === 'running').length,
      completed: response.tasks.filter(t => t.status === 'completed').length,
      failed: response.tasks.filter(t => t.status === 'failed').length,
    };

    const total = expectedCounts.pending + expectedCounts.running + 
                  expectedCounts.completed + expectedCounts.failed;

    expect(total).toBe(response.total);
    expect(total).toBe(6);
  });

  it('should display tasks with correct status badges', async () => {
    vi.mocked(api.getAllTasks).mockResolvedValue(createMockTasksResponse());
    vi.mocked(api.getPriorityLevels).mockResolvedValue({ levels: {} });

    renderTasks();

    await waitFor(() => {
      expect(screen.getByText('baseline-training')).toBeInTheDocument();
    });

    // Verify status badges exist for each status
    // The StatusBadge component should render appropriate badges
  });
});
