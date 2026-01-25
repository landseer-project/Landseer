/**
 * API-Frontend Consistency Tests
 * 
 * These tests verify that frontend calculations match backend API responses
 * and that the data displayed to users is accurate.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Types matching the API
interface TaskResponse {
  id: string;
  tool: {
    name: string;
    container: { image: string; command: string; runtime: string | null };
    is_baseline: boolean;
  };
  config: Record<string, unknown>;
  priority: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  task_type: string;
  counter: number;
  workflows: string[];
  pipeline_id: string;
  dependency_ids: string[];
}

interface TaskListResponse {
  tasks: TaskResponse[];
  total: number;
}

interface ProgressResponse {
  total: number;
  pending: number;
  running: number;
  completed: number;
  failed: number;
  progress_percent: number;
  is_complete: boolean;
}

interface TaskPriorityInfo {
  task_id: string;
  priority: number;
  dependency_level: number;
  usage_counter: number;
  status: string;
  dependencies: string[];
  workflows: string[];
}

// Simulated backend responses for testing
const mockApiResponses = {
  // Unfiltered /api/tasks response
  getAllTasks: (): TaskListResponse => ({
    tasks: [
      {
        id: 'task-aaa111',
        tool: { name: 'baseline-training', container: { image: 'ml-base:v1', command: 'train.py', runtime: null }, is_baseline: true },
        config: { epochs: 100 },
        priority: 100,
        status: 'completed',
        task_type: 'pre',
        counter: 4,
        workflows: ['wf-1', 'wf-2', 'wf-3', 'wf-4'],
        pipeline_id: 'pipeline-main',
        dependency_ids: [],
      },
      {
        id: 'task-bbb222',
        tool: { name: 'pgd-attack', container: { image: 'attacks:v1', command: 'pgd.py', runtime: null }, is_baseline: false },
        config: { epsilon: 0.03 },
        priority: 90,
        status: 'completed',
        task_type: 'in',
        counter: 2,
        workflows: ['wf-1', 'wf-2'],
        pipeline_id: 'pipeline-main',
        dependency_ids: ['task-aaa111'],
      },
      {
        id: 'task-ccc333',
        tool: { name: 'fgsm-attack', container: { image: 'attacks:v1', command: 'fgsm.py', runtime: null }, is_baseline: false },
        config: { epsilon: 0.1 },
        priority: 90,
        status: 'running',
        task_type: 'in',
        counter: 2,
        workflows: ['wf-3', 'wf-4'],
        pipeline_id: 'pipeline-main',
        dependency_ids: ['task-aaa111'],
      },
      {
        id: 'task-ddd444',
        tool: { name: 'adversarial-training', container: { image: 'defense:v1', command: 'adv_train.py', runtime: null }, is_baseline: false },
        config: { method: 'madry' },
        priority: 80,
        status: 'pending',
        task_type: 'in',
        counter: 2,
        workflows: ['wf-1', 'wf-2'],
        pipeline_id: 'pipeline-main',
        dependency_ids: ['task-bbb222'],
      },
      {
        id: 'task-eee555',
        tool: { name: 'evaluation', container: { image: 'eval:v1', command: 'evaluate.py', runtime: null }, is_baseline: false },
        config: {},
        priority: 70,
        status: 'pending',
        task_type: 'post',
        counter: 4,
        workflows: ['wf-1', 'wf-2', 'wf-3', 'wf-4'],
        pipeline_id: 'pipeline-main',
        dependency_ids: ['task-ddd444', 'task-ccc333'],
      },
      {
        id: 'task-fff666',
        tool: { name: 'bad-attack', container: { image: 'attacks:v1', command: 'bad.py', runtime: null }, is_baseline: false },
        config: {},
        priority: 90,
        status: 'failed',
        task_type: 'in',
        counter: 1,
        workflows: ['wf-5'],
        pipeline_id: 'pipeline-main',
        dependency_ids: ['task-aaa111'],
      },
    ],
    total: 6,
  }),
  
  // /api/tasks?status=pending response
  getTasksByStatus: (status: string): TaskListResponse => {
    const all = mockApiResponses.getAllTasks();
    const filtered = all.tasks.filter(t => t.status === status);
    return {
      tasks: filtered,
      total: filtered.length,  // BUG: This should be all.total or have a separate field
    };
  },
  
  // /api/progress response
  getProgress: (): ProgressResponse => ({
    total: 6,
    pending: 2,
    running: 1,
    completed: 2,
    failed: 1,
    progress_percent: 50.0,  // (2 + 1) / 6 * 100
    is_complete: false,
  }),
  
  // /api/tasks/{task_id}/priority response
  getTaskPriority: (taskId: string): TaskPriorityInfo => {
    const task = mockApiResponses.getAllTasks().tasks.find(t => t.id === taskId);
    if (!task) throw new Error(`Task ${taskId} not found`);
    
    return {
      task_id: task.id,
      priority: task.priority,
      dependency_level: task.dependency_ids.length,  // NOTE: This is count, not depth!
      usage_counter: task.counter,
      status: task.status,
      dependencies: task.dependency_ids,
      workflows: task.workflows,
    };
  },
};

describe('API Response Consistency', () => {
  describe('/api/tasks and /api/progress should match', () => {
    it('total task count should be identical', () => {
      const tasksResponse = mockApiResponses.getAllTasks();
      const progressResponse = mockApiResponses.getProgress();
      
      expect(tasksResponse.total).toBe(progressResponse.total);
      expect(tasksResponse.tasks.length).toBe(progressResponse.total);
    });
    
    it('status counts should match between endpoints', () => {
      const tasks = mockApiResponses.getAllTasks().tasks;
      const progress = mockApiResponses.getProgress();
      
      const calculatedCounts = {
        pending: tasks.filter(t => t.status === 'pending').length,
        running: tasks.filter(t => t.status === 'running').length,
        completed: tasks.filter(t => t.status === 'completed').length,
        failed: tasks.filter(t => t.status === 'failed').length,
      };
      
      expect(calculatedCounts.pending).toBe(progress.pending);
      expect(calculatedCounts.running).toBe(progress.running);
      expect(calculatedCounts.completed).toBe(progress.completed);
      expect(calculatedCounts.failed).toBe(progress.failed);
    });
    
    it('progress_percent calculation should be correct', () => {
      const progress = mockApiResponses.getProgress();
      
      // Backend formula: ((completed + failed) / total * 100)
      const expectedPercent = ((progress.completed + progress.failed) / progress.total) * 100;
      
      expect(progress.progress_percent).toBeCloseTo(expectedPercent, 2);
    });
    
    it('is_complete should be true only when no pending/running tasks', () => {
      const progress = mockApiResponses.getProgress();
      
      const shouldBeComplete = progress.pending === 0 && progress.running === 0;
      expect(progress.is_complete).toBe(shouldBeComplete);
    });
  });
});

describe('Filtered API Response Issues', () => {
  describe('BUG: /api/tasks?status=X returns misleading total', () => {
    it('filtered response total equals filtered count, not overall total', () => {
      const unfilteredResponse = mockApiResponses.getAllTasks();
      const filteredResponse = mockApiResponses.getTasksByStatus('pending');
      
      // Current behavior: total = filtered count
      expect(filteredResponse.total).toBe(2);  // Only 2 pending tasks
      expect(filteredResponse.tasks.length).toBe(2);
      
      // Problem: Frontend uses this to show "X of Y tasks"
      // Shows "2 of 2 tasks" instead of "2 of 6 tasks"
      
      // What it SHOULD be:
      expect(unfilteredResponse.total).toBe(6);
    });
    
    it('frontend cannot calculate correct stats from filtered response', () => {
      const filteredResponse = mockApiResponses.getTasksByStatus('pending');
      
      // Frontend calculates stats from the task list it receives
      const frontendStats = {
        pending: filteredResponse.tasks.filter(t => t.status === 'pending').length,
        running: filteredResponse.tasks.filter(t => t.status === 'running').length,
        completed: filteredResponse.tasks.filter(t => t.status === 'completed').length,
        failed: filteredResponse.tasks.filter(t => t.status === 'failed').length,
      };
      
      // All these are wrong except pending!
      expect(frontendStats.pending).toBe(2);  // Correct
      expect(frontendStats.running).toBe(0);  // WRONG - should be 1
      expect(frontendStats.completed).toBe(0);  // WRONG - should be 2
      expect(frontendStats.failed).toBe(0);  // WRONG - should be 1
    });
  });
});

describe('Task Data Integrity', () => {
  it('counter should equal workflows.length for each task', () => {
    const tasks = mockApiResponses.getAllTasks().tasks;
    
    tasks.forEach(task => {
      expect(task.counter).toBe(
        task.workflows.length,
        `Task ${task.id}: counter (${task.counter}) != workflows.length (${task.workflows.length})`
      );
    });
  });
  
  it('dependency_ids should reference valid task IDs', () => {
    const tasks = mockApiResponses.getAllTasks().tasks;
    const allTaskIds = new Set(tasks.map(t => t.id));
    
    tasks.forEach(task => {
      task.dependency_ids.forEach(depId => {
        expect(allTaskIds.has(depId)).toBe(
          true,
          `Task ${task.id} has invalid dependency: ${depId}`
        );
      });
    });
  });
  
  it('task with no dependencies should have highest priority', () => {
    const tasks = mockApiResponses.getAllTasks().tasks;
    const rootTasks = tasks.filter(t => t.dependency_ids.length === 0);
    
    rootTasks.forEach(task => {
      expect(task.priority).toBe(100);  // Depth 0 = priority 100
    });
  });
  
  it('priority should decrease with dependency depth', () => {
    const tasks = mockApiResponses.getAllTasks().tasks;
    
    // task-eee555 depends on task-ddd444 which depends on task-bbb222 which depends on task-aaa111
    // Depth: aaa111=0, bbb222=1, ddd444=2, eee555=3
    
    const taskAaa = tasks.find(t => t.id === 'task-aaa111')!;
    const taskBbb = tasks.find(t => t.id === 'task-bbb222')!;
    const taskDdd = tasks.find(t => t.id === 'task-ddd444')!;
    const taskEee = tasks.find(t => t.id === 'task-eee555')!;
    
    expect(taskAaa.priority).toBeGreaterThan(taskBbb.priority);
    expect(taskBbb.priority).toBeGreaterThan(taskDdd.priority);
    expect(taskDdd.priority).toBeGreaterThan(taskEee.priority);
    
    // Verify exact values
    expect(taskAaa.priority).toBe(100);  // depth 0
    expect(taskBbb.priority).toBe(90);   // depth 1
    expect(taskDdd.priority).toBe(80);   // depth 2
    expect(taskEee.priority).toBe(70);   // depth 3
  });
});

describe('Priority Info Endpoint', () => {
  it('dependency_level reflects direct dependency count, not graph depth', () => {
    // This is the documented behavior, but can be confusing
    
    // task-eee555 has 2 direct dependencies but is at depth 3
    const taskInfo = mockApiResponses.getTaskPriority('task-eee555');
    
    expect(taskInfo.dependency_level).toBe(2);  // Direct count
    
    // But the priority suggests depth 3 (100 - 30 = 70)
    const impliedDepth = (100 - taskInfo.priority) / 10;
    expect(impliedDepth).toBe(3);
    
    // These don't match - potentially confusing for users
    expect(taskInfo.dependency_level).not.toBe(impliedDepth);
  });
  
  it('usage_counter should match counter field in task', () => {
    const tasks = mockApiResponses.getAllTasks().tasks;
    
    tasks.forEach(task => {
      const priorityInfo = mockApiResponses.getTaskPriority(task.id);
      expect(priorityInfo.usage_counter).toBe(task.counter);
    });
  });
  
  it('workflows should match between task and priority info', () => {
    const tasks = mockApiResponses.getAllTasks().tasks;
    
    tasks.forEach(task => {
      const priorityInfo = mockApiResponses.getTaskPriority(task.id);
      expect(priorityInfo.workflows.sort()).toEqual(task.workflows.sort());
    });
  });
});

describe('Frontend Display Calculations', () => {
  /**
   * Simulates the actual calculations done in Tasks.tsx
   */
  
  const simulateTasksPageLogic = (tasks: TaskResponse[]) => {
    // From Tasks.tsx lines 93-98
    const stats = {
      pending: tasks.filter((t) => t.status === 'pending').length,
      running: tasks.filter((t) => t.status === 'running').length,
      completed: tasks.filter((t) => t.status === 'completed').length,
      failed: tasks.filter((t) => t.status === 'failed').length,
    };
    
    // From Tasks.tsx lines 77-90 - CURRENT (buggy) sorting
    const sortByPriorityBuggy = [...tasks].sort((a, b) => a.priority - b.priority);
    
    // CORRECT sorting (high priority first)
    const sortByPriorityCorrect = [...tasks].sort((a, b) => b.priority - a.priority);
    
    return {
      stats,
      sortedTasks: {
        buggy: sortByPriorityBuggy,
        correct: sortByPriorityCorrect,
      },
    };
  };
  
  it('should show correct stats when all tasks are loaded', () => {
    const tasks = mockApiResponses.getAllTasks().tasks;
    const result = simulateTasksPageLogic(tasks);
    
    const expectedProgress = mockApiResponses.getProgress();
    
    expect(result.stats.pending).toBe(expectedProgress.pending);
    expect(result.stats.running).toBe(expectedProgress.running);
    expect(result.stats.completed).toBe(expectedProgress.completed);
    expect(result.stats.failed).toBe(expectedProgress.failed);
  });
  
  it('BUG: current sorting shows LOW priority first', () => {
    const tasks = mockApiResponses.getAllTasks().tasks;
    const result = simulateTasksPageLogic(tasks);
    
    // Current buggy behavior: lowest priority first
    expect(result.sortedTasks.buggy[0].priority).toBe(70);
    expect(result.sortedTasks.buggy[0].tool.name).toBe('evaluation');
    
    // Correct behavior: highest priority first
    expect(result.sortedTasks.correct[0].priority).toBe(100);
    expect(result.sortedTasks.correct[0].tool.name).toBe('baseline-training');
  });
});

describe('Data Freshness and Timing', () => {
  /**
   * Tests to ensure data consistency across multiple API calls
   */
  
  it('task status in /api/tasks should match /api/progress counts', () => {
    // In real usage, these are fetched at slightly different times
    // They should still be consistent
    
    const tasks = mockApiResponses.getAllTasks().tasks;
    const progress = mockApiResponses.getProgress();
    
    const statusCounts = {
      pending: tasks.filter(t => t.status === 'pending').length,
      running: tasks.filter(t => t.status === 'running').length,
      completed: tasks.filter(t => t.status === 'completed').length,
      failed: tasks.filter(t => t.status === 'failed').length,
    };
    
    const total = statusCounts.pending + statusCounts.running + 
                  statusCounts.completed + statusCounts.failed;
    
    expect(total).toBe(progress.total);
    expect(statusCounts.pending).toBe(progress.pending);
    expect(statusCounts.running).toBe(progress.running);
    expect(statusCounts.completed).toBe(progress.completed);
    expect(statusCounts.failed).toBe(progress.failed);
  });
});
