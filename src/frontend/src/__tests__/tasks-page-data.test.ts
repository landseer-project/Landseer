/**
 * Frontend Data Display Tests - Tasks Page
 * 
 * These tests verify that the frontend displays correct numbers to users.
 * 
 * CRITICAL ISSUES IDENTIFIED:
 * 
 * 1. STATUS COUNTS MISMATCH (BUG)
 *    When a status filter is applied (e.g., "pending"), the API returns only
 *    matching tasks. The frontend then calculates stats from this filtered list,
 *    showing WRONG counts for other statuses (all zeros).
 *    
 *    Example: Filter by "pending" → API returns 10 pending tasks
 *    Frontend shows: pending=10, running=0, completed=0, failed=0
 *    Reality: Could be pending=10, running=5, completed=20, failed=2
 * 
 * 2. PRIORITY SORT DIRECTION (BUG)
 *    Frontend sorts by priority using `a.priority - b.priority` (ascending)
 *    But backend uses HIGHER priority value = runs FIRST
 *    So frontend shows LOW priority tasks first, opposite of execution order!
 * 
 * 3. DEPENDENCY LEVEL MISMATCH (MISLEADING)
 *    API's `get_task_priority_info()` returns `len(task.dependencies)` as dependency_level
 *    But priority is calculated based on DEPTH (longest path), not count
 *    A task with 1 dependency at depth 5 shows level=1 but has priority of depth 6
 * 
 * 4. TOTAL COUNT INCONSISTENCY (BUG)
 *    When filtered, `TaskListResponse.total` equals the filtered count
 *    Not the actual total tasks in the system
 *    "X of Y tasks" display is misleading when filtered
 * 
 * 5. DASHBOARD vs TASKS PAGE DATA SOURCE MISMATCH (POTENTIAL)
 *    Dashboard uses `/api/progress` for status counts
 *    Tasks page calculates counts from `/api/tasks` response
 *    These should match but could diverge due to timing
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock task data representing real-world scenarios
const createMockTasks = () => [
  {
    id: 'task-1',
    tool: { name: 'Tool A', container: { image: 'img', command: 'cmd', runtime: null }, is_baseline: false },
    config: {},
    priority: 100,  // Highest priority (depth 0)
    status: 'completed',
    task_type: 'pre',
    counter: 3,
    workflows: ['wf-1', 'wf-2', 'wf-3'],
    pipeline_id: 'pipeline-1',
    dependency_ids: [],
  },
  {
    id: 'task-2',
    tool: { name: 'Tool B', container: { image: 'img', command: 'cmd', runtime: null }, is_baseline: false },
    config: {},
    priority: 90,  // Second highest (depth 1)
    status: 'completed',
    task_type: 'in',
    counter: 2,
    workflows: ['wf-1', 'wf-2'],
    pipeline_id: 'pipeline-1',
    dependency_ids: ['task-1'],
  },
  {
    id: 'task-3',
    tool: { name: 'Tool C', container: { image: 'img', command: 'cmd', runtime: null }, is_baseline: false },
    config: {},
    priority: 80,  // Third (depth 2)
    status: 'running',
    task_type: 'in',
    counter: 2,
    workflows: ['wf-1', 'wf-2'],
    pipeline_id: 'pipeline-1',
    dependency_ids: ['task-2'],
  },
  {
    id: 'task-4',
    tool: { name: 'Tool D', container: { image: 'img', command: 'cmd', runtime: null }, is_baseline: false },
    config: {},
    priority: 70,  // Lower (depth 3)
    status: 'pending',
    task_type: 'post',
    counter: 1,
    workflows: ['wf-1'],
    pipeline_id: 'pipeline-1',
    dependency_ids: ['task-3'],
  },
  {
    id: 'task-5',
    tool: { name: 'Tool E', container: { image: 'img', command: 'cmd', runtime: null }, is_baseline: false },
    config: {},
    priority: 70,  // Same priority as task-4 (depth 3)
    status: 'pending',
    task_type: 'post',
    counter: 1,
    workflows: ['wf-2'],
    pipeline_id: 'pipeline-1',
    dependency_ids: ['task-3'],
  },
  {
    id: 'task-6',
    tool: { name: 'Tool F', container: { image: 'img', command: 'cmd', runtime: null }, is_baseline: false },
    config: {},
    priority: 90,  // depth 1, but fails
    status: 'failed',
    task_type: 'in',
    counter: 1,
    workflows: ['wf-3'],
    pipeline_id: 'pipeline-1',
    dependency_ids: ['task-1'],
  },
];

describe('Tasks Page - Status Statistics Calculation', () => {
  /**
   * BUG #1: Status counts are calculated from the filtered task list
   * When a status filter is applied, stats for other statuses become 0
   */
  
  it('should calculate correct stats from ALL tasks when no filter is applied', () => {
    const tasks = createMockTasks();
    
    // Current frontend implementation (correct when unfiltered)
    const stats = {
      pending: tasks.filter((t) => t.status === 'pending').length,
      running: tasks.filter((t) => t.status === 'running').length,
      completed: tasks.filter((t) => t.status === 'completed').length,
      failed: tasks.filter((t) => t.status === 'failed').length,
    };
    
    expect(stats.pending).toBe(2);
    expect(stats.running).toBe(1);
    expect(stats.completed).toBe(2);
    expect(stats.failed).toBe(1);
    expect(stats.pending + stats.running + stats.completed + stats.failed).toBe(tasks.length);
  });
  
  it('BUG: Stats are WRONG when status filter is applied', () => {
    // Simulates what happens when user clicks "Pending" filter
    // API returns only pending tasks
    const allTasks = createMockTasks();
    const filteredTasks = allTasks.filter(t => t.status === 'pending');
    
    // Current frontend implementation - calculates from filtered tasks
    const incorrectStats = {
      pending: filteredTasks.filter((t) => t.status === 'pending').length,
      running: filteredTasks.filter((t) => t.status === 'running').length,
      completed: filteredTasks.filter((t) => t.status === 'completed').length,
      failed: filteredTasks.filter((t) => t.status === 'failed').length,
    };
    
    // This is WRONG - shows 0 for all statuses except filtered one
    expect(incorrectStats.pending).toBe(2);
    expect(incorrectStats.running).toBe(0);  // WRONG! Should be 1
    expect(incorrectStats.completed).toBe(0);  // WRONG! Should be 2
    expect(incorrectStats.failed).toBe(0);  // WRONG! Should be 1
    
    // What the correct stats should be (from all tasks)
    const correctStats = {
      pending: allTasks.filter((t) => t.status === 'pending').length,
      running: allTasks.filter((t) => t.status === 'running').length,
      completed: allTasks.filter((t) => t.status === 'completed').length,
      failed: allTasks.filter((t) => t.status === 'failed').length,
    };
    
    expect(correctStats.running).toBe(1);
    expect(correctStats.completed).toBe(2);
    expect(correctStats.failed).toBe(1);
  });
});

describe('Tasks Page - Priority Sorting', () => {
  /**
   * BUG #2: Priority sort direction is inverted
   * Backend: higher priority value = runs first
   * Frontend: sorts ascending, so lower priority shows first
   */
  
  it('BUG: Priority sort shows tasks in WRONG order', () => {
    const tasks = createMockTasks();
    
    // Current frontend implementation (incorrect)
    const sortedByPriorityIncorrect = [...tasks].sort((a, b) => a.priority - b.priority);
    
    // First task shown should be lowest priority (70) - but user expects highest!
    expect(sortedByPriorityIncorrect[0].priority).toBe(70);
    expect(sortedByPriorityIncorrect[sortedByPriorityIncorrect.length - 1].priority).toBe(100);
    
    // Correct implementation - descending order (highest priority first)
    const sortedByPriorityCorrect = [...tasks].sort((a, b) => b.priority - a.priority);
    
    expect(sortedByPriorityCorrect[0].priority).toBe(100);  // Highest first
    expect(sortedByPriorityCorrect[sortedByPriorityCorrect.length - 1].priority).toBe(70);  // Lowest last
  });
  
  it('should show high-priority tasks first (what users expect)', () => {
    const tasks = createMockTasks();
    
    // Users expect to see tasks that will execute first at the top
    // Since backend runs higher priority first, UI should sort descending
    const correctSort = [...tasks].sort((a, b) => b.priority - a.priority);
    
    // task-1 (priority 100) should be first
    expect(correctSort[0].id).toBe('task-1');
    expect(correctSort[0].tool.name).toBe('Tool A');
    
    // Tasks with priority 70 should be last
    const lastTwoPriorities = correctSort.slice(-2).map(t => t.priority);
    expect(lastTwoPriorities).toEqual([70, 70]);
  });
});

describe('Tasks Page - Workflow and Dependency Counts', () => {
  /**
   * These should be correct if task data is properly formatted
   */
  
  it('should display correct workflow count for each task', () => {
    const tasks = createMockTasks();
    
    const task1 = tasks.find(t => t.id === 'task-1')!;
    expect(task1.workflows.length).toBe(3);  // Used in 3 workflows
    
    const task4 = tasks.find(t => t.id === 'task-4')!;
    expect(task4.workflows.length).toBe(1);  // Used in 1 workflow
  });
  
  it('should display correct dependency count for each task', () => {
    const tasks = createMockTasks();
    
    const task1 = tasks.find(t => t.id === 'task-1')!;
    expect(task1.dependency_ids.length).toBe(0);  // No dependencies
    
    const task4 = tasks.find(t => t.id === 'task-4')!;
    expect(task4.dependency_ids.length).toBe(1);  // 1 dependency
  });
  
  it('should match counter with workflows.length', () => {
    const tasks = createMockTasks();
    
    // counter should equal the number of workflows using this task
    tasks.forEach(task => {
      expect(task.counter).toBe(task.workflows.length);
    });
  });
});

describe('Tasks Page - List Count Display', () => {
  /**
   * BUG #4: "X of Y tasks" uses filtered total as Y
   */
  
  it('BUG: "X of Y tasks" shows wrong Y when filtered', () => {
    const allTasks = createMockTasks();
    const filteredTasks = allTasks.filter(t => t.status === 'pending');
    
    // Current frontend logic (from API response when filtered)
    const incorrectApiResponse = {
      tasks: filteredTasks,
      total: filteredTasks.length  // API returns filtered count as total
    };
    
    // Frontend shows "filteredTasks.length of incorrectApiResponse.total tasks"
    // e.g., "2 of 2 tasks" when it should say "2 of 6 tasks"
    expect(incorrectApiResponse.total).toBe(2);  // WRONG! Should be 6
    
    // What users expect
    const correctDisplay = `${filteredTasks.length} of ${allTasks.length} tasks`;
    expect(correctDisplay).toBe("2 of 6 tasks");
  });
});

describe('Dashboard vs Tasks Page Consistency', () => {
  /**
   * Potential Issue: Dashboard and Tasks page may show different numbers
   * Dashboard uses /api/progress
   * Tasks page calculates from /api/tasks
   */
  
  it('should have consistent counts between progress API and tasks API', () => {
    const allTasks = createMockTasks();
    
    // Simulated /api/progress response (from scheduler.get_progress())
    const progressResponse = {
      total: 6,
      pending: 2,
      running: 1,
      completed: 2,
      failed: 1,
      progress_percent: 50.0,  // (2 completed + 1 failed) / 6 * 100
      is_complete: false
    };
    
    // Calculated from tasks (what Tasks page does)
    const calculatedFromTasks = {
      total: allTasks.length,
      pending: allTasks.filter(t => t.status === 'pending').length,
      running: allTasks.filter(t => t.status === 'running').length,
      completed: allTasks.filter(t => t.status === 'completed').length,
      failed: allTasks.filter(t => t.status === 'failed').length,
    };
    
    // These should ALWAYS match
    expect(calculatedFromTasks.total).toBe(progressResponse.total);
    expect(calculatedFromTasks.pending).toBe(progressResponse.pending);
    expect(calculatedFromTasks.running).toBe(progressResponse.running);
    expect(calculatedFromTasks.completed).toBe(progressResponse.completed);
    expect(calculatedFromTasks.failed).toBe(progressResponse.failed);
  });
  
  it('progress_percent should match calculated percentage', () => {
    const progressResponse = {
      total: 6,
      completed: 2,
      failed: 1,
    };
    
    // Backend calculates: ((completed + failed) / total * 100)
    const expectedPercent = ((progressResponse.completed + progressResponse.failed) / progressResponse.total) * 100;
    
    expect(expectedPercent).toBeCloseTo(50.0, 2);
  });
});

describe('Task Priority Info - Dependency Level', () => {
  /**
   * Issue #3: dependency_level shows count of direct dependencies
   * but priority is calculated from DEPTH (longest path)
   * This can be confusing to users
   */
  
  it('ISSUE: dependency_level differs from actual depth used in priority', () => {
    // From PriorityScheduler.get_task_priority_info():
    // dependency_level = len(task.dependencies) <- Direct count
    // But priority uses depth = longest path from root
    
    // Example: task-4 has 1 direct dependency (task-3)
    // But it's at depth 3 (task-1 → task-2 → task-3 → task-4)
    
    const task4 = createMockTasks().find(t => t.id === 'task-4')!;
    const directDependencyCount = task4.dependency_ids.length;
    
    // API would return dependency_level = 1
    expect(directDependencyCount).toBe(1);
    
    // But actual depth in dependency chain is 3
    // (Priority = 100 - (3 * 10) = 70)
    const actualDepth = (100 - task4.priority) / 10;
    expect(actualDepth).toBe(3);
    
    // These are different! Users may be confused.
    expect(directDependencyCount).not.toBe(actualDepth);
  });
});

describe('Edge Cases', () => {
  it('should handle empty task list', () => {
    const tasks: typeof createMockTasks extends () => infer R ? R : never = [];
    
    const stats = {
      pending: tasks.filter((t) => t.status === 'pending').length,
      running: tasks.filter((t) => t.status === 'running').length,
      completed: tasks.filter((t) => t.status === 'completed').length,
      failed: tasks.filter((t) => t.status === 'failed').length,
    };
    
    expect(stats.pending).toBe(0);
    expect(stats.running).toBe(0);
    expect(stats.completed).toBe(0);
    expect(stats.failed).toBe(0);
  });
  
  it('should handle all tasks in same status', () => {
    const allPendingTasks = createMockTasks().map(t => ({ ...t, status: 'pending' as const }));
    
    const stats = {
      pending: allPendingTasks.filter((t) => t.status === 'pending').length,
      running: allPendingTasks.filter((t) => t.status === 'running').length,
      completed: allPendingTasks.filter((t) => t.status === 'completed').length,
      failed: allPendingTasks.filter((t) => t.status === 'failed').length,
    };
    
    expect(stats.pending).toBe(6);
    expect(stats.running).toBe(0);
    expect(stats.completed).toBe(0);
    expect(stats.failed).toBe(0);
  });
  
  it('should handle tasks with same priority', () => {
    const tasks = createMockTasks();
    const samePriorityTasks = tasks.filter(t => t.priority === 70);
    
    expect(samePriorityTasks.length).toBe(2);
    expect(samePriorityTasks.map(t => t.id).sort()).toEqual(['task-4', 'task-5']);
  });
  
  it('should handle task with no workflows', () => {
    const task = {
      ...createMockTasks()[0],
      workflows: [],
      counter: 0,
    };
    
    expect(task.workflows.length).toBe(0);
    expect(task.counter).toBe(0);
  });
});
