/**
 * E2E Data Validation Tests for Tasks Page
 * 
 * These tests can be run against the actual running application
 * to validate that displayed numbers match backend API responses.
 * 
 * To run these tests:
 * 1. Start the backend: `poetry run python -m src.backend.cli start`
 * 2. Start the frontend: `cd src/frontend && npm run dev`
 * 3. Run tests with a tool like Playwright or Cypress
 * 
 * This file provides test specifications that can be adapted to
 * your preferred E2E testing framework.
 */

/**
 * Test Specification: Tasks Page Data Validation
 */
export const TasksPageE2ETests = {
  /**
   * Verify status card counts match API /progress endpoint
   */
  'status cards should match /api/progress response': async () => {
    // 1. Fetch /api/progress to get expected counts
    const progressResponse = await fetch('/api/progress');
    const progress = await progressResponse.json();
    
    // 2. Navigate to Tasks page
    // await page.goto('/tasks');
    
    // 3. Extract displayed counts from UI
    // const pendingCount = await page.locator('.stats-card.pending .count').textContent();
    // const runningCount = await page.locator('.stats-card.running .count').textContent();
    // const completedCount = await page.locator('.stats-card.completed .count').textContent();
    // const failedCount = await page.locator('.stats-card.failed .count').textContent();
    
    // 4. Assert they match
    // expect(parseInt(pendingCount)).toBe(progress.pending);
    // expect(parseInt(runningCount)).toBe(progress.running);
    // expect(parseInt(completedCount)).toBe(progress.completed);
    // expect(parseInt(failedCount)).toBe(progress.failed);
    
    return {
      description: 'Status card counts should exactly match /api/progress endpoint values',
      expectedBehavior: 'Pending, Running, Completed, Failed counts in UI match API',
      knownBug: 'When filtered, stats show 0 for non-matching statuses',
    };
  },

  /**
   * Verify task list count matches total
   */
  'task list count should match total tasks': async () => {
    // 1. Fetch /api/tasks to get expected count
    const tasksResponse = await fetch('/api/tasks');
    const tasks = await tasksResponse.json();
    
    // 2. Navigate to Tasks page
    // await page.goto('/tasks');
    
    // 3. Count visible task items
    // const taskItems = await page.locator('.task-list-item').count();
    
    // 4. Check header text
    // const headerText = await page.locator('.task-list-header').textContent();
    // Expected format: "X of Y tasks"
    
    // 5. Assert
    // expect(taskItems).toBe(tasks.total);
    // expect(headerText).toContain(`${tasks.total} of ${tasks.total} tasks`);
    
    return {
      description: 'Task list should show correct count in header',
      expectedBehavior: 'Header shows "N of N tasks" where N is total',
      knownBug: 'When filtered, Y in "X of Y" equals filtered count, not total',
    };
  },

  /**
   * Verify priority sorting order
   */
  'priority sort should show highest priority first': async () => {
    // 1. Navigate to Tasks page
    // await page.goto('/tasks');
    
    // 2. Select priority sort
    // await page.selectOption('.sort-dropdown', 'priority');
    
    // 3. Get first visible task priority
    // const firstTaskPriority = await page.locator('.task-list-item:first-child .priority').textContent();
    
    // 4. Get last visible task priority
    // const lastTaskPriority = await page.locator('.task-list-item:last-child .priority').textContent();
    
    // 5. Assert first has higher priority than last
    // expect(parseInt(firstTaskPriority)).toBeGreaterThan(parseInt(lastTaskPriority));
    
    return {
      description: 'When sorted by priority, highest priority tasks should appear first',
      expectedBehavior: 'Task with priority 100 appears before task with priority 70',
      knownBug: 'Currently shows lowest priority first due to ascending sort',
    };
  },

  /**
   * Verify workflow count per task
   */
  'workflow count should match task.workflows.length': async () => {
    // 1. Fetch specific task from API
    const tasksResponse = await fetch('/api/tasks');
    const tasks = await tasksResponse.json();
    const testTask = tasks.tasks[0];
    
    // 2. Navigate to Tasks page
    // await page.goto('/tasks');
    
    // 3. Find the task in UI
    // const taskElement = await page.locator(`[data-task-id="${testTask.id}"]`);
    
    // 4. Get displayed workflow count
    // const workflowCount = await taskElement.locator('.workflow-count').textContent();
    
    // 5. Assert it matches API
    // expect(parseInt(workflowCount)).toBe(testTask.workflows.length);
    
    return {
      description: 'Workflow count displayed should match API response',
      expectedBehavior: 'UI shows same count as task.workflows.length from API',
    };
  },

  /**
   * Verify dependency count per task
   */
  'dependency count should match task.dependency_ids.length': async () => {
    // Similar to workflow count test
    return {
      description: 'Dependency count displayed should match API response',
      expectedBehavior: 'UI shows same count as task.dependency_ids.length from API',
    };
  },

  /**
   * Verify filtered view updates correctly
   */
  'status filter should show only matching tasks': async () => {
    // 1. Navigate to Tasks page
    // await page.goto('/tasks');
    
    // 2. Select "Pending" filter
    // await page.selectOption('.status-filter', 'pending');
    
    // 3. Wait for update
    // await page.waitForResponse(resp => resp.url().includes('/api/tasks'));
    
    // 4. Verify all visible tasks have pending status
    // const taskStatuses = await page.locator('.task-list-item .status-badge').allTextContents();
    // taskStatuses.forEach(status => expect(status.toLowerCase()).toBe('pending'));
    
    return {
      description: 'Filtering by status should only show matching tasks',
      expectedBehavior: 'Only pending tasks visible when Pending filter is selected',
    };
  },

  /**
   * Verify task detail dialog shows correct data
   */
  'task detail should show accurate information': async () => {
    // 1. Fetch specific task from API
    const tasksResponse = await fetch('/api/tasks');
    const tasks = await tasksResponse.json();
    const testTask = tasks.tasks[0];
    
    // 2. Also fetch priority info
    const priorityResponse = await fetch(`/api/tasks/${testTask.id}/priority`);
    const priorityInfo = await priorityResponse.json();
    
    // 3. Navigate to Tasks page and click on task
    // await page.goto('/tasks');
    // await page.locator(`[data-task-id="${testTask.id}"]`).click();
    
    // 4. Wait for dialog to open
    // await page.waitForSelector('.task-detail-dialog');
    
    // 5. Verify displayed values
    // const displayedPriority = await page.locator('.dialog .priority-value').textContent();
    // const displayedUsageCount = await page.locator('.dialog .usage-count').textContent();
    // const displayedDependencyLevel = await page.locator('.dialog .dependency-level').textContent();
    
    // expect(parseInt(displayedPriority)).toBe(testTask.priority);
    // expect(parseInt(displayedUsageCount)).toBe(testTask.counter);
    // expect(parseInt(displayedDependencyLevel)).toBe(priorityInfo.dependency_level);
    
    return {
      description: 'Task detail dialog should show accurate values from API',
      expectedBehavior: 'Priority, Usage Count, and Dependency Level match API',
      knownIssue: 'dependency_level shows direct count, not graph depth (may confuse users)',
    };
  },
};

/**
 * Test Specification: Dashboard vs Tasks Page Consistency
 */
export const CrossPageConsistencyTests = {
  /**
   * Dashboard and Tasks page should show same status counts
   */
  'dashboard and tasks page status counts should match': async () => {
    // 1. Navigate to Dashboard
    // await page.goto('/');
    
    // 2. Extract status counts from Dashboard
    // const dashboardCounts = {
    //   completed: await page.locator('.dashboard .completed-count').textContent(),
    //   running: await page.locator('.dashboard .running-count').textContent(),
    //   pending: await page.locator('.dashboard .pending-count').textContent(),
    //   failed: await page.locator('.dashboard .failed-count').textContent(),
    // };
    
    // 3. Navigate to Tasks page
    // await page.goto('/tasks');
    
    // 4. Extract status counts from Tasks page
    // const tasksCounts = {
    //   completed: await page.locator('.tasks .completed-count').textContent(),
    //   running: await page.locator('.tasks .running-count').textContent(),
    //   pending: await page.locator('.tasks .pending-count').textContent(),
    //   failed: await page.locator('.tasks .failed-count').textContent(),
    // };
    
    // 5. Assert they match
    // expect(dashboardCounts).toEqual(tasksCounts);
    
    return {
      description: 'Both pages should show identical status counts',
      expectedBehavior: 'Dashboard and Tasks page show same numbers',
      potentialIssue: 'Different data sources (progress API vs tasks API) could diverge',
    };
  },

  /**
   * Total task counts should be consistent everywhere
   */
  'total task count should be consistent across pages': async () => {
    // 1. Get total from Dashboard
    // 2. Get total from Tasks page
    // 3. Get total from Progress API
    // 4. Get count from Tasks API
    // All should match
    
    return {
      description: 'Total task count should be same across all pages and APIs',
      expectedBehavior: 'Dashboard, Tasks page, /api/progress, /api/tasks all show same total',
    };
  },
};

/**
 * Test Specification: Real-time Updates
 */
export const RealTimeUpdateTests = {
  /**
   * When task status changes, UI should update
   */
  'status update should reflect in UI': async () => {
    // This test requires ability to trigger status changes via API
    // 1. Get initial counts
    // 2. Complete a task via PUT /api/tasks/status
    // 3. Wait for React Query to refetch
    // 4. Verify counts updated (completed +1, running -1)
    
    return {
      description: 'UI should reflect status changes from API',
      expectedBehavior: 'After task completion, completed count increases by 1',
    };
  },
};

/**
 * Manual Validation Checklist
 * 
 * Run through these checks manually while the application is running:
 * 
 * 1. UNFILTERED VIEW
 *    [ ] Pending count matches /api/progress.pending
 *    [ ] Running count matches /api/progress.running
 *    [ ] Completed count matches /api/progress.completed
 *    [ ] Failed count matches /api/progress.failed
 *    [ ] Sum of counts equals total task count
 *    [ ] Task list shows correct "X of Y tasks" header
 * 
 * 2. FILTERED VIEW (e.g., filter by Pending)
 *    [ ] BUG CHECK: Do status card counts still show ALL statuses, not just filtered?
 *    [ ] Task list only shows tasks with selected status
 *    [ ] "X of Y tasks" shows correct X (filtered) and Y (total, not filtered)
 * 
 * 3. PRIORITY SORTING
 *    [ ] BUG CHECK: Is highest priority (100) shown first, or lowest (60)?
 *    [ ] Tasks should be sorted highest to lowest when "Priority" sort is selected
 * 
 * 4. TASK DETAILS
 *    [ ] Click on a task - does dialog show correct Priority value?
 *    [ ] Does Usage Count match the number of workflows listed?
 *    [ ] Does Dependencies tab show correct number of dependencies?
 * 
 * 5. DASHBOARD CONSISTENCY
 *    [ ] Go to Dashboard, note the status counts
 *    [ ] Go to Tasks page, verify counts are identical
 *    [ ] Both should update in sync after task status changes
 */
