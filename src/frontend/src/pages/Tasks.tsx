import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { StatusBadge } from '@/components/StatusBadge';
import { getAllTasks, getTaskPriority, getReadyTasks, getBlockedTasks, getTaskLogs } from '@/lib/api';
import { truncateId } from '@/lib/utils';
import type { TaskResponse } from '@/types/api';
import {
  Search,
  Filter,
  Layers,
  GitBranch,
  Clock,
  ChevronRight,
  ArrowUpDown,
  Loader2,
  Info,
  Box,
  PlayCircle,
  List,
  AlertTriangle,
  Ban,
} from 'lucide-react';

export function Tasks() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTask, setSelectedTask] = useState<TaskResponse | null>(null);
  const [sortBy, setSortBy] = useState<'priority' | 'status' | 'name'>('priority');
  const [showReadyQueue, setShowReadyQueue] = useState(false);
  const [showBlockedTasks, setShowBlockedTasks] = useState(false);

  const { data: tasksData, isLoading, isFetching } = useQuery({
    queryKey: ['tasks', statusFilter === 'all' ? undefined : statusFilter],
    queryFn: () => getAllTasks(statusFilter === 'all' ? undefined : statusFilter),
  });

  const { data: readyTasksData, isFetching: isReadyFetching } = useQuery({
    queryKey: ['ready-tasks'],
    queryFn: getReadyTasks,
    refetchInterval: 2000, // Refresh every 2 seconds
  });

  const { data: blockedTasksData, isFetching: isBlockedFetching } = useQuery({
    queryKey: ['blocked-tasks'],
    queryFn: getBlockedTasks,
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  const { data: taskPriority } = useQuery({
    queryKey: ['task-priority', selectedTask?.id],
    queryFn: () => (selectedTask ? getTaskPriority(selectedTask.id) : null),
    enabled: !!selectedTask,
  });

  const { data: taskLogs } = useQuery({
    queryKey: ['task-logs', selectedTask?.id],
    queryFn: () => (selectedTask ? getTaskLogs(selectedTask.id) : null),
    enabled: !!selectedTask && selectedTask.status === 'failed',
  });

  const tasks = tasksData?.tasks || [];
  const readyTasks = readyTasksData?.tasks || [];
  const blockedTasks = blockedTasksData?.tasks || [];
  const isRefreshing = isFetching && !isLoading;

  // Handle task query parameter (from worker card click)
  useEffect(() => {
    const taskId = searchParams.get('task');
    if (taskId && tasks.length > 0) {
      const task = tasks.find((t) => t.id === taskId);
      if (task) {
        setSelectedTask(task);
        // Remove query param after opening
        setSearchParams({}, { replace: true });
      }
    }
  }, [searchParams, tasks, setSearchParams]);

  // Filter and sort tasks
  const filteredTasks = tasks
    .filter((task) => {
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          task.id.toLowerCase().includes(query) ||
          task.tool.name.toLowerCase().includes(query)
        );
      }
      return true;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'priority':
          return a.priority - b.priority;
        case 'status':
          const statusOrder = { running: 0, pending: 1, completed: 2, failed: 3 };
          return (statusOrder[a.status as keyof typeof statusOrder] || 4) - 
                 (statusOrder[b.status as keyof typeof statusOrder] || 4);
        case 'name':
          return a.tool.name.localeCompare(b.tool.name);
        default:
          return 0;
      }
    });

  // Stats
  const stats = {
    pending: tasks.filter((t) => t.status === 'pending').length,
    running: tasks.filter((t) => t.status === 'running').length,
    completed: tasks.filter((t) => t.status === 'completed').length,
    failed: tasks.filter((t) => t.status === 'failed').length,
  };

  // Only show full loading on initial load
  if (isLoading && !tasksData) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2">
          <h1 className="text-3xl font-bold tracking-tight">Tasks</h1>
          {isRefreshing && (
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          )}
        </div>
        <p className="text-muted-foreground">
          View and monitor all pipeline tasks
        </p>
      </div>

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-6">
        <Card className="cursor-pointer transition-shadow hover:shadow-md" onClick={() => setShowReadyQueue(!showReadyQueue)}>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900/30">
              <PlayCircle className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <p className="text-2xl font-bold">{readyTasks.length}</p>
                {isReadyFetching && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}
              </div>
              <p className="text-sm text-muted-foreground">Ready Queue</p>
            </div>
          </CardContent>
        </Card>

        <Card className="cursor-pointer transition-shadow hover:shadow-md" onClick={() => setShowBlockedTasks(!showBlockedTasks)}>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-orange-100 dark:bg-orange-900/30">
              <Ban className="h-6 w-6 text-orange-600 dark:text-orange-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <p className="text-2xl font-bold">{blockedTasks.length}</p>
                {isBlockedFetching && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}
              </div>
              <p className="text-sm text-muted-foreground">Blocked</p>
            </div>
          </CardContent>
        </Card>

        <Card className="cursor-pointer transition-shadow hover:shadow-md" onClick={() => setStatusFilter('pending')}>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-yellow-100 dark:bg-yellow-900/30">
              <Clock className="h-6 w-6 text-yellow-600 dark:text-yellow-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{stats.pending}</p>
              <p className="text-sm text-muted-foreground">Pending</p>
            </div>
          </CardContent>
        </Card>

        <Card className="cursor-pointer transition-shadow hover:shadow-md" onClick={() => setStatusFilter('running')}>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/30">
              <Loader2 className="h-6 w-6 animate-spin text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{stats.running}</p>
              <p className="text-sm text-muted-foreground">Running</p>
            </div>
          </CardContent>
        </Card>

        <Card className="cursor-pointer transition-shadow hover:shadow-md" onClick={() => setStatusFilter('completed')}>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-green-100 dark:bg-green-900/30">
              <Layers className="h-6 w-6 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{stats.completed}</p>
              <p className="text-sm text-muted-foreground">Completed</p>
            </div>
          </CardContent>
        </Card>

        <Card className="cursor-pointer transition-shadow hover:shadow-md" onClick={() => setStatusFilter('failed')}>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-red-100 dark:bg-red-900/30">
              <Info className="h-6 w-6 text-red-600 dark:text-red-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{stats.failed}</p>
              <p className="text-sm text-muted-foreground">Failed</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Ready Queue Panel */}
      {showReadyQueue && (
        <Card className="border-purple-200 dark:border-purple-800">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <PlayCircle className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                <CardTitle>Ready Queue</CardTitle>
              </div>
              <Badge variant="secondary" className="bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300">
                {readyTasks.length} tasks ready
              </Badge>
            </div>
            <CardDescription>
              Tasks with all dependencies satisfied, waiting for a worker to claim them
            </CardDescription>
          </CardHeader>
          <CardContent>
            {readyTasks.length === 0 ? (
              <div className="flex h-24 items-center justify-center rounded-lg border border-dashed text-muted-foreground">
                <div className="text-center">
                  <List className="mx-auto h-8 w-8 mb-2 opacity-50" />
                  <p>No tasks in ready queue</p>
                  <p className="text-sm">All pending tasks are waiting for dependencies</p>
                </div>
              </div>
            ) : (
              <ScrollArea className="h-[200px]">
                <div className="space-y-2">
                  {readyTasks.map((task, index) => (
                    <div
                      key={task.id}
                      onClick={() => setSelectedTask(task)}
                      className="flex cursor-pointer items-center justify-between rounded-lg border border-purple-100 bg-purple-50/50 p-3 transition-all hover:border-purple-300 hover:bg-purple-100/50 dark:border-purple-900 dark:bg-purple-950/30 dark:hover:border-purple-700"
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-purple-200 text-sm font-medium text-purple-700 dark:bg-purple-800 dark:text-purple-300">
                          {index + 1}
                        </div>
                        <div>
                          <p className="font-medium">{task.tool.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {truncateId(task.id)} • Priority: {task.priority}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                          {task.task_type}
                        </Badge>
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </CardContent>
        </Card>
      )}

      {/* Blocked Tasks Panel */}
      {showBlockedTasks && (
        <Card className="border-orange-200 dark:border-orange-800">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Ban className="h-5 w-5 text-orange-600 dark:text-orange-400" />
                <CardTitle>Blocked Tasks</CardTitle>
              </div>
              <Badge variant="secondary" className="bg-orange-100 text-orange-700 dark:bg-orange-900/50 dark:text-orange-300">
                {blockedTasks.length} tasks blocked
              </Badge>
            </div>
            <CardDescription>
              Tasks that cannot run because one or more of their dependencies have failed
            </CardDescription>
          </CardHeader>
          <CardContent>
            {blockedTasks.length === 0 ? (
              <div className="flex h-24 items-center justify-center rounded-lg border border-dashed text-muted-foreground">
                <div className="text-center">
                  <AlertTriangle className="mx-auto h-8 w-8 mb-2 opacity-50" />
                  <p>No blocked tasks</p>
                  <p className="text-sm">All pending tasks can eventually run</p>
                </div>
              </div>
            ) : (
              <ScrollArea className="h-[200px]">
                <div className="space-y-2">
                  {blockedTasks.map((task) => (
                    <div
                      key={task.id}
                      onClick={() => setSelectedTask(task)}
                      className="flex cursor-pointer items-center justify-between rounded-lg border border-orange-100 bg-orange-50/50 p-3 transition-all hover:border-orange-300 hover:bg-orange-100/50 dark:border-orange-900 dark:bg-orange-950/30 dark:hover:border-orange-700"
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-orange-200 dark:bg-orange-800">
                          <Ban className="h-4 w-4 text-orange-700 dark:text-orange-300" />
                        </div>
                        <div>
                          <p className="font-medium">{task.tool.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {truncateId(task.id)} • {task.dependency_ids.length} dependencies
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="border-orange-300 text-orange-700 text-xs dark:border-orange-700 dark:text-orange-300">
                          blocked
                        </Badge>
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </CardContent>
        </Card>
      )}

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex flex-1 gap-2">
              <div className="relative flex-1 sm:max-w-xs">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search tasks..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            <div className="flex gap-2">
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-[140px]">
                  <Filter className="mr-2 h-4 w-4" />
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                  <SelectItem value="running">Running</SelectItem>
                  <SelectItem value="completed">Completed</SelectItem>
                  <SelectItem value="failed">Failed</SelectItem>
                </SelectContent>
              </Select>

              <Select value={sortBy} onValueChange={(v) => setSortBy(v as typeof sortBy)}>
                <SelectTrigger className="w-[140px]">
                  <ArrowUpDown className="mr-2 h-4 w-4" />
                  <SelectValue placeholder="Sort by" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="priority">Priority</SelectItem>
                  <SelectItem value="status">Status</SelectItem>
                  <SelectItem value="name">Name</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Task List */}
      <Card>
        <CardHeader>
          <CardTitle>Task List</CardTitle>
          <CardDescription>
            {filteredTasks.length} of {tasks.length} tasks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[500px]">
            <div className="space-y-2">
              {filteredTasks.map((task) => (
                <div
                  key={task.id}
                  onClick={() => setSelectedTask(task)}
                  className="flex cursor-pointer items-center justify-between rounded-lg border p-4 transition-all hover:border-primary hover:bg-muted/50"
                >
                  <div className="flex items-center gap-4">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                      <Box className="h-5 w-5" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="font-medium">{task.tool.name}</p>
                        {task.tool.is_baseline && (
                          <Badge variant="outline" className="text-xs">Baseline</Badge>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {truncateId(task.id)} • Priority: {task.priority}
                        {task.worker_id && (
                          <> • Worker: <span className="font-medium">{task.worker_id}</span></>
                        )}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    <div className="hidden text-right sm:block">
                      <p className="text-sm">
                        <span className="text-muted-foreground">Workflows:</span>{' '}
                        {task.workflow_names.length > 0
                          ? task.workflow_names.slice(0, 2).join(', ') + (task.workflow_names.length > 2 ? '...' : '')
                          : task.workflows.length}
                      </p>
                      <p className="text-sm">
                        <span className="text-muted-foreground">Dependencies:</span> {task.dependency_ids.length}
                      </p>
                      {task.worker_id && (
                        <p className="text-xs text-blue-600 dark:text-blue-400">
                          Executed by: {task.worker_id}
                        </p>
                      )}
                      {task.cache_hit && (
                        <p className="text-xs text-green-600 dark:text-green-400 flex items-center gap-1">
                          <Layers className="h-3 w-3" />
                          Cache used
                        </p>
                      )}
                    </div>
                    <StatusBadge status={task.status} />
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                </div>
              ))}

              {filteredTasks.length === 0 && (
                <div className="flex h-32 items-center justify-center text-muted-foreground">
                  No tasks found
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Task Detail Dialog */}
      <Dialog open={!!selectedTask} onOpenChange={() => setSelectedTask(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Box className="h-5 w-5" />
              {selectedTask?.tool.name}
            </DialogTitle>
            <DialogDescription>
              Task ID: {selectedTask?.id}
            </DialogDescription>
          </DialogHeader>

          {selectedTask && (
            <Tabs defaultValue="details" className="mt-4">
              <TabsList>
                <TabsTrigger value="details">Details</TabsTrigger>
                <TabsTrigger value="config">Config</TabsTrigger>
                <TabsTrigger value="dependencies">Dependencies</TabsTrigger>
                {selectedTask.status === 'failed' && (
                  <TabsTrigger value="logs" className="text-red-600 dark:text-red-400">
                    Error Logs
                  </TabsTrigger>
                )}
              </TabsList>

              <TabsContent value="details" className="space-y-4">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Status</p>
                    <StatusBadge status={selectedTask.status} showIcon={true} />
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Priority</p>
                    <p className="font-medium">{selectedTask.priority}</p>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Task Type</p>
                    <Badge variant="outline">{selectedTask.task_type}</Badge>
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Usage Count</p>
                    <p className="font-medium">{selectedTask.counter}</p>
                  </div>
                  {selectedTask.worker_id && (
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Executed By</p>
                      <p className="font-medium">{selectedTask.worker_id}</p>
                    </div>
                  )}
                  {selectedTask.execution_time_ms && (
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Execution Time</p>
                      <p className="font-medium">{selectedTask.execution_time_ms}ms</p>
                    </div>
                  )}
                  {selectedTask.cache_hit && (
                    <div className="space-y-1 sm:col-span-2">
                      <p className="text-sm text-muted-foreground">Cache</p>
                      <div className="flex items-center gap-2 rounded-lg bg-green-50 p-2 dark:bg-green-900/20">
                        <Layers className="h-4 w-4 text-green-600 dark:text-green-400" />
                        <span className="text-sm font-medium text-green-700 dark:text-green-300">
                          Cache hit
                        </span>
                        {selectedTask.cache_key && (
                          <code className="ml-auto text-xs text-muted-foreground">
                            {truncateId(selectedTask.cache_key, 12)}
                          </code>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Input/Configuration</p>
                  <div className="rounded-lg bg-muted p-3">
                    <pre className="text-xs overflow-x-auto">
                      {JSON.stringify(selectedTask.config, null, 2)}
                    </pre>
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Container</p>
                  <div className="rounded-lg bg-muted p-3">
                    <p className="text-sm">
                      <span className="text-muted-foreground">Image:</span>{' '}
                      <code className="rounded bg-background px-1">{selectedTask.tool.container.image}</code>
                    </p>
                    <p className="mt-1 text-sm">
                      <span className="text-muted-foreground">Command:</span>{' '}
                      <code className="rounded bg-background px-1">{selectedTask.tool.container.command}</code>
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">Workflows ({selectedTask.workflows.length})</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedTask.workflow_names.length > 0
                      ? selectedTask.workflow_names.map((wfName, idx) => (
                          <Badge key={selectedTask.workflows[idx] || idx} variant="secondary">
                            <GitBranch className="mr-1 h-3 w-3" />
                            {wfName}
                          </Badge>
                        ))
                      : selectedTask.workflows.map((wf) => (
                          <Badge key={wf} variant="secondary">
                            <GitBranch className="mr-1 h-3 w-3" />
                            {truncateId(wf)}
                          </Badge>
                        ))}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="config" className="space-y-4">
                <div className="space-y-2">
                  <p className="text-sm font-medium">Task Configuration</p>
                  <p className="text-xs text-muted-foreground">
                    This is the input/configuration passed to the task
                  </p>
                </div>
                <ScrollArea className="h-[300px]">
                  <pre className="rounded-lg bg-muted p-4 text-sm">
                    {JSON.stringify(selectedTask.config, null, 2)}
                  </pre>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="dependencies" className="space-y-4">
                {taskPriority && (
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Dependency Level</p>
                      <p className="font-medium">{taskPriority.dependency_level}</p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Usage Counter</p>
                      <p className="font-medium">{taskPriority.usage_counter}</p>
                    </div>
                  </div>
                )}

                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Dependencies ({selectedTask.dependency_ids.length})
                  </p>
                  {selectedTask.dependency_ids.length > 0 ? (
                    <div className="space-y-2">
                      {selectedTask.dependency_ids.map((depId) => (
                        <div
                          key={depId}
                          className="flex items-center justify-between rounded-lg border p-3"
                        >
                          <code className="text-sm">{truncateId(depId, 16)}</code>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No dependencies</p>
                  )}
                </div>
              </TabsContent>

              {selectedTask.status === 'failed' && (
                <TabsContent value="logs" className="space-y-4">
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-red-600 dark:text-red-400">Error Information</p>
                    {selectedTask.error_message && (
                      <div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900/50 dark:bg-red-900/10">
                        <p className="text-sm font-medium text-red-800 dark:text-red-200 mb-2">Error Message:</p>
                        <pre className="text-xs text-red-700 dark:text-red-300 whitespace-pre-wrap overflow-x-auto">
                          {selectedTask.error_message}
                        </pre>
                      </div>
                    )}
                    {taskLogs?.logs && (
                      <div className="rounded-lg border border-orange-200 bg-orange-50 p-4 dark:border-orange-900/50 dark:bg-orange-900/10">
                        <p className="text-sm font-medium text-orange-800 dark:text-orange-200 mb-2">Execution Logs:</p>
                        <ScrollArea className="h-[300px]">
                          <pre className="text-xs text-orange-700 dark:text-orange-300 whitespace-pre-wrap overflow-x-auto">
                            {taskLogs.logs}
                          </pre>
                        </ScrollArea>
                      </div>
                    )}
                    {!selectedTask.error_message && !taskLogs?.logs && (
                      <div className="rounded-lg border border-dashed p-4 text-center text-muted-foreground">
                        <AlertTriangle className="mx-auto h-8 w-8 mb-2 opacity-50" />
                        <p>No error logs available</p>
                        <p className="text-xs mt-1">Logs may not have been captured or task failed before execution</p>
                      </div>
                    )}
                  </div>
                </TabsContent>
              )}
            </Tabs>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
