import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { StatsCard } from '@/components/StatsCard';
import { StatusBadge } from '@/components/StatusBadge';
import { ProgressRing } from '@/components/ProgressRing';
import {
  getPipelineDetail,
  getProgress,
  getWorkers,
  getAllTasks,
  getRunningTasks,
  getSchedulerStatus,
  getReadyTasks,
  resetScheduler,
  reclaimStaleTasks,
} from '@/lib/api';
import { formatDuration, formatRelativeTime, truncateId } from '@/lib/utils';
import {
  Activity,
  CheckCircle2,
  Clock,
  AlertCircle,
  Users,
  Layers,
  PlayCircle,
  RotateCcw,
  Loader2,
  ArrowRight,
  Zap,
  Cpu,
  Circle,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

const TASK_TYPE_LABELS: Record<string, string> = {
  pre_training: 'Pre-training',
  in_training: 'During-training',
  during_training: 'During-training',
  post_training: 'Post-training',
  deployment: 'Deployment',
  evaluation: 'Evaluation',
};

export function Dashboard() {
  const { data: pipeline, isLoading: pipelineLoading, isFetching: pipelineFetching } = useQuery({
    queryKey: ['pipeline'],
    queryFn: getPipelineDetail,
    refetchInterval: 5_000,
  });

  const { data: progress, isFetching: progressFetching } = useQuery({
    queryKey: ['progress'],
    queryFn: getProgress,
    refetchInterval: 3_000,
  });

  const { data: workers } = useQuery({
    queryKey: ['workers'],
    queryFn: getWorkers,
    refetchInterval: 5_000,
  });

  const { data: runningTasksData } = useQuery({
    queryKey: ['running-tasks'],
    queryFn: getRunningTasks,
    refetchInterval: 3_000,
  });

  const { data: completedTasksData } = useQuery({
    queryKey: ['completed-tasks'],
    queryFn: () => getAllTasks('completed'),
    refetchInterval: 5_000,
  });

  const { data: scheduler, isLoading: schedulerLoading } = useQuery({
    queryKey: ['scheduler-status'],
    queryFn: getSchedulerStatus,
    refetchInterval: 15_000,
  });

  const { data: readyTasks } = useQuery({
    queryKey: ['ready-tasks'],
    queryFn: getReadyTasks,
    refetchInterval: 5_000,
  });

  const handleReset = async () => {
    if (confirm('Are you sure you want to reset the scheduler? All progress will be lost.')) {
      await resetScheduler();
    }
  };

  const handleReclaim = async () => {
    const result = await reclaimStaleTasks();
    alert(`Reclaimed ${result.reclaimed_count} stale task(s) back to pending.`);
  };

  const isRefreshing = pipelineFetching || progressFetching;

  if (pipelineLoading && !pipeline && schedulerLoading && !scheduler) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (scheduler && !scheduler.initialized) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <Card className="max-w-md">
          <CardHeader className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-yellow-100 dark:bg-yellow-900/30">
              <AlertCircle className="h-8 w-8 text-yellow-600 dark:text-yellow-400" />
            </div>
            <CardTitle>Scheduler Not Initialized</CardTitle>
            <CardDescription>
              The scheduler needs to be initialized with a pipeline before you can use the dashboard.
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center">
            <p className="text-sm text-muted-foreground">
              Start the backend with a pipeline configuration to begin.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Chart data
  const statusData = [
    { name: 'Completed', value: progress?.completed || 0, color: '#22c55e' },
    { name: 'Running', value: progress?.running || 0, color: '#3b82f6' },
    { name: 'Pending', value: progress?.pending || 0, color: '#eab308' },
    { name: 'Failed', value: progress?.failed || 0, color: '#ef4444' },
  ].filter(d => d.value > 0);

  // Live activity: running tasks first, then last 5 completed
  const runningTasks = runningTasksData?.tasks || [];
  const recentCompleted = (completedTasksData?.tasks || []).slice(-5).reverse();
  const activityItems = [
    ...runningTasks,
    ...recentCompleted.slice(0, Math.max(0, 5 - runningTasks.length)),
  ];

  // Stage breakdown from running tasks
  const stageRunningCounts: Record<string, number> = {};
  for (const t of runningTasks) {
    const label = TASK_TYPE_LABELS[t.task_type] ?? t.task_type;
    stageRunningCounts[label] = (stageRunningCounts[label] || 0) + 1;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
            {isRefreshing && (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            )}
          </div>
          <p className="text-muted-foreground">
            Pipeline: <span className="font-medium text-foreground">{pipeline?.name || 'Unknown'}</span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleReclaim}>
            <Zap className="mr-2 h-4 w-4" />
            Reclaim Stale
          </Button>
          <Button variant="outline" size="sm" onClick={handleReset}>
            <RotateCcw className="mr-2 h-4 w-4" />
            Reset
          </Button>
          {scheduler?.started_at && (
            <Badge variant="secondary" className="gap-1">
              <Clock className="h-3 w-3" />
              Started {formatRelativeTime(scheduler.started_at)}
            </Badge>
          )}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="Total Tasks"
          value={progress?.total || 0}
          subtitle={`${pipeline?.workflow_count || 0} workflows`}
          icon={Layers}
          iconClassName="bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400"
        />
        <StatsCard
          title="Completed"
          value={progress?.completed || 0}
          subtitle={`${progress?.progress_percent?.toFixed(1) || 0}% done`}
          icon={CheckCircle2}
          iconClassName="bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400"
        />
        <StatsCard
          title="Running"
          value={progress?.running || 0}
          subtitle={`${readyTasks?.total || 0} ready in queue`}
          icon={PlayCircle}
          iconClassName="bg-purple-100 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400"
        />
        <StatsCard
          title="Active Workers"
          value={workers?.active || 0}
          subtitle={`${workers?.total || 0} registered`}
          icon={Users}
          iconClassName="bg-orange-100 text-orange-600 dark:bg-orange-900/30 dark:text-orange-400"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Progress Overview */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Pipeline Progress
            </CardTitle>
            <CardDescription>
              Real-time overview of task execution status
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col gap-6 md:flex-row md:items-center">
              {/* Progress Ring */}
              <div className="flex justify-center md:justify-start">
                <ProgressRing progress={progress?.progress_percent || 0} size={160} strokeWidth={12} />
              </div>

              {/* Status Breakdown */}
              <div className="flex-1 space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <div className="h-3 w-3 rounded-full bg-green-500" />
                      Completed
                    </span>
                    <span className="font-medium">{progress?.completed || 0}</span>
                  </div>
                  <Progress
                    value={((progress?.completed || 0) / (progress?.total || 1)) * 100}
                    className="h-2"
                    indicatorClassName="bg-green-500"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <div className="h-3 w-3 rounded-full bg-blue-500" />
                      Running
                    </span>
                    <span className="font-medium">{progress?.running || 0}</span>
                  </div>
                  <Progress
                    value={((progress?.running || 0) / (progress?.total || 1)) * 100}
                    className="h-2"
                    indicatorClassName="bg-blue-500"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <div className="h-3 w-3 rounded-full bg-yellow-500" />
                      Pending
                    </span>
                    <span className="font-medium">{progress?.pending || 0}</span>
                  </div>
                  <Progress
                    value={((progress?.pending || 0) / (progress?.total || 1)) * 100}
                    className="h-2"
                    indicatorClassName="bg-yellow-500"
                  />
                </div>

                {(progress?.failed || 0) > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="flex items-center gap-2">
                        <div className="h-3 w-3 rounded-full bg-red-500" />
                        Failed
                      </span>
                      <span className="font-medium">{progress?.failed || 0}</span>
                    </div>
                    <Progress
                      value={((progress?.failed || 0) / (progress?.total || 1)) * 100}
                      className="h-2"
                      indicatorClassName="bg-red-500"
                    />
                  </div>
                )}
              </div>
            </div>

            {/* Stage breakdown (only when tasks are running) */}
            {Object.keys(stageRunningCounts).length > 0 && (
              <div className="mt-4 flex flex-wrap gap-2">
                {Object.entries(stageRunningCounts).map(([stage, count]) => (
                  <Badge key={stage} variant="secondary" className="gap-1">
                    <span className="h-2 w-2 rounded-full bg-blue-500 inline-block" />
                    {count} {stage}
                  </Badge>
                ))}
              </div>
            )}

            {/* Time Stats */}
            {pipeline?.running_time_seconds && (
              <div className="mt-6 flex items-center justify-between rounded-lg bg-muted/50 p-4">
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Running time</span>
                </div>
                <span className="font-medium">{formatDuration(pipeline.running_time_seconds)}</span>
                {pipeline.estimated_remaining_seconds && (
                  <>
                    <Separator orientation="vertical" className="h-4" />
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">Est. remaining</span>
                    </div>
                    <span className="font-medium">{formatDuration(pipeline.estimated_remaining_seconds)}</span>
                  </>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Status Distribution Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Status Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={statusData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {statusData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value, name) => [value, name]}
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-2">
              {statusData.map((item) => (
                <div key={item.name} className="flex items-center gap-2 text-sm">
                  <div
                    className="h-3 w-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-muted-foreground">{item.name}</span>
                  <span className="ml-auto font-medium">{item.value}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Bottom Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Live Activity Feed */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-base">Live Activity</CardTitle>
              <CardDescription>
                {runningTasks.length > 0
                  ? `${runningTasks.length} running now`
                  : 'Latest task executions'}
              </CardDescription>
            </div>
            <Button variant="ghost" size="sm" asChild>
              <Link to="/tasks">
                View all
                <ArrowRight className="ml-1 h-4 w-4" />
              </Link>
            </Button>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[280px]">
              {activityItems.length > 0 ? (
                <div className="space-y-3">
                  {activityItems.map((task) => (
                    <div
                      key={task.id}
                      className="flex items-center justify-between rounded-lg border p-3 transition-colors hover:bg-muted/50"
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-muted">
                          <Layers className="h-4 w-4" />
                        </div>
                        <div>
                          <p className="font-medium">{task.tool.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {TASK_TYPE_LABELS[task.task_type] ?? task.task_type}
                            {' · '}
                            {truncateId(task.id)}
                          </p>
                        </div>
                      </div>
                      <StatusBadge status={task.status} />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  No recent activity
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>

        {/* Live Workers Panel */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-base">
                Workers
                {(workers?.active || 0) > 0 && (
                  <span className="ml-2 text-sm font-normal text-muted-foreground">
                    {workers?.active} active / {workers?.total} registered
                  </span>
                )}
              </CardTitle>
              <CardDescription>Live worker status &amp; GPU assignment</CardDescription>
            </div>
            <Button variant="ghost" size="sm" asChild>
              <Link to="/workers">
                View all
                <ArrowRight className="ml-1 h-4 w-4" />
              </Link>
            </Button>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[280px]">
              {(workers?.workers || []).length > 0 ? (
                <div className="space-y-2">
                  {(workers?.workers || []).map((worker) => {
                    const gpuId = worker.capabilities
                      ? (worker.capabilities['gpu_id'] ?? worker.capabilities['gpu'] ?? null)
                      : null;
                    const statusColor =
                      worker.status === 'busy'
                        ? 'bg-blue-500'
                        : worker.status === 'idle'
                        ? 'bg-green-500'
                        : 'bg-gray-400';
                    return (
                      <div
                        key={worker.worker_id}
                        className="flex items-center gap-3 rounded-lg border p-3"
                      >
                        <Circle
                          className={`h-3 w-3 shrink-0 fill-current ${statusColor} text-transparent`}
                        />
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <p className="truncate text-sm font-medium">
                              {truncateId(worker.worker_id)}
                            </p>
                            {gpuId !== null && (
                              <Badge variant="outline" className="gap-1 shrink-0 py-0 text-xs">
                                <Cpu className="h-3 w-3" />
                                GPU {String(gpuId)}
                              </Badge>
                            )}
                          </div>
                          {worker.status === 'busy' && worker.current_task_id ? (
                            <p className="truncate text-xs text-muted-foreground">
                              Running: {truncateId(worker.current_task_id)}
                            </p>
                          ) : (
                            <p className="text-xs text-muted-foreground capitalize">{worker.status}</p>
                          )}
                        </div>
                        <div className="shrink-0 text-right">
                          <p className="text-sm font-medium">{worker.tasks_completed}</p>
                          <p className="text-xs text-muted-foreground">done</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-2 text-muted-foreground">
                  <Users className="h-8 w-8 opacity-40" />
                  <p className="text-sm">No workers registered</p>
                  <p className="text-xs">Start workers to begin processing tasks</p>
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
