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
  getSchedulerStatus,
  getReadyTasks,
  resetScheduler,
} from '@/lib/api';
import { formatDuration, formatRelativeTime, truncateId } from '@/lib/utils';
import {
  Activity,
  CheckCircle2,
  Clock,
  AlertCircle,
  Users,
  Layers,
  GitBranch,
  PlayCircle,
  RotateCcw,
  Loader2,
  ArrowRight,
  Zap,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';

export function Dashboard() {
  const { data: pipeline, isLoading: pipelineLoading, isFetching: pipelineFetching } = useQuery({
    queryKey: ['pipeline'],
    queryFn: getPipelineDetail,
  });

  const { data: progress, isFetching: progressFetching } = useQuery({
    queryKey: ['progress'],
    queryFn: getProgress,
  });

  const { data: workers } = useQuery({
    queryKey: ['workers'],
    queryFn: getWorkers,
  });

  const { data: tasks } = useQuery({
    queryKey: ['tasks'],
    queryFn: () => getAllTasks(),
  });

  const { data: scheduler, isLoading: schedulerLoading } = useQuery({
    queryKey: ['scheduler-status'],
    queryFn: getSchedulerStatus,
  });

  const { data: readyTasks } = useQuery({
    queryKey: ['ready-tasks'],
    queryFn: getReadyTasks,
  });

  const handleReset = async () => {
    if (confirm('Are you sure you want to reset the scheduler? All progress will be lost.')) {
      await resetScheduler();
    }
  };

  const isRefreshing = pipelineFetching || progressFetching;

  // Only show full loading screen on initial load (no data yet)
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

  // Recent tasks
  const recentTasks = tasks?.tasks
    .filter(t => t.status !== 'pending')
    .slice(0, 5) || [];

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
        {/* Recent Activity */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-base">Recent Activity</CardTitle>
              <CardDescription>Latest task executions</CardDescription>
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
              {recentTasks.length > 0 ? (
                <div className="space-y-3">
                  {recentTasks.map((task) => (
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

        {/* Quick Links / Workflows */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-base">Workflows</CardTitle>
              <CardDescription>{pipeline?.workflow_count || 0} workflows in pipeline</CardDescription>
            </div>
            <Button variant="ghost" size="sm" asChild>
              <Link to="/workflows">
                View all
                <ArrowRight className="ml-1 h-4 w-4" />
              </Link>
            </Button>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-2">
              <Link
                to="/tasks"
                className="flex items-center gap-3 rounded-lg border p-4 transition-all hover:border-primary hover:shadow-sm"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400">
                  <Layers className="h-5 w-5" />
                </div>
                <div>
                  <p className="font-medium">Tasks</p>
                  <p className="text-sm text-muted-foreground">{progress?.total || 0} total</p>
                </div>
              </Link>

              <Link
                to="/workflows"
                className="flex items-center gap-3 rounded-lg border p-4 transition-all hover:border-primary hover:shadow-sm"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-purple-100 text-purple-600 dark:bg-purple-900/30 dark:text-purple-400">
                  <GitBranch className="h-5 w-5" />
                </div>
                <div>
                  <p className="font-medium">Workflows</p>
                  <p className="text-sm text-muted-foreground">{pipeline?.workflow_count || 0} defined</p>
                </div>
              </Link>

              <Link
                to="/workers"
                className="flex items-center gap-3 rounded-lg border p-4 transition-all hover:border-primary hover:shadow-sm"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400">
                  <Users className="h-5 w-5" />
                </div>
                <div>
                  <p className="font-medium">Workers</p>
                  <p className="text-sm text-muted-foreground">{workers?.active || 0} active</p>
                </div>
              </Link>

              <Link
                to="/tools"
                className="flex items-center gap-3 rounded-lg border p-4 transition-all hover:border-primary hover:shadow-sm"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-orange-100 text-orange-600 dark:bg-orange-900/30 dark:text-orange-400">
                  <Zap className="h-5 w-5" />
                </div>
                <div>
                  <p className="font-medium">Tools</p>
                  <p className="text-sm text-muted-foreground">Manage tools</p>
                </div>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
