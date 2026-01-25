import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { StatusBadge } from '@/components/StatusBadge';
import { getWorkflowDetail, getWorkflowResults } from '@/lib/api';
import { truncateId, formatDuration } from '@/lib/utils';
import {
  ArrowLeft,
  GitBranch,
  Layers,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  AlertTriangle,
  Box,
} from 'lucide-react';

export function WorkflowDetail() {
  const { id } = useParams<{ id: string }>();

  const { data: workflow, isLoading, isFetching } = useQuery({
    queryKey: ['workflow-detail', id],
    queryFn: () => getWorkflowDetail(id!),
    enabled: !!id,
  });

  const { data: results } = useQuery({
    queryKey: ['workflow-results', id],
    queryFn: () => getWorkflowResults(id!),
    enabled: !!id,
  });

  const isRefreshing = isFetching && !isLoading;

  // Only show full loading on initial load
  if (isLoading && !workflow) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!workflow) {
    return (
      <div className="flex h-[60vh] flex-col items-center justify-center gap-4">
        <AlertTriangle className="h-12 w-12 text-yellow-500" />
        <p className="text-lg text-muted-foreground">Workflow not found</p>
        <Button asChild>
          <Link to="/workflows">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Workflows
          </Link>
        </Button>
      </div>
    );
  }

  const progress = (workflow.completed_tasks / workflow.task_count) * 100;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500';
      case 'running':
        return 'bg-blue-500';
      case 'failed':
        return 'bg-red-500';
      default:
        return 'bg-yellow-500';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" asChild>
            <Link to="/workflows">
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          <div>
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900/30">
                <GitBranch className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-2xl font-bold tracking-tight">{workflow.name}</h1>
                  {isRefreshing && (
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                  )}
                </div>
                <p className="text-sm text-muted-foreground">
                  {truncateId(workflow.id, 20)}
                </p>
              </div>
            </div>
          </div>
        </div>
        <StatusBadge status={workflow.status} />
      </div>

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-4">
        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/30">
              <Layers className="h-5 w-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-xl font-bold">{workflow.task_count}</p>
              <p className="text-xs text-muted-foreground">Total Tasks</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-green-100 dark:bg-green-900/30">
              <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-xl font-bold">{workflow.completed_tasks}</p>
              <p className="text-xs text-muted-foreground">Completed</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-red-100 dark:bg-red-900/30">
              <XCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
            </div>
            <div>
              <p className="text-xl font-bold">{workflow.failed_tasks}</p>
              <p className="text-xs text-muted-foreground">Failed</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Progress</span>
                <span className="font-medium">{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="tasks">
        <TabsList>
          <TabsTrigger value="tasks">Tasks ({workflow.task_count})</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
          {workflow.failure_reasons.length > 0 && (
            <TabsTrigger value="errors" className="text-red-500">
              Errors ({workflow.failure_reasons.length})
            </TabsTrigger>
          )}
        </TabsList>

        <TabsContent value="tasks" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Task Pipeline</CardTitle>
              <CardDescription>
                Visual representation of tasks in this workflow
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px]">
                <div className="space-y-3">
                  {workflow.tasks.map((task, index) => (
                    <div key={task.id} className="flex items-start gap-4">
                      {/* Timeline indicator */}
                      <div className="flex flex-col items-center">
                        <div
                          className={`flex h-8 w-8 items-center justify-center rounded-full ${getStatusColor(
                            task.status
                          )} text-white`}
                        >
                          {task.status === 'completed' ? (
                            <CheckCircle2 className="h-4 w-4" />
                          ) : task.status === 'running' ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : task.status === 'failed' ? (
                            <XCircle className="h-4 w-4" />
                          ) : (
                            <Clock className="h-4 w-4" />
                          )}
                        </div>
                        {index < workflow.tasks.length - 1 && (
                          <div className="h-8 w-0.5 bg-border" />
                        )}
                      </div>

                      {/* Task card */}
                      <div className="flex-1 rounded-lg border p-4 transition-colors hover:bg-muted/50">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <Box className="h-4 w-4 text-muted-foreground" />
                            <div>
                              <p className="font-medium">{task.tool.name}</p>
                              <p className="text-xs text-muted-foreground">
                                {truncateId(task.id)} • Type: {task.task_type}
                              </p>
                            </div>
                          </div>
                          <StatusBadge status={task.status} showIcon={false} />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Execution Results</CardTitle>
              <CardDescription>
                Detailed results for each task execution
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px]">
                <div className="space-y-4">
                  {results?.results.map((result) => (
                    <div
                      key={result.task_id}
                      className="rounded-lg border p-4"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <Box className="h-4 w-4 text-muted-foreground" />
                          <div>
                            <p className="font-medium">{result.tool_name}</p>
                            <p className="text-xs text-muted-foreground">
                              {truncateId(result.task_id)}
                            </p>
                          </div>
                        </div>
                        <StatusBadge status={result.status} />
                      </div>

                      <div className="mt-3 grid gap-2 text-sm sm:grid-cols-3">
                        <div>
                          <span className="text-muted-foreground">Execution time:</span>{' '}
                          <span className="font-medium">
                            {result.execution_time_ms
                              ? `${result.execution_time_ms}ms`
                              : '--'}
                          </span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Worker:</span>{' '}
                          <span className="font-medium">
                            {result.worker_id || '--'}
                          </span>
                        </div>
                        {result.error_message && (
                          <div className="col-span-full">
                            <span className="text-red-500">Error:</span>{' '}
                            <span className="text-red-600">{result.error_message}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )) || (
                    <div className="flex h-32 items-center justify-center text-muted-foreground">
                      No results available
                    </div>
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {workflow.failure_reasons.length > 0 && (
          <TabsContent value="errors" className="mt-4">
            <Card className="border-red-200 dark:border-red-900/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-red-600 dark:text-red-400">
                  <AlertTriangle className="h-5 w-5" />
                  Failed Tasks
                </CardTitle>
                <CardDescription>
                  {workflow.failure_reasons.length} task(s) failed in this workflow
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {workflow.failure_reasons.map((failure) => (
                    <div
                      key={failure.task_id}
                      className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900/50 dark:bg-red-900/10"
                    >
                      <div className="flex items-center gap-2">
                        <XCircle className="h-4 w-4 text-red-500" />
                        <span className="font-medium">{failure.task_name}</span>
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground">
                        ID: {truncateId(failure.task_id)}
                      </p>
                      <Separator className="my-2" />
                      <p className="text-sm text-red-600 dark:text-red-400">
                        {failure.error}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
}
