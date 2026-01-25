import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { getWorkflows, getWorkflowDetail } from '@/lib/api';
import { truncateId } from '@/lib/utils';
import type { WorkflowInfo } from '@/types/api';
import {
  Search,
  GitBranch,
  ChevronRight,
  Loader2,
  Layers,
  CheckCircle2,
  XCircle,
  Clock,
} from 'lucide-react';

export function Workflows() {
  const [searchQuery, setSearchQuery] = useState('');

  const { data: workflowsData, isLoading, isFetching } = useQuery({
    queryKey: ['workflows'],
    queryFn: getWorkflows,
  });

  const workflows = workflowsData?.workflows || [];
  const isRefreshing = isFetching && !isLoading;

  const filteredWorkflows = workflows.filter((wf) => {
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        wf.id.toLowerCase().includes(query) ||
        wf.name.toLowerCase().includes(query)
      );
    }
    return true;
  });

  // Only show full loading on initial load
  if (isLoading && !workflowsData) {
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
          <h1 className="text-3xl font-bold tracking-tight">Workflows</h1>
          {isRefreshing && (
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          )}
        </div>
        <p className="text-muted-foreground">
          View and manage pipeline workflows
        </p>
      </div>

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900/30">
              <GitBranch className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{workflows.length}</p>
              <p className="text-sm text-muted-foreground">Total Workflows</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/30">
              <Layers className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">
                {workflows.reduce((acc, wf) => acc + wf.task_count, 0)}
              </p>
              <p className="text-sm text-muted-foreground">Total Tasks</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-green-100 dark:bg-green-900/30">
              <CheckCircle2 className="h-6 w-6 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">
                {Math.round(
                  (workflows.reduce((acc, wf) => acc + wf.task_count, 0) /
                    Math.max(workflows.length, 1)) *
                    10
                ) / 10}
              </p>
              <p className="text-sm text-muted-foreground">Avg Tasks/Workflow</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <Card>
        <CardContent className="p-4">
          <div className="relative max-w-sm">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search workflows..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
        </CardContent>
      </Card>

      {/* Workflow List */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {filteredWorkflows.map((workflow) => (
          <WorkflowCard key={workflow.id} workflow={workflow} />
        ))}

        {filteredWorkflows.length === 0 && (
          <div className="col-span-full flex h-32 items-center justify-center text-muted-foreground">
            No workflows found
          </div>
        )}
      </div>
    </div>
  );
}

function WorkflowCard({ workflow }: { workflow: WorkflowInfo }) {
  const { data: detail } = useQuery({
    queryKey: ['workflow-detail', workflow.id],
    queryFn: () => getWorkflowDetail(workflow.id),
  });

  const progress = detail
    ? (detail.completed_tasks / detail.task_count) * 100
    : 0;

  const getStatusIcon = () => {
    if (!detail) return <Clock className="h-4 w-4 text-muted-foreground" />;
    switch (detail.status) {
      case 'completed':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'running':
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />;
    }
  };

  return (
    <Link to={`/workflows/${workflow.id}`}>
      <Card className="cursor-pointer transition-all hover:border-primary hover:shadow-md">
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900/30">
                <GitBranch className="h-4 w-4 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <CardTitle className="text-base">{workflow.name}</CardTitle>
                <CardDescription className="text-xs">
                  {truncateId(workflow.id)}
                </CardDescription>
              </div>
            </div>
            {getStatusIcon()}
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Tasks</span>
              <span className="font-medium">{workflow.task_count}</span>
            </div>

            {detail && (
              <>
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Progress</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>

                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-1">
                    <CheckCircle2 className="h-3 w-3 text-green-500" />
                    <span className="text-xs">{detail.completed_tasks}</span>
                  </div>
                  {detail.failed_tasks > 0 && (
                    <div className="flex items-center gap-1">
                      <XCircle className="h-3 w-3 text-red-500" />
                      <span className="text-xs">{detail.failed_tasks}</span>
                    </div>
                  )}
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                </div>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
