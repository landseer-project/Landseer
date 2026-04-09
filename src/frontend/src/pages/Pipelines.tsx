import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { StatusBadge } from '@/components/StatusBadge';
import {
  getPipelineConfigs,
  getPipelineRuns,
  startPipelineRun,
  stopPipelineRun,
} from '@/lib/api';
import { formatTimestamp, formatRelativeTime, truncateId } from '@/lib/utils';
import {
  Play,
  Square,
  Loader2,
  FolderOpen,
  Clock,
  Hash,
  ChevronDown,
  ChevronRight,
  FileText,
  RefreshCw,
  AlertCircle,
} from 'lucide-react';
import type { PipelineConfig, PipelineRun } from '@/types/api';

function RunDuration({ run }: { run: PipelineRun }) {
  if (!run.started_at) return <span className="text-muted-foreground">--</span>;
  const start = new Date(run.started_at).getTime();
  const end = run.completed_at ? new Date(run.completed_at).getTime() : Date.now();
  const secs = Math.floor((end - start) / 1000);
  if (secs < 60) return <span>{secs}s</span>;
  if (secs < 3600) return <span>{Math.floor(secs / 60)}m {secs % 60}s</span>;
  return <span>{Math.floor(secs / 3600)}h {Math.floor((secs % 3600) / 60)}m</span>;
}

function ConfigCard({
  config,
  runs,
  onStartRun,
  onStopRun,
  isStarting,
}: {
  config: PipelineConfig;
  runs: PipelineRun[];
  onStartRun: (configId: string) => void;
  onStopRun: (runId: string) => void;
  isStarting: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const configRuns = runs.filter((r) => r.pipeline_config_id === config.id);
  const activeRun = configRuns.find((r) => r.status === 'running' || r.status === 'pending' || r.status === 'stopping');
  const lastRun = configRuns[0];

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
              <FileText className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle className="text-lg">{config.name}</CardTitle>
              <CardDescription className="text-xs font-mono">
                {config.config_path.split('/').slice(-2).join('/')}
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {activeRun ? (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => onStopRun(activeRun.id)}
              >
                <Square className="mr-2 h-4 w-4" />
                Stop
              </Button>
            ) : (
              <Button
                size="sm"
                onClick={() => onStartRun(config.id)}
                disabled={isStarting}
              >
                {isStarting ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <Play className="mr-2 h-4 w-4" />
                )}
                Start Run
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-1">
            <Hash className="h-3.5 w-3.5" />
            {configRuns.length} run{configRuns.length !== 1 ? 's' : ''}
          </div>
          {lastRun && (
            <>
              <Separator orientation="vertical" className="h-4" />
              <div className="flex items-center gap-1">
                Last: <StatusBadge status={lastRun.status} />
              </div>
              <Separator orientation="vertical" className="h-4" />
              <div className="flex items-center gap-1">
                <Clock className="h-3.5 w-3.5" />
                {formatRelativeTime(lastRun.created_at)}
              </div>
            </>
          )}
        </div>

        {configRuns.length > 0 && (
          <>
            <Separator className="my-3" />
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex w-full items-center gap-1 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              {expanded ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
              Run History
            </button>

            {expanded && (
              <div className="mt-3 space-y-2">
                {configRuns.map((run) => (
                  <div
                    key={run.id}
                    className="flex items-center justify-between rounded-lg border p-3 text-sm"
                  >
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className="font-mono text-xs">
                        #{run.run_number}
                      </Badge>
                      <StatusBadge status={run.status} />
                      <span className="text-muted-foreground font-mono text-xs">
                        {truncateId(run.id, 16)}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span>{formatTimestamp(run.created_at)}</span>
                      <span className="font-medium">
                        <RunDuration run={run} />
                      </span>
                      {run.error_message && (
                        <span className="text-destructive max-w-[200px] truncate" title={run.error_message}>
                          {run.error_message}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

export function Pipelines() {
  const queryClient = useQueryClient();

  const { data: configsData, isLoading: configsLoading } = useQuery({
    queryKey: ['pipeline-configs'],
    queryFn: getPipelineConfigs,
    refetchInterval: 10_000,
  });

  const { data: runsData, isLoading: runsLoading } = useQuery({
    queryKey: ['pipeline-runs'],
    queryFn: () => getPipelineRuns(),
    refetchInterval: 5_000,
  });

  const startMutation = useMutation({
    mutationFn: (configId: string) => startPipelineRun(configId, { use_cache: true }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipeline-runs'] });
      queryClient.invalidateQueries({ queryKey: ['pipeline-configs'] });
      queryClient.invalidateQueries({ queryKey: ['scheduler-status'] });
      queryClient.invalidateQueries({ queryKey: ['progress'] });
    },
  });

  const stopMutation = useMutation({
    mutationFn: (runId: string) => stopPipelineRun(runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipeline-runs'] });
      queryClient.invalidateQueries({ queryKey: ['scheduler-status'] });
    },
  });

  const configs = configsData?.configs ?? [];
  const runs = runsData?.runs ?? [];
  const activeRun = runs.find(
    (r) => r.status === 'running' || r.status === 'pending'
  );

  if (configsLoading && runsLoading) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading pipeline configs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Pipelines</h1>
          <p className="text-muted-foreground">
            {configs.length} config{configs.length !== 1 ? 's' : ''} discovered
            {activeRun && (
              <span className="ml-2">
                 &middot; <span className="text-blue-500 font-medium">1 active run</span>
              </span>
            )}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            queryClient.invalidateQueries({ queryKey: ['pipeline-configs'] });
            queryClient.invalidateQueries({ queryKey: ['pipeline-runs'] });
          }}
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>
      </div>

      {/* Error display */}
      {startMutation.isError && (
        <Card className="border-destructive">
          <CardContent className="flex items-center gap-3 py-3">
            <AlertCircle className="h-5 w-5 text-destructive shrink-0" />
            <p className="text-sm text-destructive">
              {(startMutation.error as Error)?.message ||
                'Failed to start pipeline run'}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Config cards */}
      {configs.length > 0 ? (
        <div className="space-y-4">
          {configs.map((config) => (
            <ConfigCard
              key={config.id}
              config={config}
              runs={runs}
              onStartRun={(id) => startMutation.mutate(id)}
              onStopRun={(id) => stopMutation.mutate(id)}
              isStarting={startMutation.isPending}
            />
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12 text-center">
            <FolderOpen className="h-12 w-12 text-muted-foreground/40 mb-4" />
            <p className="font-medium">No pipeline configs found</p>
            <p className="text-sm text-muted-foreground mt-1">
              Add YAML files to <code className="text-xs bg-muted px-1 py-0.5 rounded">configs/pipeline/</code> and
              they will be discovered automatically.
            </p>
          </CardContent>
        </Card>
      )}

      {/* All runs table */}
      {runs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">All Runs</CardTitle>
            <CardDescription>
              {runs.length} total run{runs.length !== 1 ? 's' : ''} across all configs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="max-h-[400px]">
              <div className="space-y-2">
                {runs.map((run) => {
                  const cfg = configs.find((c) => c.id === run.pipeline_config_id);
                  return (
                    <div
                      key={run.id}
                      className="flex items-center justify-between rounded-lg border p-3 text-sm"
                    >
                      <div className="flex items-center gap-3">
                        <Badge variant="outline" className="font-mono text-xs">
                          #{run.run_number}
                        </Badge>
                        <span className="font-medium">{cfg?.name ?? run.pipeline_config_id}</span>
                        <StatusBadge status={run.status} />
                      </div>
                      <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        <span className="font-mono">{truncateId(run.id, 20)}</span>
                        <span>{formatTimestamp(run.started_at || run.created_at)}</span>
                        <span className="font-medium">
                          <RunDuration run={run} />
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
