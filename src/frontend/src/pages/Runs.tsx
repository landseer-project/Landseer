import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Loader2, RefreshCw } from 'lucide-react';
import { getPipelineConfigs, getPipelineRuns, getRunMetrics } from '@/lib/api';
import { formatTimestamp, truncateId } from '@/lib/utils';

const PAGE_SIZE = 25;

function getWorkflowToolsLabel(workflow: {
  workflow_tools_label?: string;
  workflow_tools?: Record<string, string[]>;
}): string {
  if (workflow.workflow_tools_label && workflow.workflow_tools_label.trim().length > 0) {
    return workflow.workflow_tools_label;
  }
  const parts: string[] = [];
  for (const stage of ['pre', 'in', 'post', 'deploy']) {
    const tools = workflow.workflow_tools?.[stage] ?? [];
    if (tools.length > 0) parts.push(`${stage}: ${tools.join(' -> ')}`);
  }
  return parts.join(' | ');
}

export function Runs() {
  const [selectedRunId, setSelectedRunId] = useState<string>('');
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);

  const {
    data: configsData,
    isLoading: configsLoading,
    refetch: refetchConfigs,
  } = useQuery({
    queryKey: ['pipeline-configs'],
    queryFn: getPipelineConfigs,
    refetchInterval: 15_000,
  });

  const {
    data: runsData,
    isLoading: runsLoading,
    refetch: refetchRuns,
  } = useQuery({
    queryKey: ['pipeline-runs'],
    queryFn: () => getPipelineRuns(),
    refetchInterval: 10_000,
  });

  const runs = runsData?.runs ?? [];
  const configs = configsData?.configs ?? [];

  const sortedRuns = useMemo(() => {
    return [...runs].sort((a, b) => {
      const aTs = Date.parse(a.created_at || '') || 0;
      const bTs = Date.parse(b.created_at || '') || 0;
      return bTs - aTs;
    });
  }, [runs]);

  const selectedRun = useMemo(() => {
    if (selectedRunId) return sortedRuns.find((r) => r.id === selectedRunId) ?? null;
    return sortedRuns[0] ?? null;
  }, [selectedRunId, sortedRuns]);

  const {
    data: metricsData,
    isLoading: metricsLoading,
    isFetching: metricsFetching,
    refetch: refetchMetrics,
  } = useQuery({
    queryKey: ['run-metrics', selectedRun?.id],
    queryFn: () => getRunMetrics(selectedRun!.id),
    enabled: !!selectedRun?.id,
    staleTime: 30_000,
  });

  const filteredWorkflows = useMemo(() => {
    const workflows = metricsData?.workflows ?? [];
    const q = search.trim().toLowerCase();
    if (!q) return workflows;
    return workflows.filter((w) => {
      const label = getWorkflowToolsLabel(w).toLowerCase();
      return (
        w.workflow_name.toLowerCase().includes(q) ||
        w.workflow_id.toLowerCase().includes(q) ||
        label.includes(q)
      );
    });
  }, [metricsData, search]);

  const totalPages = Math.max(1, Math.ceil(filteredWorkflows.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const startIdx = (currentPage - 1) * PAGE_SIZE;
  const endIdx = Math.min(startIdx + PAGE_SIZE, filteredWorkflows.length);
  const visibleWorkflows = filteredWorkflows.slice(startIdx, endIdx);

  const runName = selectedRun
    ? configs.find((c) => c.id === selectedRun.pipeline_config_id)?.name ?? selectedRun.pipeline_config_id
    : '';

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Runs</h1>
          <p className="text-muted-foreground">Analyze completed and active experiment runs in one place</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            refetchConfigs();
            refetchRuns();
            refetchMetrics();
          }}
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Run Selection</CardTitle>
          <CardDescription>Pick a run to inspect all combination metrics and tool sequences</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-[1fr_320px]">
          <Select
            value={selectedRun?.id ?? ''}
            onValueChange={(value) => {
              setSelectedRunId(value);
              setPage(1);
            }}
            disabled={runsLoading || sortedRuns.length === 0}
          >
            <SelectTrigger>
              <SelectValue placeholder={runsLoading ? 'Loading runs...' : 'Select run'} />
            </SelectTrigger>
            <SelectContent>
              {sortedRuns.map((run) => {
                const name = configs.find((c) => c.id === run.pipeline_config_id)?.name ?? run.pipeline_config_id;
                return (
                  <SelectItem key={run.id} value={run.id}>
                    {name} #{run.run_number} · {run.status}
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
          <Input
            placeholder="Search combination/tool sequence..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            disabled={!metricsData}
          />
        </CardContent>
      </Card>

      {configsLoading || runsLoading ? (
        <div className="flex h-40 items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      ) : !selectedRun ? (
        <Card>
          <CardContent className="py-8 text-sm text-muted-foreground">No runs available yet.</CardContent>
        </Card>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Run</p>
                <p className="text-sm font-semibold">{runName} #{selectedRun.run_number}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Status</p>
                <Badge variant={selectedRun.status === 'completed' ? 'default' : 'secondary'}>
                  {selectedRun.status}
                </Badge>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Created</p>
                <p className="text-xs">{formatTimestamp(selectedRun.created_at)}</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-muted-foreground">Run ID</p>
                <p className="text-xs font-mono">{truncateId(selectedRun.id, 24)}</p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <CardTitle>Combination Metrics</CardTitle>
                {metricsFetching && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
              </div>
              <CardDescription>
                Tool sequence is shown per combination. Table supports horizontal scrolling for many metrics.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {metricsLoading ? (
                <div className="flex h-48 items-center justify-center">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
              ) : !metricsData ? (
                <p className="text-sm text-muted-foreground">Metrics unavailable for this run.</p>
              ) : (
                <>
                  <div className="w-full overflow-x-auto border rounded-md">
                    <table className="text-sm min-w-max">
                      <thead>
                        <tr className="border-b bg-muted/40">
                          <th className="text-left py-2 px-3 min-w-[170px]">Combination</th>
                          <th className="text-left py-2 px-3 min-w-[430px]">Tool Sequence</th>
                          {metricsData.metric_names.map((metric) => (
                            <th key={metric} className="text-left py-2 px-3 min-w-[120px]">
                              {metric}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {visibleWorkflows.map((workflow) => (
                          <tr key={workflow.workflow_id} className="border-b last:border-0 align-top">
                            <td className="py-2 px-3">
                              <div className="space-y-0.5">
                                <p className="font-medium">{workflow.workflow_name}</p>
                                <p className="text-xs text-muted-foreground font-mono">
                                  {truncateId(workflow.workflow_id, 18)}
                                </p>
                                {workflow.is_baseline && (
                                  <Badge variant="outline" className="text-xs">
                                    baseline
                                  </Badge>
                                )}
                              </div>
                            </td>
                            <td className="py-2 px-3">
                              <pre className="text-xs whitespace-pre-wrap break-words font-mono">
                                {getWorkflowToolsLabel(workflow) || 'No sequence reported'}
                              </pre>
                            </td>
                            {metricsData.metric_names.map((metric) => {
                              const value = workflow.metrics[metric];
                              return (
                                <td key={metric} className="py-2 px-3 tabular-nums">
                                  {value == null ? '—' : `${(value * 100).toFixed(2)}%`}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                        {visibleWorkflows.length === 0 && (
                          <tr>
                            <td colSpan={2 + metricsData.metric_names.length} className="py-6 px-3 text-sm text-muted-foreground">
                              No combinations match the search.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>

                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>
                      Showing {filteredWorkflows.length === 0 ? 0 : startIdx + 1}-{endIdx} of {filteredWorkflows.length}
                    </span>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        disabled={currentPage <= 1}
                        onClick={() => setPage((p) => Math.max(1, p - 1))}
                      >
                        Previous
                      </Button>
                      <span>Page {currentPage} / {totalPages}</span>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        disabled={currentPage >= totalPages}
                        onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                      >
                        Next
                      </Button>
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
