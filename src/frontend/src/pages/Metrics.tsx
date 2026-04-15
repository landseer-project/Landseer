import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { StatsCard } from '@/components/StatsCard';
import { getPipelineMetrics, getPipelineDetail } from '@/lib/api';
import { Loader2 } from 'lucide-react';

// Sparkline with optional baseline reference line
function Sparkline({
  values,
  color = 'blue',
  baselineValue,
}: {
  values: number[];
  color?: string;
  baselineValue?: number | null;
}) {
  if (values.length === 0) return null;

  const allForRange = baselineValue != null ? [...values, baselineValue] : values;
  const min = Math.min(...allForRange);
  const max = Math.max(...allForRange);
  const range = max - min || 1;

  const points = values
    .map((v, i) => {
      const x = values.length === 1 ? 50 : (i / (values.length - 1)) * 100;
      const y = 100 - ((v - min) / range) * 100;
      return `${x},${y}`;
    })
    .join(' ');

  const baselineY =
    baselineValue != null ? 100 - ((baselineValue - min) / range) * 100 : null;

  const lineColor =
    color === 'green' ? '#22c55e' : color === 'red' ? '#ef4444' : '#3b82f6';

  return (
    <svg viewBox="0 0 100 100" className="h-8 w-24 inline-block ml-2" aria-hidden>
      {baselineY != null && (
        <line
          x1="0"
          y1={baselineY}
          x2="100"
          y2={baselineY}
          stroke="#f59e0b"
          strokeWidth="2"
          strokeDasharray="5,3"
        />
      )}
      <polyline points={points} fill="none" stroke={lineColor} strokeWidth="3" />
    </svg>
  );
}

// Heatmap cell — supports diverging mode anchored at a baseline value
function HeatmapCell({
  value,
  min,
  max,
  baselineValue,
}: {
  value: number | null;
  min: number;
  max: number;
  baselineValue?: number | null;
}) {
  if (value === null) {
    return (
      <div className="w-10 h-10 bg-gray-100 rounded flex items-center justify-center text-xs text-gray-400">
        N/A
      </div>
    );
  }

  let bgColor: string;
  if (baselineValue != null) {
    // Diverging: blue = below baseline, green = above baseline
    const delta = value - baselineValue;
    const maxDelta = Math.max(Math.abs(max - baselineValue), Math.abs(min - baselineValue)) || 1;
    const t = Math.min(Math.abs(delta) / maxDelta, 1);
    const intensity = Math.round(t * 200);
    if (delta >= 0) {
      bgColor = `rgb(${155 - intensity}, ${155 + intensity}, ${155 - intensity})`; // green tones
    } else {
      bgColor = `rgb(${155 + intensity}, ${155 - Math.round(intensity * 0.5)}, ${155 - Math.round(intensity * 0.5)})`; // red/warm tones
    }
  } else {
    const normalized = max === min ? 0.5 : (value - min) / (max - min);
    const intensity = Math.round(normalized * 255);
    bgColor = `rgb(${255 - intensity}, ${155 + intensity * 0.4}, ${155 + intensity * 0.4})`;
  }

  return (
    <div
      className="w-10 h-10 rounded flex items-center justify-center text-xs font-medium"
      style={{ backgroundColor: bgColor }}
      title={value.toFixed(4)}
    >
      {(value * 100).toFixed(0)}
    </div>
  );
}

export function Metrics() {
  const { id: pipelineIdParam } = useParams<{ id: string }>();

  const { data: pipelineDetail } = useQuery({
    queryKey: ['pipeline'],
    queryFn: getPipelineDetail,
    enabled: !pipelineIdParam,
    refetchInterval: 15_000,
  });

  const resolvedPipelineId = pipelineIdParam || pipelineDetail?.id;

  const { data: metrics, isLoading, isError, isFetching } = useQuery({
    queryKey: ['pipeline-metrics', resolvedPipelineId],
    queryFn: () => getPipelineMetrics(resolvedPipelineId!),
    enabled: !!resolvedPipelineId,
    refetchInterval: 15_000,
  });

  if (isLoading || (!metrics && !isError)) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 w-48 bg-gray-200 rounded" />
          <div className="grid grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-24 bg-gray-200 rounded" />
            ))}
          </div>
          <div className="h-64 bg-gray-200 rounded" />
        </div>
      </div>
    );
  }

  if (isError || !metrics) {
    return (
      <div className="p-6">
        <Card>
          <CardContent className="pt-6">
            <p className="text-red-500">Failed to load metrics</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Baseline workflow (first one where all non-evaluator tools are baseline noops)
  const baselineWorkflow = metrics.workflows.find((w) => w.is_baseline) ?? null;

  const avgCleanAccuracy = metrics.summary['clean_accuracy']?.avg;
  const bestPgdAccuracy = metrics.summary['pgd_accuracy']?.max;
  const completedEvals = metrics.workflows.filter((w) =>
    Object.values(w.metrics).some((v) => v !== null)
  ).length;
  const hasNoData = completedEvals === 0 || metrics.metric_names.length === 0;

  const allMetricNames = metrics.metric_names;

  // Per-metric ranges for heatmap coloring
  const metricRanges: Record<string, { min: number; max: number }> = {};
  for (const metricName of allMetricNames) {
    const values = metrics.workflows
      .map((w) => w.metrics[metricName])
      .filter((v): v is number => v !== null);
    metricRanges[metricName] =
      values.length > 0
        ? { min: Math.min(...values), max: Math.max(...values) }
        : { min: 0, max: 1 };
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold">Metrics Dashboard</h1>
          {isFetching && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
        </div>
        <p className="text-gray-600">
          {metrics.pipeline_name} · {metrics.workflow_count} workflows
        </p>
      </div>

      {/* Empty state */}
      {hasNoData && (
        <Card className="border-blue-200 bg-blue-50/80 dark:bg-blue-950/30 dark:border-blue-800">
          <CardContent className="pt-6">
            <div className="flex gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 16v-4M12 8h.01" />
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-blue-900 dark:text-blue-100">No metrics yet</h3>
                <p className="mt-1 text-sm text-blue-800 dark:text-blue-200">
                  Metrics will appear here after workflow evaluations complete. Run your pipeline (training and evaluation tasks); once evaluators run, you'll see clean accuracy, PGD accuracy, and other metrics in the cards and comparison table below.
                </p>
                <p className="mt-2 text-xs text-blue-700 dark:text-blue-300">
                  You can still see workflow status (Pending / Completed) in the table. Use the <strong>Tasks</strong> or <strong>Workflows</strong> pages to monitor progress.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <StatsCard
          title="Total Workflows"
          value={metrics.workflow_count}
          subtitle="Workflow combinations"
        />
        <StatsCard
          title="Avg Clean Accuracy"
          value={avgCleanAccuracy != null ? `${(avgCleanAccuracy * 100).toFixed(1)}%` : 'N/A'}
          subtitle={hasNoData ? 'Complete evaluations to see value' : 'Across all workflows'}
        />
        <StatsCard
          title="Best PGD Accuracy"
          value={bestPgdAccuracy != null ? `${(bestPgdAccuracy * 100).toFixed(1)}%` : 'N/A'}
          subtitle={hasNoData ? 'Complete evaluations to see value' : 'Adversarial robustness'}
        />
        <StatsCard
          title="Completed Evaluations"
          value={completedEvals}
          subtitle={`of ${metrics.workflow_count} workflows`}
        />
      </div>

      <Tabs defaultValue="table" className="space-y-4">
        <TabsList>
          <TabsTrigger value="table">Comparison Table</TabsTrigger>
          <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
          <TabsTrigger value="summary">Summary</TabsTrigger>
        </TabsList>

        {/* ── Comparison Table ── */}
        <TabsContent value="table">
          <Card>
            <CardHeader>
              <CardTitle>Workflow Comparison</CardTitle>
              <CardDescription>
                {allMetricNames.length > 0 ? (
                  <>
                    All metrics across workflows.
                    {baselineWorkflow && (
                      <span className="ml-1 text-amber-600 dark:text-amber-400">
                        <span className="inline-block w-3 h-3 rounded-sm bg-amber-400 mr-1 align-middle" />
                        Baseline row is highlighted. Deltas show change vs baseline.
                        Dashed line in sparklines marks baseline level.
                      </span>
                    )}
                  </>
                ) : (
                  'Workflow status. Metrics columns will appear once evaluations complete.'
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-3">Workflow</th>
                      {allMetricNames.length > 0 ? (
                        allMetricNames.slice(0, 6).map((name) => (
                          <th key={name} className="text-left py-2 px-3">
                            {name.replace(/_/g, ' ')}
                          </th>
                        ))
                      ) : (
                        <th className="text-left py-2 px-3 text-muted-foreground">Metrics</th>
                      )}
                      <th className="text-left py-2 px-3">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.workflows.map((workflow, idx) => {
                      const isBaseline = workflow.is_baseline;
                      return (
                        <tr
                          key={workflow.workflow_id}
                          className={
                            isBaseline
                              ? 'border-b bg-amber-50 dark:bg-amber-950/30 font-semibold'
                              : 'border-b hover:bg-gray-50 dark:hover:bg-gray-800/40'
                          }
                        >
                          <td className="py-2 px-3">
                            <span className="flex items-center gap-2">
                              {workflow.workflow_name}
                              {isBaseline && (
                                <Badge
                                  variant="outline"
                                  className="border-amber-500 text-amber-700 dark:text-amber-400 text-xs"
                                >
                                  Baseline
                                </Badge>
                              )}
                            </span>
                          </td>

                          {allMetricNames.length > 0 ? (
                            allMetricNames.slice(0, 6).map((metricName) => {
                              const value = workflow.metrics[metricName];
                              const baselineValue =
                                baselineWorkflow?.metrics[metricName] ?? null;
                              const delta =
                                !isBaseline &&
                                value !== null &&
                                baselineValue !== null
                                  ? value - baselineValue
                                  : null;

                              const allValues = metrics.workflows
                                .slice(0, idx + 1)
                                .map((w) => w.metrics[metricName])
                                .filter((v): v is number => v !== null);

                              return (
                                <td key={metricName} className="py-2 px-3">
                                  {value !== null ? (
                                    <span className="flex items-center flex-wrap gap-x-1">
                                      <span>{(value * 100).toFixed(1)}%</span>
                                      {delta !== null && (
                                        <span
                                          className={`text-xs font-normal ${
                                            delta >= 0
                                              ? 'text-green-600 dark:text-green-400'
                                              : 'text-red-500 dark:text-red-400'
                                          }`}
                                        >
                                          {delta >= 0 ? '+' : ''}
                                          {(delta * 100).toFixed(1)}%
                                        </span>
                                      )}
                                      {allValues.length > 1 && (
                                        <Sparkline
                                          values={allValues}
                                          color={value > 0.5 ? 'green' : 'blue'}
                                          baselineValue={baselineValue}
                                        />
                                      )}
                                    </span>
                                  ) : (
                                    <span className="text-gray-400">N/A</span>
                                  )}
                                </td>
                              );
                            })
                          ) : (
                            <td className="py-2 px-3 text-muted-foreground italic">
                              No evaluation data yet
                            </td>
                          )}

                          <td className="py-2 px-3">
                            {workflow.evaluators_run.length > 0 ? (
                              <Badge variant="default">Completed</Badge>
                            ) : workflow.evaluators_skipped.length > 0 ? (
                              <Badge variant="secondary">Partial</Badge>
                            ) : (
                              <Badge variant="outline">Pending</Badge>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Heatmap ── */}
        <TabsContent value="heatmap">
          <Card>
            <CardHeader>
              <CardTitle>Metrics Heatmap</CardTitle>
              <CardDescription>
                {baselineWorkflow
                  ? 'Green = above baseline · Red = below baseline · Amber border = baseline row'
                  : 'Visual comparison of metrics across workflows (darker = higher)'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {allMetricNames.length === 0 ? (
                <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-gray-200 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-900/30 py-12 px-6 text-center">
                  <p className="text-sm text-muted-foreground">No metrics to display yet.</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Complete workflow evaluations to see a heatmap here.
                  </p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <div className="min-w-max">
                    {/* Column headers */}
                    <div className="flex items-center gap-1 mb-2">
                      <div className="w-28 flex-shrink-0" />
                      {allMetricNames.map((name) => (
                        <div key={name} className="w-10 text-center">
                          <span className="text-xs text-gray-500 writing-mode-vertical transform -rotate-45 inline-block origin-bottom-left">
                            {name.slice(0, 8)}
                          </span>
                        </div>
                      ))}
                    </div>

                    {/* Rows */}
                    {metrics.workflows.map((workflow) => {
                      const isBaseline = workflow.is_baseline;
                      return (
                        <div
                          key={workflow.workflow_id}
                          className={`flex items-center gap-1 mb-1 rounded ${
                            isBaseline
                              ? 'border-l-4 border-amber-400 pl-1 bg-amber-50/60 dark:bg-amber-950/20'
                              : ''
                          }`}
                        >
                          <div
                            className="w-24 flex-shrink-0 text-xs font-medium truncate"
                            title={workflow.workflow_name}
                          >
                            {workflow.workflow_name}
                            {isBaseline && (
                              <span className="ml-1 text-amber-600 dark:text-amber-400 font-normal">
                                ★
                              </span>
                            )}
                          </div>
                          {allMetricNames.map((metricName) => {
                            const value = workflow.metrics[metricName];
                            const range = metricRanges[metricName];
                            const baselineValue =
                              baselineWorkflow?.metrics[metricName] ?? null;
                            return (
                              <HeatmapCell
                                key={metricName}
                                value={value}
                                min={range.min}
                                max={range.max}
                                baselineValue={baselineValue}
                              />
                            );
                          })}
                        </div>
                      );
                    })}

                    {/* Legend */}
                    <div className="flex items-center gap-4 mt-4 text-xs text-gray-500">
                      {baselineWorkflow ? (
                        <>
                          <div className="flex items-center gap-1">
                            <div className="w-6 h-4 rounded" style={{ backgroundColor: 'rgb(355,155,155)' /* clipped red */ }} />
                            <span>Below baseline</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <div className="w-6 h-4 rounded" style={{ backgroundColor: 'rgb(155,155,155)' }} />
                            <span>At baseline</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <div className="w-6 h-4 rounded" style={{ backgroundColor: 'rgb(155,355,155)' /* clipped green */ }} />
                            <span>Above baseline</span>
                          </div>
                        </>
                      ) : (
                        <>
                          <span>Low</span>
                          <div className="flex">
                            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                              <div
                                key={v}
                                className="w-6 h-4"
                                style={{
                                  backgroundColor: `rgb(${255 - v * 255}, ${155 + v * 100}, ${155 + v * 100})`,
                                }}
                              />
                            ))}
                          </div>
                          <span>High</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ── Summary Statistics ── */}
        <TabsContent value="summary">
          {Object.keys(metrics.summary).length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12 px-6 text-center">
                <p className="text-sm text-muted-foreground">No summary statistics yet.</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Min / avg / max per metric will appear here once evaluations complete.
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {Object.entries(metrics.summary).map(([metricName, stats]) => {
                const baselineValue =
                  baselineWorkflow?.metrics[metricName] ?? null;
                return (
                  <Card key={metricName}>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">{metricName.replace(/_/g, ' ')}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-3 gap-4 text-center">
                        <div>
                          <p className="text-2xl font-bold text-red-600">
                            {stats.min !== null ? `${(stats.min * 100).toFixed(1)}%` : 'N/A'}
                          </p>
                          <p className="text-xs text-gray-500">Min</p>
                        </div>
                        <div>
                          <p className="text-2xl font-bold text-blue-600">
                            {stats.avg !== null ? `${(stats.avg * 100).toFixed(1)}%` : 'N/A'}
                          </p>
                          <p className="text-xs text-gray-500">Avg</p>
                        </div>
                        <div>
                          <p className="text-2xl font-bold text-green-600">
                            {stats.max !== null ? `${(stats.max * 100).toFixed(1)}%` : 'N/A'}
                          </p>
                          <p className="text-xs text-gray-500">Max</p>
                        </div>
                      </div>

                      {baselineValue !== null && (
                        <div className="mt-3 pt-3 border-t border-amber-200 dark:border-amber-800 flex items-center justify-between">
                          <span className="text-xs text-amber-700 dark:text-amber-400 font-medium flex items-center gap-1">
                            <span className="inline-block w-2 h-2 rounded-full bg-amber-400" />
                            Baseline
                          </span>
                          <span className="text-sm font-semibold text-amber-700 dark:text-amber-300">
                            {(baselineValue * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}

                      <p className="text-xs text-gray-400 mt-2 text-center">
                        {stats.count} workflows with data
                      </p>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
