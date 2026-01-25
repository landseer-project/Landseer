import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { StatsCard } from '@/components/StatsCard';
import { getPipelineMetrics, getPipelineDetail, PipelineMetricsResponse, PipelineDetailResponse } from '@/lib/api';

// Simple sparkline component
function Sparkline({ values, color = 'blue' }: { values: number[]; color?: string }) {
  if (values.length === 0) return null;
  
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * 100;
    const y = 100 - ((v - min) / range) * 100;
    return `${x},${y}`;
  }).join(' ');
  
  return (
    <svg viewBox="0 0 100 100" className="h-8 w-24 inline-block ml-2">
      <polyline
        points={points}
        fill="none"
        stroke={color === 'blue' ? '#3b82f6' : color === 'green' ? '#22c55e' : '#ef4444'}
        strokeWidth="3"
      />
    </svg>
  );
}

// Heatmap cell
function HeatmapCell({ value, min, max }: { value: number | null; min: number; max: number }) {
  if (value === null) {
    return <div className="w-10 h-10 bg-gray-100 rounded flex items-center justify-center text-xs text-gray-400">N/A</div>;
  }
  
  const normalized = max === min ? 0.5 : (value - min) / (max - min);
  const intensity = Math.round(normalized * 255);
  const bgColor = `rgb(${255 - intensity}, ${155 + intensity * 0.4}, ${155 + intensity * 0.4})`;
  
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
  const { id: pipelineId } = useParams<{ id: string }>();
  const [metrics, setMetrics] = useState<PipelineMetricsResponse | null>(null);
  const [pipeline, setPipeline] = useState<PipelineDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, [pipelineId]);

  async function loadData() {
    try {
      setLoading(true);
      const [pipelineRes, metricsRes] = await Promise.all([
        getPipelineDetail(),
        pipelineId ? getPipelineMetrics(pipelineId) : Promise.resolve(null),
      ]);
      setPipeline(pipelineRes);
      
      // If no pipelineId, use the current pipeline
      if (!pipelineId && pipelineRes) {
        const metricsData = await getPipelineMetrics(pipelineRes.id);
        setMetrics(metricsData);
      } else {
        setMetrics(metricsRes);
      }
      
      setError(null);
    } catch (err) {
      setError('Failed to load metrics');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
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

  if (error || !metrics) {
    return (
      <div className="p-6">
        <Card>
          <CardContent className="pt-6">
            <p className="text-red-500">{error || 'No metrics available'}</p>
            <Button onClick={loadData} className="mt-4">Retry</Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Calculate summary statistics
  const avgCleanAccuracy = metrics.summary['clean_accuracy']?.avg;
  const bestPgdAccuracy = metrics.summary['pgd_accuracy']?.max;
  const completedEvals = metrics.workflows.filter(w => w.evaluators_run.length > 0).length;
  
  // Get all unique metrics for the heatmap
  const allMetricNames = metrics.metric_names;
  
  // Get min/max for each metric for heatmap coloring
  const metricRanges: Record<string, { min: number; max: number }> = {};
  for (const metricName of allMetricNames) {
    const values = metrics.workflows
      .map(w => w.metrics[metricName])
      .filter((v): v is number => v !== null);
    if (values.length > 0) {
      metricRanges[metricName] = {
        min: Math.min(...values),
        max: Math.max(...values),
      };
    } else {
      metricRanges[metricName] = { min: 0, max: 1 };
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Metrics Dashboard</h1>
          <p className="text-gray-600">
            {metrics.pipeline_name} - {metrics.workflow_count} workflows
          </p>
        </div>
        <Button onClick={loadData} variant="outline">Refresh</Button>
      </div>

      {/* Overview Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <StatsCard
          title="Total Workflows"
          value={metrics.workflow_count}
          description="Workflow combinations"
        />
        <StatsCard
          title="Avg Clean Accuracy"
          value={avgCleanAccuracy ? `${(avgCleanAccuracy * 100).toFixed(1)}%` : 'N/A'}
          description="Across all workflows"
        />
        <StatsCard
          title="Best PGD Accuracy"
          value={bestPgdAccuracy ? `${(bestPgdAccuracy * 100).toFixed(1)}%` : 'N/A'}
          description="Adversarial robustness"
        />
        <StatsCard
          title="Completed Evaluations"
          value={completedEvals}
          description={`of ${metrics.workflow_count} workflows`}
        />
      </div>

      <Tabs defaultValue="table" className="space-y-4">
        <TabsList>
          <TabsTrigger value="table">Comparison Table</TabsTrigger>
          <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
          <TabsTrigger value="summary">Summary</TabsTrigger>
        </TabsList>

        {/* Comparison Table with Sparklines */}
        <TabsContent value="table">
          <Card>
            <CardHeader>
              <CardTitle>Workflow Comparison</CardTitle>
              <CardDescription>All metrics across workflows with trend visualization</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-3">Workflow</th>
                      {allMetricNames.slice(0, 6).map(name => (
                        <th key={name} className="text-left py-2 px-3">{name.replace(/_/g, ' ')}</th>
                      ))}
                      <th className="text-left py-2 px-3">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.workflows.map((workflow, idx) => (
                      <tr key={workflow.workflow_id} className="border-b hover:bg-gray-50">
                        <td className="py-2 px-3 font-medium">{workflow.workflow_name}</td>
                        {allMetricNames.slice(0, 6).map(metricName => {
                          const value = workflow.metrics[metricName];
                          // Get all values for this metric to create sparkline
                          const allValues = metrics.workflows
                            .slice(0, idx + 1)
                            .map(w => w.metrics[metricName])
                            .filter((v): v is number => v !== null);
                          
                          return (
                            <td key={metricName} className="py-2 px-3">
                              {value !== null ? (
                                <span className="flex items-center">
                                  {(value * 100).toFixed(1)}%
                                  {allValues.length > 1 && (
                                    <Sparkline values={allValues} color={value > 0.5 ? 'green' : 'blue'} />
                                  )}
                                </span>
                              ) : (
                                <span className="text-gray-400">N/A</span>
                              )}
                            </td>
                          );
                        })}
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
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Heatmap */}
        <TabsContent value="heatmap">
          <Card>
            <CardHeader>
              <CardTitle>Metrics Heatmap</CardTitle>
              <CardDescription>Visual comparison of metrics across workflows (darker = higher)</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <div className="min-w-max">
                  {/* Header */}
                  <div className="flex items-center gap-1 mb-2">
                    <div className="w-24 flex-shrink-0"></div>
                    {allMetricNames.map(name => (
                      <div key={name} className="w-10 text-center">
                        <span className="text-xs text-gray-500 writing-mode-vertical transform -rotate-45 inline-block origin-bottom-left">
                          {name.slice(0, 8)}
                        </span>
                      </div>
                    ))}
                  </div>
                  
                  {/* Rows */}
                  {metrics.workflows.map(workflow => (
                    <div key={workflow.workflow_id} className="flex items-center gap-1 mb-1">
                      <div className="w-24 flex-shrink-0 text-xs font-medium truncate" title={workflow.workflow_name}>
                        {workflow.workflow_name}
                      </div>
                      {allMetricNames.map(metricName => {
                        const value = workflow.metrics[metricName];
                        const range = metricRanges[metricName];
                        return (
                          <HeatmapCell
                            key={metricName}
                            value={value}
                            min={range.min}
                            max={range.max}
                          />
                        );
                      })}
                    </div>
                  ))}
                  
                  {/* Legend */}
                  <div className="flex items-center gap-2 mt-4 text-xs text-gray-500">
                    <span>Low</span>
                    <div className="flex">
                      {[0, 0.25, 0.5, 0.75, 1].map(v => (
                        <div
                          key={v}
                          className="w-6 h-4"
                          style={{ backgroundColor: `rgb(${255 - v * 255}, ${155 + v * 100}, ${155 + v * 100})` }}
                        />
                      ))}
                    </div>
                    <span>High</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Summary Statistics */}
        <TabsContent value="summary">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {Object.entries(metrics.summary).map(([metricName, stats]) => (
              <Card key={metricName}>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">{metricName.replace(/_/g, ' ')}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-2xl font-bold text-red-600">
                        {stats.min !== null ? (stats.min * 100).toFixed(1) : 'N/A'}%
                      </p>
                      <p className="text-xs text-gray-500">Min</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-blue-600">
                        {stats.avg !== null ? (stats.avg * 100).toFixed(1) : 'N/A'}%
                      </p>
                      <p className="text-xs text-gray-500">Avg</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-green-600">
                        {stats.max !== null ? (stats.max * 100).toFixed(1) : 'N/A'}%
                      </p>
                      <p className="text-xs text-gray-500">Max</p>
                    </div>
                  </div>
                  <p className="text-xs text-gray-400 mt-2 text-center">
                    {stats.count} workflows with data
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
