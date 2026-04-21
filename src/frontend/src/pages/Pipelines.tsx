import { useState, useMemo } from 'react';
import { useQuery, useQueries, useMutation, useQueryClient } from '@tanstack/react-query';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { StatusBadge } from '@/components/StatusBadge';
import {
  getPipelineConfigs,
  getPipelineRuns,
  startPipelineRun,
  stopPipelineRun,
  getRegistryTools,
  getRunMetrics,
  getModelConfigs,
} from '@/lib/api';
import type { PipelineMetricsResponse, ModelConfigInfo } from '@/lib/api';
import { formatTimestamp, formatRelativeTime, truncateId } from '@/lib/utils';
import {
  Play,
  Square,
  Loader2,
  Clock,
  Hash,
  ChevronDown,
  ChevronRight,
  FileText,
  FolderOpen,
  RefreshCw,
  AlertCircle,
  Database,
  ShieldAlert,
  Settings2,
  Lock,
  Wrench,
  GitCompare,
  X,
  TrendingUp,
  TrendingDown,
  Minus,
  BarChart2,
} from 'lucide-react';
import type { PipelineConfig, PipelineRun, ToolInfo } from '@/types/api';

// ─── Constants ────────────────────────────────────────────────────────────────

const DATASETS = [
  { id: 'cifar10', label: 'CIFAR-10', description: 'Image classification (10 classes)' },
  { id: 'celeba', label: 'CelebA', description: 'Face attribute recognition' },
] as const;
type DatasetId = (typeof DATASETS)[number]['id'];

const STAGE_ORDER = ['pre_training', 'during_training', 'post_training', 'deployment'] as const;
type Stage = (typeof STAGE_ORDER)[number];

const STAGE_LABELS: Record<Stage, string> = {
  pre_training: 'Pre-training',
  during_training: 'During training',
  post_training: 'Post-training',
  deployment: 'Deployment',
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Permutations P(n, k) = n! / (n-k)! */
function perm(n: number, k: number): number {
  if (k > n) return 0;
  let r = 1;
  for (let i = n; i > n - k; i--) r *= i;
  return r;
}

/**
 * Number of workflow options for a stage.
 * pre/post/deploy: all permutation subsets + 1 baseline = Σ P(n,k) for k=1..n + 1
 * during_training: n non-baseline tools + 1 baseline
 */
function stageOptions(nonBaselineCount: number, isDuring: boolean): number {
  if (isDuring) return nonBaselineCount + 1;
  let total = 1; // baseline always available
  for (let k = 1; k <= nonBaselineCount; k++) total += perm(nonBaselineCount, k);
  return total;
}

function countCombinations(toolsByStage: Record<Stage, { name: string; is_baseline: boolean }[]>): number {
  return STAGE_ORDER.reduce((product, stage) => {
    const tools = toolsByStage[stage] ?? [];
    const nonBase = tools.filter((t) => !t.is_baseline).length;
    return product * stageOptions(nonBase, stage === 'during_training');
  }, 1);
}

// ─── Small reusable UI atoms ──────────────────────────────────────────────────

function OptionToggle({ label, checked, onChange, disabled = false }: {
  label: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => !disabled && onChange(!checked)}
      className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2
        ${checked ? 'bg-primary' : 'bg-input'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <span className={`pointer-events-none block h-5 w-5 rounded-full bg-white shadow-lg ring-0 transition-transform ${checked ? 'translate-x-5' : 'translate-x-0'}`} />
      <span className="sr-only">{label}</span>
    </button>
  );
}

function RunDuration({ run }: { run: PipelineRun }) {
  if (!run.started_at) return <span className="text-muted-foreground">--</span>;
  const start = new Date(run.started_at).getTime();
  const end = run.completed_at ? new Date(run.completed_at).getTime() : Date.now();
  const secs = Math.floor((end - start) / 1000);
  if (secs < 60) return <span>{secs}s</span>;
  if (secs < 3600) return <span>{Math.floor(secs / 60)}m {secs % 60}s</span>;
  return <span>{Math.floor(secs / 3600)}h {Math.floor((secs % 3600) / 60)}m</span>;
}

/** Compact read-only display of a locked tools_config snapshot */
function LockedToolsDisplay({ toolsConfig }: { toolsConfig: Record<string, string[]> }) {
  return (
    <div className="mt-2 rounded-md border border-amber-200 bg-amber-50/60 dark:bg-amber-950/20 dark:border-amber-800 px-3 py-2">
      <div className="flex items-center gap-1 mb-1.5 text-xs font-medium text-amber-700 dark:text-amber-400">
        <Lock className="h-3 w-3" />
        Locked tool configuration
      </div>
      <div className="space-y-1">
        {STAGE_ORDER.filter((s) => toolsConfig[s]?.length).map((stage) => (
          <div key={stage} className="flex items-start gap-2 text-xs">
            <span className="text-muted-foreground w-28 shrink-0">{STAGE_LABELS[stage]}</span>
            <div className="flex flex-wrap gap-1">
              {toolsConfig[stage].map((t) => (
                <Badge key={t} variant="secondary" className="text-xs px-1.5 py-0 h-4">{t}</Badge>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── CompareDialog ────────────────────────────────────────────────────────────

/** Format a delta value for display: "+5.2%" / "-3.1%" / "—" */
function DeltaCell({ value, baseline }: { value: number | null | undefined; baseline: number | null | undefined }) {
  if (value == null || baseline == null || baseline === 0) {
    return <span className="text-muted-foreground">—</span>;
  }
  const delta = ((value - baseline) / Math.abs(baseline)) * 100;
  const formatted = `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%`;
  if (Math.abs(delta) < 0.05) {
    return <span className="flex items-center gap-0.5 text-muted-foreground"><Minus className="h-3 w-3" />{formatted}</span>;
  }
  if (delta > 0) {
    return <span className="flex items-center gap-0.5 text-emerald-600 font-medium"><TrendingUp className="h-3 w-3" />{formatted}</span>;
  }
  return <span className="flex items-center gap-0.5 text-red-500 font-medium"><TrendingDown className="h-3 w-3" />{formatted}</span>;
}

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
    if (tools.length > 0) parts.push(`${stage}: ${tools.join(', ')}`);
  }
  return parts.join(' | ');
}

function CompareDialog({
  runIds,
  runs,
  configs,
  onClose,
}: {
  runIds: string[];
  runs: { id: string; run_number: number; pipeline_config_id: string; status: string }[];
  configs: { id: string; name: string }[];
  onClose: () => void;
}) {
  const results = useQueries({
    queries: runIds.map((runId) => ({
      queryKey: ['run-metrics', runId],
      queryFn: () => getRunMetrics(runId),
      retry: 1,
    })),
  });

  const isLoading = results.some((r) => r.isLoading);
  const hasError = results.some((r) => r.isError);

  // Collect all unique metric names across all loaded results
  const allMetricNames = useMemo(() => {
    const names = new Set<string>();
    for (const r of results) {
      if (r.data) r.data.metric_names.forEach((m) => names.add(m));
    }
    return Array.from(names).sort();
  }, [results]);

  // Per-run data: keyed by runId → { data, runInfo }
  const runDataMap = useMemo(() => {
    const map: Record<string, PipelineMetricsResponse> = {};
    runIds.forEach((runId, i) => {
      if (results[i]?.data) map[runId] = results[i].data!;
    });
    return map;
  }, [results, runIds]);

  // For each run, find the baseline workflow (all tools are baseline)
  function getBaseline(data: PipelineMetricsResponse): { metrics: Record<string, number | null> } | null {
    return data.workflows.find((w) => w.is_baseline) ?? null;
  }

  function getRunLabel(runId: string): string {
    const run = runs.find((r) => r.id === runId);
    if (!run) return truncateId(runId, 12);
    const cfg = configs.find((c) => c.id === run.pipeline_config_id);
    return `${cfg?.name ?? '?'} #${run.run_number}`;
  }

  // Best metric value across runs (for column highlighting)
  const bestValues = useMemo(() => {
    const best: Record<string, number | null> = {};
    for (const metric of allMetricNames) {
      let max: number | null = null;
      for (const runId of runIds) {
        const data = runDataMap[runId];
        if (!data) continue;
        const summary = data.summary[metric];
        const val = summary?.avg ?? null;
        if (val !== null && (max === null || val > max)) max = val;
      }
      best[metric] = max;
    }
    return best;
  }, [allMetricNames, runIds, runDataMap]);

  return (
    <Dialog open onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="max-w-5xl max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <GitCompare className="h-5 w-5" />
            Compare {runIds.length} Runs
          </DialogTitle>
          <DialogDescription>
            Side-by-side metric comparison. Deltas are relative to each run's own baseline workflow.
          </DialogDescription>
        </DialogHeader>

        {isLoading && (
          <div className="flex-1 flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        )}

        {hasError && !isLoading && (
          <div className="flex items-center gap-2 text-destructive py-4">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="text-sm">
              Some runs could not be loaded. They may not have completed evaluation yet.
            </span>
          </div>
        )}

        {!isLoading && allMetricNames.length === 0 && (
          <div className="py-8 text-center text-muted-foreground text-sm">
            No evaluation metrics found for the selected runs.
            <br />
            Runs must be completed with at least one evaluator.
          </div>
        )}

        {!isLoading && allMetricNames.length > 0 && (
          <ScrollArea className="flex-1">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 pr-4 font-medium text-muted-foreground w-40">Metric</th>
                    {runIds.map((runId) => (
                      <th key={runId} className="text-left py-2 px-3 font-medium min-w-[140px]">
                        {getRunLabel(runId)}
                        <span className={`ml-1 text-xs font-normal ${
                          runs.find((r) => r.id === runId)?.status === 'completed'
                            ? 'text-emerald-500'
                            : 'text-muted-foreground'
                        }`}>
                          ({runs.find((r) => r.id === runId)?.status ?? '?'})
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {allMetricNames.map((metric) => (
                    <tr key={metric} className="border-b last:border-0 hover:bg-muted/30">
                      <td className="py-2 pr-4 font-mono text-xs text-muted-foreground">{metric}</td>
                      {runIds.map((runId) => {
                        const data = runDataMap[runId];
                        if (!data) {
                          return (
                            <td key={runId} className="py-2 px-3 text-muted-foreground">—</td>
                          );
                        }
                        const summary = data.summary[metric];
                        const avg = summary?.avg ?? null;
                        const baseline = getBaseline(data);
                        const baselineVal = baseline?.metrics[metric] ?? null;
                        const isBest = avg !== null && avg === bestValues[metric];
                        return (
                          <td key={runId} className={`py-2 px-3 ${isBest ? 'bg-emerald-50/50 dark:bg-emerald-950/20' : ''}`}>
                            <div className="flex flex-col gap-0.5">
                              {avg !== null ? (
                                <span className={`font-medium tabular-nums ${isBest ? 'text-emerald-700 dark:text-emerald-400' : ''}`}>
                                  {avg.toFixed(4)}
                                </span>
                              ) : (
                                <span className="text-muted-foreground">—</span>
                              )}
                              {baselineVal !== null && avg !== null && (
                                <DeltaCell value={avg} baseline={baselineVal} />
                              )}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ScrollArea>
        )}

        <DialogFooter className="mt-3">
          <Button variant="outline" onClick={onClose}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ─── RunMetricsDialog ─────────────────────────────────────────────────────────

function RunMetricsDialog({
  runId,
  runLabel,
  onClose,
}: {
  runId: string;
  runLabel: string;
  onClose: () => void;
}) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['run-metrics', runId],
    queryFn: () => getRunMetrics(runId),
    retry: 1,
    staleTime: 30_000,
  });

  return (
    <Dialog open onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="max-w-4xl max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <BarChart2 className="h-5 w-5" />
            Metrics — {runLabel}
          </DialogTitle>
          <DialogDescription>
            Per-workflow evaluation metrics for this run.
          </DialogDescription>
        </DialogHeader>

        {isLoading && (
          <div className="flex-1 flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        )}

        {isError && !isLoading && (
          <div className="flex items-center gap-2 text-destructive py-4">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span className="text-sm">
              Metrics could not be loaded. The run may not have completed evaluation yet.
            </span>
          </div>
        )}

        {data && !isLoading && data.metric_names.length === 0 && (
          <div className="py-8 text-center text-muted-foreground text-sm">
            No evaluation metrics found for this run.
            <br />
            Runs must complete with at least one evaluator to generate metrics.
          </div>
        )}

        {data && !isLoading && data.metric_names.length > 0 && (
          <ScrollArea className="flex-1">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 pr-4 font-medium text-muted-foreground w-40">Metric</th>
                    {data.workflows.map((wf) => (
                      <th key={wf.workflow_id} className="text-left py-2 px-3 font-medium min-w-[120px]">
                        <span
                          className="block truncate max-w-[170px]"
                          title={
                            getWorkflowToolsLabel(wf)
                              ? `${wf.workflow_name}\n${getWorkflowToolsLabel(wf)}`
                              : wf.workflow_name
                          }
                        >
                          {wf.workflow_name}
                        </span>
                        {getWorkflowToolsLabel(wf) && (
                          <span className="block truncate max-w-[190px] text-[11px] font-normal text-muted-foreground" title={getWorkflowToolsLabel(wf)}>
                            {getWorkflowToolsLabel(wf)}
                          </span>
                        )}
                        {wf.is_baseline && (
                          <Badge variant="outline" className="text-xs px-1.5 py-0 h-4 border-amber-400 text-amber-600 font-normal mt-0.5">
                            baseline
                          </Badge>
                        )}
                      </th>
                    ))}
                    <th className="text-left py-2 px-3 font-medium text-muted-foreground min-w-[70px]">Avg</th>
                    <th className="text-left py-2 px-3 font-medium text-emerald-600 min-w-[70px]">Best</th>
                  </tr>
                </thead>
                <tbody>
                  {data.metric_names.map((metric) => {
                    const summary = data.summary[metric];
                    return (
                      <tr key={metric} className="border-b last:border-0 hover:bg-muted/30">
                        <td className="py-2 pr-4 font-mono text-xs text-muted-foreground">{metric}</td>
                        {data.workflows.map((wf) => {
                          const val = wf.metrics[metric];
                          return (
                            <td key={wf.workflow_id} className="py-2 px-3 tabular-nums">
                              {val != null ? (
                                <span className={wf.is_baseline ? 'text-amber-600 dark:text-amber-400' : ''}>
                                  {val.toFixed(4)}
                                </span>
                              ) : (
                                <span className="text-muted-foreground">—</span>
                              )}
                            </td>
                          );
                        })}
                        <td className="py-2 px-3 tabular-nums text-muted-foreground">
                          {summary?.avg != null ? summary.avg.toFixed(4) : '—'}
                        </td>
                        <td className="py-2 px-3 tabular-nums text-emerald-600 font-medium">
                          {summary?.max != null ? summary.max.toFixed(4) : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </ScrollArea>
        )}

        <DialogFooter className="mt-3">
          <Button variant="outline" onClick={onClose}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ─── CustomRunDialog ──────────────────────────────────────────────────────────

function CustomRunDialog({
  configs,
  allTools,
  onStartRun,
  isStarting,
  onClose,
}: {
  configs: PipelineConfig[];
  allTools: ToolInfo[];
  onStartRun: (configId: string, opts: {
    dataset_name: string;
    dataset_variant: string;
    use_cache: boolean;
    tools_override: Record<string, string[]> | null;
    model_script: string | null;
  }) => void;
  isStarting: boolean;
  onClose: () => void;
}) {
  const [selectedDataset, setSelectedDataset] = useState<DatasetId>('cifar10');
  const [poisoned, setPoisoned] = useState(false);
  const [useCache, setUseCache] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const { data: modelData } = useQuery({
    queryKey: ['model-configs'],
    queryFn: getModelConfigs,
    staleTime: 60_000,
  });
  const modelList: ModelConfigInfo[] = modelData?.models ?? [];

  const toolsByStage = useMemo<Record<Stage, ToolInfo[]>>(() => {
    const map = {} as Record<Stage, ToolInfo[]>;
    for (const stage of STAGE_ORDER) map[stage] = [];
    for (const tool of allTools) {
      const s = (tool.defense_stage ?? '') as Stage;
      if (STAGE_ORDER.includes(s)) map[s].push(tool);
    }
    return map;
  }, [allTools]);

  const defaultChecked = useMemo(() => {
    const s = {} as Record<Stage, Set<string>>;
    for (const stage of STAGE_ORDER) {
      s[stage] = new Set(toolsByStage[stage].map((t) => t.name));
    }
    return s;
  }, [toolsByStage]);
  const [checkedTools, setCheckedTools] = useState<Record<Stage, Set<string>>>(defaultChecked);

  const duringTools = toolsByStage['during_training'] ?? [];
  // Store registry key so tools_override sends the correct identifier
  const firstNonBaseline = duringTools.find((t) => !t.is_baseline);
  const defaultDuring = firstNonBaseline ? toolKey(firstNonBaseline) : (duringTools[0] ? toolKey(duringTools[0]) : '');
  const [duringSelected, setDuringSelected] = useState<string>(defaultDuring);

  function handleCheck(stage: Stage, toolName: string, checked: boolean) {
    setCheckedTools((prev) => {
      const next = { ...prev, [stage]: new Set(prev[stage]) };
      if (checked) next[stage].add(toolName);
      else next[stage].delete(toolName);
      return next;
    });
  }

  function toolKey(t: ToolInfo): string {
    return t.key ?? t.name;
  }

  function buildToolsOverride(): Record<string, string[]> {
    const override: Record<string, string[]> = {};
    for (const stage of STAGE_ORDER) {
      if (stage === 'during_training') {
        const baseline = duringTools.find((t) => t.is_baseline);
        const keys: string[] = [];
        if (duringSelected) keys.push(duringSelected);
        if (baseline && !keys.includes(toolKey(baseline))) keys.push(toolKey(baseline));
        override[stage] = keys;
      } else {
        override[stage] = toolsByStage[stage]
          .filter((t) => t.is_baseline || checkedTools[stage]?.has(t.name))
          .map(toolKey);
      }
    }
    return override;
  }

  const combinationCount = useMemo(() => {
    const byStage = {} as Record<Stage, { name: string; is_baseline: boolean }[]>;
    for (const stage of STAGE_ORDER) {
      if (stage === 'during_training') {
        const baseline = duringTools.find((t) => t.is_baseline);
        byStage[stage] = [
          ...(duringSelected ? [{ name: duringSelected, is_baseline: false }] : []),
          ...(baseline ? [{ name: baseline.name, is_baseline: true }] : []),
        ];
      } else {
        byStage[stage] = toolsByStage[stage].filter(
          (t) => t.is_baseline || checkedTools[stage]?.has(t.name)
        );
      }
    }
    return countCombinations(byStage);
  }, [toolsByStage, checkedTools, duringSelected, duringTools]);

  const baseConfig = configs[0];

  return (
    <Dialog open onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="sm:max-w-lg max-h-[90vh] flex flex-col gap-0 p-0">
        {/* Header */}
        <div className="px-6 pt-6 pb-4 border-b">
          <DialogTitle className="flex items-center gap-2 text-base font-semibold">
            <Settings2 className="h-4 w-4" />
            Custom Run
          </DialogTitle>
          <p className="text-xs text-muted-foreground mt-1">
            Configuration is locked once the run starts.
          </p>
        </div>

        {/* Tabs */}
        <Tabs defaultValue="tools" className="flex flex-col flex-1 min-h-0">
          <div className="px-6 pt-3 shrink-0">
            <TabsList className="w-full grid grid-cols-4">
              <TabsTrigger value="tools" className="gap-1.5">
                <Wrench className="h-3.5 w-3.5" />
                Tools
                <Badge variant="secondary" className="ml-0.5 text-xs px-1 py-0 h-4">
                  {combinationCount}
                </Badge>
              </TabsTrigger>
              <TabsTrigger value="dataset" className="gap-1.5">
                <Database className="h-3.5 w-3.5" />
                Dataset
              </TabsTrigger>
              <TabsTrigger value="model" className="gap-1.5">
                <FileText className="h-3.5 w-3.5" />
                Model
              </TabsTrigger>
              <TabsTrigger value="settings" className="gap-1.5">
                <Settings2 className="h-3.5 w-3.5" />
                Settings
              </TabsTrigger>
            </TabsList>
          </div>

          {/* ── Tools tab ── */}
          <TabsContent value="tools" className="flex-1 mt-0 overflow-y-auto min-h-0">
            <div className="px-6 py-4 space-y-3">
              {allTools.length === 0 ? (
                <p className="text-sm text-muted-foreground py-6 text-center">
                  No tools in registry — start the backend to load tools.
                </p>
              ) : (
                STAGE_ORDER.map((stage) => {
                  const tools = toolsByStage[stage];
                  if (tools.length === 0) return null;
                  const isDuring = stage === 'during_training';
                  return (
                    <div key={stage} className="rounded-lg border bg-muted/20">
                      <div className="flex items-center gap-2 px-3 pt-2.5 pb-1.5 border-b border-border/50">
                        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                          {STAGE_LABELS[stage]}
                        </p>
                        {isDuring && (
                          <Badge variant="outline" className="text-xs px-1.5 py-0 h-4 font-normal">single select</Badge>
                        )}
                      </div>
                      <div className="p-1">
                        {tools.map((tool) => {
                          const isBaseline = tool.is_baseline;
                          if (isDuring) {
                            return (
                              <label
                                key={tool.name}
                                className={`flex items-center gap-2.5 rounded px-2 py-2 text-sm cursor-pointer transition-colors
                                  ${isBaseline ? 'opacity-45 cursor-not-allowed' : 'hover:bg-muted/60'}`}
                              >
                                <input
                                  type="radio"
                                  name="during_training_tool"
                                  value={toolKey(tool)}
                                  checked={duringSelected === toolKey(tool)}
                                  disabled={isBaseline}
                                  onChange={() => !isBaseline && setDuringSelected(toolKey(tool))}
                                  className="h-4 w-4 accent-primary shrink-0"
                                />
                                <span className="flex-1">{tool.name}</span>
                                {isBaseline && (
                                  <Badge variant="outline" className="text-xs px-1.5 py-0 h-4 border-amber-400 text-amber-600">baseline</Badge>
                                )}
                              </label>
                            );
                          }
                          const isChecked = isBaseline || (checkedTools[stage]?.has(tool.name) ?? false);
                          return (
                            <label
                              key={tool.name}
                              className={`flex items-center gap-2.5 rounded px-2 py-2 text-sm cursor-pointer transition-colors
                                ${isBaseline ? 'opacity-45 cursor-not-allowed' : 'hover:bg-muted/60'}`}
                            >
                              <input
                                type="checkbox"
                                checked={isChecked}
                                disabled={isBaseline}
                                onChange={(e) => !isBaseline && handleCheck(stage, tool.name, e.target.checked)}
                                className="h-4 w-4 rounded border-gray-300 accent-primary shrink-0"
                              />
                              <span className="flex-1">{tool.name}</span>
                              {isBaseline && (
                                <Badge variant="outline" className="text-xs px-1.5 py-0 h-4 border-amber-400 text-amber-600">baseline</Badge>
                              )}
                            </label>
                          );
                        })}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </TabsContent>

          {/* ── Dataset tab ── */}
          <TabsContent value="dataset" className="flex-1 mt-0 overflow-y-auto min-h-0">
            <div className="px-6 py-4 space-y-3">
              <div className="grid grid-cols-2 gap-2">
                {DATASETS.map((ds) => (
                  <button
                    key={ds.id}
                    type="button"
                    onClick={() => {
                      setSelectedDataset(ds.id);
                      if (ds.id === 'celeba') setPoisoned(false);
                    }}
                    className={`rounded-lg border p-3 text-left transition-colors ${
                      selectedDataset === ds.id
                        ? 'border-primary bg-primary/5 ring-1 ring-primary'
                        : 'border-border hover:border-primary/50'
                    }`}
                  >
                    <p className="text-sm font-semibold">{ds.label}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{ds.description}</p>
                  </button>
                ))}
              </div>
              <div className={`flex items-center justify-between rounded-lg border p-3 transition-opacity ${selectedDataset === 'celeba' ? 'opacity-40 pointer-events-none' : ''}`}>
                <div className="flex items-center gap-2">
                  <ShieldAlert className="h-4 w-4 text-amber-500" />
                  <div>
                    <p className="text-sm font-medium">Poisoned dataset</p>
                    <p className="text-xs text-muted-foreground">
                      {selectedDataset === 'celeba' ? 'Not available for CelebA' : 'Use backdoor-poisoned training data'}
                    </p>
                  </div>
                </div>
                <OptionToggle label="Poisoned" checked={poisoned} onChange={setPoisoned} />
              </div>
            </div>
          </TabsContent>

          {/* ── Model tab ── */}
          <TabsContent value="model" className="flex-1 mt-0 overflow-y-auto min-h-0">
            <div className="px-6 py-4">
              {modelList.length === 0 ? (
                <p className="text-sm text-muted-foreground py-6 text-center">
                  No model configs found in configs/model/.
                </p>
              ) : (
                <div className="space-y-1.5">
                  <label className={`flex items-center gap-3 rounded-lg border p-3 text-sm cursor-pointer transition-colors ${selectedModel === null ? 'border-primary bg-primary/5' : 'hover:bg-muted/40'}`}>
                    <input
                      type="radio"
                      name="model"
                      checked={selectedModel === null}
                      onChange={() => setSelectedModel(null)}
                      className="h-4 w-4 accent-primary shrink-0"
                    />
                    <div>
                      <p className="font-medium">From config file</p>
                      <p className="text-xs text-muted-foreground">Use the model defined in the base pipeline config</p>
                    </div>
                  </label>
                  {modelList.map((m) => (
                    <label
                      key={m.path}
                      className={`flex items-center gap-3 rounded-lg border p-3 text-sm cursor-pointer transition-colors ${
                        selectedModel === m.path ? 'border-primary bg-primary/5' : 'hover:bg-muted/40'
                      }`}
                    >
                      <input
                        type="radio"
                        name="model"
                        value={m.path}
                        checked={selectedModel === m.path}
                        onChange={() => setSelectedModel(m.path)}
                        className="h-4 w-4 accent-primary shrink-0"
                      />
                      <div>
                        <p className="font-medium">{m.name}</p>
                        <p className="text-xs text-muted-foreground font-mono">{m.path}</p>
                      </div>
                    </label>
                  ))}
                </div>
              )}
            </div>
          </TabsContent>

          {/* ── Settings tab ── */}
          <TabsContent value="settings" className="flex-1 mt-0 overflow-y-auto min-h-0">
            <div className="px-6 py-4 space-y-4">
              <div className="flex items-center justify-between rounded-lg border p-4">
                <div className="flex items-center gap-2">
                  <Settings2 className="h-4 w-4 text-muted-foreground shrink-0" />
                  <div>
                    <p className="text-sm font-medium">Reuse cached results</p>
                    <p className="text-xs text-muted-foreground">
                      Skip tasks whose tool + inputs haven't changed. Disable for a fully fresh run.
                    </p>
                  </div>
                </div>
                <OptionToggle label="Use cache" checked={useCache} onChange={setUseCache} />
              </div>
            </div>
          </TabsContent>

        </Tabs>

        {/* Fixed footer: summary + actions */}
        <div className="border-t px-6 py-4 space-y-3">
          <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <Database className="h-3 w-3" />
              <span className={poisoned ? 'text-amber-500 font-medium' : ''}>
                {DATASETS.find((d) => d.id === selectedDataset)?.label}{poisoned ? ' (poisoned)' : ''}
              </span>
            </span>
            <span className="text-border">·</span>
            <span className="flex items-center gap-1">
              <FileText className="h-3 w-3" />
              {selectedModel ? modelList.find((m) => m.path === selectedModel)?.name ?? selectedModel : 'default model'}
            </span>
            <span className="text-border">·</span>
            <span className="flex items-center gap-1">
              <Wrench className="h-3 w-3" />
              {combinationCount} combo{combinationCount !== 1 ? 's' : ''}
            </span>
            <span className="text-border">·</span>
            <span>cache {useCache ? 'on' : 'off'}</span>
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={onClose}>Cancel</Button>
            <Button
              disabled={!baseConfig || isStarting}
              onClick={() => {
                if (!baseConfig) return;
                onClose();
                onStartRun(baseConfig.id, {
                  dataset_name: selectedDataset,
                  dataset_variant: poisoned ? 'poisoned' : 'clean',
                  use_cache: useCache,
                  tools_override: buildToolsOverride(),
                  model_script: selectedModel,
                });
              }}
            >
              {isStarting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
              Start Custom Run
            </Button>
          </div>
        </div>

      </DialogContent>
    </Dialog>
  );
}

// ─── ConfigCard ───────────────────────────────────────────────────────────────

function ConfigCard({
  config,
  runs,
  onStartRun,
  onStopRun,
  isStarting,
}: {
  config: PipelineConfig;
  runs: PipelineRun[];
  onStartRun: (configId: string, opts: {
    dataset_name: string;
    dataset_variant: string;
    use_cache: boolean;
    tools_override: Record<string, string[]> | null;
  }) => void;
  onStopRun: (runId: string) => void;
  isStarting: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const [quickOpen, setQuickOpen] = useState(false);
  const [useCache, setUseCache] = useState(true);
  const [metricsRunId, setMetricsRunId] = useState<string | null>(null);

  const configRuns = runs.filter((r) => r.pipeline_config_id === config.id);
  const activeRun = configRuns.find(
    (r) => r.status === 'running' || r.status === 'pending' || r.status === 'stopping'
  );
  const lastRun = configRuns[0];

  return (
    <Card className={activeRun ? 'ring-1 ring-blue-400/50' : ''}>
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
              <Button variant="destructive" size="sm" onClick={() => onStopRun(activeRun.id)}>
                <Square className="mr-2 h-4 w-4" />
                Stop
              </Button>
            ) : (
              <>
                <Button size="sm" onClick={() => setQuickOpen(true)} disabled={isStarting}>
                  {isStarting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                  Start Run
                </Button>
              </>
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

        {activeRun?.tools_config && (
          <LockedToolsDisplay toolsConfig={activeRun.tools_config} />
        )}

        {configRuns.length > 0 && (
          <>
            <Separator className="my-3" />
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex w-full items-center gap-1 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              Run History
            </button>
            {expanded && (
              <div className="mt-3 space-y-2">
                {configRuns.map((run) => (
                  <div key={run.id} className="rounded-lg border p-3 text-sm space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Badge variant="outline" className="font-mono text-xs">#{run.run_number}</Badge>
                        <StatusBadge status={run.status} />
                        <span className="text-muted-foreground font-mono text-xs">{truncateId(run.id, 16)}</span>
                        {!run.use_cache && (
                          <Badge variant="secondary" className="text-xs">no cache</Badge>
                        )}
                      </div>
                      <div className="flex items-center gap-3 text-xs text-muted-foreground">
                        <span>{formatTimestamp(run.created_at)}</span>
                        <span className="font-medium"><RunDuration run={run} /></span>
                        {run.error_message && (
                          <span className="text-destructive max-w-[200px] truncate" title={run.error_message}>
                            {run.error_message}
                          </span>
                        )}
                        {run.status === 'completed' && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2 text-xs gap-1"
                            onClick={() => setMetricsRunId(run.id)}
                          >
                            <BarChart2 className="h-3 w-3" />
                            Metrics
                          </Button>
                        )}
                      </div>
                    </div>
                    {run.tools_config && (
                      <LockedToolsDisplay toolsConfig={run.tools_config} />
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </CardContent>

      {/* ── Run Metrics dialog ── */}
      {metricsRunId && (() => {
        const run = configRuns.find((r) => r.id === metricsRunId);
        return (
          <RunMetricsDialog
            runId={metricsRunId}
            runLabel={`${config.name} #${run?.run_number ?? '?'}`}
            onClose={() => setMetricsRunId(null)}
          />
        );
      })()}

      {/* ── Quick Start dialog (cache only) ── */}
      <Dialog open={quickOpen} onOpenChange={setQuickOpen}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Play className="h-4 w-4" />
              Start Run — {config.name}
            </DialogTitle>
            <DialogDescription>
              Uses the dataset and tools defined in the config file.
            </DialogDescription>
          </DialogHeader>

          <div className="py-2">
            <div className="flex items-center justify-between rounded-lg border p-4">
              <div className="flex items-center gap-2">
                <Settings2 className="h-4 w-4 text-muted-foreground shrink-0" />
                <div>
                  <p className="text-sm font-medium">Reuse cached results</p>
                  <p className="text-xs text-muted-foreground">
                    Skip tasks whose inputs haven't changed.
                  </p>
                </div>
              </div>
              <OptionToggle label="Use cache" checked={useCache} onChange={setUseCache} />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setQuickOpen(false)}>Cancel</Button>
            <Button
              onClick={() => {
                setQuickOpen(false);
                onStartRun(config.id, {
                  dataset_name: 'cifar10',
                  dataset_variant: 'clean',
                  use_cache: useCache,
                  tools_override: null,
                });
              }}
              disabled={isStarting}
            >
              {isStarting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
              Start
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

    </Card>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function Pipelines() {
  const queryClient = useQueryClient();
  const [compareRunIds, setCompareRunIds] = useState<Set<string>>(new Set());
  const [compareOpen, setCompareOpen] = useState(false);
  const [customRunOpen, setCustomRunOpen] = useState(false);
  const [allRunsMetricsId, setAllRunsMetricsId] = useState<string | null>(null);

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

  // Fetch tools from the registry (always populated from tools.yaml at startup)
  const { data: toolsData } = useQuery({
    queryKey: ['registry-tools'],
    queryFn: getRegistryTools,
    staleTime: 60_000,
  });

  const startMutation = useMutation({
    mutationFn: ({
      configId,
      dataset_name,
      dataset_variant,
      use_cache,
      tools_override,
      model_script,
    }: {
      configId: string;
      dataset_name: string;
      dataset_variant: string;
      use_cache: boolean;
      tools_override: Record<string, string[]> | null;
      model_script?: string | null;
    }) =>
      startPipelineRun(configId, { use_cache, dataset_name, dataset_variant, tools_override, model_script }),
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
  const allTools = toolsData?.tools ?? [];
  const activeRun = runs.find((r) => r.status === 'running' || r.status === 'pending');

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
        <div className="flex items-center gap-2">
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
          <Button
            size="sm"
            onClick={() => setCustomRunOpen(true)}
            disabled={startMutation.isPending || configs.length === 0}
          >
            <Settings2 className="mr-2 h-4 w-4" />
            Custom Run
          </Button>
        </div>
      </div>

      {startMutation.isError && (
        <Card className="border-destructive">
          <CardContent className="flex items-center gap-3 py-3">
            <AlertCircle className="h-5 w-5 text-destructive shrink-0" />
            <p className="text-sm text-destructive">
              {(startMutation.error as Error)?.message || 'Failed to start pipeline run'}
            </p>
          </CardContent>
        </Card>
      )}

      {configs.length > 0 ? (
        <div className="space-y-4">
          {configs.map((config) => (
            <ConfigCard
              key={config.id}
              config={config}
              runs={runs}
              onStartRun={(id, opts) => startMutation.mutate({ configId: id, model_script: null, ...opts })}
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

      {runs.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">All Runs</CardTitle>
                <CardDescription>
                  {runs.length} total run{runs.length !== 1 ? 's' : ''} across all configs
                  {compareRunIds.size > 0 && (
                    <span className="ml-2 text-primary font-medium">
                      · {compareRunIds.size} selected
                    </span>
                  )}
                </CardDescription>
              </div>
              <div className="flex items-center gap-2">
                {compareRunIds.size > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setCompareRunIds(new Set())}
                    className="text-muted-foreground"
                  >
                    <X className="mr-1 h-3.5 w-3.5" />
                    Clear
                  </Button>
                )}
                <Button
                  variant={compareRunIds.size >= 2 ? 'default' : 'outline'}
                  size="sm"
                  disabled={compareRunIds.size < 2}
                  onClick={() => setCompareOpen(true)}
                >
                  <GitCompare className="mr-2 h-4 w-4" />
                  Compare{compareRunIds.size >= 2 ? ` (${compareRunIds.size})` : ''}
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="max-h-[400px]">
              <div className="space-y-2">
                {runs.map((run) => {
                  const cfg = configs.find((c) => c.id === run.pipeline_config_id);
                  const canCompare = run.status === 'completed' || run.status === 'failed';
                  const isSelected = compareRunIds.has(run.id);
                  return (
                    <div
                      key={run.id}
                      className={`rounded-lg border p-3 text-sm space-y-1.5 transition-colors ${
                        isSelected ? 'border-primary bg-primary/5' : ''
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          {canCompare && (
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={(e) => {
                                setCompareRunIds((prev) => {
                                  const next = new Set(prev);
                                  if (e.target.checked) next.add(run.id);
                                  else next.delete(run.id);
                                  return next;
                                });
                              }}
                              className="h-4 w-4 rounded border-gray-300 accent-primary shrink-0"
                              title="Select for comparison"
                            />
                          )}
                          {!canCompare && <div className="w-4 shrink-0" />}
                          <Badge variant="outline" className="font-mono text-xs">#{run.run_number}</Badge>
                          <span className="font-medium">{cfg?.name ?? run.pipeline_config_id}</span>
                          <StatusBadge status={run.status} />
                          {!run.use_cache && (
                            <Badge variant="secondary" className="text-xs">no cache</Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground">
                          <span className="font-mono">{truncateId(run.id, 20)}</span>
                          <span>{formatTimestamp(run.started_at || run.created_at)}</span>
                          <span className="font-medium"><RunDuration run={run} /></span>
                          {run.status === 'completed' && (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-6 px-2 text-xs gap-1"
                              onClick={() => setAllRunsMetricsId(run.id)}
                            >
                              <BarChart2 className="h-3 w-3" />
                              Metrics
                            </Button>
                          )}
                        </div>
                      </div>
                      {run.tools_config && (
                        <LockedToolsDisplay toolsConfig={run.tools_config} />
                      )}
                    </div>
                  );
                })}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {allRunsMetricsId && (() => {
        const run = runs.find((r) => r.id === allRunsMetricsId);
        const cfg = configs.find((c) => c.id === run?.pipeline_config_id);
        return (
          <RunMetricsDialog
            runId={allRunsMetricsId}
            runLabel={`${cfg?.name ?? '?'} #${run?.run_number ?? '?'}`}
            onClose={() => setAllRunsMetricsId(null)}
          />
        );
      })()}

      {compareOpen && (
        <CompareDialog
          runIds={Array.from(compareRunIds)}
          runs={runs}
          configs={configs}
          onClose={() => setCompareOpen(false)}
        />
      )}

      {customRunOpen && (
        <CustomRunDialog
          configs={configs}
          allTools={allTools}
          onStartRun={(id, opts) => startMutation.mutate({ configId: id, ...opts })}
          isStarting={startMutation.isPending}
          onClose={() => setCustomRunOpen(false)}
        />
      )}

    </div>
  );
}
