/**
 * Pipelines Page Component Tests
 *
 * Covers:
 * - Helper functions: perm, stageOptions, countCombinations
 * - CustomRunDialog: tool listing per stage (checkboxes / radio buttons)
 * - ConfigCard: quick-start dialog, cache toggle, stop button for active run
 * - CompareDialog: loading state, metrics table, no-metrics fallback
 * - Pipelines page: header, loading state, "Custom Run" button
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, within, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';

// ── Mock API ──────────────────────────────────────────────────────────────────

vi.mock('../lib/api', () => ({
  getPipelineConfigs: vi.fn(),
  getPipelineRuns: vi.fn(),
  startPipelineRun: vi.fn(),
  stopPipelineRun: vi.fn(),
  getRegistryTools: vi.fn(),
  getRunMetrics: vi.fn(),
  getModelConfigs: vi.fn(),
  setPipelineKey: vi.fn(),
  getPipelineKey: vi.fn(),
  clearPipelineKey: vi.fn(),
  isPipelineKeyAuthError: vi.fn(),
}));

import * as api from '../lib/api';

// ── Test helpers ──────────────────────────────────────────────────────────────

function makeClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: false },
    },
  });
}

function wrap(ui: React.ReactElement) {
  const client = makeClient();
  return render(
    <QueryClientProvider client={client}>
      <BrowserRouter>{ui}</BrowserRouter>
    </QueryClientProvider>
  );
}

// ── Fixtures ──────────────────────────────────────────────────────────────────

const TOOL_PRE_NOOP = {
  name: 'noop',
  container: { image: 'ghcr.io/landseer-project/pre_noop:v2', command: 'python main.py', runtime: null },
  is_baseline: true,
  defense_stage: 'pre_training',
};
const TOOL_PRE_XGBOD = {
  name: 'pre-xgbod',
  container: { image: 'ghcr.io/landseer-project/pre_xgbod:v2', command: 'python3 main.py', runtime: null },
  is_baseline: false,
  defense_stage: 'pre_training',
};
const TOOL_IN_NOOP = {
  name: 'in_noop',
  container: { image: 'ghcr.io/landseer-project/in_noop:v7', command: 'python main.py', runtime: null },
  is_baseline: true,
  defense_stage: 'during_training',
};
const TOOL_IN_DP = {
  name: 'in-dp',
  container: { image: 'ghcr.io/landseer-project/in_dp:v10', command: 'python3 main.py', runtime: null },
  is_baseline: false,
  defense_stage: 'during_training',
};
const TOOL_POST_NOOP = {
  name: 'post_noop',
  container: { image: 'ghcr.io/landseer-project/post_noop_new:v1', command: 'python main.py', runtime: null },
  is_baseline: true,
  defense_stage: 'post_training',
};
const TOOL_DEPLOY_NOOP = {
  name: 'deploy_noop',
  container: { image: 'ghcr.io/landseer-project/deploy_noop_docker:v1', command: 'python main.py', runtime: null },
  is_baseline: true,
  defense_stage: 'deployment',
};

const ALL_MOCK_TOOLS = [TOOL_PRE_NOOP, TOOL_PRE_XGBOD, TOOL_IN_NOOP, TOOL_IN_DP, TOOL_POST_NOOP, TOOL_DEPLOY_NOOP];

const MOCK_CONFIG = {
  id: 'cfg-001',
  name: 'mini',
  config_path: 'configs/pipeline/mini.yaml',
  description: 'Mini test config',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
};

const MOCK_RUN_COMPLETED = {
  id: 'run-001',
  run_number: 1,
  pipeline_config_id: 'cfg-001',
  status: 'completed' as const,
  use_cache: true,
  dataset_name: 'cifar10',
  dataset_variant: 'clean',
  tools_config: null,
  started_at: '2024-01-01T10:00:00Z',
  completed_at: '2024-01-01T11:00:00Z',
  created_at: '2024-01-01T10:00:00Z',
  error_message: null,
};

const MOCK_METRICS_RESPONSE = {
  pipeline_id: 'cfg-001',
  pipeline_name: 'mini',
  workflow_count: 2,
  metric_names: ['accuracy', 'asr'],
  workflows: [
    {
      workflow_id: 'wf-1',
      workflow_name: 'baseline',
      metrics: { accuracy: 0.9, asr: 0.1 },
      evaluators_run: ['eval'],
      evaluators_skipped: [],
      is_baseline: true,
    },
    {
      workflow_id: 'wf-2',
      workflow_name: 'defended',
      metrics: { accuracy: 0.85, asr: 0.05 },
      evaluators_run: ['eval'],
      evaluators_skipped: [],
      is_baseline: false,
    },
  ],
  summary: {
    accuracy: { min: 0.85, max: 0.9, avg: 0.875, count: 2 },
    asr: { min: 0.05, max: 0.1, avg: 0.075, count: 2 },
  },
};

// ── Import after mocks ────────────────────────────────────────────────────────

import { Pipelines } from '../pages/Pipelines';

// ── Helper function unit tests ─────────────────────────────────────────────────

// These are not exported so we test them indirectly via countCombinations.
// We also verify them through rendered combination counts in the UI.

describe('combination count logic', () => {
  beforeEach(() => {
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [MOCK_CONFIG], total: 1 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs: [], total: 0 });
    vi.mocked(api.getModelConfigs).mockResolvedValue({ models: [], total: 0 });
    vi.mocked(api.getPipelineKey).mockReturnValue(null);
    vi.mocked(api.isPipelineKeyAuthError).mockReturnValue(false);
  });

  it('shows 1 combo when only pre_training baseline tool is registered', async () => {
    // No during_training tools → duringSelected is '' (falsy) → 0 non-baseline → 1 option
    // Only a baseline pre_training tool → stageOptions(0) = 1 for every stage
    // Total = 1 × 1 × 1 × 1 = 1
    vi.mocked(api.getRegistryTools).mockResolvedValue({
      tools: [TOOL_PRE_NOOP],
      total: 1,
    });

    wrap(<Pipelines />);
    const customRunBtn = await screen.findByRole('button', { name: /custom run/i });
    await userEvent.click(customRunBtn);
    await screen.findByRole('heading', { name: /custom run/i });

    await waitFor(() => expect(document.body.textContent).toContain('1 combo'));
  });

  it('shows more than 1 combo when non-baseline tools are present', async () => {
    vi.mocked(api.getRegistryTools).mockResolvedValue({ tools: ALL_MOCK_TOOLS, total: ALL_MOCK_TOOLS.length });

    wrap(<Pipelines />);
    const customRunBtn = await screen.findByRole('button', { name: /custom run/i });
    await userEvent.click(customRunBtn);
    await screen.findByRole('heading', { name: /custom run/i });

    // With pre-xgbod and in-dp as non-baselines:
    // pre: 1 baseline + P(1,1)=1 non-baseline = 2 options
    // during: 1 baseline + 1 non-baseline = 2 options (radio, not permutation)
    // post: 1 option (baseline only), deploy: 1 option (baseline only)
    // total = 2 * 2 * 1 * 1 = 4
    await waitFor(() => expect(document.body.textContent).toContain('4 combos'));
  });
});

// ── CustomRunDialog ───────────────────────────────────────────────────────────

describe('CustomRunDialog', () => {
  beforeEach(() => {
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [MOCK_CONFIG], total: 1 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs: [], total: 0 });
    vi.mocked(api.getModelConfigs).mockResolvedValue({ models: [], total: 0 });
    vi.mocked(api.getPipelineKey).mockReturnValue(null);
    vi.mocked(api.isPipelineKeyAuthError).mockReturnValue(false);
  });

  async function openCustomRunDialog() {
    vi.mocked(api.getRegistryTools).mockResolvedValue({ tools: ALL_MOCK_TOOLS, total: ALL_MOCK_TOOLS.length });
    wrap(<Pipelines />);
    const btn = await screen.findByRole('button', { name: /custom run/i });
    await userEvent.click(btn);
    // Wait for dialog title heading to appear
    await screen.findByRole('heading', { name: /custom run/i });
  }

  it('renders four tabs: Tools, Dataset, Model, Settings', async () => {
    await openCustomRunDialog();
    expect(screen.getByRole('tab', { name: /tools/i })).toBeDefined();
    expect(screen.getByRole('tab', { name: /dataset/i })).toBeDefined();
    expect(screen.getByRole('tab', { name: /model/i })).toBeDefined();
    expect(screen.getByRole('tab', { name: /settings/i })).toBeDefined();
  });

  it('shows pre_training tools with checkboxes', async () => {
    await openCustomRunDialog();

    // Tools tab is the default view
    const preSection = screen.getByText('Pre-training').closest('div')!;
    expect(preSection).toBeDefined();

    // noop (baseline) should appear as a checked, disabled checkbox
    const noopLabel = screen.getByText('noop').closest('label')!;
    const noopCheckbox = within(noopLabel).getByRole('checkbox');
    expect(noopCheckbox).toBeDefined();
    expect((noopCheckbox as HTMLInputElement).checked).toBe(true);
    expect((noopCheckbox as HTMLInputElement).disabled).toBe(true);

    // pre-xgbod (non-baseline) should appear as an enabled checkbox
    const xgbodLabel = screen.getByText('pre-xgbod').closest('label')!;
    const xgbodCheckbox = within(xgbodLabel).getByRole('checkbox');
    expect((xgbodCheckbox as HTMLInputElement).disabled).toBe(false);
  });

  it('shows during_training tools as radio buttons (single select)', async () => {
    await openCustomRunDialog();

    // in_noop (baseline) should be a disabled radio
    const inNoopLabel = screen.getByText('in_noop').closest('label')!;
    const inNoopRadio = within(inNoopLabel).getByRole('radio');
    expect(inNoopRadio).toBeDefined();
    expect((inNoopRadio as HTMLInputElement).disabled).toBe(true);

    // in-dp (non-baseline) should be an enabled radio
    const inDpLabel = screen.getByText('in-dp').closest('label')!;
    const inDpRadio = within(inDpLabel).getByRole('radio');
    expect((inDpRadio as HTMLInputElement).disabled).toBe(false);

    // The "single select" badge should be present
    expect(screen.getByText('single select')).toBeDefined();
  });

  it('shows "No tools in registry" when tool list is empty', async () => {
    vi.mocked(api.getRegistryTools).mockResolvedValue({ tools: [], total: 0 });
    wrap(<Pipelines />);
    const btn = await screen.findByRole('button', { name: /custom run/i });
    await userEvent.click(btn);

    await screen.findByText(/no tools in registry/i);
  });

  it('footer summary shows selected dataset and cache status', async () => {
    await openCustomRunDialog();

    // Default: CIFAR-10, cache on — check via body since some text spans multiple nodes
    await waitFor(() => {
      expect(document.body.textContent).toContain('CIFAR-10');
      expect(document.body.textContent).toContain('cache on');
    });
  });

  it('can toggle cache off in Settings tab', async () => {
    await openCustomRunDialog();

    await userEvent.click(screen.getByRole('tab', { name: /settings/i }));

    // Find the cache toggle switch
    const cacheToggle = await screen.findByRole('switch', { name: /use cache/i });
    expect((cacheToggle as HTMLElement).getAttribute('aria-checked')).toBe('true');

    await userEvent.click(cacheToggle);
    expect((cacheToggle as HTMLElement).getAttribute('aria-checked')).toBe('false');

    // Footer should now say "cache off"
    await waitFor(() => expect(document.body.textContent).toContain('cache off'));
  });

  it('switches dataset to CelebA and disables poisoned toggle', async () => {
    await openCustomRunDialog();

    await userEvent.click(screen.getByRole('tab', { name: /dataset/i }));

    const celebaBtn = await screen.findByRole('button', { name: /celeba/i });
    await userEvent.click(celebaBtn);

    // Poisoned toggle row becomes non-interactive
    expect(screen.getByText(/not available for celeba/i)).toBeDefined();
  });

  it('calls onStartRun with tools_override when Start Custom Run is clicked', async () => {
    vi.mocked(api.startPipelineRun).mockResolvedValue({ ...MOCK_RUN_COMPLETED, status: 'pending' });
    await openCustomRunDialog();

    const startBtn = screen.getByRole('button', { name: /start custom run/i });
    await userEvent.click(startBtn);

    await waitFor(() => {
      expect(api.startPipelineRun).toHaveBeenCalledWith(
        MOCK_CONFIG.id,
        expect.objectContaining({
          tools_override: expect.any(Object),
          use_cache: true,
        })
      );
    });
  });
});

// ── ConfigCard quick-start dialog ─────────────────────────────────────────────

describe('ConfigCard', () => {
  beforeEach(() => {
    vi.mocked(api.getRegistryTools).mockResolvedValue({ tools: ALL_MOCK_TOOLS, total: ALL_MOCK_TOOLS.length });
    vi.mocked(api.getModelConfigs).mockResolvedValue({ models: [], total: 0 });
    vi.mocked(api.getPipelineKey).mockReturnValue(null);
    vi.mocked(api.isPipelineKeyAuthError).mockReturnValue(false);
  });

  it('shows Start Run button when no active run', async () => {
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [MOCK_CONFIG], total: 1 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs: [], total: 0 });

    wrap(<Pipelines />);

    const startBtn = await screen.findByRole('button', { name: /start run/i });
    expect(startBtn).toBeDefined();
  });

  it('opens quick-start dialog with cache toggle on click', async () => {
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [MOCK_CONFIG], total: 1 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs: [], total: 0 });

    wrap(<Pipelines />);

    const startBtn = await screen.findByRole('button', { name: /start run/i });
    await userEvent.click(startBtn);

    // Dialog should open with config name in title
    await screen.findByText(`Start Run — ${MOCK_CONFIG.name}`);
    expect(screen.getByRole('switch', { name: /use cache/i })).toBeDefined();
  });

  it('shows Stop button when there is an active run', async () => {
    const activeRun = { ...MOCK_RUN_COMPLETED, id: 'run-active', status: 'running' as const };
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [MOCK_CONFIG], total: 1 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs: [activeRun], total: 1 });

    wrap(<Pipelines />);

    await screen.findByRole('button', { name: /stop/i });
    // Start Run button should not be present
    expect(screen.queryByRole('button', { name: /^start run$/i })).toBeNull();
  });

  it('submits quick-start with null tools_override and default dataset', async () => {
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [MOCK_CONFIG], total: 1 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs: [], total: 0 });
    vi.mocked(api.startPipelineRun).mockResolvedValue({ ...MOCK_RUN_COMPLETED, status: 'pending' });

    wrap(<Pipelines />);

    const startBtn = await screen.findByRole('button', { name: /start run/i });
    await userEvent.click(startBtn);

    // Click Start in the dialog
    const dialogStartBtn = await screen.findByRole('button', { name: /^start$/i });
    await userEvent.click(dialogStartBtn);

    await waitFor(() => {
      expect(api.startPipelineRun).toHaveBeenCalledWith(
        MOCK_CONFIG.id,
        expect.objectContaining({
          tools_override: null,
          dataset_name: 'cifar10',
          dataset_variant: 'clean',
        })
      );
    });
  });
});

// ── CompareDialog ─────────────────────────────────────────────────────────────

describe('CompareDialog', () => {
  const twoCompletedRuns = [
    { ...MOCK_RUN_COMPLETED, id: 'run-001', run_number: 1 },
    { ...MOCK_RUN_COMPLETED, id: 'run-002', run_number: 2 },
  ];

  beforeEach(() => {
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [MOCK_CONFIG], total: 1 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs: twoCompletedRuns, total: 2 });
    vi.mocked(api.getRegistryTools).mockResolvedValue({ tools: [], total: 0 });
    vi.mocked(api.getModelConfigs).mockResolvedValue({ models: [], total: 0 });
    vi.mocked(api.getPipelineKey).mockReturnValue(null);
    vi.mocked(api.isPipelineKeyAuthError).mockReturnValue(false);
  });

  async function openCompareDialog() {
    wrap(<Pipelines />);

    // Select both completed runs
    const checkboxes = await screen.findAllByTitle('Select for comparison');
    await userEvent.click(checkboxes[0]);
    await userEvent.click(checkboxes[1]);

    // Compare button should be enabled now
    const compareBtn = screen.getByRole('button', { name: /compare \(2\)/i });
    await userEvent.click(compareBtn);

    await screen.findByText('Compare 2 Runs');
  }

  it('shows loading spinner while fetching metrics', async () => {
    // Never resolves — stays loading
    vi.mocked(api.getRunMetrics).mockReturnValue(new Promise(() => {}));

    await openCompareDialog();

    // Spinner should be visible
    const spinner = document.querySelector('.animate-spin');
    expect(spinner).not.toBeNull();
  });

  it('shows "no metrics" message when runs have no evaluation data', async () => {
    vi.mocked(api.getRunMetrics).mockResolvedValue({
      ...MOCK_METRICS_RESPONSE,
      metric_names: [],
      workflows: [],
      summary: {},
    });

    await openCompareDialog();

    await screen.findByText(/no evaluation metrics found/i);
  });

  it('renders metric rows with run labels when data loads', async () => {
    vi.mocked(api.getRunMetrics).mockResolvedValue(MOCK_METRICS_RESPONSE);

    await openCompareDialog();

    // Metric names should appear
    await screen.findByText('accuracy');
    await screen.findByText('asr');

    // Both run labels should appear in the header row
    expect(screen.getByText(`mini #1`)).toBeDefined();
    expect(screen.getByText(`mini #2`)).toBeDefined();
  });

  it('shows error message when metrics fetch fails', async () => {
    vi.mocked(api.getRunMetrics).mockRejectedValue(new Error('not found'));

    await openCompareDialog();

    await screen.findByText(/could not be loaded/i);
  });
});

// ── Pipelines page ─────────────────────────────────────────────────────────────

describe('Pipelines page', () => {
  beforeEach(() => {
    vi.mocked(api.getRegistryTools).mockResolvedValue({ tools: [], total: 0 });
    vi.mocked(api.getModelConfigs).mockResolvedValue({ models: [], total: 0 });
    vi.mocked(api.getPipelineKey).mockReturnValue(null);
    vi.mocked(api.isPipelineKeyAuthError).mockReturnValue(false);
  });

  it('shows loading state while queries are in flight', () => {
    vi.mocked(api.getPipelineConfigs).mockReturnValue(new Promise(() => {}));
    vi.mocked(api.getPipelineRuns).mockReturnValue(new Promise(() => {}));

    wrap(<Pipelines />);

    expect(screen.getByText(/loading pipeline configs/i)).toBeDefined();
  });

  it('shows "no pipeline configs found" when configs list is empty', async () => {
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [], total: 0 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs: [], total: 0 });

    wrap(<Pipelines />);

    await screen.findByText(/no pipeline configs found/i);
  });

  it('renders a card for each config', async () => {
    const configs = [
      MOCK_CONFIG,
      { ...MOCK_CONFIG, id: 'cfg-002', name: 'full', config_path: 'configs/pipeline/full.yaml' },
    ];
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs, total: 2 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs: [], total: 0 });

    wrap(<Pipelines />);

    await screen.findByText('mini');
    expect(screen.getByText('full')).toBeDefined();
  });

  it('disables Custom Run button when configs list is empty', async () => {
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [], total: 0 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs: [], total: 0 });

    wrap(<Pipelines />);

    const btn = await screen.findByRole('button', { name: /custom run/i });
    expect((btn as HTMLButtonElement).disabled).toBe(true);
  });

  it('Compare button is disabled until 2+ runs are selected', async () => {
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [MOCK_CONFIG], total: 1 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({
      runs: [MOCK_RUN_COMPLETED],
      total: 1,
    });

    wrap(<Pipelines />);

    // One run visible, compare button disabled initially
    const compareBtn = await screen.findByRole('button', { name: /^compare$/i });
    expect((compareBtn as HTMLButtonElement).disabled).toBe(true);
  });

  it('enables Compare button after selecting 2 completed runs', async () => {
    const runs = [
      { ...MOCK_RUN_COMPLETED, id: 'run-001', run_number: 1 },
      { ...MOCK_RUN_COMPLETED, id: 'run-002', run_number: 2 },
    ];
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [MOCK_CONFIG], total: 1 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs, total: 2 });

    wrap(<Pipelines />);

    const checkboxes = await screen.findAllByTitle('Select for comparison');
    await userEvent.click(checkboxes[0]);
    await userEvent.click(checkboxes[1]);

    const compareBtn = screen.getByRole('button', { name: /compare \(2\)/i });
    expect((compareBtn as HTMLButtonElement).disabled).toBe(false);
  });

  it('clears run selection when Clear button is clicked', async () => {
    const runs = [
      { ...MOCK_RUN_COMPLETED, id: 'run-001', run_number: 1 },
      { ...MOCK_RUN_COMPLETED, id: 'run-002', run_number: 2 },
    ];
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [MOCK_CONFIG], total: 1 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs, total: 2 });

    wrap(<Pipelines />);

    const checkboxes = await screen.findAllByTitle('Select for comparison');
    await userEvent.click(checkboxes[0]);
    await userEvent.click(checkboxes[1]);

    await screen.findByText('· 2 selected');

    const clearBtn = screen.getByRole('button', { name: /clear/i });
    await userEvent.click(clearBtn);

    // Selection cleared, "2 selected" text should be gone
    expect(screen.queryByText('· 2 selected')).toBeNull();
    expect(screen.getByRole('button', { name: /^compare$/i })).toBeDefined();
  });
});

describe('Pipeline key auth flow', () => {
  beforeEach(() => {
    vi.mocked(api.getPipelineConfigs).mockResolvedValue({ configs: [MOCK_CONFIG], total: 1 });
    vi.mocked(api.getPipelineRuns).mockResolvedValue({ runs: [], total: 0 });
    vi.mocked(api.getRegistryTools).mockResolvedValue({ tools: ALL_MOCK_TOOLS, total: ALL_MOCK_TOOLS.length });
    vi.mocked(api.getModelConfigs).mockResolvedValue({ models: [], total: 0 });
    vi.mocked(api.getPipelineKey).mockReturnValue(null);
    vi.mocked(api.isPipelineKeyAuthError).mockReturnValue(false);
  });

  it('prompts for pipeline key on auth error and retries start', async () => {
    const forbidden = new Error('forbidden');
    vi.mocked(api.startPipelineRun)
      .mockRejectedValueOnce(forbidden)
      .mockResolvedValueOnce({ ...MOCK_RUN_COMPLETED, status: 'pending' });
    vi.mocked(api.isPipelineKeyAuthError).mockImplementation((err) => err === forbidden);

    wrap(<Pipelines />);

    const startBtn = await screen.findByRole('button', { name: /^start run$/i });
    await userEvent.click(startBtn);
    await screen.findByText(`Start Run — ${MOCK_CONFIG.name}`);
    await userEvent.click(screen.getByRole('button', { name: /^start$/i }));

    await screen.findByRole('heading', { name: /pipeline access key/i });
    await userEvent.type(screen.getByPlaceholderText(/enter pipeline key/i), 'secret-key');
    await userEvent.click(screen.getByRole('button', { name: /save and retry/i }));

    await waitFor(() => {
      expect(api.setPipelineKey).toHaveBeenCalledWith('secret-key');
      expect(api.startPipelineRun).toHaveBeenCalled();
    });

    const lastCall = vi.mocked(api.startPipelineRun).mock.calls.at(-1);
    expect(lastCall?.[0]).toBe(MOCK_CONFIG.id);
  });

  it('supports clearing an existing session key', async () => {
    vi.mocked(api.getPipelineKey).mockReturnValue('existing-key');

    wrap(<Pipelines />);

    await userEvent.click(await screen.findByRole('button', { name: /pipeline key set/i }));
    await screen.findByRole('heading', { name: /pipeline access key/i });
    await userEvent.click(screen.getByRole('button', { name: /clear key/i }));

    expect(api.clearPipelineKey).toHaveBeenCalled();
  });
});
