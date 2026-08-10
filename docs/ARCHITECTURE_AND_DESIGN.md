# Landseer: Architecture, Design Reasoning & Future Work

> This document explains the **why** behind the distributed rewrite, how the new system works end-to-end, and concrete improvements that would make the system more capable for the research use case.

---

## 1. What Landseer Does (One-Page Summary)

Landseer studies whether ML defense techniques (adversarial robustness, differential privacy, outlier removal, watermarking, fingerprinting, fairness, explainability) **compose** with each other when applied at different stages of an ML pipeline (pre-training → during-training → post-training → deployment).

The core computational problem is:

- There are ~14 tools across 7 categories and 4 pipeline stages.
- Each tool is a Docker container with standardized inputs (`/data`, `/config`) and outputs (`/output`).
- Every valid ordered permutation of tools across stages is a **workflow** (~250+ total).
- Many workflows share sub-sequences (e.g., every workflow that uses `pre_xgbod` + `in_trades` shares the same "pre + during" subtree).
- For each workflow, we run attacks and evaluate a vector of metrics: `(clean_accuracy, pgd_accuracy, attack_success_rate, dp_epsilon, fingerprint_score, ...)`.
- Results are analyzed via **interference graphs** to identify which tool combinations degrade each other's guarantees.

---

## 2. Why We Rewrote: From Monolithic to Distributed

### What the old system did

`src_old/` runs as a single Python process:

```
main.py → PipelineExecutor → ThreadPoolExecutor → DockerRunner per thread
                                                 → ArtifactCache (disk)
                                                 → ModelEvaluator (after each combo)
→ Results written to CSV
```

This works for a single machine but has hard limits:

| Problem | Impact |
|---------|--------|
| Single-machine execution | All 250 workflows on one GPU cluster |
| In-memory state only | Crash = restart everything |
| No task-level recovery | Rerunning always re-executes cached hits locally, but cache isn't shared |
| CSV-only results | No queryable state for dashboard or partial analysis |
| Docker-only | Won't work on HPC clusters that use Apptainer/Singularity |
| No live progress | Must wait for full run to finish to see anything |

### What the new system provides

`src/` decomposes the system into three separately deployable services:

1. **Backend** (`src/backend/`) — FastAPI REST API + priority scheduler. Knows about all tasks, their dependencies, and which are ready to run.
2. **Workers** (`src/worker/`) — Stateless executors. Poll the backend for tasks, run containers, report results. Can be scaled horizontally.
3. **Database** (`src/db/`) — MySQL via SQLAlchemy. All task state is durable. A worker crash just leaves a task in `RUNNING` state; the backend reassigns it.

The two-level cache (`src/store/`) adds:
- **Level 1**: Local worker disk — fast, zero-network retrieval.
- **Level 2**: MinIO (S3-compatible) — shared across all workers on a cluster; enables cross-worker deduplication.

---

## 3. Core Data Model

```
PipelineRun
  └── Pipeline
        ├── config (dataset, model, tool lists)
        └── Workflows [1..N]
              └── ordered list of Tasks

Task (atomic unit)
  ├── tool (name, Docker image, command)
  ├── dependencies: [Task, ...] (forms a DAG)
  ├── task_hash: SHA256(tool + config + sorted dep hashes)
  ├── cache_key: used for artifact lookup
  ├── status: PENDING | QUEUED | RUNNING | COMPLETED | FAILED
  ├── priority: integer (scheduler sorts by this)
  └── result_metadata: JSON (metrics, artifact path, logs)
```

**Key invariant**: If two workflows share the same tool applied to the same prior results, they share **one Task object** in the database. The task runs once; both workflows consume its output. This is the deduplication that makes the whole system efficient.

---

## 4. How Workflows Are Generated

Given a pipeline config like:
```yaml
pre_training:  [pre_xgbod, pre_noop]
during_training: [in_trades, in_noop]
post_training: [post_fineprune, post_noop]
deployment: [deploy_dp, deploy_noop]
```

`WorkflowGenerator` does:

1. **Pre/Post/Deployment stages**: generate all **permutations** of all non-empty subsets + the baseline (noop alone). Order matters because applying tool A before tool B produces a different model than B before A.

   For `[post_fineprune, post_noop]` (where `post_noop` is baseline):
   - `[post_fineprune, post_noop]` — both, in this order
   - `[post_noop, post_fineprune]` — both, reversed
   - `[post_fineprune]` — fineprune only
   - `[post_noop]` — baseline only

2. **During-training stage**: **single tools only** (no permutations). The training loop is a single function; you can't interleave two optimizers. Options: `[in_trades]` or `[in_noop]`.

3. **Cross-stage cross product**: all combinations of (pre options) × (during options) × (post options) × (deploy options) → each tuple is one workflow.

4. **Task deduplication**: `TaskFactory.get_or_create_task()` hashes `(tool, config, sorted dependency hashes)`. If two workflows would create the same task, they share a single `Task` row in the DB.

The result is a DAG where upstream shared tasks are parents to many downstream tasks.

---

## 5. Scheduling and Priority

The `PriorityScheduler` maintains a ready queue: tasks whose dependencies are all `COMPLETED`.

Priority formula:
```
depth       = longest path from this task to a root (no-dependency) task
base        = 100 - (depth × 10)          # roots get 100, deeper = lower base
counter     = number of workflows sharing this task (capped at 9)
priority    = base + counter               # shared tasks get a slight boost

# Evaluation tasks get an extra +32 boost (capped at 108)
# so metrics run as soon as the model is ready
```

**Why depth-based priority?** Earlier-stage tasks unblock more downstream work. Running `pre_xgbod` first means all workflows that need it can immediately proceed rather than waiting. This minimizes the critical path.

**Why counter bonus?** A task shared by many workflows is on the critical path for many results. Running it first produces the most unblocking per unit of compute.

---

## 6. Caching Architecture (Two-Level)

```
Worker receives Task T
  │
  ├─ Check local cache (~/.landseer_cache/<cache_key>/)
  │    Hit → return artifact path directly (zero network)
  │
  ├─ Check MinIO bucket landseer-artifacts/<task_id>/<cache_key>/
  │    Hit → download to local cache → return path
  │
  └─ Miss → execute container → save to local + upload to MinIO
```

**Cache key** = `task.cache_key` (derived from task hash). Two tasks with the same tool, config, and inputs produce the same key.

**LRU eviction**: When local cache exceeds 90% of `max_local_size_gb` (default 50 GB), oldest-accessed entries are removed. MinIO retains everything.

**Why MinIO?** A second worker that needs the same artifact (e.g., starting from `pre_xgbod` output) downloads it from MinIO instead of re-running the container. This enables true cross-worker sharing without a shared filesystem.

---

## 7. Worker Execution Loop

```python
while running:
    heartbeat(backend)                    # keeps worker registration alive
    
    task = get_next_task(backend)         # backend returns highest-priority ready task
    if task is None:
        sleep(poll_interval)
        continue
    
    if cache_hit(task.cache_key):
        report_completion(task, from_cache=True)
        continue
    
    # prepare workspace: mount /data (previous outputs), /output, /config
    workspace = prepare_workspace(task)
    
    # detect runtime: Docker > Apptainer > Singularity
    runner = ContainerRuntime.detect()
    result = runner.run(task.tool.image, workspace, gpu_id=gpu_id)
    
    cache.put(task.cache_key, result.output_dir)  # local + MinIO
    report_completion(task, result)               # updates DB via REST
```

**Fault tolerance**: If a worker dies mid-task, the backend sees a missing heartbeat and marks the task `FAILED` (or back to `PENDING` after a timeout). Another worker picks it up. No duplicate execution because only one worker holds the task at a time.

---

## 8. Container Interface (Tool Contract)

Every defense and evaluator is a Docker/OCI image that obeys this interface:

| Mount | Direction | Contents |
|-------|-----------|----------|
| `/data` | Input | Outputs from all upstream tasks: `data.npy`, `test_data.npy`, `labels.npy`, `model.pt`, etc |
| `/config` | Input | `config_model.py` (model architecture), dataset params |
| `/output` | Output | Tool writes its artifacts here: `model.pt` (updated model) and/or `data.npy` (transformed dataset) |

**Image labels** (used by scheduler for validation):
```
stage=pre|during|post|deployment
dataset=cifar10|mnist|celeba|...
defense_type=adversarial|outlier|privacy|watermark|fingerprint|fairness|explanation
```

This means adding a new defense = write a Dockerfile, push to a registry, add entry to `configs/tools.yaml`. No code changes to Landseer core.

---

## 9. Evaluation

After the last deployment-stage task in a workflow completes, evaluator tasks are queued automatically. Each evaluator is also a container:

| Evaluator | Metrics Produced |
|-----------|-----------------|
| `clean` | `clean_accuracy` |
| `adversarial` | `pgd_accuracy`, `fgsm_accuracy`, `carlini_l2_accuracy` |
| `backdoor` | `attack_success_rate`, `backdoor_robustness` |
| `fairness` | `demographic_parity`, `equalized_odds_diff` |
| `fingerprinting` | `mingd_score`, `fingerprint_accuracy` |
| `ood` | `ood_auc`, `fpr_at_95_tpr` |
| `watermark` | `watermark_accuracy`, `bit_accuracy`, `detection_rate` |

Evaluators get a +32 priority boost so they run before new tool tasks. This means you start seeing metrics rolling in as soon as early workflows complete, rather than waiting for the entire run.

---

## 10. Database Schema (Key Tables)

```sql
-- Tasks: the atomic unit of work
tasks (id, tool_name, tool_image, tool_command, task_type,
       task_hash, cache_key, priority, status,
       counter,          -- # workflows sharing this task
       attempts,         -- for retry logic
       assigned_worker_id, pipeline_id, run_id,
       artifact_path, execution_time_ms, result_metadata,
       created_at, updated_at, started_at, completed_at)

-- Many-to-many: task depends on task
task_dependencies (task_id, dependency_id)

-- Many-to-many: task participates in workflow
task_workflows (task_id, workflow_id)

-- Workflows: ordered list of tasks
workflows (id, name, pipeline_id, run_id, status, workflow_json)

-- Pipeline: one config instantiation
pipelines (id, name, run_id, config, status)

-- Workers: registered executors
workers (id, hostname, status, capabilities,
         current_task_id, tasks_completed, tasks_failed,
         last_heartbeat)

-- Run: top-level execution tracking
pipeline_runs (id, pipeline_id, status, total_tasks, started_at, completed_at)
```

---

## 11. API Endpoints Reference

```
# Health
GET  /health

# Pipeline runs
POST /api/pipeline-configs/{config_name}/runs      # Start a new run
GET  /api/pipeline-configs/{config_name}/runs      # List all runs
GET  /api/pipeline-configs/{config_name}/runs/{run_id}

# Task management (used by workers)
GET  /api/tasks?status=pending                     # Get next task
POST /api/tasks/{task_id}/status                   # Report completion
GET  /api/tasks/{task_id}                          # Task details
GET  /api/tasks/{task_id}/priority                 # Priority debug info

# Worker management
POST /api/workers/register                         # Register worker
POST /api/workers/{worker_id}/heartbeat
GET  /api/workers

# Scheduler status
GET  /api/scheduler/status
GET  /api/scheduler/progress/{pipeline_id}
```

---

## 12. How to Add a New Defense Tool

1. **Write the defense** in a Docker container obeying the `/data → /output` interface.
2. **Add image labels**: `stage`, `dataset`, `defense_type`.
3. **Add entry to `configs/tools.yaml`**:
   ```yaml
   tools:
     my_new_defense:
       name: my_new_defense
       is_baseline: false
       container:
         image: ghcr.io/landseer-project/my_new_defense:v1
         command: python main.py
   ```
4. **Add to pipeline config** (`configs/pipeline/my_experiment.yaml`):
   ```yaml
   pipeline:
     post_training:
       tools: [post_fineprune, my_new_defense, post_noop]
   ```
5. Landseer will automatically generate all permutations including the new tool and handle caching/scheduling.

---

## 13. Current Limitations

These are gaps in the current implementation relative to the research goals stated in the paper:

| Limitation | Impact |
|-----------|--------|
| No hyperparameter sweep | Can't study how DP epsilon, AT perturbation budget, etc. affect composability |
| No interference graph builder | The paper describes Algorithm 1 & 2 but no code implements the graph traversal |
| No automated threshold detection | Table III in paper shows interference levels; these must be computed manually |
| During-stage limited to one tool | Paper acknowledges this; real pipelines might need multiple in-training techniques |
| No cross-dataset evaluation | Each run fixes one dataset; composability may differ by dataset |
| No model architecture sweep | Results depend on ResNet vs VGG vs ViT; not explored |
| No optimizer variation | Each tool uses its own optimizer; interaction between optimizer choices not studied |
| Workers don't validate image digests | Image can be updated under the same tag, breaking reproducibility |
| No experiment versioning | Rerunning the same config overwrites previous results in DB |

---

## 14. Future Work: Concrete Improvements

### 14.1 Interference Graph Builder

The paper (Algorithms 1 & 2) describes the full interference analysis pipeline but this is not implemented in `src/`. It should be a post-processing step that:

1. Loads all completed workflow results from the DB.
2. For each tool, builds a directed graph where nodes = `Comb` (workflow ID) and edges = vertical (tool added) or horizontal (reordering).
3. Traverses the graph to find root-cause interference: lowest-cardinality `Comb` where a metric drops by more than threshold `t`.
4. Classifies interference as: Global / Signal-chain, Dominating / Cross / Performance.

**File to create**: `src/analysis/interference_graph.py`

### 14.2 Hyperparameter Sweep Support

Extend the YAML config to allow parameter grids:

```yaml
pipeline:
  during_training:
    tools:
      - name: in_dp
        params:
          epsilon: [0.1, 1.0, 10.0]   # sweep these values
          delta: 1e-5
      - in_noop
```

The `WorkflowGenerator` would instantiate one task per parameter combination. Each gets a unique task hash (params are part of the hash), so caching still works correctly.

### 14.3 Worker Task Assignment (Pull → Push)

Currently workers poll (`GET /api/tasks?status=pending`). For large clusters, polling creates backend load. An alternative:

- Backend maintains a WebSocket or SSE stream per worker.
- When a task becomes ready, backend pushes assignment directly.
- Workers ACK and begin immediately.

This reduces latency from poll-interval to near-zero and eliminates redundant requests.

### 14.4 Automatic Metric Threshold Detection

After a run, the system should compute interference significance automatically:

```python
# For each (workflow_A, workflow_B) where B = A + one tool:
delta = metric_B - metric_A
severity = "negligible" if |delta| < 0.02 else "moderate" if |delta| < 0.05 else "severe"
direction = "positive" if delta > 0 else "negative"
```

This mirrors Table III in the paper and should be surfaced in the web dashboard.

### 14.5 Experiment Versioning

Add a `version` or `tag` field to `PipelineRun`. Allow running the same config multiple times (e.g., with different seeds) without overwriting results. Store results per `(pipeline_config, run_id, seed)` tuple.

### 14.6 Image Digest Pinning

Tools should be referenced by digest, not tag:

```yaml
container:
  image: ghcr.io/landseer-project/in_trades@sha256:abc123...
```

The backend verifies the digest at task submission time. This prevents "same tag, different image" reproducibility failures — one of the key problems identified in Section IV of the paper.

### 14.7 Cross-Dataset Evaluation

The current design runs one dataset per `PipelineRun`. To study cross-dataset composability, allow:

```yaml
datasets:
  - name: cifar10
  - name: mnist
  - name: celeba
```

The workflow generator would produce one set of workflows per dataset. Tasks for different datasets never share cache entries (dataset name is part of the task hash).

### 14.8 Optimizer Variation Study

The paper notes (Section IV.B) that different tools use different optimizers (SGD, Adam, DP-SGD) and this complicates comparisons. A future study could:

1. Normalize all during-training tools to accept an `optimizer` parameter.
2. Sweep optimizer choices as part of the hyperparameter grid.
3. Report whether metric differences between tool combinations are attributable to the optimizer choice vs the defense itself.

### 14.9 Result Export & Comparison

Add an export endpoint:

```
GET /api/results/export?format=csv&pipeline_id=42
GET /api/results/interference-table?pipeline_id=42
```

That produces the equivalent of Table III / Table IV from the paper in machine-readable form for use in analysis notebooks.

### 14.10 Heartbeat-Based Dead Worker Detection

Currently a worker that dies mid-task leaves the task stuck in `RUNNING`. The backend needs a background job:

```python
# Every 60s:
for task in db.query(Task).filter(status=RUNNING):
    worker = task.assigned_worker
    if now() - worker.last_heartbeat > HEARTBEAT_TIMEOUT:
        task.status = PENDING
        task.attempts += 1
        if task.attempts > MAX_ATTEMPTS:
            task.status = FAILED
```

This makes the system self-healing without manual intervention.

### 14.11 Parallel During-Training

The paper restricts the during-training stage to at most one tool. This is reasonable for the current tools (each fully replaces the training loop). However, some future tools (e.g., regularization terms that can be added on top of adversarial training) could be composed. A flag `composable: true` in the tool YAML could mark tools that support stacking, allowing the generator to produce multi-tool during-training sequences for those.

---

## 15. Directory Reference

```
src/
├── backend/
│   ├── api.py                  # FastAPI routes
│   ├── db_service.py           # DB read/write operations
│   └── scheduler/
│       └── priority_scheduler.py   # Priority queue + task assignment
├── worker/
│   ├── cli.py                  # Worker entry point + main loop
│   ├── runner.py               # Container execution (Docker/Apptainer)
│   └── client.py               # HTTP client for backend API
├── pipeline/
│   ├── pipeline.py             # Pipeline / Workflow / Task classes
│   ├── workflow_generator.py   # Permutation generation
│   ├── workflow_factory.py     # Workflow construction
│   ├── task_factory.py         # Task deduplication
│   └── config_loader.py        # YAML parsing + tool registry
├── db/
│   └── models.py               # SQLAlchemy ORM models
├── store/
│   └── cache.py                # TwoLevelCache (local + MinIO)
├── common/
│   └── ...                     # Shared types, logging, config
└── frontend/
    └── ...                     # React dashboard

src_old/                        # Reference: original monolithic system
├── landseer_pipeline/
│   ├── main.py
│   ├── pipeline/
│   │   ├── runner.py           # PipelineExecutor
│   │   └── artifact_cache.py  # Content-addressable cache
│   ├── scheduler/
│   │   └── dependency_scheduler.py
│   ├── evaluator/
│   │   └── model_evaluator.py
│   ├── container_handler/
│   │   └── docker_impl.py
│   └── dataset_handler/
│       └── manager.py

configs/
├── tools.yaml                  # Tool registry (name → image)
├── evaluators.yaml             # Evaluator definitions + metrics
├── pipeline/                   # Per-experiment pipeline configs
└── model/
    └── config_model.py         # Model architecture
```

---

## 16. Quick-Start Mental Model

```
You write:  configs/pipeline/my_experiment.yaml
            configs/tools.yaml (tool registry)

Landseer computes: all valid ordered permutations of tools across stages
                   = ~250 workflows for the current 14-tool corpus

For each workflow: runs each tool's Docker container in sequence
                   (pre → during → post → deploy)
                   caching intermediate results shared across workflows

For each completed workflow: runs evaluator containers
                             collects (clean_acc, pgd_acc, ASR, DP_epsilon, ...)

You analyze: which tool combinations drop which metrics by how much
             = interference graph traversal (Algorithm 1 & 2 from paper)
```

The distributed rewrite makes step 3 horizontally scalable: add more workers → more parallel workflows → shorter wall-clock time for the full 250-combination study.
