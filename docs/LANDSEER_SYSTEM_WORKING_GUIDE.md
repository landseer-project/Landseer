# Landseer System Working Guide

This guide documents the current Landseer code structure and runtime behavior end-to-end.
It is intended to be the "single source of truth" for how pipeline execution works today.

---

## 1) High-Level Architecture

Landseer is a backend-worker pipeline orchestration system with containerized task execution and optional two-level caching.

Core runtime components:

- `src/backend/*`: backend bootstrap, API, scheduler, pipeline run control, DB sync.
- `src/pipeline/*`: pipeline/workflow/task models, config loading, combination generation.
- `src/worker/*`: worker loop, task claiming, container execution, cache interaction.
- `src/store/*`: MinIO integration, two-level cache, artifact manager.
- `src/db/*`: SQLAlchemy models/repositories and persistence layer.
- `configs/*`: tool definitions and pipeline definitions.

Execution topology:

1. Backend starts, loads config, builds pipeline, prepares dataset, initializes scheduler.
2. Workers register, heartbeat, claim tasks, run containers, report status/results.
3. Scheduler advances task states based on dependencies and priority.
4. Cache is consulted before execution and updated after successful execution.
5. API exposes task/workflow/pipeline run state and control endpoints.

---

## 2) Code Structure Map

### Backend Layer

- `src/backend/cli.py`
  - CLI entrypoint (`landseer-backend`).
  - Calls backend initialization and starts FastAPI server.

- `src/backend/initialization.py`
  - Initializes tool registry.
  - Loads pipeline config and creates pipeline/workflows/tasks.
  - Prepares dataset (host side) and optionally uploads to MinIO.
  - Initializes DB service and syncs pipeline metadata.

- `src/backend/api.py`
  - FastAPI app and all runtime endpoints.
  - Maintains `SchedulerState` (scheduler + workers + DB/store references).
  - Exposes:
    - health/info,
    - task scheduling/status APIs,
    - worker registration/claim/heartbeat APIs,
    - pipeline config/run APIs (`/api/pipeline-configs/...` and `/api/pipeline-runs/...`).

- `src/backend/scheduler/base_scheduler.py`
  - Generic scheduler behavior:
    - ready check = `PENDING` + all deps `COMPLETED`,
    - task status update,
    - progress accounting.

- `src/backend/scheduler/priority_scheduler.py`
  - Concrete scheduler:
    - priority based on dependency depth (`100 - depth*10`, floor 10),
    - tie-break via task reuse counter (`counter` bonus),
    - returns highest-priority ready task and marks it `RUNNING`.

### Pipeline Modeling Layer

- `src/pipeline/tasks.py`
  - Task model and task types:
    - `PRE_TRAINING`, `IN_TRAINING`, `POST_TRAINING`, `DEPLOYMENT`, `EVALUATION`.
  - Task hash identity from tool image/command/config/dependencies.
  - Global dedup registry (`get_or_create_task`) for task reuse across workflows.

- `src/pipeline/workflow.py`
  - Workflow = ordered set of tasks (one tool-chain combination).
  - Supports lookup helpers and restart integration hooks.

- `src/pipeline/pipeline.py`
  - Pipeline = collection of workflows.
  - `DefenseEvaluationPipeline` is current concrete implementation.

- `src/pipeline/workflow_generator.py`
  - Generates stage permutations and all workflow combinations.
  - Applies stage-specific rules and task dedup.

- `src/pipeline/config_loader.py`
  - Validates YAML via Pydantic models.
  - Converts stage config into workflow combinations.
  - Builds pipeline/workflows/tasks and attaches evaluation tasks if configured.

- `src/pipeline/tools.py`
  - Tool definition registry loaded from `configs/tools.yaml`.

### Worker Execution Layer

- `src/worker/cli.py`
  - CLI entrypoint (`landseer-worker`).
  - Worker loop:
    1. connect/register,
    2. fetch dataset metadata,
    3. claim task,
    4. check cache,
    5. execute task via `TaskRunner`,
    6. report status,
    7. repeat until complete.

- `src/worker/client.py`
  - HTTP client for API endpoints:
    - register/heartbeat/claim,
    - task complete/failed reporting,
    - progress and dataset info retrieval.

- `src/worker/runner.py`
  - Container execution abstraction (Docker / Apptainer / Singularity).
  - Prepares per-task workspace and mounts:
    - `/input`, `/output`, and `/data`.
  - Pulls images, runs tool command, captures logs, enforces timeout.

### Storage & Cache Layer

- `src/store/minio_store.py`
  - MinIO client wrapper:
    - bucket ensure,
    - file/directory upload and download,
    - object listing and metadata.

- `src/store/artifact_manager.py`
  - Coordinates local + MinIO artifact operations.
  - Stores manifest and `.success` markers in local cache.

- `src/store/cache.py`
  - Two-level cache facade:
    - L1 local cache,
    - L2 MinIO cache.
  - LRU-like eviction, prefetch support, optional auto-upload.

### Persistence Layer

- `src/db/models.py`
  - SQLAlchemy models for tasks, workers, workflows, pipelines, runs, artifacts, evaluation results.

- `src/backend/db_service.py`
  - Syncs in-memory scheduler state to DB.
  - Tracks worker/task assignments and execution outcomes.

---

## 3) Scheduling: How It Works

Scheduling in practice is pull-based by workers:

1. Worker calls `POST /workers/{worker_id}/claim`.
2. Backend delegates to scheduler `get_next_task()`.
3. Scheduler computes ready set (`PENDING` + deps completed).
4. Ready tasks are sorted by priority and first is marked `RUNNING`.
5. Worker executes and reports completion/failure via `PUT /tasks/status`.
6. Scheduler updates status and recomputes priorities.

Priority model:

- Primary: dependency depth (root tasks first).
- Secondary: workflow reuse count (`counter`) so shared tasks are preferred.

Task state model:

- `pending -> running -> completed|failed`
- `cancelled` used for stop/cancel flows.

Important behavior:

- No task runs until all direct dependencies are `COMPLETED`.
- Failed dependencies block downstream tasks from becoming ready.
- Worker claim endpoint naturally distributes work across active workers.

---

## 4) Cache System: How It Works

Landseer supports two cache modes:

1. Local-only cache (worker filesystem).
2. Two-level cache (local + MinIO shared cache).

Lookup path on worker:

1. Compute cache key from task identity:
   - tool name/image/command,
   - task config,
   - parent hashes.
2. Check local cache (`L1`).
3. If miss and MinIO enabled, check/download from MinIO (`L2`).
4. On successful task execution, store output to local cache and optionally upload to MinIO.

Local cache layout:

- per cache key directory,
- `output/` artifact payload,
- `manifest.json` metadata,
- `.success` marker to indicate complete cache entry.

Eviction:

- Triggered by configured size threshold.
- Removes least-recently-used local entries.
- If MinIO is enabled, attempts upload before local eviction.

Key environment variables:

- `LANDSEER_CACHE_DIR`
- `LANDSEER_CACHE_MAX_SIZE_GB`
- `LANDSEER_USE_MINIO`
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`, `MINIO_SECURE`

---

## 5) One Workflow Lifecycle (Concrete)

A workflow (for example `comb_001`) is an ordered chain of stage tasks:

- pre-training tools (possibly multiple in permutation order),
- exactly one during-training tool,
- post-training tools (possibly multiple),
- deployment tools (possibly multiple),
- optional evaluation tasks.

For each workflow task:

1. Scheduler marks task `RUNNING` when claimed.
2. Worker prepares workspace:
   - copies dataset/model/dependency outputs into task input dir.
3. Worker runs container command with mounted input/output.
4. On success:
   - worker reports `completed`,
   - logs and artifact metadata are attached to result payload,
   - cache can be populated.
5. On failure:
   - worker reports `failed` and error message/log snippet.

Dependencies:

- first tool in a stage depends on final task of previous stage,
- within stage, tools are chained (`tool_2` depends on `tool_1`).

---

## 5.1) Input/Output Handling Between Tasks (Critical)

This is one of the most important and easiest-to-misunderstand parts of the system.

### Per-task workspace contract

For each claimed task, worker creates:

- `<workspace>/<task_id>/input`
- `<workspace>/<task_id>/output`
- `<workspace>/<task_id>/logs`

Before container execution:

1. Dataset / base input is copied into task `input`.
2. Outputs from dependency tasks are copied into the same `input`.
3. Model script (for example `config_model.py`) is copied/mounted for container imports.

### Container mount contract

Task runner mounts:

- `input` -> `/input` (read-only)
- `output` -> `/output` (read-write)
- `input` -> `/data` (read-only alias for tools expecting `/data`)

Also sets:

- `INPUT_DIR=/input`
- `OUTPUT_DIR=/output`
- `PYTHONPATH` includes `/input:/app` for config/script imports.

### Dependency artifact chaining

When a task has `dependency_ids`, worker resolves each dependency output dir and copies artifacts forward.
This enables stage-to-stage chaining (for example trained model from in-training stage used by post-training stage).

### What tools should assume

Tool containers should:

- read all incoming artifacts from `/input` (or `/data`),
- write all produced artifacts to `/output`,
- avoid writing critical outputs outside `/output`.

### Known caveat

Dependency output resolution currently relies on workspace output directories of dependency task IDs.
If outputs are not present there (for example due to cleanup, relocation, or incomplete cache backfill), chaining can fail or be partial.

---

## 6) Pipeline Execution Model

Pipeline creation (`create_pipeline_from_config`):

1. Load tools from `configs/tools.yaml`.
2. Load pipeline YAML (for example `configs/pipeline/trades.yaml`).
3. Build stage options:
   - pre/post/deployment: permutations of subsets + baseline substitution,
   - during-training: single tool option only.
4. Cartesian product across stages to create all workflow combinations.
5. Deduplicate tasks across workflows when identity matches.
6. Optionally append evaluator tasks.

Run control (API):

- Start run: `POST /api/pipeline-configs/{config_id}/runs`
- Get run: `GET /api/pipeline-runs/{run_id}`
- List runs: `GET /api/pipeline-configs/{config_id}/runs`
- Stop run: `POST /api/pipeline-runs/{run_id}/stop`
- Restart run: `POST /api/pipeline-runs/{run_id}/restart`

Run semantics:

- Active-run guard prevents starting duplicate active runs for same config.
- Run ID is propagated to pipeline/workflows/tasks for traceability.
- Restart creates a new run with a new run ID.

---

## 7) Config Model and Semantics

### Tool Config (`configs/tools.yaml`)

Defines each tool:

- `name`
- `is_baseline` (noop substitute)
- container:
  - `image`
  - `command`
  - `runtime` (optional)

### Pipeline Config (`configs/pipeline/*.yaml`)

Defines:

- `dataset`:
  - `name`, `variant`, optional params
- `model`:
  - script path and framework
- `pipeline` stages:
  - `pre_training`, `during_training`, `post_training`, `deployment`
  - each stage lists tool IDs from tools registry

Current `trades.yaml` maps to 4 stages and generates many workflow combinations due to permutation/cartesian logic.

### Evaluator Config (`configs/evaluators.yaml`)

- Optional.
- If present, evaluator tasks are added after deployment.
- Missing required artifacts are expected to skip gracefully at runtime.

---

## 8) Dataset and MinIO Flow

Backend startup can prepare dataset on host:

1. Resolve dataset config from pipeline YAML.
2. Prepare dataset locally under configured data path.
3. If MinIO available, upload dataset under:
   - `datasets/{dataset_name}/{variant}`
4. Expose dataset metadata to workers via API.

Worker dataset resolution:

1. Use manual `--data-path` if provided.
2. Else query backend dataset info.
3. Prefer local path if accessible.
4. Else download from MinIO into worker cache area.

---

## 9) API Surface (Operationally Important)

### Health and Info

- `GET /health`
- `GET /info/pipeline`
- `GET /info/workflows`

### Task Operations

- `GET /tasks/next` (anonymous claim)
- `PUT /tasks/status` (completed/failed)
- `GET /tasks`
- `GET /tasks/{task_id}`
- `GET /tasks/{task_id}/logs`
- `GET /tasks/{task_id}/priority`

### Worker Operations

- `POST /workers/register`
- `POST /workers/{worker_id}/heartbeat`
- `POST /workers/{worker_id}/claim`
- `GET /workers`
- `GET /workers/{worker_id}`

### Progress and Scheduler

- `GET /progress`
- `GET /progress/ready`
- `GET /progress/blocked`
- `POST /scheduler/initialize`
- `POST /scheduler/reset`
- `GET /scheduler/status`

### Pipeline Configs / Runs

- `GET /api/pipeline-configs`
- `GET /api/pipeline-configs/{config_id}`
- `POST /api/pipeline-configs/{config_id}/runs`
- `GET /api/pipeline-runs/{run_id}`
- `GET /api/pipeline-configs/{config_id}/runs`
- `POST /api/pipeline-runs/{run_id}/stop`
- `POST /api/pipeline-runs/{run_id}/restart`

---

## 10) Database and State Persistence

Persistent entities include:

- pipelines,
- workflows,
- tasks,
- workers,
- pipeline runs,
- artifacts,
- evaluation results.

State synchronization:

- Backend syncs pipeline/workflow/task models to DB on initialization/start-run.
- Worker/task events update task and worker records.
- Run status transitions are stored in `pipeline_runs`.

This enables:

- history,
- observability,
- partial recovery and debugging after crashes.

---

## 11) Failure/Recovery Behavior

Task-level failures:

- worker reports `failed` with error message and logs snippet,
- dependent tasks remain blocked unless restarted/replanned.

Pipeline stop:

- run status transitions to stopping/cancelled,
- pending tasks for that run are marked `CANCELLED`.

Restart:

- creates a new run with a fresh run ID,
- rebuilds pipeline/workflows/tasks for that run.

Workflow restart hooks exist (`workflow_restart.py` + workflow APIs), but run-level restart is primarily handled via pipeline-run endpoints.

---

## 12) Typical End-to-End Runtime Sequence

1. Start MinIO (optional but recommended for shared cache).
2. Start backend (`poetry run landseer-backend ...`).
3. Backend loads config, generates workflows/tasks, uploads dataset (if configured).
4. Start one or more workers (`poetry run landseer-worker ...`).
5. Workers register and start claiming ready tasks.
6. Scheduler advances DAG as task completions arrive.
7. Progress and details are visible via API/UI endpoints.
8. Run can be stopped/restarted through run-control endpoints.

---

## 13) Practical Notes and Current Limitations

- Python/package compatibility is sensitive for heavy deps (`torch`, `torchvision`, `scipy`); keep lockfile aligned with interpreter.
- Some cleanup endpoints currently update DB but still mark TODO for full filesystem/MinIO cleanup.
- If MinIO credentials mismatch persisted storage, use a fresh MinIO data directory or matching credentials.
- Worker currently defaults to CPU unless `--gpu` is provided.
- Missing evaluator config file is tolerated (logged warning).

---

## 13.1) Challenging Parts and Pitfalls (Read Carefully)

### A) Cache key and parent hash semantics

- Worker cache key is computed from:
  - tool identity,
  - task config,
  - `parent_hashes`.
- In current worker path, `parent_hashes` are still effectively not populated in the normal execution path.
- Implication: cache correctness can be weaker for tools whose true input state depends on parent artifacts not reflected in key.

### B) Workflow/task dedup vs run boundaries

- Task dedup in pipeline generation reuses same task object identity across workflows when hash matches.
- Run APIs reassign `run_id`/`pipeline_id` over generated objects at run creation.
- Any future changes to dedup behavior must preserve clear run isolation semantics in DB and scheduler state.

### C) API scheduler state is in-memory-first

- Scheduler truth is in API process memory; DB mirrors state.
- After backend restart, scheduler reconstruction depends on initialization flow, not DB replay alone.
- Operationally, treat DB as persistence/observability layer, not fully authoritative scheduler engine.

### D) Stop/restart and cleanup are partially implemented

- Stop/cancel marks statuses and cancels pending tasks correctly.
- Full cache/object-store/filesystem cleanup for run-specific artifacts is not fully end-to-end in all paths.
- Do not assume `stop` or `delete cache` purges everything physically from all storage tiers.

### E) MinIO availability and credential drift

- If MinIO endpoint points to an old persisted volume with different credentials, backend/worker can connect but fail bucket ops (`AccessDenied`).
- In practice, this is a frequent environment issue. Validate with a test bucket operation, not only TCP connectivity.

### F) Container runtime/GPU behavior

- Runtime detection and GPU pass-through are environment dependent (Docker vs Apptainer/Singularity).
- Worker can register as CPU-only if `--gpu` not provided.
- Container-level CUDA visibility can fail despite host GPU availability if runtime settings are incomplete.

### G) Evaluator task assumptions

- Evaluator tasks are appended when evaluator config is available.
- They expect specific artifacts; missing artifacts should skip gracefully but still affect metrics completeness.
- Missing `configs/evaluators.yaml` means no evaluator tasks are appended.

---

## 14) Quick Reference: Most Important Files

- Scheduling:
  - `src/backend/scheduler/base_scheduler.py`
  - `src/backend/scheduler/priority_scheduler.py`
- Workflow/pipeline generation:
  - `src/pipeline/config_loader.py`
  - `src/pipeline/workflow_generator.py`
  - `src/pipeline/tasks.py`
  - `src/pipeline/workflow.py`
- Runtime orchestration:
  - `src/backend/api.py`
  - `src/backend/initialization.py`
  - `src/worker/cli.py`
  - `src/worker/runner.py`
- Caching/storage:
  - `src/store/cache.py`
  - `src/store/artifact_manager.py`
  - `src/store/minio_store.py`
- Persistence:
  - `src/backend/db_service.py`
  - `src/db/models.py`

---

If you update scheduler policy, cache keys, stage generation rules, or run lifecycle APIs, update this document in the same PR so architecture docs stay in sync with behavior.
