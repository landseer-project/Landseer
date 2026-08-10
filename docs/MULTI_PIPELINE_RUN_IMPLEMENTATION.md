# Multi-Pipeline Run System Implementation

## Overview

This implementation adds run-aware pipeline management so Landseer can:
- Track multiple runs per pipeline config
- Prevent concurrent runs of the same config
- Tag cache/results with run identity
- Support run stop/restart and per-run cleanup

## Implemented Changes

### 1) Database Model Updates

Added in `src/db/models.py`:
- `PipelineRunStatus` enum (`pending`, `running`, `completed`, `failed`, `cancelled`, `stopping`)
- `PipelineConfigModel` (`pipeline_configs` table)
- `PipelineRunModel` (`pipeline_runs` table)

Extended existing models:
- `PipelineModel`: added `run_id`
- `TaskModel`: added `run_id`
- `WorkflowModel`: added `run_id`
- `EvaluationResultModel`: added `run_id`
- `ArtifactModel`: added `created_by_run_id`

### 2) DB Repository Layer

Added in `src/db/repository.py`:
- `PipelineConfigRepository`
- `PipelineRunRepository`
- `ArtifactRepository.get_by_run_id()`
- `ArtifactRepository.delete_by_run_id()`

### 3) DB Module Exports

Updated `src/db/__init__.py` exports for:
- New models (`PipelineConfigModel`, `PipelineRunModel`, `PipelineRunStatus`, etc.)
- New repositories (`PipelineConfigRepository`, `PipelineRunRepository`)

### 4) Config Discovery Service

Added `src/backend/config_discovery.py`:
- `discover_configs()`
- `sync_configs_to_db()`
- `get_config_by_id()`
- `get_all_configs()`
- config hashing for change detection

Scans:
- `configs/pipeline/*.yaml`
- optional `configs/attack/*.yaml`

### 5) Backend API (Pipeline Config + Run APIs)

Extended `src/backend/api.py` with endpoints:
- `GET /api/pipeline-configs`
- `GET /api/pipeline-configs/{config_id}`
- `POST /api/pipeline-configs/{config_id}/runs`
- `GET /api/pipeline-runs/{run_id}`
- `GET /api/pipeline-configs/{config_id}/runs`
- `POST /api/pipeline-runs/{run_id}/stop`
- `POST /api/pipeline-runs/{run_id}/restart`
- `DELETE /api/pipeline-runs/{run_id}/cache`
- `DELETE /api/pipeline-runs/{run_id}/workspace`

Also added request/response models for config/run payloads.

### 6) DB Service Run-Aware Sync

Updated `src/backend/db_service.py`:
- `sync_pipeline_to_db(..., run_id=None)` now stores run linkage in pipelines/tasks/workflows
- `save_evaluation_result(..., run_id=None)` now tags evaluation records per run

### 7) Worker + Cache Run Propagation

Updated `src/worker/client.py`:
- `TaskInfo` includes `run_id`

Updated `src/worker/cli.py`:
- cache store path now passes run_id through

Updated `src/worker/db.py`:
- `ArtifactMetadata` includes `run_id`
- `ArtifactCacheDB.store_artifact(..., run_id=None)` persists run metadata in `manifest.json`
- `CacheManager.store_result(..., run_id=None)` forwards run_id

### 8) Task API Response Run Field

Updated `src/backend/api.py`:
- `TaskResponse` includes `run_id`
- `task_to_response()` populates run_id

## What This Enables

- Multiple run records for same config
- Active-run guard per config
- Run-specific visibility/status APIs
- Run-tagged cache metadata for cleanup and traceability

## Remaining Follow-ups (Recommended)

1. **Filesystem/MinIO cache cleanup completion**
   - API deletes DB entries by run; extend to remove object storage/local artifact paths too.

2. **Run-completion status auto-finalization**
   - Ensure pipeline run transitions to `completed`/`failed` reliably at scheduler end-state.

3. **Result directory normalization**
   - Fully standardize result output to:
   - `results/<config_id>/<run_id>/...`

4. **Worker stop-signal hardening**
   - Ensure in-flight task handling and cancellation reporting are robust during stop/restart.

## Quick Verification Commands

```bash
# list discovered configs
curl http://localhost:8000/api/pipeline-configs

# start a run
curl -X POST http://localhost:8000/api/pipeline-configs/config_trades/runs \
  -H "Content-Type: application/json" \
  -d '{"use_cache": true}'

# check run status
curl http://localhost:8000/api/pipeline-runs/<run_id>

# stop run
curl -X POST http://localhost:8000/api/pipeline-runs/<run_id>/stop
```
