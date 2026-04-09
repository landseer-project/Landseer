# Backend

Backend CLI entrypoint: `src/backend/cli.py`

## What it does

- Parses runtime options (`--host`, `--port`, `--config`, `--tools-config`).
- Initializes backend context (`initialize_backend`) from `src/backend/initialization.py`.
- Starts FastAPI server (`src/backend/api.py`).

## Initialization stages

1. Tool registry from `configs/tools.yaml`
2. Optional MinIO store setup
3. Optional dataset preparation (if data module is available)
4. Pipeline creation from YAML (unless running headless)
5. Database service initialization and sync

## Headless mode

If `--config` is omitted, backend starts without an active pipeline and waits for run creation via:

- `POST /api/pipeline-configs/{config_id}/runs`

## Core backend APIs

- Health/info: `/health`, `/info/pipeline`, `/info/workflows`
- Task operations: `/tasks/*`, `/progress*`
- Worker operations: `/workers/*`
- Run lifecycle: `/api/pipeline-configs/*/runs`, `/api/pipeline-runs/*`
