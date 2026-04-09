# CLI Reference

Entry points are declared in `pyproject.toml`.

## `landseer-backend`

Primary options:

- `--host` (default `0.0.0.0`)
- `--port` (default `8000`)
- `--debug`
- `--config` (pipeline config path, optional for headless)
- `--tools-config` (default `configs/tools.yaml`)

## `landseer-worker`

Primary options:

- `--backend-url`
- `--worker-id`
- `--gpu`
- `--timeout`
- `--runtime`
- `--workspace`
- `--cache-dir`
- `--no-cache`
- `--data-path`
- `--no-minio`
- `--minio-endpoint`
- `--poll-interval`
- `--heartbeat-interval`
- `--debug`, `--log-level`

## `landseer-frontend`

Frontend wrapper command is also exposed in `pyproject.toml` via `src/frontend/cli.py`.
