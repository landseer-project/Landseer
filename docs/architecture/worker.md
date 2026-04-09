# Worker

Worker CLI entrypoint: `src/worker/cli.py`

## Worker loop

1. Initialize client/runner/cache
2. Wait for backend health
3. Register worker capabilities
4. Fetch dataset info
5. Repeatedly claim -> execute -> report
6. Send heartbeats while idle and while executing

## Execution details

- Uses `TaskRunner` to execute tasks in container runtime (`docker`, `apptainer`, `singularity`, or auto).
- Builds cache keys from tool identity + parent hashes.
- Tracks full artifact ancestry so downstream stages receive complete upstream outputs.
- Reports structured task payloads (logs, cache metadata, output path, optional evaluation result).

## Important CLI flags

- `--backend-url`: backend endpoint
- `--gpu`: GPU id
- `--runtime`: runtime selection (`docker|apptainer|singularity|auto`)
- `--cache-dir` / `--no-cache`: caching behavior
- `--data-path`: manual dataset override
- `--no-minio`: force local-only cache
