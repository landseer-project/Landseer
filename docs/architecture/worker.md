# Worker

Worker CLI entrypoint: `src/worker/cli.py`

## Worker loop

1. Initialize client/runner/cache
2. Wait for backend health
3. Register worker capabilities
4. Fetch dataset info (the backend's latest dataset; each run's own dataset is
   resolved when its first task arrives)
5. Repeatedly claim -> execute -> report
6. Send heartbeats while idle and while executing

## Execution details

- Uses `TaskRunner` to execute tasks in container runtime (`docker`, `apptainer`, `singularity`, or auto).
- Builds cache keys from tool identity + parent hashes + the run's dataset
  content id and model script hash.
- Resolves dataset and model script per run (`GET /dataset?run_id=`), so runs
  on different datasets share workers; see
  [Datasets on workers](../how-to/run-a-pipeline.md#datasets-on-workers).
- Tracks full artifact ancestry so downstream stages receive complete upstream outputs.
- Reports structured task payloads (logs, cache metadata, output path, optional evaluation result).

## Important CLI flags

- `--backend-url`: backend endpoint
- `--gpu`: GPU id
- `--runtime`: runtime selection (`docker|apptainer|singularity|auto`)
- `--cache-dir` / `--no-cache`: caching behavior
- `--data-path` (`LANDSEER_DATA_PATH`): one local dataset dir, used for runs whose dataset it matches
- `--data-root` (`LANDSEER_DATA_ROOT`): dir of local dataset copies as `<name>/<variant>/`
- `--no-minio`: force local-only cache
