# Run a Pipeline

## 1. Start backend (headless)

```bash
uv run landseer-backend
```

Optional: preload a config with `--config configs/pipeline/trades.yaml`.

## 2. Start workers

```bash
uv run landseer-worker --gpu 0
```

## 3. Submit a run

```bash
uv run landseer run start configs/pipeline/trades.yaml
```

Runs on different datasets can run at the same time on the same backend and
workers (e.g. `trades.yaml` on CIFAR-10 next to `celeba_trades.yaml` on CelebA).
The backend prepares and publishes each run's dataset and model script to MinIO
when the run starts or is restarted; every task is executed on the dataset of
its own run. See [Datasets on workers](#datasets-on-workers).

## 4. Watch it

```bash
uv run landseer run list
uv run landseer progress --run-id <run_id>
uv run landseer task list <run_id>
uv run landseer task list <run_id> --status failed
uv run landseer worker list
```

## 5. Stop a run

```bash
uv run landseer run stop <run_id>
```

Command sheet: [CLI reference](../reference/cli.md)

## Datasets on workers

A worker looks up the dataset of each task's run (`GET /dataset?run_id=...`)
and uses the first of these that holds exactly that dataset (checked against the
run's `dataset_manifest.json`: sizes first, then sha256 via the
`.landseer_hashes.json` sidecar):

1. `LANDSEER_DATA_PATH` / `--data-path`: one dataset directory.
2. `LANDSEER_DATA_ROOT` / `--data-root`: a directory with one subdirectory per
   dataset, `<root>/<name>/<variant>/` (e.g. `<root>/cifar10/clean`,
   `<root>/celeba/clean`).
3. The backend's own dataset directory (worker on the backend host).
4. A verified copy synced from MinIO into `<cache>/datasets/<name>/<variant>`.

A manual directory that does not match a run's dataset is skipped for that run
(logged), not fatal: that run uses the MinIO copy. Tasks are refused only when
no verified copy can be found at all; `LANDSEER_ALLOW_DATASET_MISMATCH=1` then
uses `LANDSEER_DATA_PATH` anyway. The resolved directory is remembered per
dataset and per run, so a worker alternating between runs does not re-verify or
re-download. The model script is resolved per run too (by sha256, from MinIO
when the local copy differs), and cache keys include both the dataset's
content id and the model script hash.

Example, a host with both CIFAR-10 and CelebA copies:

```bash
LANDSEER_DATA_ROOT=/data/landseer/landseer_old_data/data uv run landseer-worker --gpu 0
```

