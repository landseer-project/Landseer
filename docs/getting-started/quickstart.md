# Quickstart

This path gets a new engineer from clone to first run with backend and worker.

## Prerequisites

- Python 3.11+
- Poetry
- Docker (or another supported runtime such as Apptainer/Singularity)

## Install

```bash
git clone <repo-url>
cd Landseer
poetry install
```

## Start backend

```bash
poetry run landseer-backend --config configs/pipeline/trades.yaml
```

By default, backend serves on `http://localhost:8000`.

## Start a worker

In another terminal:

```bash
poetry run landseer-worker --backend-url http://localhost:8000 --gpu 0
```

For CPU-only mode, omit `--gpu`.

## Trigger a pipeline run

```bash
curl -X POST http://localhost:8000/api/pipeline-configs/trades/runs \
  -H "Content-Type: application/json" \
  -d '{"use_cache": true}'
```

## Monitor progress

```bash
curl http://localhost:8000/progress
curl http://localhost:8000/api/pipeline-runs
```

## Smoke-test endpoints

```bash
curl http://localhost:8000/health
curl http://localhost:8000/info/pipeline
curl http://localhost:8000/workers
```
