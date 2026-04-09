# Run a Pipeline

## 1. Start backend with config

```bash
poetry run landseer-backend --config configs/pipeline/trades.yaml
```

## 2. Start one or more workers

```bash
poetry run landseer-worker --backend-url http://localhost:8000 --gpu 0
```

Run additional workers on other GPUs or hosts for parallel throughput.

## 3. Trigger run

```bash
curl -X POST http://localhost:8000/api/pipeline-configs/trades/runs \
  -H "Content-Type: application/json" \
  -d '{"use_cache": true}'
```

## 4. Observe and inspect

```bash
curl http://localhost:8000/progress
curl http://localhost:8000/api/pipeline-runs
curl http://localhost:8000/tasks?status=failed
```

## 5. Stop or restart a run

```bash
curl -X POST http://localhost:8000/api/pipeline-runs/<run_id>/stop
curl -X POST http://localhost:8000/api/pipeline-runs/<run_id>/restart
```
