# REST API Reference

Base URL: `http://localhost:8000`

## Info

- `GET /`
- `GET /health`
- `GET /info/pipeline`
- `GET /info/workflows`

## Tasks

- `GET /tasks/next`
- `PUT /tasks/status`
- `GET /tasks`
- `GET /tasks/{task_id}`
- `GET /tasks/{task_id}/logs`

## Progress

- `GET /progress`
- `GET /progress/levels`
- `GET /progress/ready`
- `GET /progress/blocked`

## Workers

- `POST /workers/register`
- `GET /workers`
- `GET /workers/{worker_id}`
- `POST /workers/{worker_id}/heartbeat`
- `POST /workers/{worker_id}/claim`

## Workflows and pipeline

- `GET /workflows/{workflow_id}`
- `GET /workflows/{workflow_id}/results`
- `GET /pipeline`

## Registry and tools

- `GET /tools`, `GET /tools/{tool_name}`, `POST /tools`
- `GET /registry/tools`, `POST /registry/tools`
- `GET /registry/evaluators`, `POST /registry/evaluators`

## Dataset and stats

- `GET /dataset`
- `GET /dataset/download-url`
- `GET /stats/database`
- `GET /stats/store`
- `GET /stats/system`

## Pipeline config/run lifecycle

- `GET /api/pipeline-configs`
- `GET /api/pipeline-configs/{config_id}`
- `POST /api/pipeline-configs/{config_id}/runs`
- `GET /api/pipeline-runs/{run_id}`
- `GET /api/pipeline-configs/{config_id}/runs`
- `GET /api/pipeline-runs`
- `POST /api/pipeline-runs/{run_id}/stop`
- `POST /api/pipeline-runs/{run_id}/restart`
- `DELETE /api/pipeline-runs/{run_id}/cache`
- `DELETE /api/pipeline-runs/{run_id}/workspace`
