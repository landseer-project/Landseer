# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation
```bash
# Recommended
pip install -e .

# Alternative
poetry install
```

### Running the System
```bash
# Backend (scheduler + API server)
PYTHONPATH=./src:. python -m src.backend.cli --host 0.0.0.0 --port 8000 --config configs/pipeline/trades.yaml

# Worker (one per GPU; scales horizontally)
PYTHONPATH=./src:. python -m src.worker.cli --backend-url http://localhost:8000 --workspace /tmp/landseer_worker_1 --gpu 0

# Frontend
PYTHONPATH=./src:. python -m src.frontend.cli

# Or with Poetry entry points
poetry run landseer-backend --config configs/pipeline/trades.yaml
poetry run landseer-worker --backend-url http://localhost:8000 --gpu 0
poetry run landseer-frontend
```

### Triggering a Pipeline Run (via REST API)
```bash
# Start a run for a named pipeline config
curl -X POST http://localhost:8000/api/pipeline-configs/trades/runs \
  -H "Content-Type: application/json" -d '{"use_cache": true}'

# Check health
curl http://localhost:8000/health
```

### Tests
```bash
# Run all tests
poetry run pytest

# Run by domain
poetry run pytest tests/worker
poetry run pytest tests/backend
poetry run pytest tests/pipeline
poetry run pytest tests/evaluators

# Pre-deploy minimum (evaluators require NumPy + PyTorch in the test env)
poetry run pytest tests/worker tests/evaluators

# Single test function
poetry run pytest tests/backend/scheduler/test_scheduler_comprehensive.py::test_blocked_when_dependency_failed -q

# Marker-based
poetry run pytest -m "backend and api"
poetry run pytest -m scheduler
```

### Building Evaluator Containers
```bash
# If Docker build paths change for evaluators:
docker build -f containers/evals/clean/Dockerfile containers/evals
```

## Architecture

### Three-Process System

Landseer is split into three separately deployable services that communicate over HTTP:

1. **Backend** (`src/backend/`) — FastAPI REST API + in-memory `PriorityScheduler`. Loads the pipeline config at startup, generates all workflow/task combinations, and serves tasks to workers via `/workers/{id}/claim`. Also manages a SQLite/MySQL database (`src/db/`) for durability.

2. **Workers** (`src/worker/`) — Stateless. Poll the backend, run tool containers (Docker or Apptainer), report results. Each worker maintains an in-memory `_task_outputs` dict that chains artifact directories across sequential tasks it has handled.

3. **Frontend** (`src/frontend/`) — React/TypeScript dashboard served by a FastAPI static file server.

### Pipeline Generation

Given a YAML config (`configs/pipeline/*.yaml`), `src/pipeline/config_loader.py:create_pipeline_from_config` generates all workflows:
- **Pre/Post/Deployment stages**: all permutations of all non-empty tool subsets + the noop baseline.
- **During-training stage**: single-tool options only (can't interleave two training objectives).
- **Cartesian product** across stages = total workflows (~250+ for a full config).
- **Task deduplication**: `get_or_create_task()` hashes `(tool, config, sorted dep IDs)`. Two workflows sharing the same tool applied to the same prior results share one `Task` row.

### Artifact Chaining (Critical for Correctness)

Each tool container reads from `/input` (and `/data`) and writes to `/output`. The worker assembles the input directory by merging ancestor outputs in pipeline order (earliest first, later overrides on collision). This is implemented in `src/worker/runner.py:TaskRunner.run_task` via the `ancestor_dirs` argument.

The ancestry chain is tracked in `Worker._task_outputs` (dict: `task_id → (cache_key, output_path, [ancestor_paths])`). For each dep of a new task, it extends `ancestor_dirs` with `dep_anc + [dep_out]`. If a dep ran on a **different worker**, `_task_outputs` won't have it — `_collect_remote_ancestry()` in `src/worker/cli.py` fetches the task's `output_path` from the backend via `GET /tasks/{id}` and reconstructs the chain recursively.

**Key invariant**: `model.pt` produced by a during-training tool must survive intact all the way to the deployment and evaluation stages. If any stage in the chain has an empty or wrong output, downstream tasks fail with "model.pt not found".

### Tool Container Contract

Every defense tool and evaluator is a Docker/OCI image:
- **Input**: `/input` (= `/data`) — merged outputs of all upstream tasks: `data.npy`, `test_data.npy`, `labels.npy`, `model.pt`, etc.
- **Config**: `config_model.py` mounted into `/app/` and `/input/`
- **Output**: `/output` — tool writes `model.pt` (during/post/deploy stages) or transformed `data.npy` (pre-training stage)

Adding a new defense = push a Docker image, add an entry to `configs/tools.yaml`, list it in a pipeline YAML. No core code changes.

### Scheduling and Priority

`PriorityScheduler` (`src/backend/scheduler/priority_scheduler.py`) maintains a ready queue (tasks with all deps `COMPLETED`). Priority = `100 - (depth × 10) + counter_bonus`, where depth = longest path from the task to a root, and counter = number of workflows sharing the task. Evaluator tasks get an additional +32 boost so metrics run as soon as a model is ready.

### Two-Level Artifact Cache

Cache key = hash of `(tool_name, image, command, config, sorted parent hashes)`.
1. **Local**: `cache_dir/<cache_key>/` on the worker disk — zero network.
2. **MinIO** (`src/store/`): shared S3-compatible store — enables cross-worker deduplication without a shared filesystem.

On a cache hit, the worker symlinks the cached path and records it in `_task_outputs` to preserve the ancestry chain for downstream tasks.

### Database Layer

`src/db/models.py` defines SQLAlchemy models: `TaskModel`, `WorkflowModel`, `PipelineModel`, `PipelineRunModel`, `ArtifactModel`, `EvaluationResultModel`. Tasks store direct `dependencies` (many-to-many via `task_dependencies` table). Results and metrics land in `EvaluationResultModel` (keyed per workflow × evaluator, not per task).

### Config Files

- `configs/tools.yaml` — all tool definitions (name, container image, command, `is_baseline` flag)
- `configs/pipeline/*.yaml` — dataset, model script, and per-stage tool lists
- `configs/model/config_model_*.py` — model architecture, mounted into containers as `config_model.py`
- `configs/evaluators.yaml` (optional) — override built-in evaluator definitions
