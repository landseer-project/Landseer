# Testing

## Test layout

- `tests/backend/api/`
- `tests/backend/scheduler/`
- `tests/pipeline/`
- `tests/worker/`
- `tests/evaluators/`

## Core commands

```bash
uv run pytest
uv run pytest tests/backend
uv run pytest tests/backend/api
uv run pytest tests/backend/scheduler
uv run pytest tests/pipeline
uv run pytest tests/worker
uv run pytest tests/evaluators
```

## Marker-based commands

```bash
uv run pytest -m backend
uv run pytest -m "backend and api"
uv run pytest -m scheduler
uv run pytest -m db
uv run pytest -m integration
uv run pytest -m security
```
