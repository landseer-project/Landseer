# Testing

## Test layout

- `tests/backend/api/`
- `tests/backend/scheduler/`
- `tests/pipeline/`
- `tests/worker/`
- `tests/evaluators/`

## Core commands

```bash
poetry run pytest
poetry run pytest tests/backend
poetry run pytest tests/backend/api
poetry run pytest tests/backend/scheduler
poetry run pytest tests/pipeline
poetry run pytest tests/worker
poetry run pytest tests/evaluators
```

## Marker-based commands

```bash
poetry run pytest -m backend
poetry run pytest -m "backend and api"
poetry run pytest -m scheduler
poetry run pytest -m db
poetry run pytest -m integration
poetry run pytest -m security
```
