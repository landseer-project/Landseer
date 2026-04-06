# Test Suite Guide

This repository uses a domain-based test layout:

- `tests/backend/api/` - Backend API contract and behavior
- `tests/backend/scheduler/` - Scheduler logic and task ordering
- `tests/pipeline/` - Pipeline config/task/workflow construction
- `tests/worker/` - Worker client/runtime/cache behavior
- `tests/evaluators/` - Evaluator-specific logic
- `tests/test_*.py` - cross-cutting top-level tests

## Quick Start

Run everything:

```bash
poetry run pytest
```

Run by domain:

```bash
poetry run pytest tests/backend
poetry run pytest tests/backend/api
poetry run pytest tests/backend/scheduler
poetry run pytest tests/pipeline
poetry run pytest tests/worker
poetry run pytest tests/evaluators
```

## Marker-based Runs

Markers are auto-applied from folder/file path via `tests/conftest.py`.

```bash
poetry run pytest -m backend
poetry run pytest -m "backend and api"
poetry run pytest -m scheduler
poetry run pytest -m db
poetry run pytest -m integration
poetry run pytest -m security
```

## Fast Local Workflow

```bash
# Scheduler-only loop
poetry run pytest tests/backend/scheduler -q

# API-only loop
poetry run pytest tests/backend/api -q

# One specific test function
poetry run pytest tests/backend/scheduler/test_scheduler_comprehensive.py::test_blocked_when_dependency_failed -q
```

## Conventions

- File names: `test_<feature>.py`
- One primary concern per file
- Keep fixtures close to their domain (`tests/<domain>/conftest.py`)
- Use marker-filter runs for focused debugging
