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
uv run pytest
```

Run by domain:

```bash
uv run pytest tests/backend
uv run pytest tests/backend/api
uv run pytest tests/backend/scheduler
uv run pytest tests/pipeline
uv run pytest tests/worker
uv run pytest tests/evaluators
```

## Marker-based Runs

Markers are auto-applied from folder/file path via `tests/conftest.py`.

```bash
uv run pytest -m backend
uv run pytest -m "backend and api"
uv run pytest -m scheduler
uv run pytest -m db
uv run pytest -m integration
uv run pytest -m security
```

## Fast Local Workflow

```bash
# Scheduler-only loop
uv run pytest tests/backend/scheduler -q

# API-only loop
uv run pytest tests/backend/api -q

# One specific test function
uv run pytest tests/backend/scheduler/test_scheduler_comprehensive.py::test_blocked_when_dependency_failed -q
```

## Conventions

- File names: `test_<feature>.py`
- One primary concern per file
- Keep fixtures close to their domain (`tests/<domain>/conftest.py`)
- Use marker-filter runs for focused debugging
