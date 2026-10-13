# Backend Test Organization

Backend tests are split into two explicit suites:

- `api/` - FastAPI endpoint behavior, request/response validation, integration paths
- `scheduler/` - Task scheduling rules, dependency gating, priority correctness

## Commands

```bash
# All backend tests
uv run pytest tests/backend

# API suite only
uv run pytest tests/backend/api

# Scheduler suite only
uv run pytest tests/backend/scheduler

# Marker equivalent
uv run pytest -m "backend and api"
uv run pytest -m "backend and scheduler"
```

## Suggested CI Order

1. `tests/backend/scheduler` (fastest logic checks)
2. `tests/backend/api` (higher-level behavior)
3. Remaining integration-heavy suites
