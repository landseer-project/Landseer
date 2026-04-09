# Architecture Overview

Landseer runs as a backend + N workers model.

Execution path:

1. User/API client starts runs in backend.
2. Backend scheduler exposes runnable tasks.
3. Workers claim tasks, execute containers, and publish artifacts.
4. Backend persists run/task/workflow state in SQLite/MySQL.
5. Progress/metrics endpoints provide system visibility.

## High-level flow

1. Backend loads tool registry and pipeline config.
2. Backend expands configuration into workflows/tasks and tracks them in memory and DB.
3. Workers register and claim runnable tasks.
4. Worker executes each tool in container runtime and reports completion/failure.
5. Backend updates task/workflow/run state and exposes metrics/progress APIs.

## Key design choices

- **Task deduplication:** equivalent tasks are reused across workflows.
- **Dependency-aware scheduling:** tasks run only when parent dependencies complete.
- **Cache-first execution:** workers compute cache keys from tool identity + parents.
- **Headless backend mode:** backend can start without active pipeline and run on-demand via API.
