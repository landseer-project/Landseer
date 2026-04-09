# Project Structure

Landseer is organized by runtime responsibility.

## Top-level

- `src/backend/`: FastAPI scheduler and orchestration APIs
- `src/worker/`: worker process that claims and executes tasks
- `src/pipeline/`: config loading, workflow generation, and task graph creation
- `src/db/`: SQLAlchemy models, sessions, persistence layer
- `src/store/`: MinIO/local cache integration
- `configs/`: pipeline, tools, evaluator configuration
- `tests/`: domain-based test suites
- `docs/`: Sphinx documentation site

## Runtime responsibilities

- Backend owns pipeline lifecycle, worker registration, task status, and run APIs.
- Workers execute tool containers, report results, and handle cache read/write.
- Pipeline modules expand YAML into workflows and deduplicated tasks.
- DB and store modules persist run metadata and artifacts.
