# Testing guidelines before deploy

This document describes how to test Landseer so pipeline regressions (missing artifacts, bad evaluator inputs, cross-worker gaps) are caught **before** production runs.

## Layers

1. **Unit tests** — Pure functions, loaders, and small modules with no I/O or mocked I/O. Fast; run on every commit.
2. **Component tests** — Worker `TaskRunner` merge behavior, scheduler ordering, API request/response shapes. Use temporary directories and mocks for HTTP/container runs.
3. **Integration tests** — Real `torch.save` / `torch.load` round-trips on merged paths, evaluator `main()` with a synthetic workspace, multi-task ancestry with `Worker._execute_task` patched.
4. **End-to-end (optional)** — Full stack: backend, workers, containers. Run before release or when changing wire formats.

## Conventions

- Prefer **real file formats** over byte placeholders (`b"fake"`) when testing tools that parse those files (PyTorch, NumPy).
- When adding a new evaluator or training stage, add: (1) input contract test (required files), (2) success-path output JSON schema, (3) one failure path (missing file / bad checkpoint).
- Keep domain layout under `tests/` (`tests/worker`, `tests/evaluators`, `tests/backend`, …); see `tests/README.md` for commands.

## Pre-deploy checklist (minimum)let's run the pi
- `poetry run pytest tests/worker tests/evaluators` passes (evaluator tests require **NumPy** and **PyTorch** in the environment used for pytest).
- No skipped tests that encode required safety properties for your change.
- If Docker build paths change (e.g. evaluator `COPY`), build once:  
  `docker build -f containers/evals/clean/Dockerfile containers/evals`.
