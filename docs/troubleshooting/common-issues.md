# Common Issues

## Worker cannot connect to backend

- Confirm backend is running: `curl http://localhost:8000/health`
- Verify `--backend-url` points to reachable host/port

## No tasks are claimed

- Check run state: `GET /api/pipeline-runs`
- Check blocked tasks: `GET /progress/blocked`
- Confirm at least one run was triggered

## Dataset not available warnings

- Backend may not have prepared dataset, or `src.data` is not installed/available
- Provide explicit worker `--data-path` when needed

## Cache confusion

- Disable cache for diagnosis: run worker with `--no-cache`
- Clear cache for a run with `DELETE /api/pipeline-runs/{run_id}/cache`

## DB connection failures

- For MySQL: validate `LANDSEER_DB_*` variables and service availability
- For local development, default to SQLite for simpler startup
