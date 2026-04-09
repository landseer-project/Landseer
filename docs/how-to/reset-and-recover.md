# Reset and Recover

For a clean slate, use `helper_scripts/reset_pipeline.sh`.

## Destructive reset

```bash
bash helper_scripts/reset_pipeline.sh --yes
```

This removes:

- `/tmp/landseer_worker_*`
- `/tmp/landseer_cache`
- local SQLite DB (`landseer.db`)
- exited docker containers

## Typical recovery flow

1. Reset local state
2. Start backend
3. Start workers
4. Re-trigger pipeline run

If only one run is broken, prefer API-level stop/restart endpoints before full reset.
