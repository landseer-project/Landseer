# Run a Pipeline

## 1. Start backend (headless)

```bash
uv run landseer-backend
```

Optional: preload a config with `--config configs/pipeline/trades.yaml`.

## 2. Start workers

```bash
uv run landseer-worker --gpu 0
```

## 3. Submit a run

```bash
uv run landseer run start configs/pipeline/trades.yaml
```

## 4. Watch it

```bash
uv run landseer run list
uv run landseer progress --run-id <run_id>
uv run landseer task list <run_id>
uv run landseer task list <run_id> --status failed
uv run landseer worker list
```

## 5. Stop a run

```bash
uv run landseer run stop <run_id>
```

Command sheet: [CLI reference](../reference/cli.md)
