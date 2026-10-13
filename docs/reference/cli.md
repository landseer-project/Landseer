# CLI

Prefix everything with `uv run`. Backend must be up for `landseer …` commands.

## Control plane

```bash
landseer run start configs/pipeline/trades.yaml
landseer run list
landseer run show <run_id>
landseer run stop <run_id>

landseer task list <run_id>
landseer task list <run_id> --status running   # pending|running|completed|failed

landseer progress
landseer progress --run-id <run_id>
landseer status
landseer health
landseer worker list
landseer config list
```

Global flags: `--backend-url`, `--pipeline-key`, `--json`

## Processes

```bash
landseer-backend                          # headless
landseer-backend --config path/to.yaml
landseer-worker --gpu 0
landseer-frontend install|dev|build|preview
```

## Stack helper

```bash
bash .cursor/skills/landseer-tmux-run/scripts/landseer-tmux.sh
bash .cursor/skills/landseer-tmux-run/scripts/landseer-tmux.sh mini
```
