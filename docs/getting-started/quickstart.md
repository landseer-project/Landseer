# Quickstart

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker (or Apptainer/Singularity)

## Install

```bash
git clone <repo-url>
cd Landseer
uv sync --all-groups
cp .env.example .env.db   # fill DB + MinIO secrets
```

## Start processes

```bash
# Terminal 1 — headless backend (recommended)
uv run landseer-backend

# Terminal 2 — worker(s)
uv run landseer-worker --gpu 0
```

Or all-in-one: `bash .cursor/skills/landseer-tmux-run/scripts/landseer-tmux.sh`

## Run a pipeline

```bash
uv run landseer run start configs/pipeline/trades.yaml
uv run landseer progress --run-id <run_id>
uv run landseer task list <run_id>
uv run landseer run list
```

## Useful checks

```bash
uv run landseer health
uv run landseer status
uv run landseer worker list
```

Full command list: [CLI reference](../reference/cli.md)
