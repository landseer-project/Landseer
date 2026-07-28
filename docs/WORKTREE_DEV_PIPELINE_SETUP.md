# Landseer Worktree Setup (Pipeline + Development)

This setup keeps `main` stable for pipeline runs while using `dev/restructure` for ongoing development, without duplicating heavy data.

## Current Layout

- Dev worktree (already exists): `/share/landseer/workspace-ayushi/Landseer` (`dev/restructure`)
- Main worktree (created): `/share/landseer/workspace-ayushi/landseer-main` (`main`)
- Shared virtualenv root: `/share/landseer/workspace-ayushi/.shared/envs`
- Shared data root: `/share/landseer/workspace-ayushi/.shared/data/landseer-shared`

## One-Time Setup Commands

```bash
# From the dev worktree repo
cd /share/landseer/workspace-ayushi/Landseer

# Create main worktree (dev/restructure is this current directory)
git worktree add ../landseer-main main

# Shared dirs for environments and heavy data
mkdir -p /share/landseer/workspace-ayushi/.shared/envs/landseer-main
mkdir -p /share/landseer/workspace-ayushi/.shared/envs/landseer-dev-restructure
mkdir -p /share/landseer/workspace-ayushi/.shared/data/landseer-shared/{datasets,models,artifacts,cache,logs,tmp}

# Separate env per worktree
python -m venv /share/landseer/workspace-ayushi/.shared/envs/landseer-main
python -m venv /share/landseer/workspace-ayushi/.shared/envs/landseer-dev-restructure
```

## Activate the Right Environment

```bash
# Pipeline run environment (main)
source /share/landseer/workspace-ayushi/.shared/envs/landseer-main/bin/activate
cd /share/landseer/workspace-ayushi/landseer-main

# Development environment (dev/restructure)
source /share/landseer/workspace-ayushi/.shared/envs/landseer-dev-restructure/bin/activate
cd /share/landseer/workspace-ayushi/Landseer
```

## Install Dependencies Per Worktree

Use your preferred installer in each worktree after activating its env.

```bash
# Main
source /share/landseer/workspace-ayushi/.shared/envs/landseer-main/bin/activate
cd /share/landseer/workspace-ayushi/landseer-main
poetry install

# Dev
source /share/landseer/workspace-ayushi/.shared/envs/landseer-dev-restructure/bin/activate
cd /share/landseer/workspace-ayushi/Landseer
poetry install
```

If Poetry is unavailable/broken on your machine, use:

```bash
pip install -e .
```

## Shared Data Configuration (Avoid Duplicate Heavy Data)

Set these variables in each shell before backend/worker runs:

```bash
export LANDSEER_DATA_ROOT=/share/landseer/workspace-ayushi/.shared/data/landseer-shared
export LANDSEER_CACHE_DIR=/share/landseer/workspace-ayushi/.shared/data/landseer-shared/cache
export LANDSEER_ARTIFACT_DIR=/share/landseer/workspace-ayushi/.shared/data/landseer-shared/artifacts
export TMPDIR=/share/landseer/workspace-ayushi/.shared/data/landseer-shared/tmp
```

Recommended practice:
- Keep dataset/model downloads under `LANDSEER_DATA_ROOT`.
- Keep worker `--cache-dir` under the shared cache path.
- Keep run logs under the shared logs path when possible.

## Daily Workflow

### 1) Stable pipeline runs on `main`

```bash
source /share/landseer/workspace-ayushi/.shared/envs/landseer-main/bin/activate
cd /share/landseer/workspace-ayushi/landseer-main
git pull
```

Run backend/workers from this worktree for reproducible runs.

### 2) Development on `dev/restructure`

```bash
source /share/landseer/workspace-ayushi/.shared/envs/landseer-dev-restructure/bin/activate
cd /share/landseer/workspace-ayushi/Landseer
git status
```

Do all refactors and experiments here.

## Disk Hygiene (Important for Landseer)

Run periodically:

```bash
du -sh /share/landseer/workspace-ayushi/.shared/data/landseer-shared/*
```

Clear only safe transient data when needed:

```bash
rm -rf /share/landseer/workspace-ayushi/.shared/data/landseer-shared/tmp/*
```

Optional old artifact cleanup (example: older than 14 days):

```bash
find /share/landseer/workspace-ayushi/.shared/data/landseer-shared/artifacts -type f -mtime +14 -delete
```

## Quick Verification

```bash
git -C /share/landseer/workspace-ayushi/Landseer worktree list
```

Expected output includes:
- `/share/landseer/workspace-ayushi/Landseer` on `dev/restructure`
- `/share/landseer/workspace-ayushi/landseer-main` on `main`
