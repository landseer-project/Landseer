# Merge Plan: main <- dev/restructure

This runbook resolves the currently observed conflicts when merging `origin/dev/restructure` into `origin/main`.

## Scope

Observed conflicts:
- `.gitignore`
- `configs/pipeline/dnn_watermarking.yaml`
- `configs/pipeline/dp.yaml`
- `configs/pipeline/fairness.yaml`
- `configs/pipeline/teaching.yaml`
- `configs/pipeline/trades.yaml`
- `pyproject.toml`
- `poetry.lock`
- file-location conflicts for:
  - `src/landseer_pipeline/evaluator/badnets.py`
  - `src/landseer_pipeline/evaluator/faithfullness_eval.py`
  - `src/landseer_pipeline/evaluator/mia.py`

## Strategy

Use `dev/restructure` as the structural baseline because it introduces:
- new package layout under `src/*`
- new pipeline config format using tool IDs from `configs/tools.yaml`
- backend/frontend/worker entrypoints

Then selectively carry forward important capability additions from `main`.

## Recommended Resolution Per File

### 1) Pipeline YAMLs (schema conflict)

Use the `dev/restructure` schema (`tools: [id, ...]`), not inline container blocks.

- `configs/pipeline/dnn_watermarking.yaml`
  - Keep `dev/restructure` version.
  - Reason: matches centralized tool registry (`configs/tools.yaml`) and uses `watermarknn` ID.

- `configs/pipeline/fairness.yaml`
  - Keep `dev/restructure` version.
  - Reason: same dataset intent, migrated schema only.

- `configs/pipeline/teaching.yaml`
  - Keep `dev/restructure` version.
  - Reason: migrated schema only.

- `configs/pipeline/trades.yaml`
  - Keep `dev/restructure` version.
  - Reason: migrated schema + updated tool composition.

- `configs/pipeline/dp.yaml`
  - Start from `dev/restructure` version, then review whether to restore `main` behavior:
    - `model.script: configs/model/config_model_resnet20_dp.py` (from main)
    - additional tool IDs from main-equivalent intent (`pre_xgbod`, `pre_watermarkbn`, `deploy_dataset_inference`, possibly `deploy_dp`).
  - Reason: this file has semantic drift beyond formatting; treat as manual reconciliation, not blind ours/theirs.

### 2) `pyproject.toml` (layout + dependencies conflict)

Use `dev/restructure` file as base, then add missing runtime dependencies still required by active code paths if needed:
- likely candidates from `main`: `torch`, `numpy`, `scipy`, `h5py`, `scikit-learn`, `torchattacks`, `filelock`, `pynvml`, `colorlogs`, `fairlearn`, `kaggle`, `opacus`.

Important: keep `dev/restructure` script entrypoints:
- `landseer-backend`
- `landseer-frontend`
- `landseer-worker`

### 3) `poetry.lock` (many conflict blocks)

Do not hand-merge lockfile.

After `pyproject.toml` is resolved:
1. delete conflicted lockfile
2. regenerate lockfile with Poetry

Example:
```bash
rm -f poetry.lock
poetry lock
```

### 4) `.gitignore`

Union both sides. Keep new ignores from each branch (including local/dev environment artifacts).

### 5) File-location conflicts (`src/...` vs renamed tree)

Git reports files added on `main` under `src/landseer_pipeline/evaluator/*` while `dev/restructure` renamed directories.

Recommended conservative approach:
- keep these files (do not drop):
  - `badnets.py`
  - `faithfullness_eval.py`
  - `mia.py`
- accept Git-suggested placement under `src_old/landseer_pipeline/evaluator/` during merge, then run import/usage scan.
- if truly unused after validation, remove in a follow-up cleanup PR.

## Execution Commands

Use a clean integration branch from `origin/main` (avoid your dirty working tree):

```bash
git fetch origin
git switch -c merge/dev-restructure-into-main origin/main
git merge --no-ff origin/dev/restructure
```

Resolve files per guidance above, then:

```bash
git add .
git status
```

## Validation Checklist

Run at minimum:

```bash
poetry check
poetry install
pytest -q tests/pipeline tests/worker tests/store
```

And specifically verify pipeline config loading for touched files:

```bash
pytest -q tests/pipeline/test_config_loader_unit.py
```

If available in this repo version, run frontend/backend smoke checks as well.

## Risk Notes

Highest risk areas:
- `configs/pipeline/dp.yaml` semantic drift
- dependency set in `pyproject.toml`
- lockfile regeneration side effects

Treat this merge as an integration PR with focused review, not a fast-forward maintenance merge.
