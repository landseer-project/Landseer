# Landseer: Exploring the Machine Learning Defense Landscape

## Overview

Landseer composes and evaluates ML defenses across pipeline stages (pre-training, during-training, post-training, deployment) with Docker-isolated tools and an evaluation engine.

**Python:** [Poetry](https://python-poetry.org/) only (`poetry install`)
**Container tags:** use local builds—see `configs/evaluators.example.yaml` and `configs/tools/example_*.yaml` (no org registry URLs in examples).


## Layout

| Path | Role |
|------|------|
| **`src/`** | `backend/`, `worker/`, `frontend/`, `pipeline/`, `data/`, `db/`, … |
| **`configs/`** | `pipeline/`, `tools.yaml`, `evaluators.yaml`, `model/`, `attack/` |
| **`tools/`** | Vendored XGBOD / MagNet Docker + patches, **`evals/`** — see [`tools/README.md`](tools/README.md) |
| **`minimal-working demo/`** | Legacy single-worker + [`README`](minimal-working%20demo/README.md) |

## Project tree (sketch)

```
.
├── src/           # backend, worker, frontend, pipeline
├── configs/       # pipeline, tools, evaluators, model, attack
├── tools/         # xgbod_v2, MagNet, etc + evals/ — see tools/README.md
└── pyproject.toml
```

Tools consume **`/data`**, write **`/output`** and get model script from pipeline YAML.

## Install

1. `pip install poetry && poetry install` (from repo root).  
2. Docker + optional NVIDIA toolkit for GPU tools.  
3. `cp` example configs to `configs/tools.yaml`, `configs/evaluators.yaml` (see `configs/pipeline/example_*.yaml`, `configs/tools/example_*.yaml`, `configs/evaluators.example.yaml`).  
4. MySQL / registry / MinIO only if your deployment needs them.

## Run (stack)

```bash
poetry run landseer-backend --config configs/pipeline/<file>.yaml --tools-config configs/tools.yaml
poetry run landseer-worker --backend-url http://localhost:8000 --gpu 0
```

Pipeline config id: `config_<yaml_stem>` (e.g. `trades.yaml` → `config_trades`).

```bash
curl -s http://localhost:8000/api/pipeline-configs
curl -X POST "http://localhost:8000/api/pipeline-configs/config_trades/runs" \
  -H "Content-Type: application/json" -d '{"use_cache": true}'
```

If `LANDSEER_PIPELINE_KEYS` is set, send `X-Pipeline-Key`. Frontend: `poetry run landseer-frontend install && poetry run landseer-frontend dev`

**Legacy demo:** `minimal-working demo/README.md`.

## `tools/` (Docker sources)

| Path | Content |
|------|---------|
| `tools/xgbod_v2/` | pre_xgbod: Dockerfiles, patches, `LANDSEER_USAGE.md` |
| `tools/MagNet/` | post_magnet: Dockerfile, patches, `LANDSEER_USAGE.md` |
| `tools/evals/` | Evaluator images — [`tools/evals/README.md`](tools/evals/README.md) |

```bash
cd tools/xgbod_v2 && docker build -f Dockerfile-main -t pre_xgbod:artifact .
cd ../MagNet   && docker build -f Dockerfile -t post_magnet:artifact .
cd ../evals    # see tools/evals/README.md for all `docker build` lines
cp configs/evaluators.example.yaml configs/evaluators.yaml
```

Other tool ids in `configs/tools/example_registry.yaml` need images you build or provide separately.

## Config snippets

Pipeline / attack YAML shapes live under `configs/pipeline/` and `configs/attack/`; use the `example_*.yaml` files as templates.

## Results

Under `results/` (and optionally DB): combination and per-tool CSVs; see `src/backend` / run logs for details.
