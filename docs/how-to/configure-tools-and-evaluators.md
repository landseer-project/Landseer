# Configure Tools and Evaluators

## Tool registry (`configs/tools.yaml`)

Each tool includes:

- `name`: display/tool identity
- `is_baseline`: whether this is the baseline/noop option
- `container`: image, command, runtime

Use baseline tools to keep stage options explicit and enable null-equivalent workflows.

## Pipeline config (`configs/pipeline/*.yaml`)

Define:

- dataset (`name`, `variant`, `params`)
- model (`script`, `framework`, `params`)
- tools per stage

Example starter: `configs/pipeline/trades.yaml`.

## Evaluators (`configs/evaluators.yaml`)

Evaluators are merged on top of built-in evaluator defaults.

Fields:

- `container.image`, `container.command`
- `required_artifacts`
- `metrics`
- `defense_types`

Evaluator tasks run after deployment and produce per-workflow metrics.

### `required_artifacts` (non-tool / sidecar inputs)

List relative file or directory names the evaluator expects under `/input`.

At runtime the worker ensures each name is present by checking, in order:

1. Already in `/input` (dataset copy and/or upstream tool `/output` merge)
2. Prepared dataset directory (`--data-path` / backend dataset path)
3. `artifact_roots` from `configs/evaluators.yaml` (global and/or per-evaluator)
4. `$LANDSEER_EVAL_ARTIFACTS_DIR/<name>` (optional env override)

If any required artifact is still missing, the evaluator task skips gracefully
(writes `evaluation_results.json` with `skipped: true`) and does not run the container.

**Preferred (no env var):** put sidecars on disk and declare the root in YAML:

```yaml
# configs/evaluators.yaml
artifact_roots:
  - /data/landseer/eval_artifacts

evaluators:
  clean:
    required_artifacts:
      - celeba_shadow
    # optional per-evaluator extra roots:
    # artifact_roots:
    #   - /mnt/celeba_extras
```

Then place files at `/data/landseer/eval_artifacts/celeba_shadow/...`.

**Also works without any roots config:** put the folder directly under the prepared
dataset dir (`<data_dir>/celeba/clean/celeba_shadow/...`).

Do **not** put evaluator-only folders in `datasets.yaml` `required_dirs` unless the
dataset prep container creates them; that field is only for dataset readiness.
