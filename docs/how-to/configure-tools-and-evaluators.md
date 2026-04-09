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
