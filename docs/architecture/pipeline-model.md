# Pipeline Model

Pipeline config loader: `src/pipeline/config_loader.py`

## Configuration model

A pipeline config provides:

- `dataset`
- `model`
- `pipeline` stages: `pre_training`, `during_training`, `post_training`, `deployment`

## Workflow generation rules

- **Pre/post/deployment:** all permutations of tool subsets plus baseline option
- **During training:** single-tool options only plus baseline
- **Cartesian expansion:** stage options are combined into full workflows
- **Task deduplication:** same tool + config + dependencies reuses a single task id
- **Evaluation stage:** evaluator tasks attach to every workflow (if enabled)

## Implication

Workflow counts can grow quickly. Use small stage tool sets when iterating locally, and scale up progressively.
