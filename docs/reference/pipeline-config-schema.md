# Pipeline Config Schema

Landseer validates pipeline YAML via Pydantic models in `src/pipeline/config_loader.py`.

## Required top-level keys

- `dataset`
- `model`
- `pipeline`

## `dataset`

- `name` (string)
- `variant` (default `clean`)
- `params` (object, optional)

## `model`

- `script` (path to model script)
- `framework` (default `pytorch`)
- `params` (object, optional)

## `pipeline`

Required stage keys:

- `pre_training`
- `during_training`
- `post_training`
- `deployment`

Each stage shape:

```yaml
stage_name:
  tools:
    - tool_id_from_tools_yaml
```

Use `configs/pipeline/trades.yaml` as a canonical example.
