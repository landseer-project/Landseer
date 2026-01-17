# Landseer Pipeline Infrastructure

This document describes the 3-layer pipeline infrastructure implemented for the Landseer project.

## Architecture Overview

The Landseer infrastructure is organized in three layers:

### 1. Tasks Layer
**Individual units of work that can be executed independently.**

- **Location**: `src/pipeline/tasks.py`
- **Components**:
  - Abstract `Task` base class
  - Four concrete task types:
    - `PreTrainingTask` - Pre-processing, data preparation
    - `InTrainingTask` - Training defenses, watermarking
    - `PostTrainingTask` - Post-processing, fine-tuning
    - `DeploymentTask` - Deployment defenses, runtime protections
  - `TaskFactory` - Factory pattern for creating tasks

Each task has:
- `priority`: Integer for scheduling (updated by scheduler)
- `specification`: Contains tool definition and configuration
- `dependencies`: List of other tasks that must complete first
- `task_type`: Type of task (pre/in/post/deploy)

### 2. Workflows Layer
**Sequences of tasks executed in order to achieve a larger goal.**

- **Location**: `src/pipeline/workflow.py`
- **Components**:
  - `Workflow` class - Represents one combination of tools
  - `WorkflowFactory` - Factory for creating workflows

Each workflow:
- Has a unique identifier (e.g., "comb_001")
- Contains an ordered list of tasks
- Represents one combination of tools from the pipeline stages
- Can be executed independently

### 3. Pipelines Layer
**Collections of workflows for ML defense evaluation.**

- **Location**: `src/pipeline/pipeline.py`
- **Components**:
  - Abstract `Pipeline` base class
  - `DefenseEvaluationPipeline` - Concrete implementation
  - `PipelineFactory` - Factory for creating pipelines

Each pipeline:
- Contains multiple workflows (combinations)
- Manages workflow execution
- Tracks results and status

## Tool Definitions

**Location**: `src/pipeline/tools.py`

Tools are defined using Pydantic models:
- `ContainerConfig` - Container image and command
- `ToolDefinition` - Tool name and container config
- `Specification` - Tool + configuration parameters

Tools are loaded from `configs/tools.yaml` at pipeline import time.

## Configuration Loading

**Location**: `src/pipeline/config_loader.py`

The config loader:
1. Validates pipeline configurations using Pydantic
2. Generates all tool combinations (Cartesian product across stages)
3. Creates workflows for each combination
4. Returns a ready-to-execute Pipeline instance

### Key Functions:
- `load_pipeline_config()` - Load and validate YAML config
- `make_combinations()` - Generate all tool combinations
- `create_workflow_from_combination()` - Create workflow from combination
- `create_pipeline_from_config()` - Main entry point

## Configuration Files

### Tools Configuration: `configs/tools.yaml`

Defines all available tools with their container configurations:

```yaml
tools:
  pre_noop:
    name: noop
    container:
      image: ghcr.io/landseer-project/pre_noop:v1
      command: python main.py
      runtime: null
  # ... more tools
```

### Pipeline Configuration: `configs/pipeline/*.yaml`

Defines the dataset, model, and tools for each stage:

```yaml
dataset:
  name: cifar10
  variant: clean

model:
  script: configs/model/config_model.py

pipeline:
  pre_training:
    tools:
      - name: pre-xgbod
        container:
          image: ghcr.io/landseer-project/pre_xgbod:v2
          command: python3 main.py
    noop:
      name: noop
      container:
        image: ghcr.io/landseer-project/pre_noop:v1
        command: python main.py
  
  during_training:
    # ... similar structure
  
  post_training:
    # ... similar structure
  
  deployment:
    # ... similar structure
```

## Backend Integration

**Location**: `src/backend/`

The backend initialization:
1. Loads tool registry from `tools.yaml`
2. Loads pipeline configuration
3. Creates Pipeline instance with all workflows
4. Provides API access to pipelines

### Files:
- `initialization.py` - Backend initialization logic
- `cli.py` - Command-line interface

### Usage:

```bash
# Start backend with default configuration
python -m src.backend.cli

# Start with custom pipeline
python -m src.backend.cli --config configs/pipeline/watermarknn.yaml

# Enable debug mode
python -m src.backend.cli --debug
```

## Example Usage

### Python API

```python
from src.pipeline.config_loader import create_pipeline_from_config

# Load pipeline
pipeline = create_pipeline_from_config(
    "configs/pipeline/trades.yaml",
    "configs/tools.yaml"
)

# Run all workflows
results = pipeline.run()

# Run single workflow
result = pipeline.run_single_workflow("comb_001")

# Inspect structure
print(f"Pipeline: {pipeline.name}")
print(f"Workflows: {len(pipeline.workflows)}")

for workflow in pipeline.workflows:
    print(f"\n{workflow.name}:")
    for task in workflow.tasks:
        tool = task.specification.tool.name
        task_type = task.task_type.value
        print(f"  [{task_type}] {tool}")
```

### Example Script

Run the example script to see the pipeline structure:

```bash
cd Landseer
python example_pipeline.py
```

## Design Patterns Used

1. **Factory Pattern**: Used for creating Tasks, Workflows, and Pipelines
2. **Abstract Base Class**: Task and Pipeline use ABC for extensibility
3. **Dataclasses**: Used for Task and Workflow data structures
4. **Pydantic Models**: Used for configuration validation and tool definitions
5. **Singleton Pattern**: Tool registry is global and initialized once

## Combination Generation

Combinations are generated using the Cartesian product of tools across all stages:

- For each stage, tools + noop are considered as options
- All possible combinations are generated (e.g., if 2 tools in each of 4 stages with noop, that's 3^4 = 81 combinations)
- Each combination becomes a workflow with ordered tasks

Example:
- **pre_training**: [pre-xgbod, noop]
- **during_training**: [in-trades, noop]
- **post_training**: [fine pruning]
- **deployment**: [deploy_dp, post_magnet]

Generates combinations like:
- comb_001: [pre-xgbod, in-trades, fine pruning, deploy_dp]
- comb_002: [pre-xgbod, in-trades, fine pruning, post_magnet]
- comb_003: [pre-xgbod, noop, fine pruning, deploy_dp]
- ... and so on

## Migration from Old Code

The new structure reuses concepts from the old codebase:

- **Old**: `src_old/landseer_pipeline/config/loaders.py`
  **New**: `src/pipeline/config_loader.py`

- **Old**: `src_old/landseer_pipeline/config/schemas/pipeline.py`
  **New**: `src/pipeline/config_loader.py` (Pydantic models)

- **Old**: Combination creation in runner
  **New**: `make_combinations()` in config_loader

## Future Enhancements

- [ ] Implement actual tool execution (container runner)
- [ ] Add dependency resolution and scheduling
- [ ] Implement caching and artifact management
- [ ] Add result tracking and evaluation
- [ ] Implement parallel execution
- [ ] Add web UI integration
- [ ] Implement monitoring and logging

## Testing

Example test files are in `tests/`:
- `test_artifact_cache_basic.py`
- `test_phase1_artifact_cache.py`

Add new tests for the pipeline infrastructure as needed.
